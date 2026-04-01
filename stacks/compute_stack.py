"""
Quick Suite Compute — CDK Stack

Deploys ephemeral analytics compute as AgentCore Gateway Lambda targets.

Resources:
  - S3: compute data bucket (inputs 24h lifecycle, results 90d lifecycle)
  - DynamoDB: spend tracking table (user ARN PK, month SK)
  - SNS: job notification topic
  - Lambda Layer: scikit-learn, pandas, statsmodels, prophet, lifelines (Docker)
  - Lambda: compute-profiles, compute-run, compute-status (AgentCore targets)
  - Lambda: check-budget, extract, runner, deliver, record-spend, handle-failure
  - Step Functions: compute job state machine
  - IAM: three roles (tool Lambdas, runner Lambda, deliver Lambda)
  - CfnOutputs: tool Lambda ARNs for AgentCore Gateway registration

Build context vars (cdk deploy --context key=value):
  enable_emr         false    Enable EMR Serverless for transform-spark profile
  monthly_budget_usd 50       Per-user monthly compute budget ceiling (USD)
  notification_email ""       SNS email subscription for job notifications
  agentcore_gateway_role_arn  Gateway execution role ARN for invoke permissions
"""

import json
from pathlib import Path

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
)
from aws_cdk import (
    aws_dynamodb as dynamodb,
)
from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_lambda as lambda_,
)
from aws_cdk import (
    aws_s3 as s3,
)
from aws_cdk import (
    aws_sns as sns,
)
from aws_cdk import (
    aws_sns_subscriptions as sns_subscriptions,
)
from aws_cdk import (
    aws_stepfunctions as sfn,
)
from aws_cdk import (
    aws_stepfunctions_tasks as tasks,
)
from constructs import Construct


def _load_profiles(config_dir: Path) -> list[dict]:
    """Load all profile JSON files from config/profiles/."""
    profiles = []
    profiles_dir = config_dir / "profiles"
    if profiles_dir.exists():
        for p in sorted(profiles_dir.glob("*.json")):
            with open(p) as f:
                profiles.append(json.load(f))
    return profiles


class ComputeStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        prefix = "qs-compute"
        account_id = self.account
        region = self.region
        qs_region = self.node.try_get_context("quicksight_region") or region
        qs_user = self.node.try_get_context("quicksight_user") or "Admin"
        enable_emr = bool(self.node.try_get_context("enable_emr"))
        monthly_budget_usd = int(self.node.try_get_context("monthly_budget_usd") or 50)
        notification_email = self.node.try_get_context("notification_email") or ""

        config_dir = Path(__file__).parent.parent / "config"
        profiles = _load_profiles(config_dir)
        profiles_config_json = json.dumps(profiles)

        # -----------------------------------------------------------------
        # S3: Compute Data Bucket
        # -----------------------------------------------------------------
        compute_bucket = s3.Bucket(
            self,
            "ComputeBucket",
            bucket_name=f"{prefix}-{account_id}-{region}",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="DeleteInputs",
                    prefix="inputs/",
                    expiration=Duration.days(1),
                ),
                s3.LifecycleRule(
                    id="ExpireResults",
                    prefix="results/",
                    expiration=Duration.days(90),
                ),
            ],
        )

        # -----------------------------------------------------------------
        # DynamoDB: Spend Tracking
        # -----------------------------------------------------------------
        spend_table = dynamodb.Table(
            self,
            "SpendTable",
            table_name=f"{prefix}-spend",
            partition_key=dynamodb.Attribute(
                name="user_arn", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="month", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # -----------------------------------------------------------------
        # SNS: Job Notification Topic
        # -----------------------------------------------------------------
        notification_topic = sns.Topic(
            self,
            "NotificationTopic",
            topic_name=f"{prefix}-notifications",
            display_name="Quick Suite Compute Job Notifications",
        )

        if notification_email:
            notification_topic.add_subscription(
                sns_subscriptions.EmailSubscription(notification_email)
            )

        # -----------------------------------------------------------------
        # Lambda Layer: Heavy Analytics Dependencies (Docker bundled)
        # -----------------------------------------------------------------
        analytics_layer = lambda_.LayerVersion(
            self,
            "AnalyticsLayer",
            code=lambda_.Code.from_docker_build(
                path="lambdas/layer",
                build_args={},
            ),
            compatible_runtimes=[lambda_.Runtime.PYTHON_3_12],
            description=(
                "scikit-learn, pandas, statsmodels, prophet, lifelines, "
                "scipy, pyarrow for Quick Suite Compute runner profiles"
            ),
        )

        # Lambda Layer: Profile modules (clustering, regression, forecast, etc.)
        # Runner dispatches via importlib.import_module(profile.entrypoint.split(".")[0])
        # so these modules must be on sys.path via a Layer.
        profiles_layer = lambda_.LayerVersion(
            self,
            "ProfilesLayer",
            code=lambda_.Code.from_asset("lambdas/profiles"),
            compatible_runtimes=[lambda_.Runtime.PYTHON_3_12],
            description="Quick Suite Compute per-profile analysis modules",
        )

        # -----------------------------------------------------------------
        # IAM Role 1: Tool Lambdas (compute_profiles, compute_run, compute_status)
        # -----------------------------------------------------------------
        tool_role = iam.Role(
            self,
            "ToolLambdaRole",
            role_name=f"{prefix}-tool-role",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                )
            ],
        )
        spend_table.grant_read_write_data(tool_role)

        # -----------------------------------------------------------------
        # IAM Role 2: Runner Lambda (inside Step Functions)
        # -----------------------------------------------------------------
        runner_role = iam.Role(
            self,
            "RunnerLambdaRole",
            role_name=f"{prefix}-runner-role",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                )
            ],
        )
        runner_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject"],
                resources=[f"{compute_bucket.bucket_arn}/inputs/*"],
            )
        )
        runner_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:PutObject"],
                resources=[f"{compute_bucket.bucket_arn}/results/*"],
            )
        )
        # Census Bureau API access (geo-enrich profile) — outbound HTTPS only
        runner_role.add_to_policy(
            iam.PolicyStatement(
                actions=["ec2:CreateNetworkInterface",
                         "ec2:DescribeNetworkInterfaces",
                         "ec2:DeleteNetworkInterface"],
                resources=["*"],
            )
        )

        # -----------------------------------------------------------------
        # IAM Role 3: Deliver Lambda
        # -----------------------------------------------------------------
        deliver_role = iam.Role(
            self,
            "DeliverLambdaRole",
            role_name=f"{prefix}-deliver-role",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                )
            ],
        )
        compute_bucket.grant_read_write(deliver_role)
        notification_topic.grant_publish(deliver_role)
        deliver_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "quicksight:CreateDataSource",
                    "quicksight:CreateDataSet",
                    "quicksight:DescribeDataSource",
                    "quicksight:DescribeDataSet",
                    "quicksight:UpdateDataSet",
                    "quicksight:PassDataSource",
                ],
                resources=[f"arn:aws:quicksight:{qs_region}:{account_id}:*"],
            )
        )

        # -----------------------------------------------------------------
        # Common Lambda env vars
        # -----------------------------------------------------------------
        common_env = {
            "COMPUTE_BUCKET": compute_bucket.bucket_name,
            "SPEND_TABLE": spend_table.table_name,
            "NOTIFICATION_TOPIC_ARN": notification_topic.topic_arn,
            "QUICKSIGHT_ACCOUNT_ID": account_id,
            "QUICKSIGHT_REGION": qs_region,
            "QUICKSIGHT_USER": qs_user,
            "MONTHLY_BUDGET_USD": str(monthly_budget_usd),
            "ENABLE_EMR": "true" if enable_emr else "false",
        }

        # -----------------------------------------------------------------
        # Lambda: Check Budget (Step Functions step)
        # -----------------------------------------------------------------
        check_budget_fn = lambda_.Function(
            self,
            "CheckBudget",
            function_name=f"{prefix}-check-budget",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="handler.handler",
            code=lambda_.Code.from_asset("lambdas/check-budget"),
            timeout=Duration.seconds(10),
            memory_size=128,
            environment=common_env,
        )
        spend_table.grant_read_data(check_budget_fn)

        # -----------------------------------------------------------------
        # Lambda: Extract (Step Functions step: QS dataset → S3 Parquet)
        # -----------------------------------------------------------------
        extract_fn = lambda_.Function(
            self,
            "Extract",
            function_name=f"{prefix}-extract",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="handler.handler",
            code=lambda_.Code.from_asset("lambdas/extract"),
            timeout=Duration.minutes(5),
            memory_size=512,
            environment=common_env,
        )
        compute_bucket.grant_write(extract_fn)
        extract_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["quicksight:DescribeDataSet", "quicksight:DescribeDataSource"],
                resources=[f"arn:aws:quicksight:{qs_region}:{account_id}:*"],
            )
        )
        # Extract Lambda needs S3 read access to manifest and source data buckets.
        # The specific buckets are not known at synth time (they come from QS dataset
        # configuration), so grant read on all buckets. Scope further post-deploy
        # by updating the IAM policy with specific bucket ARNs.
        extract_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject"],
                resources=["arn:aws:s3:::*"],
            )
        )

        # -----------------------------------------------------------------
        # Lambda: Runner (Step Functions step: dispatches to profile modules)
        # -----------------------------------------------------------------
        runner_fn = lambda_.Function(
            self,
            "Runner",
            function_name=f"{prefix}-runner",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="handler.handler",
            code=lambda_.Code.from_asset("lambdas/runner"),
            layers=[analytics_layer, profiles_layer],
            timeout=Duration.minutes(15),
            memory_size=3008,
            role=runner_role,
            environment=common_env,
        )

        # -----------------------------------------------------------------
        # Lambda: Deliver (Step Functions step: S3 → QS dataset)
        # -----------------------------------------------------------------
        deliver_fn = lambda_.Function(
            self,
            "Deliver",
            function_name=f"{prefix}-deliver",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="handler.handler",
            code=lambda_.Code.from_asset("lambdas/deliver"),
            timeout=Duration.minutes(2),
            memory_size=256,
            role=deliver_role,
            environment={
                **common_env,
                "MANIFEST_BUCKET": compute_bucket.bucket_name,
            },
        )

        # -----------------------------------------------------------------
        # Lambda: Record Spend (Step Functions step)
        # -----------------------------------------------------------------
        record_spend_fn = lambda_.Function(
            self,
            "RecordSpend",
            function_name=f"{prefix}-record-spend",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="handler.handler",
            code=lambda_.Code.from_asset("lambdas/record-spend"),
            timeout=Duration.seconds(10),
            memory_size=128,
            environment=common_env,
        )
        spend_table.grant_write_data(record_spend_fn)

        # -----------------------------------------------------------------
        # Lambda: Handle Failure (Step Functions catch)
        # -----------------------------------------------------------------
        handle_failure_fn = lambda_.Function(
            self,
            "HandleFailure",
            function_name=f"{prefix}-handle-failure",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="handler.handler",
            code=lambda_.Code.from_asset("lambdas/handle-failure"),
            timeout=Duration.seconds(10),
            memory_size=128,
            environment=common_env,
        )

        # -----------------------------------------------------------------
        # Step Functions: Compute Job State Machine
        # -----------------------------------------------------------------
        check_budget_task = tasks.LambdaInvoke(
            self,
            "CheckBudgetTask",
            lambda_function=check_budget_fn,
            result_selector={"budget_ok.$": "$.Payload.budget_ok",
                             "spend_usd.$": "$.Payload.spend_usd",
                             "user_arn.$": "$.Payload.user_arn"},
            result_path="$.budget",
            payload=sfn.TaskInput.from_object({
                "user_arn.$": "$.user_arn",
                "estimated_cost_usd.$": "$.profile.cost_estimate.typical_cost_usd",
            }),
        )

        budget_exceeded = sfn.Fail(
            self,
            "BudgetExceeded",
            cause="Monthly compute budget exceeded",
            error="BudgetExceeded",
        )

        extract_task = tasks.LambdaInvoke(
            self,
            "ExtractDatasetTask",
            lambda_function=extract_fn,
            result_path="$.extract",
            payload_response_only=True,
            timeout=Duration.minutes(5),
        )

        compute_lambda_task = tasks.LambdaInvoke(
            self,
            "ComputeLambdaTask",
            lambda_function=runner_fn,
            result_path="$.compute",
            payload_response_only=True,
            timeout=Duration.minutes(15),
            retry_on_service_exceptions=False,
        )
        compute_lambda_task.add_retry(
            errors=["Lambda.TooManyRequestsException"],
            max_attempts=2,
            interval=Duration.seconds(5),
            backoff_rate=2,
        )

        handle_failure_task = tasks.LambdaInvoke(
            self,
            "HandleFailureTask",
            lambda_function=handle_failure_fn,
            result_path="$.failure",
            payload_response_only=True,
        )

        notify_failure = tasks.SnsPublish(
            self,
            "NotifyFailure",
            topic=notification_topic,
            subject=sfn.JsonPath.string_at("States.Format('Analysis job failed: {}', $.profile.display_name)"),
            message=sfn.TaskInput.from_json_path_at(
                "States.Format('Your {} analysis could not be completed. Error: {}', $.profile.display_name, $.failure.error_message)"
            ),
        )
        handle_failure_task.next(notify_failure)

        compute_lambda_task.add_catch(
            handler=handle_failure_task,
            errors=["States.ALL"],
            result_path="$.error_info",
        )

        deliver_task = tasks.LambdaInvoke(
            self,
            "DeliverResultsTask",
            lambda_function=deliver_fn,
            result_path="$.deliver",
            payload_response_only=True,
            timeout=Duration.minutes(2),
        )
        deliver_task.add_catch(
            handler=handle_failure_task,
            errors=["States.ALL"],
            result_path="$.error_info",
        )

        record_spend_task = tasks.LambdaInvoke(
            self,
            "RecordSpendTask",
            lambda_function=record_spend_fn,
            result_path="$.spend",
            payload_response_only=True,
        )

        notify_success = tasks.SnsPublish(
            self,
            "NotifyUser",
            topic=notification_topic,
            subject=sfn.JsonPath.string_at(
                "States.Format('Your {} analysis is ready', $.profile.display_name)"
            ),
            message=sfn.TaskInput.from_json_path_at(
                "States.Format('Your {} analysis is complete. Results are available as a new dataset in Quick Sight: {}', $.profile.display_name, $.deliver.result_dataset_name)"
            ),
        )

        emr_not_enabled = sfn.Fail(
            self,
            "EmrNotEnabled",
            cause="EMR Serverless is not enabled in this deployment",
            error="EmrNotEnabled",
        )

        route_compute = sfn.Choice(self, "RouteCompute")
        route_compute.when(
            sfn.Condition.string_equals("$.profile.backend", "lambda"),
            compute_lambda_task,
        )
        route_compute.when(
            sfn.Condition.and_(
                sfn.Condition.string_equals("$.profile.backend", "emr_serverless"),
                sfn.Condition.boolean_equals("$.enable_emr", True),
            ),
            # Placeholder: real EMR Serverless task would go here
            # For initial build, route to handle-failure with requires_emr status
            handle_failure_task,
        )
        route_compute.otherwise(emr_not_enabled)

        budget_decision = sfn.Choice(self, "BudgetDecision")
        budget_decision.when(
            sfn.Condition.boolean_equals("$.budget.budget_ok", False),
            budget_exceeded,
        )
        budget_decision.otherwise(extract_task)

        chain = (
            check_budget_task
            .next(budget_decision)
        )

        extract_task.add_catch(
            handler=handle_failure_task,
            errors=["States.ALL"],
            result_path="$.error_info",
        )
        extract_task.next(route_compute)
        compute_lambda_task.next(deliver_task)
        deliver_task.next(record_spend_task)
        record_spend_task.next(notify_success)

        state_machine = sfn.StateMachine(
            self,
            "ComputeStateMachine",
            state_machine_name=f"{prefix}-job",
            definition_body=sfn.DefinitionBody.from_chainable(chain),
            timeout=Duration.hours(4),
        )

        # -----------------------------------------------------------------
        # Lambda: Compute Profiles (AgentCore tool: compute_profiles)
        # -----------------------------------------------------------------
        profiles_fn = lambda_.Function(
            self,
            "ComputeProfiles",
            function_name=f"{prefix}-compute-profiles",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="handler.handler",
            code=lambda_.Code.from_asset("lambdas/compute-profiles"),
            timeout=Duration.seconds(10),
            memory_size=128,
            role=tool_role,
            environment={
                **common_env,
                "PROFILES_CONFIG": profiles_config_json,
            },
        )

        # -----------------------------------------------------------------
        # Lambda: Compute Run (AgentCore tool: compute_run)
        # -----------------------------------------------------------------
        run_fn = lambda_.Function(
            self,
            "ComputeRun",
            function_name=f"{prefix}-compute-run",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="handler.handler",
            code=lambda_.Code.from_asset("lambdas/compute-run"),
            timeout=Duration.seconds(30),
            memory_size=256,
            role=tool_role,
            environment={
                **common_env,
                "PROFILES_CONFIG": profiles_config_json,
                "STATE_MACHINE_ARN": state_machine.state_machine_arn,
            },
        )
        state_machine.grant_start_execution(run_fn)

        # -----------------------------------------------------------------
        # Lambda: Compute Status (AgentCore tool: compute_status)
        # -----------------------------------------------------------------
        status_fn = lambda_.Function(
            self,
            "ComputeStatus",
            function_name=f"{prefix}-compute-status",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="handler.handler",
            code=lambda_.Code.from_asset("lambdas/compute-status"),
            timeout=Duration.seconds(10),
            memory_size=128,
            role=tool_role,
            environment={
                **common_env,
                "STATE_MACHINE_ARN": state_machine.state_machine_arn,
            },
        )
        status_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["states:DescribeExecution", "states:GetExecutionHistory"],
                resources=[state_machine.state_machine_arn.replace(
                    "stateMachine", "execution"
                ) + ":*"],
            )
        )

        # -----------------------------------------------------------------
        # AgentCore Gateway invoke permissions
        # -----------------------------------------------------------------
        gateway_role_arn = self.node.try_get_context("agentcore_gateway_role_arn")
        if gateway_role_arn:
            for fn in [profiles_fn, run_fn, status_fn]:
                fn.add_permission(
                    "AgentCoreInvoke",
                    principal=iam.ArnPrincipal(gateway_role_arn),
                    action="lambda:InvokeFunction",
                )

        # -----------------------------------------------------------------
        # CloudFormation Outputs
        # -----------------------------------------------------------------
        tool_arns = {
            "compute_profiles": profiles_fn.function_arn,
            "compute_run": run_fn.function_arn,
            "compute_status": status_fn.function_arn,
        }

        for tool_name, arn_value in tool_arns.items():
            CfnOutput(
                self,
                f"{tool_name.title().replace('_', '')}Arn",
                value=arn_value,
                description=f"Register as AgentCore Gateway Lambda target: {tool_name}",
                export_name=f"qs-compute-{tool_name.replace('_', '-')}-arn",
            )

        CfnOutput(
            self,
            "ToolArns",
            value=json.dumps(tool_arns),
            description="All tool Lambda ARNs — register each as AgentCore Gateway Lambda target",
        )

        CfnOutput(self, "ComputeBucketName", value=compute_bucket.bucket_name)
        CfnOutput(self, "SpendTableName", value=spend_table.table_name)
        CfnOutput(self, "StateMachineArn", value=state_machine.state_machine_arn)
        CfnOutput(self, "NotificationTopicArn", value=notification_topic.topic_arn)
