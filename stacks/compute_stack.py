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
    aws_cloudwatch as cw,
)
from aws_cdk import (
    aws_dynamodb as dynamodb,
)
from aws_cdk import (
    aws_emrserverless as emrs,
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
    aws_s3_deployment as s3deploy,
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
        claws_resolver_arn = self.node.try_get_context("claws_resolver_arn") or ""
        router_spend_table_arn = self.node.try_get_context("router_spend_table_arn") or ""

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
        # DynamoDB: Named Result Snapshots (Issue 19)
        # -----------------------------------------------------------------
        snapshots_table = dynamodb.Table(
            self,
            "SnapshotsTable",
            table_name=f"{prefix}-snapshots",
            partition_key=dynamodb.Attribute(
                name="user_arn", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="label", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # -----------------------------------------------------------------
        # DynamoDB: Job History
        # -----------------------------------------------------------------
        history_table = dynamodb.Table(
            self,
            "HistoryTable",
            table_name=f"{prefix}-history",
            partition_key=dynamodb.Attribute(
                name="user_arn", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="started_at", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            time_to_live_attribute="ttl",
            removal_policy=RemovalPolicy.DESTROY,
        )
        history_table.add_global_secondary_index(
            index_name="by-execution-arn",
            partition_key=dynamodb.Attribute(
                name="execution_arn", type=dynamodb.AttributeType.STRING
            ),
            projection_type=dynamodb.ProjectionType.INCLUDE,
            non_key_attributes=["cost_usd", "duration_seconds", "profile_id"],
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
            "HISTORY_TABLE": history_table.table_name,
            "SNAPSHOTS_TABLE": snapshots_table.table_name,
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
        notification_topic.grant_publish(check_budget_fn)

        # -----------------------------------------------------------------
        # Lambda: Extract (Step Functions step: QS dataset → S3 Parquet)
        # -----------------------------------------------------------------
        extract_env = {
            **common_env,
            "CLAWS_RESOLVER_ARN": claws_resolver_arn,
        }
        extract_fn = lambda_.Function(
            self,
            "Extract",
            function_name=f"{prefix}-extract",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="handler.handler",
            code=lambda_.Code.from_asset("lambdas/extract"),
            timeout=Duration.minutes(5),
            memory_size=512,
            environment=extract_env,
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
        # Allow extract Lambda to invoke the claws-resolver Lambda for claws:// URIs
        if claws_resolver_arn:
            extract_fn.add_to_role_policy(
                iam.PolicyStatement(
                    actions=["lambda:InvokeFunction"],
                    resources=[claws_resolver_arn],
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
        history_table.grant_write_data(record_spend_fn)
        snapshots_table.grant_write_data(record_spend_fn)
        record_spend_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["cloudwatch:PutMetricData"],
                resources=["*"],
            )
        )

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

        # -----------------------------------------------------------------
        # EMR Serverless (optional — enabled when enable_emr context var = true)
        # -----------------------------------------------------------------
        if enable_emr:
            emr_app = emrs.CfnApplication(
                self,
                "EmrSparkApp",
                type="SPARK",
                release_label="emr-7.1.0",
                name=f"{prefix}-spark",
                maximum_capacity=emrs.CfnApplication.MaximumAllowedResourcesProperty(
                    cpu="16 vCPU",
                    memory="64 GB",
                ),
                auto_stop_configuration=emrs.CfnApplication.AutoStopConfigurationProperty(
                    enabled=True,
                    idle_timeout_minutes=15,
                ),
            )

            emr_job_role = iam.Role(
                self,
                "EmrJobRole",
                role_name=f"{prefix}-emr-job-role",
                assumed_by=iam.ServicePrincipal("emr-serverless.amazonaws.com"),
            )
            emr_job_role.add_to_policy(
                iam.PolicyStatement(
                    actions=["s3:GetObject", "s3:ListBucket"],
                    resources=[
                        compute_bucket.bucket_arn,
                        f"{compute_bucket.bucket_arn}/inputs/*",
                        f"{compute_bucket.bucket_arn}/spark/*",
                    ],
                )
            )
            emr_job_role.add_to_policy(
                iam.PolicyStatement(
                    actions=["s3:PutObject", "s3:GetObject"],
                    resources=[f"{compute_bucket.bucket_arn}/results/*"],
                )
            )

            # Upload Spark job script to compute bucket at deploy time
            s3deploy.BucketDeployment(
                self,
                "SparkScriptDeploy",
                sources=[s3deploy.Source.asset("spark")],
                destination_bucket=compute_bucket,
                destination_key_prefix="spark",
            )

            # Grant Step Functions permission to start EMR Serverless jobs
            emr_start_role = iam.Role(
                self,
                "EmrStartRole",
                assumed_by=iam.ServicePrincipal(
                    "states.amazonaws.com",
                    conditions={"StringEquals": {"aws:SourceAccount": self.account}},
                ),
            )
            emr_start_role.add_to_policy(
                iam.PolicyStatement(
                    actions=["emr-serverless:StartJobRun", "emr-serverless:GetJobRun"],
                    resources=[f"arn:aws:emr-serverless:{self.region}:{self.account}:/applications/{emr_app.ref}/*"],
                )
            )
            emr_start_role.add_to_policy(
                iam.PolicyStatement(
                    actions=["iam:PassRole"],
                    resources=[emr_job_role.role_arn],
                )
            )

            emr_spark_task = tasks.EmrServerlessStartJobRun(
                self,
                "ComputeEmrTask",
                application_id=emr_app.ref,
                execution_role_arn=emr_job_role.role_arn,
                job_driver=tasks.JobDriver(
                    spark_submit=tasks.SparkSubmit(
                        entry_point=f"s3://{compute_bucket.bucket_name}/spark/transform.py",
                        entry_point_arguments=[
                            "--execution-id", sfn.JsonPath.string_at("$.execution_id"),
                            "--input-s3", sfn.JsonPath.string_at("$.extract.input_s3_uri"),
                            "--output-s3", sfn.JsonPath.format(
                                "s3://{}/results/{}",
                                compute_bucket.bucket_name,
                                sfn.JsonPath.string_at("$.execution_id"),
                            ),
                            "--join-keys", sfn.JsonPath.json_to_string(
                                sfn.JsonPath.list_at("$.parameters.join_keys")
                            ),
                            "--join-type", sfn.JsonPath.string_at("$.parameters.join_type"),
                            "--select-cols", sfn.JsonPath.json_to_string(
                                sfn.JsonPath.list_at("$.parameters.select_cols")
                            ),
                            "--filter-expr", sfn.JsonPath.string_at("$.parameters.filter_expr"),
                            "--additional-s3", sfn.JsonPath.json_to_string(
                                sfn.JsonPath.list_at("$.parameters.additional_s3")
                            ),
                        ],
                        spark_submit_parameters=(
                            "--conf spark.executor.cores=2 "
                            "--conf spark.executor.memory=4g "
                            "--conf spark.driver.memory=2g"
                        ),
                    )
                ),
                result_path="$.compute",
                timeout=Duration.hours(2),
            )
            emr_spark_task.add_catch(
                handler=handle_failure_task,
                errors=["States.ALL"],
                result_path="$.error_info",
            )
            emr_spark_task.next(deliver_task)

            emr_route_target = emr_spark_task
        else:
            emr_route_target = emr_not_enabled

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
            emr_route_target,
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
        run_env = {
            **common_env,
            "PROFILES_CONFIG": profiles_config_json,
            "STATE_MACHINE_ARN": state_machine.state_machine_arn,
            "MAX_CONCURRENT_JOBS_PER_USER": str(self.node.try_get_context("max_concurrent_jobs_per_user") or "2"),
        }
        if router_spend_table_arn:
            # Issue #23: cross-stack router spend table name derived from ARN
            # ARN format: arn:aws:dynamodb:region:account:table/TABLE_NAME
            run_env["ROUTER_SPEND_TABLE"] = router_spend_table_arn.split("/")[-1]

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
            environment=run_env,
        )
        state_machine.grant_start_execution(run_fn)

        # Issue #23: grant compute-run Lambda read access to the router spend table
        if router_spend_table_arn:
            run_fn.add_to_role_policy(
                iam.PolicyStatement(
                    actions=["dynamodb:Scan"],
                    resources=[router_spend_table_arn],
                )
            )
        run_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["states:ListExecutions", "states:DescribeExecution"],
                resources=[
                    state_machine.state_machine_arn,
                    state_machine.state_machine_arn.replace("stateMachine", "execution") + ":*",
                ],
            )
        )

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
        history_table.grant_read_data(status_fn)

        # -----------------------------------------------------------------
        # Lambda: Compute History (AgentCore tool: compute_history)
        # -----------------------------------------------------------------
        history_fn = lambda_.Function(
            self,
            "ComputeHistory",
            function_name=f"{prefix}-compute-history",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="handler.handler",
            code=lambda_.Code.from_asset("lambdas/compute-history"),
            timeout=Duration.seconds(10),
            memory_size=128,
            role=tool_role,
            environment={
                **common_env,
            },
        )
        history_table.grant_read_data(history_fn)

        # -----------------------------------------------------------------
        # Lambda: Compute Cancel (AgentCore tool: compute_cancel)
        # -----------------------------------------------------------------
        cancel_fn = lambda_.Function(
            self,
            "ComputeCancel",
            function_name=f"{prefix}-compute-cancel",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="handler.handler",
            code=lambda_.Code.from_asset("lambdas/compute-cancel"),
            timeout=Duration.seconds(10),
            memory_size=128,
            role=tool_role,
            environment={
                **common_env,
                "STATE_MACHINE_ARN": state_machine.state_machine_arn,
            },
        )
        cancel_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["states:StopExecution"],
                resources=[state_machine.state_machine_arn.replace(
                    "stateMachine", "execution"
                ) + ":*"],
            )
        )

        # -----------------------------------------------------------------
        # Lambda: Compute Snapshots (AgentCore tool: compute_snapshots) — Issue 19
        # -----------------------------------------------------------------
        snapshots_fn = lambda_.Function(
            self,
            "ComputeSnapshots",
            function_name=f"{prefix}-compute-snapshots",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="handler.handler",
            code=lambda_.Code.from_asset("lambdas/compute-snapshots"),
            timeout=Duration.seconds(10),
            memory_size=128,
            role=tool_role,
            environment={
                **common_env,
            },
        )
        snapshots_table.grant_read_data(snapshots_fn)

        # -----------------------------------------------------------------
        # Lambda: Compute Compare (AgentCore tool: compute_compare) — Issue 20
        # -----------------------------------------------------------------
        compare_fn = lambda_.Function(
            self,
            "ComputeCompare",
            function_name=f"{prefix}-compute-compare",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="handler.handler",
            code=lambda_.Code.from_asset("lambdas/compute-compare"),
            timeout=Duration.seconds(30),
            memory_size=256,
            role=tool_role,
            environment={
                **common_env,
            },
        )
        snapshots_table.grant_read_data(compare_fn)
        compare_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject"],
                resources=["arn:aws:s3:::*"],
            )
        )

        # -----------------------------------------------------------------
        # AgentCore Gateway invoke permissions
        # -----------------------------------------------------------------
        gateway_role_arn = self.node.try_get_context("agentcore_gateway_role_arn")
        if gateway_role_arn:
            for fn in [profiles_fn, run_fn, status_fn, history_fn, cancel_fn,
                       snapshots_fn, compare_fn]:
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
            "compute_history": history_fn.function_arn,
            "compute_cancel": cancel_fn.function_arn,
            "compute_snapshots": snapshots_fn.function_arn,
            "compute_compare": compare_fn.function_arn,
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

        # -----------------------------------------------------------------
        # CloudWatch Dashboard
        # -----------------------------------------------------------------
        dashboard = cw.Dashboard(
            self,
            "ComputeDashboard",
            dashboard_name=f"{prefix}-usage",
        )

        dashboard.add_widgets(
            cw.GraphWidget(
                title="Job Cost by Profile (USD, 24h sum)",
                left=[
                    cw.Metric(
                        namespace="QuickSuiteCompute",
                        metric_name="JobCost",
                        statistic="Sum",
                        period=Duration.hours(24),
                        label="Total Cost",
                    )
                ],
                width=12,
            ),
            cw.GraphWidget(
                title="Job Duration by Profile (seconds, p99)",
                left=[
                    cw.Metric(
                        namespace="QuickSuiteCompute",
                        metric_name="JobDuration",
                        statistic="p99",
                        period=Duration.hours(1),
                        label="p99 Duration",
                    )
                ],
                width=12,
            ),
        )

        dashboard.add_widgets(
            cw.GraphWidget(
                title="State Machine Executions",
                left=[
                    state_machine.metric_started(period=Duration.hours(1), label="Started"),
                    state_machine.metric_succeeded(period=Duration.hours(1), label="Succeeded"),
                    state_machine.metric_failed(period=Duration.hours(1), label="Failed"),
                ],
                width=12,
            ),
        )

        # Per-user spend row (compute#12)
        dashboard.add_widgets(
            cw.GraphWidget(
                title="Spend by User (USD, 24h sum)",
                left=[
                    cw.MathExpression(
                        expression=(
                            "SEARCH('{QuickSuiteCompute,UserArn}"
                            " MetricName=\"JobCost\"', 'Sum', 86400)"
                        ),
                        label="User Spend",
                        period=Duration.hours(24),
                    )
                ],
                width=12,
            ),
            cw.GraphWidget(
                title="Jobs Submitted by User (24h count)",
                left=[
                    cw.MathExpression(
                        expression=(
                            "SEARCH('{QuickSuiteCompute,UserArn}"
                            " MetricName=\"JobCost\"', 'SampleCount', 86400)"
                        ),
                        label="Job Count",
                        period=Duration.hours(24),
                    )
                ],
                width=12,
            ),
        )

        # Per-profile cost and duration rows (CP-14) + cumulative cost widget (Issue #25)
        for profile in profiles:
            profile_id = profile["profile_id"]
            display_name = profile.get("display_name", profile_id)
            dashboard.add_widgets(
                cw.GraphWidget(
                    title=f"{display_name} — Cost (USD/24h)",
                    left=[cw.Metric(
                        namespace="QuickSuiteCompute",
                        metric_name="JobCost",
                        statistic="Sum",
                        period=Duration.hours(24),
                        dimensions_map={"ProfileId": profile_id},
                        label=profile_id,
                    )],
                    width=8,
                ),
                cw.GraphWidget(
                    title=f"{display_name} — Duration (p99)",
                    left=[cw.Metric(
                        namespace="QuickSuiteCompute",
                        metric_name="JobDuration",
                        statistic="p99",
                        period=Duration.hours(1),
                        dimensions_map={"ProfileId": profile_id},
                        label=profile_id,
                    )],
                    width=8,
                ),
                # Issue #25: cumulative cost over 30 days per profile
                cw.GraphWidget(
                    title=f"{display_name} — Cumulative Cost (USD, 30d)",
                    left=[cw.Metric(
                        namespace="QuickSuiteCompute",
                        metric_name="JobCost",
                        statistic="Sum",
                        period=Duration.days(30),
                        dimensions_map={"ProfileId": profile_id},
                        label=f"{profile_id} 30d",
                    )],
                    width=8,
                ),
            )

        # -----------------------------------------------------------------
        # Outputs
        # -----------------------------------------------------------------
        CfnOutput(self, "ComputeBucketName", value=compute_bucket.bucket_name)
        CfnOutput(self, "SpendTableName", value=spend_table.table_name)
        CfnOutput(self, "HistoryTableName", value=history_table.table_name)
        CfnOutput(self, "SnapshotsTableName", value=snapshots_table.table_name)
        CfnOutput(self, "StateMachineArn", value=state_machine.state_machine_arn)
        CfnOutput(self, "NotificationTopicArn", value=notification_topic.topic_arn)
        CfnOutput(
            self,
            "DashboardUrl",
            value=(
                f"https://{self.region}.console.aws.amazon.com"
                f"/cloudwatch/home#dashboards:name={prefix}-usage"
            ),
        )
