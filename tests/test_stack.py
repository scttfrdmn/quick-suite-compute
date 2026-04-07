"""
CDK stack synthesis test for quick-suite-compute.

Validates that the stack synthesizes without errors and that key
CloudFormation resources are present.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


try:
    import aws_cdk as cdk
    from aws_cdk.assertions import Template

    from stacks.compute_stack import ComputeStack
    _CDK_AVAILABLE = True
except ImportError:
    _CDK_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _CDK_AVAILABLE, reason="aws-cdk-lib not installed")


@pytest.fixture(scope="module")
def template():
    app = cdk.App(context={
        "enable_emr": False,
        "monthly_budget_usd": 50,
        "notification_email": "",
    })
    stack = ComputeStack(app, "TestComputeStack")
    return Template.from_stack(stack)


class TestComputeStackSynthesis:
    def test_s3_bucket_created(self, template):
        template.resource_count_is("AWS::S3::Bucket", 1)

    def test_dynamodb_tables_created(self, template):
        # spend, history, snapshots, schedules (#56)
        template.resource_count_is("AWS::DynamoDB::Table", 4)

    def test_sns_topic_created(self, template):
        template.resource_count_is("AWS::SNS::Topic", 1)

    def test_state_machine_created(self, template):
        template.resource_count_is("AWS::StepFunctions::StateMachine", 1)

    def test_tool_lambdas_created(self, template):
        # compute-profiles, compute-run, compute-status, compute-history, compute-cancel,
        # compute-snapshots, compute-compare, compute-schedule (#56)
        # + check-budget, extract, runner, deliver, record-spend, handle-failure,
        #   audit-log, schedule-trigger (#56) = 16
        lambdas = template.find_resources("AWS::Lambda::Function")
        # Filter out CDK-internal auto-delete-objects Lambda
        compute_lambdas = {
            k: v for k, v in lambdas.items()
            if "qs-compute" in json.dumps(v.get("Properties", {}).get("FunctionName", ""))
        }
        assert len(compute_lambdas) >= 11

    def test_outputs_include_tool_arns(self, template):
        outputs = template.find_outputs("*")
        output_keys = set(outputs.keys())
        assert "ComputeprofilesArn" in output_keys or any(
            "compute" in k.lower() for k in output_keys
        )
        assert "ToolArns" in output_keys

    def test_state_machine_name_in_outputs(self, template):
        template.has_output("StateMachineArn", {})

    def test_iam_roles_created(self, template):
        # tool-role, runner-role, deliver-role, sfn-role (#74), scheduler-invoke-role (#56)
        roles = template.find_resources("AWS::IAM::Role")
        named = {
            k: v for k, v in roles.items()
            if "qs-compute" in json.dumps(v.get("Properties", {}).get("RoleName", ""))
        }
        assert len(named) >= 5

    def test_dashboard_has_cumulative_cost_widget(self, template):
        """Issue #25: each profile gets a 30-day cumulative cost widget in the dashboard."""
        import json as _json
        from pathlib import Path as _Path
        profiles_dir = _Path(__file__).parent.parent / "config" / "profiles"
        profiles = [_json.load(open(p)) for p in sorted(profiles_dir.glob("*.json"))]
        assert len(profiles) > 0, "No profiles found in config/profiles/"
        dashboards = template.find_resources("AWS::CloudWatch::Dashboard")
        assert len(dashboards) >= 1
        dashboard_body = _json.dumps(list(dashboards.values())[0])
        # Verify at least one profile's cumulative cost widget title is present
        assert "Cumulative Cost" in dashboard_body

    def test_run_lambda_env_has_router_spend_table_when_context_set(self):
        """Issue #23: ROUTER_SPEND_TABLE env var is set on compute-run when context var provided."""
        import aws_cdk as cdk
        from aws_cdk.assertions import Template

        from stacks.compute_stack import ComputeStack
        app = cdk.App(context={
            "enable_emr": False,
            "monthly_budget_usd": 50,
            "notification_email": "",
            "router_spend_table_arn": "arn:aws:dynamodb:us-east-1:123456789012:table/qs-router-spend",
        })
        stack = ComputeStack(app, "TestComputeStackRouterSpend")
        tpl = Template.from_stack(stack)
        lambdas = tpl.find_resources("AWS::Lambda::Function")
        run_lambdas = {
            k: v for k, v in lambdas.items()
            if "qs-compute-compute-run" in json.dumps(v.get("Properties", {}).get("FunctionName", ""))
        }
        assert len(run_lambdas) >= 1
        run_env_vars = list(run_lambdas.values())[0]["Properties"]["Environment"]["Variables"]
        assert "ROUTER_SPEND_TABLE" in run_env_vars

    def test_audit_log_lambda_present(self, template):
        """Issue #28: audit-log Lambda is synthesized in the stack."""
        lambdas = template.find_resources("AWS::Lambda::Function")
        audit_lambdas = {
            k: v for k, v in lambdas.items()
            if "audit-log" in json.dumps(v.get("Properties", {}).get("FunctionName", ""))
        }
        assert len(audit_lambdas) >= 1

    def test_state_machine_definition_includes_audit_log(self, template):
        """Issue #28: state machine definition references the audit-log Lambda."""
        state_machines = template.find_resources("AWS::StepFunctions::StateMachine")
        sm_body = json.dumps(list(state_machines.values())[0])
        assert "AuditLog" in sm_body

    def test_enable_vpc_synthesizes_vpc_resources(self):
        """Issue #26: enable_vpc=true produces a VPC and VPC Gateway endpoint."""
        import aws_cdk as cdk
        from aws_cdk.assertions import Template

        from stacks.compute_stack import ComputeStack
        app = cdk.App(context={
            "enable_vpc": True,
            "enable_emr": False,
            "monthly_budget_usd": 50,
            "notification_email": "",
        })
        stack = ComputeStack(app, "TestComputeStackVpc")
        tpl = Template.from_stack(stack)
        tpl.resource_count_is("AWS::EC2::VPC", 1)
        # S3 Gateway endpoint appears as AWS::EC2::VPCEndpoint
        tpl.resource_count_is("AWS::EC2::VPCEndpoint", 1)

    def test_enable_kms_synthesizes_kms_keys(self):
        """Issue #27: enable_kms=true produces KMS keys for HistoryTable and bucket."""
        import aws_cdk as cdk
        from aws_cdk.assertions import Template

        from stacks.compute_stack import ComputeStack
        app = cdk.App(context={
            "enable_kms": True,
            "enable_emr": False,
            "monthly_budget_usd": 50,
            "notification_email": "",
        })
        stack = ComputeStack(app, "TestComputeStackKms")
        tpl = Template.from_stack(stack)
        keys = tpl.find_resources("AWS::KMS::Key")
        assert len(keys) >= 2  # one for HistoryTable, one for bucket

    def test_explicit_sfn_role_present(self, template):
        """Issue #74: state machine has an explicit IAM role named qs-compute-sfn-role."""
        roles = template.find_resources("AWS::IAM::Role")
        sfn_roles = {
            k: v for k, v in roles.items()
            if "qs-compute-sfn-role" in json.dumps(v.get("Properties", {}).get("RoleName", ""))
        }
        assert len(sfn_roles) >= 1

    def test_state_machine_definition_includes_chain_states(self, template):
        """Issue #55: state machine definition includes HasChainProfile and PrepareChainInput."""
        state_machines = template.find_resources("AWS::StepFunctions::StateMachine")
        sm_body = json.dumps(list(state_machines.values())[0])
        assert "HasChainProfile" in sm_body
        assert "PrepareChainInput" in sm_body

    def test_scheduler_group_present(self, template):
        """Issue #56: EventBridge Scheduler group is created."""
        groups = template.find_resources("AWS::Scheduler::ScheduleGroup")
        assert len(groups) >= 1

    def test_schedules_table_present(self, template):
        """Issue #56: DynamoDB schedules table is created."""
        tables = template.find_resources("AWS::DynamoDB::Table")
        tables_body = json.dumps(tables)
        assert "qs-compute-schedules" in tables_body

    def test_schedule_lambda_present(self, template):
        """Issue #56: compute-schedule AgentCore tool Lambda is created."""
        lambdas = template.find_resources("AWS::Lambda::Function")
        schedule_lambdas = {
            k: v for k, v in lambdas.items()
            if "compute-schedule" in json.dumps(v.get("Properties", {}).get("FunctionName", ""))
            and "trigger" not in json.dumps(v.get("Properties", {}).get("FunctionName", ""))
        }
        assert len(schedule_lambdas) >= 1

    def test_schedule_trigger_lambda_present(self, template):
        """Issue #56: schedule-trigger internal Lambda is created."""
        lambdas = template.find_resources("AWS::Lambda::Function")
        trigger_lambdas = {
            k: v for k, v in lambdas.items()
            if "schedule-trigger" in json.dumps(v.get("Properties", {}).get("FunctionName", ""))
        }
        assert len(trigger_lambdas) >= 1

    def test_emr_results_policy_has_only_put_object(self):
        """Issue #77: EMR Serverless job role must NOT have s3:GetObject on results/*.

        The EMR Serverless CDK construct (EmrServerlessStartJobRun) is not available
        in the installed CDK version, so we cannot synthesize enable_emr=True.
        Instead, verify the stack source directly: the policy statement for results/*
        must only list s3:PutObject.
        """
        import ast
        stack_src = (REPO_ROOT / "stacks" / "compute_stack.py").read_text()
        tree = ast.parse(stack_src)
        # Walk all list literals that contain "results/*" as part of a string
        # and verify none of them include "s3:GetObject" alongside "s3:PutObject".
        # Simpler: check that GetObject never appears on the same line as results/*
        lines = stack_src.splitlines()
        results_block_lines: list[int] = []
        for i, line in enumerate(lines):
            if "results/*" in line:
                results_block_lines.append(i)

        assert results_block_lines, "Expected to find 'results/*' in compute_stack.py"

        for lineno in results_block_lines:
            # Grab a small window around the results/* reference
            window_start = max(0, lineno - 5)
            window_end = min(len(lines), lineno + 5)
            window = "\n".join(lines[window_start:window_end])
            assert "s3:GetObject" not in window, (
                f"s3:GetObject must not appear alongside results/* (lines {window_start}–{window_end}):\n{window}"
            )

    def test_router_invoke_arn_grants_runner_lambda_invoke_when_set(self):
        """Issue #74: runner Lambda role gets lambda:InvokeFunction on router ARN."""
        import aws_cdk as cdk
        from aws_cdk.assertions import Match, Template

        from stacks.compute_stack import ComputeStack
        router_arn = "arn:aws:lambda:us-east-1:123456789012:function:qs-router"
        app = cdk.App(context={
            "enable_emr": False,
            "monthly_budget_usd": 50,
            "notification_email": "",
            "router_invoke_arn": router_arn,
        })
        stack = ComputeStack(app, "TestComputeStackRouterInvoke")
        tpl = Template.from_stack(stack)
        # The runner Lambda role should have a policy statement with lambda:InvokeFunction
        tpl.has_resource_properties(
            "AWS::IAM::Policy",
            {
                "PolicyDocument": {
                    "Statement": Match.array_with([
                        Match.object_like({
                            "Action": "lambda:InvokeFunction",
                            "Effect": "Allow",
                            "Resource": router_arn,
                        })
                    ])
                }
            }
        )
