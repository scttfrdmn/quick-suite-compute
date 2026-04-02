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
        template.resource_count_is("AWS::DynamoDB::Table", 2)

    def test_sns_topic_created(self, template):
        template.resource_count_is("AWS::SNS::Topic", 1)

    def test_state_machine_created(self, template):
        template.resource_count_is("AWS::StepFunctions::StateMachine", 1)

    def test_tool_lambdas_created(self, template):
        # compute-profiles, compute-run, compute-status
        # + check-budget, extract, runner, deliver, record-spend, handle-failure = 9
        lambdas = template.find_resources("AWS::Lambda::Function")
        # Filter out CDK-internal auto-delete-objects Lambda
        compute_lambdas = {
            k: v for k, v in lambdas.items()
            if "qs-compute" in json.dumps(v.get("Properties", {}).get("FunctionName", ""))
        }
        assert len(compute_lambdas) >= 9

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
        # tool-role, runner-role, deliver-role + CDK service roles
        roles = template.find_resources("AWS::IAM::Role")
        # At minimum our 3 named roles
        named = {
            k: v for k, v in roles.items()
            if "qs-compute" in json.dumps(v.get("Properties", {}).get("RoleName", ""))
        }
        assert len(named) >= 3
