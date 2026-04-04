"""
E2E conftest for quick-suite-compute.

Runs against a deployed QuickSuiteCompute CloudFormation stack.
All tests skip automatically when the stack is not deployed or credentials are absent.

Required environment:
  AWS_PROFILE=aws  (or other standard AWS credential env vars)

Optional environment:
  QS_E2E_COMPUTE_STACK_NAME   CloudFormation stack name (default: QuickSuiteCompute)
  QS_E2E_REGION               AWS region (default: us-west-2)

Run:
  AWS_PROFILE=aws python3 -m pytest tests/e2e/ -v -m e2e
"""

import json
import os

import boto3
import pytest
from botocore.config import Config

STACK_NAME = os.environ.get("QS_E2E_COMPUTE_STACK_NAME", "QuickSuiteCompute")
REGION = os.environ.get("QS_E2E_REGION", "us-west-2")
_AWS_PROFILE = os.environ.get("AWS_PROFILE")

# Minimal CSV uploaded as test input for compute_run
_TEST_CSV = """x,y,z
1.0,2.0,3.0
4.0,5.0,6.0
7.0,8.0,9.0
10.0,11.0,12.0
20.0,22.0,24.0
30.0,33.0,36.0
40.0,44.0,48.0
50.0,55.0,60.0
60.0,66.0,72.0
70.0,77.0,84.0
80.0,88.0,96.0
90.0,99.0,108.0
100.0,110.0,120.0
110.0,121.0,132.0
120.0,132.0,144.0
130.0,143.0,156.0
140.0,154.0,168.0
150.0,165.0,180.0
160.0,176.0,192.0
170.0,187.0,204.0
180.0,198.0,216.0
190.0,209.0,228.0
200.0,220.0,240.0
210.0,231.0,252.0
220.0,242.0,264.0
230.0,253.0,276.0
240.0,264.0,288.0
250.0,275.0,300.0
260.0,286.0,312.0
270.0,297.0,324.0
280.0,308.0,336.0
290.0,319.0,348.0
300.0,330.0,360.0
310.0,341.0,372.0
320.0,352.0,384.0
"""

_E2E_USER_ARN = "arn:aws:iam::942542972736:user/e2e-test-user"
_TEST_EXECUTION_ID_PREFIX = "e2e-test-"


def _session() -> boto3.Session:
    if _AWS_PROFILE:
        return boto3.Session(profile_name=_AWS_PROFILE, region_name=REGION)
    return boto3.Session(region_name=REGION)


def invoke(lam_client, function_name: str, payload: dict) -> dict:
    """Invoke a Lambda and return the parsed JSON response."""
    resp = lam_client.invoke(
        FunctionName=function_name,
        InvocationType="RequestResponse",
        Payload=json.dumps(payload).encode(),
    )
    raw = resp["Payload"].read()
    if resp.get("FunctionError"):
        pytest.fail(f"Lambda {function_name} returned FunctionError: {raw.decode()}")
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Session-scoped AWS clients
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def lam():
    return _session().client("lambda", config=Config(read_timeout=300, connect_timeout=10))


@pytest.fixture(scope="session")
def s3_client():
    return _session().client("s3")


@pytest.fixture(scope="session")
def sfn_client():
    return _session().client("stepfunctions")


# ---------------------------------------------------------------------------
# CloudFormation outputs
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def cfn_outputs() -> dict[str, str]:
    cfn = _session().client("cloudformation")
    try:
        resp = cfn.describe_stacks(StackName=STACK_NAME)
    except Exception as exc:
        pytest.skip(
            f"Stack '{STACK_NAME}' not found or no AWS credentials "
            f"(set AWS_PROFILE=aws): {exc}"
        )
    raw = resp["Stacks"][0].get("Outputs", [])
    return {o["OutputKey"]: o["OutputValue"] for o in raw}


@pytest.fixture(scope="session")
def tool_arns(cfn_outputs) -> dict[str, str]:
    return json.loads(cfn_outputs["ToolArns"])


@pytest.fixture(scope="session")
def compute_bucket_name(cfn_outputs) -> str:
    return cfn_outputs["ComputeBucketName"]


@pytest.fixture(scope="session")
def state_machine_arn(cfn_outputs) -> str:
    return cfn_outputs["StateMachineArn"]


@pytest.fixture(scope="session")
def history_table_name(cfn_outputs) -> str:
    return cfn_outputs["HistoryTableName"]


@pytest.fixture(scope="session")
def snapshots_table_name(cfn_outputs) -> str:
    return cfn_outputs["SnapshotsTableName"]


# ---------------------------------------------------------------------------
# Test input CSV uploaded to compute bucket
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_input_s3_uri(request, s3_client, compute_bucket_name) -> str:
    """Upload a small test CSV to the compute bucket and return its S3 URI.
    Deleted on session teardown."""
    key = "e2e-test/inputs/test_data.csv"
    s3_client.put_object(
        Bucket=compute_bucket_name,
        Key=key,
        Body=_TEST_CSV.encode(),
        ContentType="text/csv",
    )
    uri = f"s3://{compute_bucket_name}/{key}"

    def cleanup():
        try:
            s3_client.delete_object(Bucket=compute_bucket_name, Key=key)
        except Exception:
            pass

    request.addfinalizer(cleanup)
    return uri


# ---------------------------------------------------------------------------
# Session-scoped run result (one job started, shared across run/status tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def run_result(request, lam, tool_arns, test_input_s3_uri, sfn_client) -> dict:
    """Start one explore-correlations job. Cancel it after all run/status tests."""
    result = invoke(lam, tool_arns["compute_run"], {
        "profile_id": "explore-correlations",
        "source_uri": test_input_s3_uri,
        "user_arn": _E2E_USER_ARN,
        "parameters": {
            "features": ["x", "y", "z"],
            "method": "pearson",
            "top_n": 3,
        },
    })

    def cancel():
        exec_arn = result.get("execution_arn")
        if exec_arn:
            try:
                sfn_client.stop_execution(
                    executionArn=exec_arn,
                    cause="E2E test teardown",
                )
            except Exception:
                pass

    request.addfinalizer(cancel)
    return result
