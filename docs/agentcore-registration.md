# Registering Compute Tools with AgentCore Gateway

After deploying the CDK stack, register the three tool Lambdas as AgentCore Gateway
Lambda targets so Quick Suite can discover and invoke them.

---

## Prerequisites

- `cdk deploy` completed successfully
- An AgentCore Gateway already exists (or you are creating one)
- You have IAM permissions to call `bedrock-agentcore:CreateTarget`

---

## Step 1 — Grant AgentCore Invoke Permission (pre-deploy)

The cleanest approach is to pass the Gateway execution role ARN **before** deploying
so the CDK stack auto-grants `lambda:InvokeFunction`:

```bash
cdk deploy --context agentcore_gateway_role_arn=arn:aws:iam::<ACCOUNT>:role/<GATEWAY_ROLE>
```

If you didn't do this at deploy time, add permissions manually:

```bash
# Retrieve tool Lambda ARNs
aws cloudformation describe-stacks \
  --stack-name QuickSuiteCompute \
  --query 'Stacks[0].Outputs[?OutputKey==`ToolArns`].OutputValue' \
  --output text | python3 -m json.tool
```

Then for each Lambda ARN:
```bash
aws lambda add-permission \
  --function-name <LAMBDA_ARN> \
  --statement-id AgentCoreInvoke \
  --action lambda:InvokeFunction \
  --principal bedrock-agentcore.amazonaws.com \
  --source-arn arn:aws:bedrock-agentcore:<REGION>:<ACCOUNT>:gateway/<GATEWAY_ID>
```

---

## Step 2 — Retrieve Lambda ARNs

```bash
# All tool ARNs as JSON
aws cloudformation describe-stacks \
  --stack-name QuickSuiteCompute \
  --query 'Stacks[0].Outputs[?OutputKey==`ToolArns`].OutputValue' \
  --output text

# Individual ARNs by tool name
aws cloudformation describe-stacks \
  --stack-name QuickSuiteCompute \
  --query 'Stacks[0].Outputs[?contains(OutputKey,`Arn`)].{Key:OutputKey,Value:OutputValue}' \
  --output table
```

Or use `scripts/post-deploy.sh` which prints all values with instructions.

---

## Step 3 — Register Lambda Targets via Console

In the AWS Console → Amazon Bedrock → AgentCore → Gateways → [your gateway]:

1. Click **Add target**
2. Select **Lambda function**
3. For each of the three tools:

| Target name | Lambda output key | Tool name in Quick Suite |
|------------|-------------------|--------------------------|
| `qs-compute-profiles` | `ComputeprofilesArn` | `compute_profiles` |
| `qs-compute-run` | `ComputerunArn` | `compute_run` |
| `qs-compute-status` | `ComputestatusArn` | `compute_status` |

For each target:
- **Lambda ARN**: paste the ARN from the CloudFormation output
- **Tool schema**: upload `config/compute-tools.yaml`
- Leave auth as **IAM** (AgentCore Gateway uses its execution role)

---

## Step 4 — Register via AWS CLI

```bash
GATEWAY_ID="<YOUR_GATEWAY_ID>"
REGION="<YOUR_REGION>"

# Read ARNs
TOOL_ARNS=$(aws cloudformation describe-stacks \
  --stack-name QuickSuiteCompute \
  --query 'Stacks[0].Outputs[?OutputKey==`ToolArns`].OutputValue' \
  --output text)

PROFILES_ARN=$(echo "$TOOL_ARNS" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['compute_profiles'])")
RUN_ARN=$(echo "$TOOL_ARNS" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['compute_run'])")
STATUS_ARN=$(echo "$TOOL_ARNS" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['compute_status'])")

aws bedrock-agentcore create-gateway-target \
  --gateway-identifier "$GATEWAY_ID" \
  --name "qs-compute-profiles" \
  --target-configuration "{\"lambdaConfiguration\":{\"lambdaArn\":\"$PROFILES_ARN\"}}"

aws bedrock-agentcore create-gateway-target \
  --gateway-identifier "$GATEWAY_ID" \
  --name "qs-compute-run" \
  --target-configuration "{\"lambdaConfiguration\":{\"lambdaArn\":\"$RUN_ARN\"}}"

aws bedrock-agentcore create-gateway-target \
  --gateway-identifier "$GATEWAY_ID" \
  --name "qs-compute-status" \
  --target-configuration "{\"lambdaConfiguration\":{\"lambdaArn\":\"$STATUS_ARN\"}}"
```

---

## Step 5 — Verify

Test each tool from the AgentCore Gateway console:

1. Go to your Gateway → Targets → select a target → **Test**
2. Send a sample payload:
   - `compute_profiles`: `{}` (no arguments required)
   - `compute_run`: `{"intent": "cluster the enrollment dataset", "dataset_id": "test-ds-001", "user_id": "user@example.edu"}`
   - `compute_status`: `{"job_id": "<execution_arn>"}`
3. Confirm a valid JSON response (not an error)

For `compute_run`, the job will start a real Step Functions execution. Use
`compute_status` with the returned `job_id` to poll progress.

---

## Quick Suite Integration

After registering all three targets, connect them in Quick Suite via
AgentCore Gateway's MCP endpoint. The compute extension works best
alongside the Open Data extension — users can browse RODA or institutional
S3 data, then immediately request an analysis on the discovered dataset.

---

## Automation

`scripts/post-deploy.sh` prints all ARNs and the registration commands with your
actual values filled in. Run it after every `cdk deploy`.
