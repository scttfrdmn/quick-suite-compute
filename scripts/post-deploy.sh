#!/usr/bin/env bash
# post-deploy.sh — Quick Suite Compute post-deployment configuration helper
#
# Run after `cdk deploy` to retrieve stack outputs and print the commands
# needed to register Lambda targets with AgentCore Gateway.
#
# Usage:
#   bash scripts/post-deploy.sh
#   bash scripts/post-deploy.sh --stack-name MyCustomStackName
#   bash scripts/post-deploy.sh --region us-west-2

set -euo pipefail

STACK_NAME="QuickSuiteCompute"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stack-name) STACK_NAME="$2"; shift 2 ;;
    --region)     REGION="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

echo ""
echo "========================================================"
echo "  Quick Suite Compute — Post-Deploy Configuration"
echo "========================================================"
echo ""

echo "Fetching stack outputs from CloudFormation..."
OUTPUTS_JSON=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --region "$REGION" \
  --query 'Stacks[0].Outputs' \
  --output json 2>/dev/null) || {
    echo ""
    echo "ERROR: Could not retrieve stack '$STACK_NAME' in region '$REGION'."
    echo "  - Verify the stack deployed: cdk deploy"
    echo "  - Check region: export AWS_DEFAULT_REGION=<your-region>"
    exit 1
  }

get_output() {
  echo "$OUTPUTS_JSON" | python3 -c \
    "import json,sys; d={o['OutputKey']:o['OutputValue'] for o in json.load(sys.stdin)}; print(d.get('$1','<NOT_FOUND>'))"
}

TOOL_ARNS_JSON=$(get_output "ToolArns")
COMPUTE_BUCKET=$(get_output "ComputeBucketName")
SPEND_TABLE=$(get_output "SpendTableName")
STATE_MACHINE=$(get_output "StateMachineArn")
SNS_TOPIC=$(get_output "NotificationTopicArn")

echo ""
echo "--------------------------------------------------------"
echo "  Stack Outputs"
echo "--------------------------------------------------------"
echo "  Compute S3 bucket   : $COMPUTE_BUCKET"
echo "  Spend DynamoDB table: $SPEND_TABLE"
echo "  Step Functions ARN  : $STATE_MACHINE"
echo "  SNS notification    : $SNS_TOPIC"
echo ""
echo "  Tool Lambda ARNs (JSON):"
echo "$TOOL_ARNS_JSON" | python3 -m json.tool 2>/dev/null || echo "  $TOOL_ARNS_JSON"
echo ""

PROFILES_ARN=$(echo "$TOOL_ARNS_JSON" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['compute_profiles'])" 2>/dev/null || echo "<compute_profiles_arn>")
RUN_ARN=$(echo "$TOOL_ARNS_JSON" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['compute_run'])" 2>/dev/null || echo "<compute_run_arn>")
STATUS_ARN=$(echo "$TOOL_ARNS_JSON" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['compute_status'])" 2>/dev/null || echo "<compute_status_arn>")

echo "--------------------------------------------------------"
echo "  Step 1 of 2 — Register Lambda Targets with AgentCore Gateway"
echo "  Replace GATEWAY_ID with your AgentCore Gateway identifier."
echo "--------------------------------------------------------"
echo ""
echo "  GATEWAY_ID=\"<YOUR_GATEWAY_ID>\""
echo ""
echo "  aws bedrock-agentcore create-gateway-target \\"
echo "    --gateway-identifier \"\$GATEWAY_ID\" \\"
echo "    --name qs-compute-profiles \\"
echo "    --target-configuration '{\"lambdaConfiguration\":{\"lambdaArn\":\"$PROFILES_ARN\"}}'"
echo ""
echo "  aws bedrock-agentcore create-gateway-target \\"
echo "    --gateway-identifier \"\$GATEWAY_ID\" \\"
echo "    --name qs-compute-run \\"
echo "    --target-configuration '{\"lambdaConfiguration\":{\"lambdaArn\":\"$RUN_ARN\"}}'"
echo ""
echo "  aws bedrock-agentcore create-gateway-target \\"
echo "    --gateway-identifier \"\$GATEWAY_ID\" \\"
echo "    --name qs-compute-status \\"
echo "    --target-configuration '{\"lambdaConfiguration\":{\"lambdaArn\":\"$STATUS_ARN\"}}'"
echo ""

echo "--------------------------------------------------------"
echo "  Step 2 of 2 — Subscribe to job notifications (optional)"
echo "--------------------------------------------------------"
echo ""
echo "  Subscribe an email address to receive job completion alerts:"
echo "  aws sns subscribe \\"
echo "    --topic-arn \"$SNS_TOPIC\" \\"
echo "    --protocol email \\"
echo "    --notification-endpoint you@example.edu"
echo ""
echo "  Full details: docs/agentcore-registration.md"
echo ""
echo "========================================================"
echo "  Done."
echo "========================================================"
echo ""
