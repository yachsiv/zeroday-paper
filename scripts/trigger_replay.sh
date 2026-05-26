#!/usr/bin/env bash
# One-shot launch the replay task on AWS.
#
# Usage:
#   ./scripts/trigger_replay.sh [days_back]
#
# Default: 365.
set -euo pipefail

cd "$(dirname "$0")/.."

DAYS="${1:-365}"
CLUSTER="ZerodayPaperCluster"

OUTPUTS_FILE="cdk-outputs.json"
if [ ! -f "$OUTPUTS_FILE" ]; then
  echo "cdk-outputs.json not found; run scripts/deploy.sh first." >&2
  exit 1
fi

TASK_DEF_ARN=$(python3 -c "import json,sys; print(json.load(open('$OUTPUTS_FILE'))['ZerodayPaperStack']['TaskDefArn'])")

read -r SUBNET_A SUBNET_B <<< "subnet-047ba927ecc16c67c subnet-0f8fb515c6319eb93"

TASK_SG=$(aws ec2 describe-security-groups \
  --region us-east-1 \
  --filters "Name=group-name,Values=ZerodayPaperStack-TaskSg-*" \
  --query 'SecurityGroups[0].GroupId' --output text)

echo "==> Launching replay (days=$DAYS, taskdef=$TASK_DEF_ARN)"

aws ecs run-task \
  --region us-east-1 \
  --cluster "$CLUSTER" \
  --launch-type FARGATE \
  --task-definition "$TASK_DEF_ARN" \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_A,$SUBNET_B],securityGroups=[$TASK_SG],assignPublicIp=DISABLED}" \
  --overrides "{\"containerOverrides\":[{\"name\":\"app\",\"environment\":[{\"name\":\"MODE\",\"value\":\"replay\"}],\"command\":[\"--days\",\"$DAYS\"]}]}" \
  --query 'tasks[0].taskArn' --output text
