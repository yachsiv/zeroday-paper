#!/usr/bin/env bash
# Trigger one-shot MODE=status task on Fargate.
#
# No arguments — fires the latest active task definition for `zeroday-paper`
# with MODE=status overrides and prints the resulting task ARN. Safe to run
# any time (read-only DuckDB connection inside the task).
#
# Usage:
#   ./scripts/post-status.sh
#
# Mirrors scripts/trigger_replay.sh shape. Network config (subnets/SG) is
# pinned to the same values the EventBridge rules use.
set -euo pipefail

cd "$(dirname "$0")/.."

CLUSTER="ZerodayPaperCluster"
REGION="us-east-1"

# Resolve the latest active task definition for the family. CDK-published
# revisions are immutable, so taking the latest is exactly what production
# uses on its next scheduled fire.
TASK_DEF_ARN=$(aws ecs describe-task-definition \
  --region "$REGION" \
  --task-definition zeroday-paper \
  --query 'taskDefinition.taskDefinitionArn' --output text)

# Pinned subnet IDs match infra/stack.py PRIVATE_SUBNETS.
SUBNET_A="subnet-047ba927ecc16c67c"
SUBNET_B="subnet-0f8fb515c6319eb93"

TASK_SG=$(aws ec2 describe-security-groups \
  --region "$REGION" \
  --filters "Name=group-name,Values=ZerodayPaperStack-TaskSg*" \
  --query 'SecurityGroups[0].GroupId' --output text)

echo "==> Launching status task (taskdef=$TASK_DEF_ARN sg=$TASK_SG)" >&2

aws ecs run-task \
  --region "$REGION" \
  --cluster "$CLUSTER" \
  --launch-type FARGATE \
  --task-definition "$TASK_DEF_ARN" \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_A,$SUBNET_B],securityGroups=[$TASK_SG],assignPublicIp=DISABLED}" \
  --overrides "{\"containerOverrides\":[{\"name\":\"app\",\"environment\":[{\"name\":\"MODE\",\"value\":\"status\"}]}]}" \
  --query 'tasks[0].taskArn' --output text
