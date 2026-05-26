"""ZerodayPaperStack — CDK stack for the paper trading engine.

Reuses the main zeroday VPC. Provisions:

    ECR repository           zeroday-paper
    ECS Fargate cluster      ZerodayPaperCluster
    EFS file system          /data persistent storage, encrypted
    Task definition          single image, MODE env dispatch
    EventBridge rules        live-start (09:20 ET Mon-Fri), report (16:30 ET Mon-Fri)
    Manual-trigger task      replay (run-task with MODE=replay)
    S3 bucket                zeroday-paper-backup-{account}
    IAM role                 read-only on zeroday/* secrets
    CloudWatch log group     30-day retention
    CloudWatch alarm         scanner heartbeat staleness (single alarm only)
"""

from __future__ import annotations

from aws_cdk import (
    Aws,
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
)
from aws_cdk import aws_cloudwatch as cw
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_efs as efs
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_secretsmanager as secretsmanager
from constructs import Construct


# ----- Pinned discoverable values -------------------------------------------

VPC_ID = "vpc-0632149c60aeaa91c"
PRIVATE_SUBNETS = [
    {"id": "subnet-047ba927ecc16c67c", "az": "us-east-1a"},
    {"id": "subnet-0f8fb515c6319eb93", "az": "us-east-1b"},
]

ZERODAY_SECRET_IDS = [
    "zeroday/polygon",
    "zeroday/flashalpha",
    "zeroday/anthropic",
    "zeroday/discord",
    "zeroday/perplexity",
]


class ZerodayPaperStack(Stack):
    def __init__(self, scope: Construct, id_: str, **kwargs) -> None:
        super().__init__(scope, id_, **kwargs)

        # ----- VPC + networking --------------------------------------------
        vpc = ec2.Vpc.from_vpc_attributes(
            self,
            "ZerodayVpc",
            vpc_id=VPC_ID,
            availability_zones=[s["az"] for s in PRIVATE_SUBNETS],
            private_subnet_ids=[s["id"] for s in PRIVATE_SUBNETS],
        )

        # ----- EFS for /data DuckDB ---------------------------------------
        efs_sg = ec2.SecurityGroup(
            self, "EfsSg",
            vpc=vpc, allow_all_outbound=False,
            description="zeroday-paper EFS access",
        )

        file_system = efs.FileSystem(
            self, "DataFs",
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnets=[
                ec2.Subnet.from_subnet_id(self, f"EfsSubnet{i}", sn["id"])
                for i, sn in enumerate(PRIVATE_SUBNETS)
            ]),
            security_group=efs_sg,
            removal_policy=RemovalPolicy.RETAIN,
            encrypted=True,
            performance_mode=efs.PerformanceMode.GENERAL_PURPOSE,
            throughput_mode=efs.ThroughputMode.BURSTING,
            lifecycle_policy=efs.LifecyclePolicy.AFTER_30_DAYS,
        )

        access_point = efs.AccessPoint(
            self, "DataAccessPoint",
            file_system=file_system,
            path="/paper",
            posix_user=efs.PosixUser(uid="1000", gid="1000"),
            create_acl=efs.Acl(owner_uid="1000", owner_gid="1000", permissions="755"),
        )

        # ----- S3 backup bucket --------------------------------------------
        backup_bucket = s3.Bucket(
            self, "BackupBucket",
            bucket_name=f"zeroday-paper-backup-{Aws.ACCOUNT_ID}",
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_rules=[s3.LifecycleRule(
                id="expire-old-backups",
                enabled=True,
                expiration=Duration.days(90),
            )],
            versioned=False,
        )

        # ----- ECR + image --------------------------------------------------
        # Build the Docker image from the repo root and push to a fresh ECR repo.
        import os
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        image_asset = ecr_assets.DockerImageAsset(
            self, "AppImage",
            directory=repo_root,
            platform=ecr_assets.Platform.LINUX_ARM64,
        )

        # ----- ECS cluster --------------------------------------------------
        cluster = ecs.Cluster(
            self, "Cluster",
            cluster_name="ZerodayPaperCluster",
            vpc=vpc,
            container_insights=True,
        )

        log_group = logs.LogGroup(
            self, "TaskLogs",
            log_group_name="/zeroday-paper/tasks",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.RETAIN,
        )

        # ----- IAM roles ---------------------------------------------------
        task_role = iam.Role(
            self, "TaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
        )

        secret_arns = [
            f"arn:aws:secretsmanager:{self.region}:{self.account}:secret:{name}-*"
            for name in ZERODAY_SECRET_IDS
        ]
        task_role.add_to_policy(iam.PolicyStatement(
            actions=["secretsmanager:GetSecretValue", "secretsmanager:DescribeSecret"],
            resources=secret_arns,
        ))
        task_role.add_to_policy(iam.PolicyStatement(
            actions=["cloudwatch:PutMetricData"],
            resources=["*"],
            conditions={"StringEquals": {"cloudwatch:namespace": "zeroday/paper"}},
        ))
        backup_bucket.grant_put(task_role)
        backup_bucket.grant_read_write(task_role)

        exec_role = iam.Role(
            self, "ExecRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[iam.ManagedPolicy.from_aws_managed_policy_name(
                "service-role/AmazonECSTaskExecutionRolePolicy"
            )],
        )

        # ----- Task definition (single image, MODE-dispatched) -------------
        task_def = ecs.FargateTaskDefinition(
            self, "TaskDef",
            cpu=512,
            memory_limit_mib=1024,
            task_role=task_role,
            execution_role=exec_role,
            family="zeroday-paper",
            runtime_platform=ecs.RuntimePlatform(
                cpu_architecture=ecs.CpuArchitecture.ARM64,
                operating_system_family=ecs.OperatingSystemFamily.LINUX,
            ),
        )

        task_def.add_volume(
            name="data",
            efs_volume_configuration=ecs.EfsVolumeConfiguration(
                file_system_id=file_system.file_system_id,
                transit_encryption="ENABLED",
                authorization_config=ecs.AuthorizationConfig(
                    access_point_id=access_point.access_point_id,
                    iam="ENABLED",
                ),
            ),
        )

        container = task_def.add_container(
            "app",
            image=ecs.ContainerImage.from_docker_image_asset(image_asset),
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="zp",
                log_group=log_group,
            ),
            environment={
                "AWS_REGION": self.region,
                "TZ": "America/New_York",
                "ZP_DUCKDB_PATH": "/data/paper.duckdb",
                "ZP_REPORT_DIR": "/data/reports",
                "MODE": "live",
            },
            essential=True,
        )

        container.add_mount_points(ecs.MountPoint(
            container_path="/data",
            source_volume="data",
            read_only=False,
        ))

        # EFS access requires the task to be in subnets that can reach the
        # mount targets, and the EFS SG to allow inbound NFS (2049) from the
        # task SG.
        task_sg = ec2.SecurityGroup(
            self, "TaskSg",
            vpc=vpc, allow_all_outbound=True,
            description="zeroday-paper task SG",
        )
        efs_sg.add_ingress_rule(
            peer=task_sg,
            connection=ec2.Port.tcp(2049),
            description="NFS from paper task",
        )

        subnets = ec2.SubnetSelection(subnets=[
            ec2.Subnet.from_subnet_attributes(
                self, f"RunSubnet{i}",
                subnet_id=sn["id"], availability_zone=sn["az"],
            )
            for i, sn in enumerate(PRIVATE_SUBNETS)
        ])

        # ----- EventBridge schedules ---------------------------------------
        # 09:20 ET Mon-Fri → live task (self-exits at session_end)
        # 16:30 ET Mon-Fri → report task
        # Crons are UTC; we lock to EDT until DST rollover (Nov), then add EST rules.
        live_rule = events.Rule(
            self, "LiveStartRule",
            description="Start paper-live at 09:28 ET (EDT cron; code-side waits until 09:30 session_start)",
            schedule=events.Schedule.cron(minute="28", hour="13", week_day="MON-FRI"),
        )
        live_rule.add_target(targets.EcsTask(
            cluster=cluster,
            task_definition=task_def,
            task_count=1,
            subnet_selection=subnets,
            security_groups=[task_sg],
            assign_public_ip=False,
            container_overrides=[targets.ContainerOverride(
                container_name="app",
                environment=[targets.TaskEnvironmentVariable(name="MODE", value="live")],
            )],
        ))

        report_rule = events.Rule(
            self, "ReportRule",
            description="Build + post daily report at 16:30 ET (EDT cron)",
            schedule=events.Schedule.cron(minute="30", hour="20", week_day="MON-FRI"),
        )
        report_rule.add_target(targets.EcsTask(
            cluster=cluster,
            task_definition=task_def,
            task_count=1,
            subnet_selection=subnets,
            security_groups=[task_sg],
            assign_public_ip=False,
            container_overrides=[targets.ContainerOverride(
                container_name="app",
                environment=[targets.TaskEnvironmentVariable(name="MODE", value="report")],
            )],
        ))

        # 08:00 ET Mon-Fri → morning brief (12:00 UTC during EDT).
        morning_rule = events.Rule(
            self, "MorningRule",
            description="Build + post pre-market brief at 08:00 ET (EDT cron)",
            schedule=events.Schedule.cron(minute="0", hour="12", week_day="MON-FRI"),
        )
        morning_rule.add_target(targets.EcsTask(
            cluster=cluster,
            task_definition=task_def,
            task_count=1,
            subnet_selection=subnets,
            security_groups=[task_sg],
            assign_public_ip=False,
            container_overrides=[targets.ContainerOverride(
                container_name="app",
                environment=[targets.TaskEnvironmentVariable(name="MODE", value="morning")],
            )],
        ))

        # ----- CloudWatch alarm + Discord notifier -------------------------
        # The scanner emits cycle.complete metric on every successful cycle.
        # If we don't see >=1 in 15 min during market hours, alarm fires.
        scanner_metric = cw.Metric(
            namespace="zeroday/paper",
            metric_name="cycle.complete",
            statistic="Sum",
            period=Duration.minutes(15),
        )
        scanner_alarm = cw.Alarm(
            self, "ScannerSilentAlarm",
            metric=scanner_metric,
            alarm_name="zeroday-paper-scanner-silent",
            threshold=1,
            evaluation_periods=2,
            comparison_operator=cw.ComparisonOperator.LESS_THAN_THRESHOLD,
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
            alarm_description="Paper scanner silent during market hours",
        )

        alarm_lambda = lambda_.Function(
            self, "AlarmNotifier",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="index.handler",
            timeout=Duration.seconds(15),
            description="Posts ScannerSilentAlarm state changes to Discord webhook_monitor",
            environment={
                "DISCORD_SECRET_ID": "zeroday/discord",
                "DISCORD_KEY": "webhook_monitor",
            },
            code=lambda_.Code.from_inline(
                """
import json
import os
import urllib.request
import boto3

_client = boto3.client('secretsmanager')
_webhook_cache = None


def _webhook():
    global _webhook_cache
    if _webhook_cache:
        return _webhook_cache
    secret_id = os.environ['DISCORD_SECRET_ID']
    key = os.environ['DISCORD_KEY']
    resp = _client.get_secret_value(SecretId=secret_id)
    data = json.loads(resp['SecretString'])
    _webhook_cache = data[key]
    return _webhook_cache


def handler(event, _ctx):
    detail = event.get('detail', {})
    alarm_name = detail.get('alarmName') or event.get('alarmName') or 'unknown'
    state = (detail.get('state') or {}).get('value') or event.get('NewStateValue') or 'UNKNOWN'
    reason = (detail.get('state') or {}).get('reason') or event.get('NewStateReason') or ''
    if state == 'ALARM':
        emoji = '[ALARM]'
    elif state == 'OK':
        emoji = '[OK]'
    else:
        emoji = '[INFO]'
    content = f"{emoji} **{alarm_name}** -> {state}\\n{reason[:1500]}"
    body = json.dumps({'content': content}).encode()
    req = urllib.request.Request(_webhook(), data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=10) as r:
        return {'status': r.status}
"""
            ),
        )

        alarm_lambda.role.add_to_principal_policy(iam.PolicyStatement(
            actions=["secretsmanager:GetSecretValue"],
            resources=[
                f"arn:aws:secretsmanager:{self.region}:{self.account}:secret:zeroday/discord-*"
            ],
        ))

        alarm_event_rule = events.Rule(
            self, "AlarmToDiscordRule",
            description="Forward ScannerSilentAlarm state changes to Discord",
            event_pattern=events.EventPattern(
                source=["aws.cloudwatch"],
                detail_type=["CloudWatch Alarm State Change"],
                detail={"alarmName": [scanner_alarm.alarm_name]},
            ),
        )
        alarm_event_rule.add_target(targets.LambdaFunction(alarm_lambda))

        # ----- Outputs ------------------------------------------------------
        CfnOutput(self, "OutClusterName", value=cluster.cluster_name)
        CfnOutput(self, "OutTaskDefArn", value=task_def.task_definition_arn)
        CfnOutput(self, "OutBackupBucket", value=backup_bucket.bucket_name)
        CfnOutput(self, "OutEfsId", value=file_system.file_system_id)
        CfnOutput(self, "OutImageUri", value=image_asset.image_uri)
        CfnOutput(self, "OutLogGroup", value=log_group.log_group_name)
        CfnOutput(self, "OutTaskSgId", value=task_sg.security_group_id)
