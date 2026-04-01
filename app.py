#!/usr/bin/env python3
import aws_cdk as cdk
from stacks.compute_stack import ComputeStack

app = cdk.App()
ComputeStack(
    app,
    "QuickSuiteCompute",
    description=(
        "Quick Suite Compute — ephemeral analytics jobs (clustering, regression, "
        "forecasting, and more) as AgentCore Gateway Lambda targets. Results "
        "land back as Quick Sight datasets."
    ),
)
app.synth()
