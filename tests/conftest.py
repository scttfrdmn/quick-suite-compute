"""
Shared test fixtures for quick-suite-compute.

- Fake AWS credentials so boto3 doesn't complain in unit tests.
- Default Lambda environment variables.
- PROFILES_CONFIG loaded from config/profiles/*.json.
- substrate_url and reset_substrate fixtures for integration tests.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Set fake credentials before any handler modules are imported at collection time.
# boto3 clients are created at module level in handlers; without these,
# botocore tries the macOS keychain provider (requires botocore[crt] on Python 3.14).
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

REPO_ROOT = Path(__file__).parent.parent
PROFILES_DIR = REPO_ROOT / "config" / "profiles"

# Profile modules are in lambdas/profiles/ (would be in Layer at runtime)
sys.path.insert(0, str(REPO_ROOT / "lambdas" / "profiles"))

_SUBSTRATE_BIN = os.path.expanduser("~/src/substrate/bin/substrate")


def _load_profiles() -> list[dict]:
    profiles = []
    for p in sorted(PROFILES_DIR.glob("*.json")):
        with open(p) as f:
            profiles.append(json.load(f))
    return profiles


_PROFILES = _load_profiles()
_PROFILES_JSON = json.dumps(_PROFILES)


@pytest.fixture(autouse=True)
def aws_env(monkeypatch):
    """Fake AWS credentials + standard Lambda environment variables."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("COMPUTE_BUCKET", "qs-compute-test-bucket")
    monkeypatch.setenv("SPEND_TABLE", "qs-compute-spend")
    monkeypatch.setenv("NOTIFICATION_TOPIC_ARN", "arn:aws:sns:us-east-1:123456789012:qs-compute-notifications")
    monkeypatch.setenv("QUICKSIGHT_ACCOUNT_ID", "123456789012")
    monkeypatch.setenv("QUICKSIGHT_REGION", "us-east-1")
    monkeypatch.setenv("MONTHLY_BUDGET_USD", "50")
    monkeypatch.setenv("ENABLE_EMR", "false")
    monkeypatch.setenv("STATE_MACHINE_ARN", "arn:aws:states:us-east-1:123456789012:stateMachine:qs-compute-job")
    monkeypatch.setenv("PROFILES_CONFIG", _PROFILES_JSON)


@pytest.fixture
def profiles():
    return _PROFILES


@pytest.fixture
def profiles_json():
    return _PROFILES_JSON


# ---------------------------------------------------------------------------
# Substrate fixtures (integration tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def substrate_url():
    import requests

    url = os.environ.get("SUBSTRATE_ENDPOINT", "http://localhost:4566")
    try:
        requests.get(f"{url}/health", timeout=1)
        yield url
        return
    except Exception:
        pass

    if not os.path.exists(_SUBSTRATE_BIN):
        pytest.skip("Substrate binary not found at ~/src/substrate/bin/substrate")
        return

    proc = subprocess.Popen(
        [_SUBSTRATE_BIN, "server"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(40):
        try:
            requests.get(f"{url}/health", timeout=0.5)
            break
        except Exception:
            time.sleep(0.25)
    else:
        proc.terminate()
        pytest.skip("Substrate did not become healthy in time")
        return

    yield url
    proc.terminate()
    proc.wait(timeout=5)


@pytest.fixture
def reset_substrate(substrate_url):
    import requests

    requests.post(f"{substrate_url}/v1/state/reset", timeout=5)
    yield
    requests.post(f"{substrate_url}/v1/state/reset", timeout=5)
