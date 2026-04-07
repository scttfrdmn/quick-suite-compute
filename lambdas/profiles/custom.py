"""
custom.py — Custom execution profiles

custom_python_handler   — Execute a user-supplied Python script via RestrictedPython sandbox (custom-python)
custom_generated_handler — Generate a Python transform from a natural-language objective, then execute it (custom-generated)

custom_python_handler downloads the script from S3, compiles it with RestrictedPython,
and calls the transform(df) function it must define.

custom_generated_handler invokes the quick-suite-router 'code' tool to generate the
script, writes it to S3, then delegates to custom_python_handler.

Called by runner/handler.py via entrypoints in config/profiles/*.json.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import signal
import time
import uuid
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Libraries made available inside the sandbox namespace
_SANDBOX_ALLOWED_IMPORTS = {
    "pd": "pandas",
    "np": "numpy",
}

_MAX_TIMEOUT_SECONDS = 300


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse 's3://bucket/key' into (bucket, key)."""
    if not uri.startswith("s3://"):
        raise ValueError(f"script_uri must be an s3:// path (got '{uri}')")
    parts = uri[len("s3://"):].split("/", 1)
    if len(parts) != 2 or not parts[1]:
        raise ValueError(f"script_uri must be s3://bucket/key (got '{uri}')")
    return parts[0], parts[1]


_NETWORK_SCHEMES = ("http://", "https://", "ftp://", "s3://", "gs://", "az://")

# pandas read_* methods that accept a URL as their first positional argument
_PANDAS_NETWORK_METHODS = frozenset({
    "read_csv", "read_json", "read_excel", "read_html", "read_sql",
    "read_parquet", "read_pickle", "read_feather", "read_orc",
    "read_fwf", "read_table", "read_clipboard",
})

_AST_FORBIDDEN_NAMES = frozenset({
    "os", "sys", "subprocess", "socket", "urllib", "requests", "builtins",
})

_AST_FORBIDDEN_CALLS = frozenset({
    "eval", "exec", "compile", "open", "__import__",
})


def _make_safe_pandas(pd_module):
    """
    Return a proxy object that wraps pandas and blocks URL-based read_* calls.

    RestrictedPython blocks __import__ at AST level but cannot block method
    calls on already-imported objects. This proxy intercepts every read_*
    call and raises PermissionError when the first argument is a network URI.
    In-memory sources (BytesIO, StringIO, DataFrame) are allowed.
    """

    class _SafePandasProxy:
        def __getattr__(self, name):
            attr = getattr(pd_module, name)
            if name in _PANDAS_NETWORK_METHODS:
                def _guarded(*args, **kwargs):
                    if args:
                        first = args[0]
                        if isinstance(first, str) and any(
                            first.lower().startswith(s) for s in _NETWORK_SCHEMES
                        ):
                            raise PermissionError(
                                f"pd.{name}() with a network URI is not allowed "
                                f"in the sandbox (got '{first[:60]}')"
                            )
                    path_arg = kwargs.get("filepath_or_buffer") or kwargs.get("path_or_buf")
                    if isinstance(path_arg, str) and any(
                        path_arg.lower().startswith(s) for s in _NETWORK_SCHEMES
                    ):
                        raise PermissionError(
                            f"pd.{name}() with a network URI is not allowed "
                            f"in the sandbox (got '{path_arg[:60]}')"
                        )
                    return attr(*args, **kwargs)
                return _guarded
            return attr

        def __repr__(self):
            return f"<SafePandasProxy wrapping {pd_module}>"

    return _SafePandasProxy()


def _analyze_generated_code(code: str) -> list[str]:
    """
    Static analysis of LLM-generated code before sandbox execution.

    Returns a list of violation strings. An empty list means the code passed.
    Detects:
      - Import statements (sandbox provides everything; imports are unnecessary
        and could indicate an attempted bypass)
      - Dunder attribute traversal (e.g. df.__class__.__bases__)
      - Calls to eval/exec/compile/open/__import__
      - References to forbidden module names (os, sys, subprocess, etc.)
    """
    violations: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"SyntaxError: {exc}"]

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            violations.append(f"Import not allowed: {', '.join(names)}")

        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                violations.append(
                    f"Dunder attribute access not allowed: '.{node.attr}'"
                )

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _AST_FORBIDDEN_CALLS:
                violations.append(f"Call to '{node.func.id}()' not allowed")

        elif isinstance(node, ast.Name):
            if node.id in _AST_FORBIDDEN_NAMES:
                violations.append(f"Reference to '{node.id}' not allowed")

    return violations


def _build_safe_globals() -> dict:
    """
    Build the restricted execution namespace.

    Imports are resolved lazily so the sandbox does not fail on
    environments missing optional heavy deps; the script itself
    will raise ImportError if it tries to use them.

    Compatible with RestrictedPython >= 7.x. The Guard API changed in v8
    (safe_iter / guarded_getiter / guarded_getattr removed); we use the
    stable symbols that exist across both major versions.
    """
    import builtins

    import numpy as np
    import pandas as pd
    from RestrictedPython import safe_builtins, safe_globals
    from RestrictedPython.Guards import (
        guarded_delattr,
        guarded_setattr,
        guarded_unpack_sequence,
        safer_getattr,
    )

    glb = dict(safe_globals)
    glb["__builtins__"] = dict(safe_builtins)
    # Allow a few extra builtins useful for data transforms
    for _name in ("print", "len", "range", "enumerate", "zip", "map", "filter",
                  "list", "dict", "set", "tuple", "str", "int", "float", "bool",
                  "sorted", "reversed", "min", "max", "sum", "abs", "round",
                  "isinstance", "type", "hasattr", "getattr"):
        if _name in vars(builtins):
            glb["__builtins__"][_name] = vars(builtins)[_name]

    # Data-science libraries available in the sandbox
    glb["pd"] = _make_safe_pandas(pd)
    glb["np"] = np

    # Optional heavy deps — fail gracefully if not present in deployment
    for _mod_name in ("scipy", "sklearn"):
        try:
            glb[_mod_name] = __import__(_mod_name)
        except ImportError:
            pass

    # RestrictedPython guard hooks (stable across v7 and v8)
    glb["_iter_"] = iter
    glb["_getiter_"] = iter
    glb["_getattr_"] = safer_getattr
    glb["_setattr_"] = guarded_setattr
    glb["_delattr_"] = guarded_delattr
    glb["_inplacevar_"] = lambda op, x, y: x  # no in-place mutation guard needed
    glb["_getitem_"] = lambda obj, key: obj[key]
    glb["_write_"] = lambda x: x
    glb["_unpack_sequence_"] = guarded_unpack_sequence

    return glb


class _TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TimeoutError("Script execution timed out")


# ---------------------------------------------------------------------------
# custom-python
# ---------------------------------------------------------------------------

def custom_python_handler(df: pd.DataFrame, parameters: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    """
    Execute a user-supplied Python transform script via RestrictedPython.

    The script must define a function: transform(df: pd.DataFrame) -> pd.DataFrame
    The function receives the input DataFrame and must return a DataFrame.

    Downloads the script from S3, compiles it with RestrictedPython's
    compile_restricted(), and executes it in a hardened namespace that
    provides pd, np, scipy, sklearn. Network access, file I/O, subprocess,
    and class-hierarchy traversal are blocked by RestrictedPython's AST
    transforms.

    A SIGALRM-based wall-clock timeout is enforced on Linux/macOS.
    On platforms without SIGALRM (e.g. Windows), the timeout is skipped
    with a warning.
    """
    import boto3
    from RestrictedPython import compile_restricted

    script_uri = parameters.get("script_uri", "")
    timeout_seconds = min(int(parameters.get("timeout_seconds", 120)), _MAX_TIMEOUT_SECONDS)

    if not script_uri:
        raise ValueError("script_uri parameter is required")

    bucket, key = _parse_s3_uri(script_uri)

    s3 = boto3.client("s3")
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        script_text = body.decode("utf-8")
    except Exception as exc:
        raise ValueError(f"Could not download script from {script_uri}: {exc}") from exc

    # Compile under RestrictedPython — raises SyntaxError on policy violations
    try:
        byte_code = compile_restricted(script_text, filename=script_uri, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"Script failed RestrictedPython compilation: {exc}") from exc

    safe_ns = _build_safe_globals()
    safe_ns["__name__"] = "__restricted__"

    # Execute the script to define 'transform'
    try:
        exec(byte_code, safe_ns)  # noqa: S102 — intentional sandbox exec
    except Exception as exc:
        raise ValueError(f"Script raised an error during load: {exc}") from exc

    transform_fn = safe_ns.get("transform")
    if transform_fn is None or not callable(transform_fn):
        raise ValueError("Script must define a callable named 'transform(df)'")

    input_rows = len(df)
    start = time.monotonic()

    # SIGALRM timeout (POSIX only)
    has_alarm = hasattr(signal, "SIGALRM")
    if has_alarm:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)

    try:
        result = transform_fn(df)
    except _TimeoutError:
        raise ValueError(
            f"Script exceeded the {timeout_seconds}s wall-clock limit"
        )
    except Exception as exc:
        raise ValueError(f"Script transform() raised an error: {exc}") from exc
    finally:
        if has_alarm:
            signal.alarm(0)  # disarm

    elapsed = round(time.monotonic() - start, 3)

    if not isinstance(result, pd.DataFrame):
        raise ValueError(
            f"Script transform() must return a pandas DataFrame "
            f"(got {type(result).__name__})"
        )

    diagnostics = {
        "script_uri": script_uri,
        "elapsed_seconds": elapsed,
        "input_rows": input_rows,
        "output_rows": len(result),
        "output_columns": list(result.columns),
        "timeout_seconds": timeout_seconds,
    }

    logger.info(
        f"custom-python complete: {input_rows} → {len(result)} rows in {elapsed}s "
        f"(script={script_uri})"
    )
    return result, diagnostics


# ---------------------------------------------------------------------------
# custom-generated
# ---------------------------------------------------------------------------

def custom_generated_handler(df: pd.DataFrame, parameters: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    """
    Generate a Python transform from a natural-language objective, then execute it.

    Invokes the quick-suite-router 'code' tool via Lambda.invoke() with a
    constrained code-generation prompt built from `objective` and optional
    `schema_hint`. The generated script is written to S3, then executed via
    custom_python_handler.

    Requires the ROUTER_INVOKE_ARN environment variable to be set to the
    router Lambda's ARN. The runner Lambda must have lambda:InvokeFunction
    permission on that ARN.
    """
    import boto3

    router_arn = os.environ.get("ROUTER_INVOKE_ARN", "")
    if not router_arn:
        raise ValueError("ROUTER_INVOKE_ARN environment variable not set")

    results_bucket = os.environ.get("RESULTS_BUCKET", "")
    if not results_bucket:
        raise ValueError("RESULTS_BUCKET environment variable not set")

    objective = parameters.get("objective", "").strip()
    schema_hint = parameters.get("schema_hint")  # optional list of column names
    timeout_seconds = min(int(parameters.get("timeout_seconds", 120)), _MAX_TIMEOUT_SECONDS)

    if not objective:
        raise ValueError("objective parameter is required")

    # Build a constrained code-generation prompt
    schema_str = ""
    if schema_hint:
        cols = schema_hint if isinstance(schema_hint, list) else [schema_hint]
        schema_str = f"\nInput DataFrame columns: {', '.join(str(c) for c in cols)}"
    elif not df.empty:
        schema_str = f"\nInput DataFrame columns: {', '.join(str(c) for c in df.columns)}"

    prompt = (
        "Write a Python function named `transform` that accepts a single pandas DataFrame "
        "argument `df` and returns a pandas DataFrame.\n"
        f"{schema_str}\n"
        f"Objective: {objective}\n\n"
        "Rules:\n"
        "- The function signature must be: def transform(df):\n"
        "- Only use pd (pandas) and np (numpy) — no other imports.\n"
        "- Do not use open(), subprocess, os, sys, socket, requests, or any I/O.\n"
        "- Return only the Python code, no explanation or markdown fences."
    )

    # Invoke router Lambda
    lambda_client = boto3.client("lambda")
    router_payload = {
        "tool": "code",
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": 2048,
    }

    try:
        resp = lambda_client.invoke(
            FunctionName=router_arn,
            InvocationType="RequestResponse",
            Payload=json.dumps(router_payload).encode(),
        )
        resp_body = json.loads(resp["Payload"].read())
    except Exception as exc:
        raise ValueError(f"Router invocation failed: {exc}") from exc

    if resp.get("FunctionError"):
        raise ValueError(f"Router Lambda returned an error: {resp_body}")

    # Extract generated code from router response
    generated_code = None
    if isinstance(resp_body, dict):
        generated_code = (
            resp_body.get("content")
            or resp_body.get("text")
            or resp_body.get("result")
        )
    if not generated_code or not isinstance(generated_code, str):
        raise ValueError(f"Router returned no usable code in response: {resp_body}")

    # Issue #73: static analysis before execution — reject forbidden patterns
    violations = _analyze_generated_code(generated_code)
    if violations:
        raise ValueError(
            f"Generated code failed static analysis ({len(violations)} violation(s)): "
            + "; ".join(violations[:5])
        )

    # Write generated script to S3
    script_key = f"results/generated-scripts/{uuid.uuid4()}.py"
    s3 = boto3.client("s3")
    try:
        s3.put_object(
            Bucket=results_bucket,
            Key=script_key,
            Body=generated_code.encode("utf-8"),
            ContentType="text/x-python",
        )
    except Exception as exc:
        raise ValueError(f"Could not write generated script to S3: {exc}") from exc

    script_s3_uri = f"s3://{results_bucket}/{script_key}"

    # Delegate to custom_python_handler
    result_df, inner_diag = custom_python_handler(
        df,
        {"script_uri": script_s3_uri, "timeout_seconds": timeout_seconds},
    )

    diagnostics = dict(inner_diag)
    diagnostics.update({
        "objective": objective,
        "generated_script": generated_code,
        "script_s3_uri": script_s3_uri,
        "router_arn": router_arn,
    })

    logger.info(
        f"custom-generated complete: objective='{objective[:60]}', "
        f"script={script_s3_uri}"
    )
    return result_df, diagnostics
