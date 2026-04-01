"""
compute_profiles — AgentCore Gateway Lambda target

Lists available analysis profiles. Reads profile definitions from the
PROFILES_CONFIG environment variable (JSON array set by CDK at deploy time).

Event args:
  category  (str, optional)   Filter by profile category
  tags      (list, optional)  Filter by tags (any match)

Returns:
  {
    "profiles": [...],
    "count": int,
    "applied_category": str | null,
    "applied_tags": list
  }
"""

import json
import logging
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)

_PROFILES: list[dict] | None = None


def _load_profiles() -> list[dict]:
    global _PROFILES
    if _PROFILES is None:
        raw = os.environ.get("PROFILES_CONFIG", "[]")
        _PROFILES = json.loads(raw)
    return _PROFILES


def _project_profile(p: dict) -> dict:
    """Return the fields useful to the agent (omit internal entrypoint)."""
    return {
        "profile_id": p.get("profile_id"),
        "display_name": p.get("display_name"),
        "description": p.get("description"),
        "category": p.get("category"),
        "backend": p.get("backend"),
        "parameters": p.get("parameters", {}),
        "input_requirements": p.get("input_requirements", {}),
        "output_schema": p.get("output_schema", {}),
        "cost_estimate": p.get("cost_estimate", {}),
        "tags": p.get("tags", []),
    }


def handler(event: dict, context) -> dict:
    _tool_name = "unknown"
    try:
        raw = context.client_context.custom["bedrockAgentCoreToolName"]
        _tool_name = raw.split("___")[-1]
    except Exception:
        pass
    logger.info(json.dumps({"tool": _tool_name, "event": event}))

    category = (event.get("category") or "").strip().lower() or None
    tags = [t.lower() for t in (event.get("tags") or []) if t]

    profiles = _load_profiles()

    # Category filter
    if category:
        profiles = [p for p in profiles if p.get("category", "").lower() == category]

    # Tag filter (any match)
    if tags:
        profiles = [
            p for p in profiles
            if any(t in [x.lower() for x in p.get("tags", [])] for t in tags)
        ]

    projected = [_project_profile(p) for p in profiles]

    return {
        "profiles": projected,
        "count": len(projected),
        "applied_category": category,
        "applied_tags": tags,
    }
