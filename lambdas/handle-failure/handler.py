"""
handle-failure — Step Functions Lambda (error catch)

Normalizes error information from Step Functions catch blocks into a
consistent structure for the NotifyFailure SNS step.

Input (from Step Functions catch, via result_path=$.error_info):
  The full execution context plus $.error_info from the catch block:
  {
    "execution_id": str,
    "profile": {"display_name": str},
    "user_arn": str,
    "error_info": {
      "Error": str,
      "Cause": str
    },
    ...
  }

Output:
  {
    "error_message": str,
    "error_code": str,
    "execution_id": str
  }
"""

import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def handler(event: dict, context) -> dict:
    execution_id = event.get("execution_id", "unknown")
    profile = event.get("profile") or {}
    error_info = event.get("error_info") or {}

    error_code = error_info.get("Error", "UnknownError")
    cause = error_info.get("Cause", "")

    # Attempt to parse cause as JSON (Lambda errors arrive this way)
    error_message = cause
    try:
        cause_obj = json.loads(cause)
        error_message = cause_obj.get("errorMessage") or cause
    except (json.JSONDecodeError, TypeError):
        pass

    if not error_message:
        error_message = f"Job failed with error: {error_code}"

    logger.error(json.dumps({
        "execution_id": execution_id,
        "profile_id": profile.get("profile_id"),
        "error_code": error_code,
        "error_message": error_message,
    }))

    return {
        "error_message": error_message,
        "error_code": error_code,
        "execution_id": execution_id,
    }
