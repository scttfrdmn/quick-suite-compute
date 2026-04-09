"""
peer_cohort.py — Peer institution identification for IPEDS benchmarking.

find_peer_cohort() identifies institutions similar to a given IPEDS unit
by Carnegie classification, enrollment band (±20%), control type, and
Pell concentration (±10 pts). Results are cached 30 days in DynamoDB.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import boto3
import pandas as pd

logger = logging.getLogger(__name__)

PEER_COHORT_TABLE = os.environ.get("PEER_COHORT_TABLE", "")
_dynamodb = None


def _ddb():
    global _dynamodb
    if _dynamodb is None:
        _dynamodb = boto3.resource("dynamodb")
    return _dynamodb


def find_peer_cohort(
    ipeds_unit_id: str,
    candidate_df: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Identify peer institutions for ipeds_unit_id from candidate_df.

    candidate_df must have columns: unit_id, carnegie_class, total_enrollment,
    control_type, pell_pct. Any column missing → filtered on available columns only.

    weights keys: enrollment (default 0.4), mission (default 0.4), pell (default 0.2).
    Returns {"peers": [unit_ids], "criteria_used": {...}, "cached": bool}.
    """
    if weights is None:
        weights = {}
    w_enrollment = float(weights.get("enrollment", 0.4))
    w_mission = float(weights.get("mission", 0.4))
    w_pell = float(weights.get("pell", 0.2))

    # Cache check
    if PEER_COHORT_TABLE:
        try:
            table = _ddb().Table(PEER_COHORT_TABLE)
            resp = table.get_item(Key={"unit_id": ipeds_unit_id})
            if "Item" in resp:
                item = resp["Item"]
                return {
                    "peers": json.loads(item.get("peers_json", "[]")),
                    "criteria_used": json.loads(item.get("criteria_json", "{}")),
                    "cached": True,
                }
        except Exception as exc:
            logger.warning(f"peer_cohort cache read failed: {exc}")

    # Find the institution row
    cols = candidate_df.columns.tolist()
    if "unit_id" not in cols:
        return {"peers": [], "criteria_used": {}, "cached": False}

    anchor_rows = candidate_df[candidate_df["unit_id"] == ipeds_unit_id]
    if anchor_rows.empty:
        return {"peers": [], "criteria_used": {}, "cached": False}
    anchor = anchor_rows.iloc[0]

    df = candidate_df[candidate_df["unit_id"] != ipeds_unit_id].copy()
    criteria_used: dict[str, Any] = {}

    # Carnegie classification filter
    if "carnegie_class" in cols and pd.notna(anchor.get("carnegie_class")):
        df = df[df["carnegie_class"] == anchor["carnegie_class"]]
        criteria_used["carnegie_class"] = anchor["carnegie_class"]

    # Enrollment band ±20%
    if "total_enrollment" in cols and pd.notna(anchor.get("total_enrollment")):
        enr = float(anchor["total_enrollment"])
        df = df[
            (df["total_enrollment"] >= enr * 0.80) &
            (df["total_enrollment"] <= enr * 1.20)
        ]
        criteria_used["enrollment_band"] = [enr * 0.80, enr * 1.20]

    # Control type filter
    if "control_type" in cols and pd.notna(anchor.get("control_type")):
        df = df[df["control_type"] == anchor["control_type"]]
        criteria_used["control_type"] = anchor["control_type"]

    # Pell concentration ±10 pts
    if "pell_pct" in cols and pd.notna(anchor.get("pell_pct")):
        pell = float(anchor["pell_pct"])
        df = df[
            (df["pell_pct"] >= pell - 10.0) &
            (df["pell_pct"] <= pell + 10.0)
        ]
        criteria_used["pell_band"] = [pell - 10.0, pell + 10.0]

    # Weighted score (proximity to anchor)
    scores = pd.Series(0.0, index=df.index)
    if "total_enrollment" in df.columns and pd.notna(anchor.get("total_enrollment")):
        enr = float(anchor["total_enrollment"])
        enr_diff = (df["total_enrollment"] - enr).abs() / max(enr, 1)
        scores += w_enrollment * (1 - enr_diff.clip(0, 1))
    if "pell_pct" in df.columns and pd.notna(anchor.get("pell_pct")):
        pell = float(anchor["pell_pct"])
        pell_diff = (df["pell_pct"] - pell).abs() / 100.0
        scores += w_pell * (1 - pell_diff.clip(0, 1))
    # mission similarity: same carnegie = max score already filtered above
    scores += w_mission * 1.0

    df = df.copy()
    df["_score"] = scores
    df = df.sort_values("_score", ascending=False)
    peers = df["unit_id"].tolist()

    # Write cache
    if PEER_COHORT_TABLE and peers:
        try:
            table = _ddb().Table(PEER_COHORT_TABLE)
            table.put_item(Item={
                "unit_id": ipeds_unit_id,
                "peers_json": json.dumps(peers),
                "criteria_json": json.dumps(criteria_used),
                "ttl": int(time.time()) + 30 * 86400,
            })
        except Exception as exc:
            logger.warning(f"peer_cohort cache write failed: {exc}")

    return {"peers": peers, "criteria_used": criteria_used, "cached": False}
