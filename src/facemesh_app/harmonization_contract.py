"""Provide one shared harmonization capture and evaluation contract."""

from typing import Any, Dict, List

HARMONIZATION_SCHEMA_VERSION = 2

HARMONIZATION_PROMPTS: List[Dict[str, str]] = [
    {
        "name": "head-left",
        "instruction": "Turn your HEAD to the LEFT",
        "type": "head",
        "axis": "yaw",
        "direction": "negative",
    },
    {
        "name": "head-right",
        "instruction": "Turn your HEAD to the RIGHT",
        "type": "head",
        "axis": "yaw",
        "direction": "positive",
    },
    {
        "name": "head-up",
        "instruction": "Tilt your HEAD UP",
        "type": "head",
        "axis": "pitch",
        "direction": "positive",
    },
    {
        "name": "head-down",
        "instruction": "Tilt your HEAD DOWN",
        "type": "head",
        "axis": "pitch",
        "direction": "negative",
    },
    {
        "name": "eye-left",
        "instruction": "Look with your EYES to the LEFT (keep head still)",
        "type": "eye",
        "axis": "yaw",
        "direction": "negative",
    },
    {
        "name": "eye-right",
        "instruction": "Look with your EYES to the RIGHT (keep head still)",
        "type": "eye",
        "axis": "yaw",
        "direction": "positive",
    },
    {
        "name": "eye-up",
        "instruction": "Look with your EYES UP (keep head still)",
        "type": "eye",
        "axis": "pitch",
        "direction": "positive",
    },
    {
        "name": "eye-down",
        "instruction": "Look with your EYES DOWN (keep head still)",
        "type": "eye",
        "axis": "pitch",
        "direction": "negative",
    },
]

HARMONIZATION_TEST_CASE: Dict[str, Any] = {
    "id": "harmonization-axis-sign-v1",
    "description": "Validate head and eye sign conventions from harmonization points.",
    "comparisons": [
        {
            "name": "head_yaw_right_gt_left",
            "lhs": {"point": "head-right", "metric": "head_yaw"},
            "op": "gt",
            "rhs": {"point": "head-left", "metric": "head_yaw"},
            "minDelta": 2.0,
        },
        {
            "name": "head_pitch_up_gt_down",
            "lhs": {"point": "head-up", "metric": "head_pitch"},
            "op": "gt",
            "rhs": {"point": "head-down", "metric": "head_pitch"},
            "minDelta": 2.0,
        },
        {
            "name": "eye_yaw_right_gt_left",
            "lhs": {"point": "eye-right", "metric": "eye_yaw"},
            "op": "gt",
            "rhs": {"point": "eye-left", "metric": "eye_yaw"},
            "minDelta": 1.0,
        },
        {
            "name": "eye_pitch_up_gt_down",
            "lhs": {"point": "eye-up", "metric": "eye_pitch"},
            "op": "gt",
            "rhs": {"point": "eye-down", "metric": "eye_pitch"},
            "minDelta": 1.0,
        },
    ],
}
