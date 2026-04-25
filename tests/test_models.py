"""
Tests for Pydantic models (action validation, observation prompt rendering).
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from dataclean.models import DataCleanAction, ActionType


def test_action_default_confidence():
    a = DataCleanAction(action_type=ActionType.DONE)
    assert a.confidence == pytest.approx(0.5)


def test_confidence_clamped_lower():
    with pytest.raises(ValidationError):
        DataCleanAction(action_type=ActionType.DONE, confidence=-0.1)


def test_confidence_clamped_upper():
    with pytest.raises(ValidationError):
        DataCleanAction(action_type=ActionType.DONE, confidence=1.5)


def test_column_strip_whitespace():
    a = DataCleanAction(action_type=ActionType.FILL_NULLS, column="  age  ")
    assert a.column == "age"


def test_action_type_enum_serialised():
    a = DataCleanAction(action_type="fill_nulls")
    dumped = a.model_dump()
    assert dumped["action_type"] == "fill_nulls"
