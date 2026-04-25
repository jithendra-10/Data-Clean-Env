"""
Unit tests for task registry, generators, and graders.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dataclean.tasks import (
    TASK_REGISTRY,
    _null_score,
    _dtype_score,
    _outlier_score,
    _dup_score,
    _business_rule_score,
)


def test_registry_has_three_tasks():
    assert set(TASK_REGISTRY) == {"task_1", "task_2", "task_3"}


@pytest.mark.parametrize("task_id", list(TASK_REGISTRY))
def test_generator_produces_dataframe(task_id: str):
    rng = np.random.default_rng(42)
    df = TASK_REGISTRY[task_id].generate(rng)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert df.duplicated().mean() > 0  # tasks intentionally inject dupes


@pytest.mark.parametrize("task_id", list(TASK_REGISTRY))
def test_grader_returns_unit_interval(task_id: str):
    rng = np.random.default_rng(42)
    df = TASK_REGISTRY[task_id].generate(rng)
    score = TASK_REGISTRY[task_id].grade(df)
    assert 0.0 <= score <= 1.0


def test_dtype_score_perfect_when_correct():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    assert _dtype_score(df, {"a": "numeric"}) == pytest.approx(1.0)


def test_dtype_score_zero_when_wrong():
    df = pd.DataFrame({"a": ["x", "y", "z"]})
    assert _dtype_score(df, {"a": "numeric"}) == pytest.approx(0.0)


def test_dup_score_full_when_no_duplicates():
    df = pd.DataFrame({"a": list(range(100))})
    assert _dup_score(df) == pytest.approx(1.0)


def test_dup_score_drops_with_duplicates():
    df = pd.concat([pd.DataFrame({"a": [1] * 50}), pd.DataFrame({"a": [2] * 50})])
    assert _dup_score(df) < 1.0


def test_null_score_full_when_no_nulls():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    assert _null_score(df, ["a"]) == pytest.approx(1.0)


def test_null_score_drops_with_nulls():
    df = pd.DataFrame({"a": [1.0, np.nan, np.nan, np.nan]})
    assert _null_score(df, ["a"]) < 1.0


def test_outlier_score_clean_data_full():
    df = pd.DataFrame({"a": np.random.default_rng(0).normal(0, 1, 1000)})
    assert _outlier_score(df, ["a"]) >= 0.9


def test_business_rule_in_range_full():
    df = pd.DataFrame({"age": [20, 30, 40, 50]})
    assert _business_rule_score(df, {"age": "min:18,max:100"}) == pytest.approx(1.0)


def test_business_rule_out_of_range_lower():
    df = pd.DataFrame({"age": [200, 300, 400]})
    assert _business_rule_score(df, {"age": "min:18,max:100"}) == pytest.approx(0.0)
