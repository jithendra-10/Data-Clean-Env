"""
Unit tests for DataCleanEnv core behaviour.
Covers: reset, step, reward shape, terminal logic, provenance, ordering bonus.
"""
from __future__ import annotations

import pytest

from dataclean.env import (
    DataCleanEnv,
    R_NULL_FIXED,
    R_DUPLICATE_REMOVED,
    R_STEP_PENALTY,
    R_EARLY_EXIT,
    R_DONE_BONUS,
    QUALITY_THRESHOLD,
    CONFIDENCE_HIGH,
)
from dataclean.models import DataCleanAction, ActionType
from dataclean.tasks import TASK_REGISTRY


# ── reset ──────────────────────────────────────────────────────────────────

def test_reset_returns_valid_observation(env: DataCleanEnv, task_id: str):
    obs = env.reset(task_id, seed=42)
    assert obs.task_id == task_id
    assert obs.step == 0
    assert obs.n_rows > 0
    assert obs.n_cols > 0
    assert obs.budget_remaining == env._task.max_steps
    assert isinstance(obs.columns, list)
    assert "overall" in obs.quality_scores


def test_reset_unknown_task_raises(env: DataCleanEnv):
    with pytest.raises(ValueError):
        env.reset("does_not_exist", seed=1)


def test_reset_is_deterministic(task_id: str):
    e1, e2 = DataCleanEnv(), DataCleanEnv()
    o1 = e1.reset(task_id, seed=42)
    o2 = e2.reset(task_id, seed=42)
    assert o1.n_rows == o2.n_rows
    assert o1.n_cols == o2.n_cols
    assert o1.duplicate_rate == pytest.approx(o2.duplicate_rate)


# ── step ───────────────────────────────────────────────────────────────────

def test_step_without_reset_raises(env: DataCleanEnv):
    with pytest.raises(RuntimeError):
        env.step(DataCleanAction(action_type=ActionType.DONE))


def test_remove_duplicates_yields_positive_reward(env: DataCleanEnv):
    env.reset("task_1", seed=42)
    result = env.step(DataCleanAction(
        action_type=ActionType.REMOVE_DUPLICATES,
        confidence=0.9,
    ))
    assert result.observation.duplicate_rate < 0.01
    assert result.reward > 0
    assert "action" in result.reward_breakdown


def test_step_penalty_always_applied(env: DataCleanEnv):
    env.reset("task_1", seed=42)
    result = env.step(DataCleanAction(
        action_type=ActionType.REMOVE_DUPLICATES,
        confidence=0.9,
    ))
    assert result.reward_breakdown["step_penalty"] == pytest.approx(R_STEP_PENALTY)


def test_invalid_column_returns_negative(env: DataCleanEnv):
    env.reset("task_1", seed=42)
    result = env.step(DataCleanAction(
        action_type=ActionType.FILL_NULLS,
        column="this_column_does_not_exist",
        params={"strategy": "mean"},
        confidence=0.9,
    ))
    assert result.reward < 0
    assert result.info.get("success") is False


def test_fill_nulls_reduces_null_rate(env: DataCleanEnv):
    obs = env.reset("task_1", seed=42)
    null_col = next(
        (c.name for c in obs.columns if c.null_rate > 0.05 and c.mean is not None),
        None,
    )
    assert null_col, "expected a numeric null column in task_1"
    result = env.step(DataCleanAction(
        action_type=ActionType.FILL_NULLS,
        column=null_col,
        params={"strategy": "median"},
        confidence=0.9,
    ))
    new_col = next(c for c in result.observation.columns if c.name == null_col)
    assert new_col.null_rate < 0.05


# ── confidence calibration ─────────────────────────────────────────────────

def test_high_confidence_correct_gets_bonus(env: DataCleanEnv):
    env.reset("task_1", seed=42)
    result = env.step(DataCleanAction(
        action_type=ActionType.REMOVE_DUPLICATES,
        confidence=CONFIDENCE_HIGH + 0.1,
    ))
    assert result.reward_breakdown.get("confidence", 0) > 0


def test_high_confidence_wrong_gets_penalty(env: DataCleanEnv):
    env.reset("task_1", seed=42)
    result = env.step(DataCleanAction(
        action_type=ActionType.FILL_NULLS,
        column="nonexistent_col",
        params={"strategy": "mean"},
        confidence=CONFIDENCE_HIGH + 0.1,
    ))
    assert result.reward_breakdown.get("confidence", 0) < 0


# ── terminal / early exit ──────────────────────────────────────────────────

def test_early_done_triggers_penalty(env: DataCleanEnv):
    env.reset("task_3", seed=42)
    result = env.step(DataCleanAction(action_type=ActionType.DONE, confidence=0.5))
    assert result.done is True
    assert result.reward_breakdown.get("early_exit_penalty", 0) == pytest.approx(R_EARLY_EXIT)


def test_done_after_clean_yields_done_bonus(env: DataCleanEnv):
    env.reset("task_1", seed=42)
    for _ in range(5):
        env.step(DataCleanAction(
            action_type=ActionType.REMOVE_DUPLICATES,
            confidence=0.9,
        ))
        if env._state.done:
            break
    if not env._state.done:
        result = env.step(DataCleanAction(action_type=ActionType.DONE, confidence=0.9))
        assert "done_bonus" in result.reward_breakdown


# ── provenance ─────────────────────────────────────────────────────────────

def test_provenance_replay_matches_state(env: DataCleanEnv):
    env.reset("task_1", seed=42)
    env.step(DataCleanAction(
        action_type=ActionType.REMOVE_DUPLICATES,
        confidence=0.9,
    ))
    assert env.verify_provenance() is True


# ── grade ──────────────────────────────────────────────────────────────────

def test_grade_in_unit_interval(env: DataCleanEnv, task_id: str):
    env.reset(task_id, seed=42)
    score = env.grade()
    assert 0.0 <= score <= 1.0


def test_quality_scores_have_expected_keys(env: DataCleanEnv, task_id: str):
    obs = env.reset(task_id, seed=42)
    expected_keys = {"null_score", "type_score", "outlier_score", "dup_score", "overall"}
    assert expected_keys.issubset(obs.quality_scores.keys())
