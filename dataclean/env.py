"""
DataClean-Env — Core Environment  v2.0
Upgrades over v1:
  - Early-exit penalty: calling done before step 3 with quality < 0.60 → -0.10
  - Causal ordering bonus: dtype fix → null fill → outlier clip in correct order → +0.02
  - Business rule signals fed into observation (constraint violations count)
  - /reward_history and /explain data stored on state for new endpoints
"""
from __future__ import annotations

import time
import uuid
import hashlib
from typing import Any

import numpy as np
import pandas as pd

from dataclean.models import (
    ActionType, ColumnProfile, DataCleanAction,
    EpisodeState, Observation, OpsLogEntry, StepResult,
)
from dataclean.tasks import TASK_REGISTRY, Task


# ── Reward constants ──────────────────────────────────────────────────────────
R_NULL_FIXED        =  0.10
R_DTYPE_FIXED       =  0.10
R_OUTLIER_CLIPPED   =  0.08
R_DUPLICATE_REMOVED =  0.12
R_RENAME_CORRECT    =  0.05
R_DROP_CORRECT      =  0.04
R_DONE_BONUS        =  0.15
R_STEP_PENALTY      = -0.01
R_WRONG_COLUMN      = -0.05
R_INVALID_OP        = -0.03
R_PROVENANCE_BONUS  =  0.05
R_EARLY_EXIT        = -0.10   # NEW: done before step 3 with quality < 0.60
R_ORDER_BONUS       =  0.02   # NEW: correct causal ordering of operations

QUALITY_THRESHOLD   =  0.80
CONFIDENCE_BONUS    =  0.04
CONFIDENCE_PENALTY  = -0.06
CONFIDENCE_HIGH     =  0.75

# Causal ordering: dtype should precede nulls, nulls should precede outliers
CAUSAL_ORDER = ["fix_dtype", "fill_nulls", "clip_outliers"]


class DataCleanEnv:
    def __init__(self) -> None:
        self._df: pd.DataFrame | None = None
        self._raw_df: pd.DataFrame | None = None
        self._task: Task | None = None
        self._state: EpisodeState | None = None
        self._ops_log: list[OpsLogEntry] = []
        self._action_type_history: list[str] = []   # for ordering bonus
        self._reward_history: list[dict] = []        # for /reward_history endpoint
        self._explain_log: list[str] = []            # for /explain endpoint

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, task_id: str, seed: int = 42,
              custom_df: pd.DataFrame | None = None) -> Observation:
        rng = np.random.default_rng(seed)

        if custom_df is not None:
            class _CustomTask:
                description = "User-uploaded sandbox dataset."
                max_steps   = 30
                canonical_column_names: dict = {}
                irrelevant_columns: list     = []
                def generate(self, r): return custom_df.copy(deep=True)
                def grade(self, df):
                    null_s = float(1 - df.isna().mean().mean())
                    dup_s  = float(1 - df.duplicated().mean())
                    return round(float(np.mean([null_s, dup_s])), 4)
            self._task = _CustomTask()
            effective_task_id = "custom_upload"
        else:
            if task_id not in TASK_REGISTRY:
                raise ValueError(f"Unknown task '{task_id}'. Available: {list(TASK_REGISTRY)}")
            self._task = TASK_REGISTRY[task_id]
            effective_task_id = task_id

        self._df              = self._task.generate(rng)
        self._raw_df          = self._df.copy(deep=True)
        self._ops_log         = []
        self._action_type_history = []
        self._reward_history  = []
        self._explain_log     = []
        self._state           = EpisodeState(
            task_id=effective_task_id,
            episode_id=str(uuid.uuid4()),
            seed=seed,
            step=0,
            max_steps=self._task.max_steps,
        )
        return self._build_observation("Episode started. Inspect columns and begin cleaning.")

    def step(self, action: DataCleanAction) -> StepResult:
        if self._state is None or self._df is None:
            raise RuntimeError("Call reset() first.")
        if self._state.done:
            raise RuntimeError("Episode done. Call reset().")

        t0            = time.monotonic_ns()
        shape_before  = self._df.shape
        reward        = R_STEP_PENALTY
        breakdown: dict[str, float] = {"step_penalty": R_STEP_PENALTY}
        success       = True
        error_msg     = None
        feedback      = ""
        action_reward = 0.0

        # Dispatch
        try:
            action_reward, feedback = self._dispatch(action)
            reward += action_reward
            breakdown["action"] = action_reward
        except Exception as exc:
            reward    += R_INVALID_OP
            breakdown["invalid_op"] = R_INVALID_OP
            success    = False
            error_msg  = str(exc)
            feedback   = f"Error: {exc}"

        # Confidence calibration
        conf_r = self._confidence_reward(action.confidence,
                                         action_reward if success else R_INVALID_OP)
        reward += conf_r
        breakdown["confidence"] = conf_r

        # Causal ordering bonus (NEW)
        if success and action_reward > 0:
            order_r = self._ordering_bonus(action.action_type)
            reward  += order_r
            if order_r > 0:
                breakdown["ordering_bonus"] = order_r

        # Track action history for ordering
        self._action_type_history.append(str(action.action_type))

        # Ops log
        entry = OpsLogEntry(
            step=self._state.step,
            action_type=str(action.action_type),
            column=action.column,
            params=action.params,
            confidence=action.confidence,
            reward_delta=round(reward, 6),
            timestamp_ns=t0,
            df_shape_before=shape_before,
            df_shape_after=self._df.shape,
            success=success,
            error_message=error_msg,
        )
        self._ops_log.append(entry)
        self._state.ops_log.append(entry.model_dump())

        self._state.step         += 1
        self._state.total_reward += reward

        # Terminal
        done = False
        if action.action_type == ActionType.DONE:
            done = True
            quality = self.grade()

            # Early-exit penalty: done before step 4 with quality < 0.80
            if self._state.step <= 4 and quality < 0.80:
                early_pen = R_EARLY_EXIT
                reward    += early_pen
                breakdown["early_exit_penalty"] = early_pen
                feedback  += f" | Early-exit penalty applied (quality={quality:.2f}, step={self._state.step})."
                self._explain_log.append(
                    f"Step {self._state.step}: agent called DONE too early "
                    f"(quality={quality:.2f}). Penalty {early_pen} applied."
                )

            done_r, prov_r = self._terminal_rewards()
            reward        += done_r + prov_r
            breakdown["done_bonus"]   = done_r
            breakdown["provenance"]   = prov_r
            self._state.total_reward += done_r + prov_r

        if self._state.step >= self._state.max_steps:
            done = True

        self._state.done = done

        # Explain log entry
        self._explain_log.append(
            f"Step {self._state.step}: {action.action_type}"
            + (f" on '{action.column}'" if action.column else "")
            + f" → reward {reward:+.4f}. {feedback}"
        )

        # Reward history (for /reward_history endpoint)
        self._reward_history.append({
            "step":        self._state.step,
            "reward":      round(reward, 6),
            "breakdown":   breakdown,
            "cumulative":  round(self._state.total_reward, 6),
            "quality":     self._compute_quality_scores().get("overall", 0),
        })

        obs = self._build_observation(feedback)
        return StepResult(
            observation=obs,
            reward=round(reward, 6),
            done=done,
            info={
                "step": self._state.step,
                "total_reward": round(self._state.total_reward, 6),
                "success": success,
                "error": error_msg,
            },
            reward_breakdown=breakdown,
        )

    def grade(self) -> float:
        if self._df is None or self._task is None:
            raise RuntimeError("No active episode.")
        return self._task.grade(self._df)

    def verify_provenance(self) -> bool:
        if self._raw_df is None or not self._ops_log:
            return False
        try:
            replay = self._raw_df.copy(deep=True)
            for entry in self._ops_log:
                if not entry.success:
                    continue
                action = DataCleanAction(
                    action_type=entry.action_type,
                    column=entry.column,
                    params=entry.params,
                    confidence=entry.confidence,
                )
                replay = self._apply_to(replay, action)
            return self._df_hash(replay) == self._df_hash(self._df)
        except Exception:
            return False

    def get_ops_log(self)      -> list[dict]: return [e.model_dump() for e in self._ops_log]
    def get_reward_history(self) -> list[dict]: return self._reward_history
    def get_explain_log(self)  -> list[str]:  return self._explain_log

    # ── Ordering bonus ────────────────────────────────────────────────────────

    def _ordering_bonus(self, action_type: str) -> float:
        """
        +0.02 if this action type follows the correct causal order.
        fix_dtype should come before fill_nulls; fill_nulls before clip_outliers.
        """
        at = str(action_type)
        if at not in CAUSAL_ORDER:
            return 0.0
        idx = CAUSAL_ORDER.index(at)
        if idx == 0:
            return R_ORDER_BONUS  # first in order — always valid
        # Check that the preceding causal step was done at some point
        preceding = CAUSAL_ORDER[idx - 1]
        if preceding in self._action_type_history:
            return R_ORDER_BONUS
        return 0.0

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def _dispatch(self, action: DataCleanAction) -> tuple[float, str]:
        at = action.action_type
        if at == ActionType.DONE:
            return 0.0, "Agent signalled DONE."
        if at == ActionType.REMOVE_DUPLICATES:
            return self._act_remove_duplicates(action)
        if not action.column:
            raise ValueError(f"Action '{at}' requires a column name.")
        if action.column not in self._df.columns:
            raise ValueError(f"Column '{action.column}' not found.")
        if at == ActionType.FILL_NULLS:     return self._act_fill_nulls(action)
        if at == ActionType.FIX_DTYPE:      return self._act_fix_dtype(action)
        if at == ActionType.CLIP_OUTLIERS:  return self._act_clip_outliers(action)
        if at == ActionType.RENAME_COLUMN:  return self._act_rename_column(action)
        if at == ActionType.DROP_COLUMN:    return self._act_drop_column(action)
        raise ValueError(f"Unknown action_type: {at}")

    def _apply_to(self, df: pd.DataFrame, action: DataCleanAction) -> pd.DataFrame:
        env = DataCleanEnv.__new__(DataCleanEnv)
        env._df   = df
        env._task = self._task
        env._dispatch(action)
        return env._df

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _act_fill_nulls(self, action: DataCleanAction) -> tuple[float, str]:
        col = action.column
        strategy = action.params.get("strategy", "mean")
        null_before = self._df[col].isna().mean()
        if null_before < 0.005:
            return R_WRONG_COLUMN, f"'{col}' already clean (null={null_before:.1%})."
        if strategy == "mean":    val = self._df[col].mean()
        elif strategy == "median": val = self._df[col].median()
        elif strategy == "mode":   val = self._df[col].mode().iloc[0] if not self._df[col].mode().empty else None
        elif strategy == "constant": val = action.params.get("value", 0)
        elif strategy == "ffill":
            self._df[col] = self._df[col].ffill()
            null_after = self._df[col].isna().mean()
            return R_NULL_FIXED * (1 - null_after), f"ffill '{col}'. {null_before:.1%}→{null_after:.1%}"
        else:
            raise ValueError(f"Unknown strategy '{strategy}'.")
        if val is None:
            raise ValueError(f"No fill value for '{col}'.")
        self._df[col] = self._df[col].fillna(val)
        null_after = self._df[col].isna().mean()
        return R_NULL_FIXED * (1 - null_after), f"Filled '{col}' ({strategy}={val:.4g}). {null_before:.1%}→{null_after:.1%}"

    def _act_remove_duplicates(self, action: DataCleanAction) -> tuple[float, str]:
        subset = action.params.get("subset", None)
        before = len(self._df)
        dup_rate = self._df.duplicated(subset=subset).mean()
        if dup_rate < 0.005:
            return R_WRONG_COLUMN, f"No significant duplicates (rate={dup_rate:.1%})."
        self._df = self._df.drop_duplicates(subset=subset).reset_index(drop=True)
        removed = before - len(self._df)
        return R_DUPLICATE_REMOVED, f"Removed {removed} duplicates. {before}→{len(self._df)} rows."

    def _act_fix_dtype(self, action: DataCleanAction) -> tuple[float, str]:
        col = action.column
        target = action.params.get("target_dtype")
        if not target:
            raise ValueError("fix_dtype requires params.target_dtype.")
        before = str(self._df[col].dtype)
        if before == target:
            return R_WRONG_COLUMN, f"'{col}' already {target}."
        try:
            if target in ("int64","int32","int"):
                self._df[col] = pd.to_numeric(self._df[col], errors="coerce").astype("Int64")
            elif target in ("float64","float32","float"):
                self._df[col] = pd.to_numeric(self._df[col], errors="coerce").astype("float64")
            elif target in ("str","string","object"):
                self._df[col] = self._df[col].astype(str)
            elif target in ("datetime64","datetime"):
                self._df[col] = pd.to_datetime(self._df[col], errors="coerce")
            else:
                self._df[col] = self._df[col].astype(target)
        except Exception as exc:
            raise ValueError(f"Cannot cast '{col}' to '{target}': {exc}")
        return R_DTYPE_FIXED, f"'{col}': {before}→{str(self._df[col].dtype)}"

    def _act_clip_outliers(self, action: DataCleanAction) -> tuple[float, str]:
        col = action.column
        method = action.params.get("method", "iqr")
        if not pd.api.types.is_numeric_dtype(self._df[col]):
            raise ValueError(f"'{col}' not numeric.")
        before_std = self._df[col].std()
        if method == "iqr":
            q1, q3 = self._df[col].quantile(0.25), self._df[col].quantile(0.75)
            iqr = q3 - q1; lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
        elif method == "zscore":
            mu, sigma = self._df[col].mean(), self._df[col].std()
            t = action.params.get("threshold", 3.0)
            lo, hi = mu - t*sigma, mu + t*sigma
        elif method == "percentile":
            lo = self._df[col].quantile(action.params.get("lower_pct", 1)/100)
            hi = self._df[col].quantile(action.params.get("upper_pct", 99)/100)
        else:
            raise ValueError(f"Unknown method '{method}'.")
        n_out = ((self._df[col] < lo) | (self._df[col] > hi)).sum()
        if n_out == 0:
            return R_WRONG_COLUMN, f"No outliers in '{col}' ({method})."
        self._df[col] = self._df[col].clip(lo, hi)
        after_std = self._df[col].std()
        red = 1 - (after_std/before_std) if before_std > 0 else 0
        return R_OUTLIER_CLIPPED*(1+red), f"Clipped {n_out} outliers in '{col}'. std {before_std:.3g}→{after_std:.3g}"

    def _act_rename_column(self, action: DataCleanAction) -> tuple[float, str]:
        col = action.column
        new = action.params.get("new_name")
        if not new: raise ValueError("rename_column needs params.new_name.")
        if new in self._df.columns: raise ValueError(f"'{new}' already exists.")
        canonical = getattr(self._task, "canonical_column_names", {})
        self._df.rename(columns={col: new}, inplace=True)
        if new in canonical.values():
            return R_RENAME_CORRECT, f"Renamed '{col}'→'{new}' (canonical ✓)."
        return R_RENAME_CORRECT*0.5, f"Renamed '{col}'→'{new}' (non-canonical)."

    def _act_drop_column(self, action: DataCleanAction) -> tuple[float, str]:
        col = action.column
        irrelevant = getattr(self._task, "irrelevant_columns", [])
        self._df.drop(columns=[col], inplace=True)
        if col in irrelevant:
            return R_DROP_CORRECT, f"Dropped irrelevant '{col}' ✓."
        return R_WRONG_COLUMN*0.5, f"Dropped '{col}' — was not irrelevant (penalty)."

    # ── Reward helpers ────────────────────────────────────────────────────────

    def _confidence_reward(self, confidence: float, action_reward: float) -> float:
        if confidence >= CONFIDENCE_HIGH:
            return CONFIDENCE_BONUS if action_reward > 0 else CONFIDENCE_PENALTY
        return 0.0

    def _terminal_rewards(self) -> tuple[float, float]:
        quality = self.grade()
        done_r  = R_DONE_BONUS if quality >= QUALITY_THRESHOLD else R_DONE_BONUS * quality
        prov_r  = R_PROVENANCE_BONUS if self.verify_provenance() else 0.0
        return done_r, prov_r

    # ── Observation ───────────────────────────────────────────────────────────

    def _build_observation(self, last_action_result: str | None = None) -> Observation:
        cols = []
        for col in self._df.columns:
            series    = self._df[col]
            null_rate = float(series.isna().mean())
            flags: list[str] = []
            if null_rate > 0.20: flags.append("heavy_nulls")
            if null_rate > 0.01: flags.append("has_nulls")
            mean_v = std_v = min_v = max_v = None
            if pd.api.types.is_numeric_dtype(series):
                clean = series.dropna()
                if len(clean) > 0:
                    mean_v = float(clean.mean()); std_v = float(clean.std()) if len(clean)>1 else 0.0
                    min_v  = float(clean.min());  max_v = float(clean.max())
                    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
                    iqr = q3 - q1
                    if ((clean < q1-1.5*iqr) | (clean > q3+1.5*iqr)).mean() > 0.05:
                        flags.append("heavy_outliers")
            if series.dtype == object:
                if pd.to_numeric(series.dropna(), errors="coerce").notna().mean() > 0.7:
                    flags.append("type_chaos")
            cols.append(ColumnProfile(
                name=col, dtype=str(series.dtype), null_rate=null_rate,
                n_unique=int(series.nunique()), mean=mean_v, std=std_v,
                min=min_v, max=max_v,
                sample_values=series.dropna().head(3).tolist(),
                corruption_flags=flags,
            ))

        return Observation(
            task_id=self._state.task_id,
            episode_id=self._state.episode_id,
            step=self._state.step,
            budget_remaining=self._state.max_steps - self._state.step,
            n_rows=len(self._df), n_cols=len(self._df.columns),
            duplicate_rate=float(self._df.duplicated().mean()),
            columns=cols,
            last_action_result=last_action_result,
            ops_log=self._state.ops_log,
            quality_scores=self._compute_quality_scores(),
        )

    def _compute_quality_scores(self) -> dict[str, float]:
        df = self._df
        null_s = float(1 - df.isna().mean().mean())
        dup_s  = float(1 - df.duplicated().mean())
        type_issues = sum(
            1 for col in df.columns
            if df[col].dtype == object
            and pd.to_numeric(df[col].dropna(), errors="coerce").notna().mean() > 0.7
        )
        numeric_cols = sum(1 for col in df.columns if pd.api.types.is_numeric_dtype(df[col]))
        type_s = 1.0 - (type_issues / max(numeric_cols, 1))
        fracs  = []
        for col in df.select_dtypes(include="number").columns:
            clean = df[col].dropna()
            if len(clean) < 4: continue
            q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
            iqr = q3 - q1
            fracs.append(((clean < q1-1.5*iqr) | (clean > q3+1.5*iqr)).mean())
        outlier_s = float(1 - np.mean(fracs)) if fracs else 1.0
        return {
            "null_score":    round(null_s, 4),
            "type_score":    round(type_s, 4),
            "outlier_score": round(outlier_s, 4),
            "dup_score":     round(dup_s, 4),
            "overall":       round(float(np.mean([null_s, type_s, outlier_s, dup_s])), 4),
        }

    @staticmethod
    def _df_hash(df: pd.DataFrame) -> str:
        s = df.sort_values(by=list(df.columns)).reset_index(drop=True)
        return hashlib.md5(s.to_csv(index=False).encode()).hexdigest()