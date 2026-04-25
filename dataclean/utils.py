"""
dataclean/utils.py
Shared prompt builder, JSON parser, and system prompt.
Imported by both inference.py and server.py — no circular deps.
"""
from __future__ import annotations
import json


SYSTEM_PROMPT = """You are an expert data cleaning agent inside DataClean-Env.

STRATEGY (strict order — ordering matters for reward):
1. remove_duplicates first — always, even if dup_rate looks small
2. fix_dtype on any column with flag type_chaos
3. fill_nulls — heaviest null_rate column first
4. clip_outliers — columns with heavy_outliers flag
5. drop_column — only columns explicitly flagged irrelevant
6. done — only when overall quality >= 0.90

CONFIDENCE RULES (directly affect your score):
- 0.85–0.95 when certain this action helps this column
- 0.50–0.70 when column is borderline or you are unsure
- confidence >= 0.75 on CORRECT action = +0.04 bonus
- confidence >= 0.75 on WRONG action  = -0.06 penalty

NEVER call done before step 3 unless quality is already >= 0.90.
Calling done on step 1 or 2 triggers an early-exit penalty.

Respond ONLY with valid JSON. No markdown, no explanation.
{
  "action_type": "fill_nulls|remove_duplicates|fix_dtype|clip_outliers|rename_column|drop_column|done",
  "column": "<column name or null>",
  "params": {},
  "confidence": <0.0-1.0>
}"""


def obs_to_prompt(obs) -> str:
    """Works with both Observation object (env direct) and dict (server API)."""
    # Handle both object and dict
    if hasattr(obs, "columns"):
        cols       = obs.columns
        step       = obs.step
        budget     = obs.budget_remaining
        n_rows     = obs.n_rows
        dup_rate   = obs.duplicate_rate
        scores     = obs.quality_scores
        last_act   = obs.last_action_result or "N/A"
        ops_count  = len(obs.ops_log)
        get_flags  = lambda c: c.corruption_flags
        get_name   = lambda c: c.name
        get_dtype  = lambda c: c.dtype
        get_nulls  = lambda c: c.null_rate
        get_unique = lambda c: c.n_unique
        get_mean   = lambda c: c.mean
        get_std    = lambda c: c.std
        get_min    = lambda c: c.min
        get_max    = lambda c: c.max
    else:
        cols       = obs.get("columns", [])
        step       = obs.get("step", 0)
        budget     = obs.get("budget_remaining", "?")
        n_rows     = obs.get("n_rows", 0)
        dup_rate   = obs.get("duplicate_rate", 0)
        scores     = obs.get("quality_scores", {})
        last_act   = obs.get("last_action_result", "N/A")
        ops_count  = len(obs.get("ops_log", []))
        get_flags  = lambda c: c.get("corruption_flags", [])
        get_name   = lambda c: c.get("name", "")
        get_dtype  = lambda c: c.get("dtype", "")
        get_nulls  = lambda c: c.get("null_rate", 0)
        get_unique = lambda c: c.get("n_unique", 0)
        get_mean   = lambda c: c.get("mean")
        get_std    = lambda c: c.get("std")
        get_min    = lambda c: c.get("min")
        get_max    = lambda c: c.get("max")

    col_lines = []
    for c in cols:
        flags = ", ".join(get_flags(c)) if get_flags(c) else "clean"
        stats = ""
        if get_mean(c) is not None:
            stats = (f" mean={get_mean(c):.2f} std={get_std(c):.2f}"
                     f" min={get_min(c):.2f} max={get_max(c):.2f}")
        col_lines.append(
            f"  {get_name(c):25s} dtype={get_dtype(c):8s}"
            f" nulls={get_nulls(c):.1%} unique={get_unique(c):4d}"
            f"{stats} flags=[{flags}]"
        )

    overall = scores.get("overall", 0)
    score_parts = " | ".join(f"{k}={v:.3f}" for k, v in scores.items() if k != "overall")

    return (
        f"Step {step} | Budget: {budget} | Rows: {n_rows} | Dup rate: {dup_rate:.1%}\n"
        f"Quality: overall={overall:.3f} | {score_parts}\n"
        f"\nColumn profiles:\n" + "\n".join(col_lines)
        + f"\n\nLast action: {last_act}"
        + f"\nOps applied: {ops_count}"
        + "\n\nChoose your next action (JSON only, no markdown):"
    )


def parse_action(raw: str) -> dict:
    """Safe JSON parser — strips markdown fences, finds first {...} block."""
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text  = parts[1] if len(parts) > 1 else text
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]
    return json.loads(text)