"""
DataClean-Env — Gradio UI (v2)
==============================
Professional, branded UI: custom theme, custom CSS, live status bar,
reward timeline, provenance badges. Mounted by `server.py` at /.

Public surface:
    build_gradio_app(env_factory) -> gr.Blocks
        env_factory: callable that returns a fresh DataCleanEnv.

Design notes
------------
- Single source of truth for visual identity: BRAND_CSS + GR_THEME.
- All callbacks are pure functions of (state_dict, *inputs) -> outputs.
- The chat copilot streams thought/action/env-response as separate messages
  for a clearer "agent loop" feel.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable

import pandas as pd

from dataclean.models import DataCleanAction
from dataclean.utils import obs_to_prompt, parse_action, SYSTEM_PROMPT


# ── Brand CSS ─────────────────────────────────────────────────────────────────

BRAND_CSS = """
:root {
    --bg-0: #0a0c10;
    --bg-1: #0f1218;
    --bg-2: #161b22;
    --bg-3: #1c222b;
    --line: #232a35;
    --text-0: #e6edf3;
    --text-1: #a8b3bf;
    --text-2: #6e7781;
    --accent: #7c5cff;
    --accent-2: #00d4ff;
    --good: #3fb950;
    --warn: #d29922;
    --bad: #f85149;
    --radius: 10px;
    --mono: ui-monospace, SFMono-Regular, "JetBrains Mono", Menlo, Consolas, monospace;
    --display: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
}

/* ── Global background ───────────────────────────────────────────────────── */
.gradio-container, .app, body {
    background: radial-gradient(1200px 600px at 80% -10%, rgba(124,92,255,.10), transparent 60%),
                radial-gradient(900px 500px at -10% 110%, rgba(0,212,255,.08), transparent 60%),
                var(--bg-0) !important;
    color: var(--text-0) !important;
    font-family: var(--display) !important;
}

/* Hide default Gradio header noise */
footer { display: none !important; }

/* ── Top brand bar ───────────────────────────────────────────────────────── */
.dc-topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 18px; margin: 0 0 18px 0;
    background: linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,0));
    border: 1px solid var(--line); border-radius: var(--radius);
    backdrop-filter: blur(8px);
}
.dc-brand { display: flex; align-items: center; gap: 12px; }
.dc-logo {
    width: 28px; height: 28px; border-radius: 8px;
    background: conic-gradient(from 200deg, var(--accent), var(--accent-2), var(--accent));
    box-shadow: 0 0 18px rgba(124,92,255,.55), inset 0 0 12px rgba(0,212,255,.35);
}
.dc-brand-name {
    font-family: var(--mono); font-size: 14px; letter-spacing: .14em;
    color: var(--text-0); text-transform: uppercase;
}
.dc-brand-sub {
    font-family: var(--mono); font-size: 11px; color: var(--text-2);
    letter-spacing: .14em; text-transform: uppercase;
}
.dc-pill {
    font-family: var(--mono); font-size: 11px; letter-spacing: .12em;
    text-transform: uppercase; padding: 4px 10px; border-radius: 999px;
    border: 1px solid var(--line); color: var(--text-1);
    background: rgba(255,255,255,.02);
}
.dc-pill-good { color: var(--good); border-color: rgba(63,185,80,.4); }
.dc-pill-warn { color: var(--warn); border-color: rgba(210,153,34,.4); }
.dc-pill-bad  { color: var(--bad);  border-color: rgba(248,81,73,.4); }

/* ── Status strip (live KPIs) ────────────────────────────────────────────── */
.dc-status {
    display: grid; grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 10px; margin-bottom: 18px;
}
.dc-stat {
    background: var(--bg-1); border: 1px solid var(--line);
    border-radius: var(--radius); padding: 12px 14px;
}
.dc-stat-label {
    font-family: var(--mono); font-size: 10px; letter-spacing: .14em;
    color: var(--text-2); text-transform: uppercase;
}
.dc-stat-value {
    font-family: var(--mono); font-size: 22px; color: var(--text-0); margin-top: 4px;
    font-weight: 600;
}
.dc-stat-value.accent { color: var(--accent-2); }
.dc-stat-value.good   { color: var(--good); }
.dc-stat-value.warn   { color: var(--warn); }
.dc-stat-value.bad    { color: var(--bad); }

/* ── Cards & panels ──────────────────────────────────────────────────────── */
.dc-card {
    background: var(--bg-1); border: 1px solid var(--line);
    border-radius: var(--radius); padding: 18px;
}
.dc-card h3 {
    font-family: var(--mono); font-size: 11px; letter-spacing: .16em;
    color: var(--text-2); text-transform: uppercase; margin: 0 0 10px 0;
    font-weight: 600;
}
.dc-section-title {
    display: flex; align-items: center; gap: 10px;
    font-family: var(--mono); font-size: 11px; letter-spacing: .16em;
    color: var(--text-2); text-transform: uppercase; margin: 16px 0 8px 0;
}
.dc-section-title::before {
    content: ""; display: block; width: 6px; height: 6px;
    background: var(--accent); border-radius: 999px;
    box-shadow: 0 0 8px var(--accent);
}

/* ── Override Gradio component styling ─────────────────────────────────── */
.gr-button-primary, button.primary {
    background: linear-gradient(180deg, #8b6cff, #6a48f5) !important;
    border: 1px solid rgba(255,255,255,.08) !important;
    color: white !important; font-weight: 600 !important;
    box-shadow: 0 1px 0 rgba(255,255,255,.08) inset, 0 8px 22px rgba(124,92,255,.25) !important;
}
.gr-button-primary:hover { filter: brightness(1.08); }
.gr-button-secondary {
    background: var(--bg-2) !important; border: 1px solid var(--line) !important;
    color: var(--text-0) !important;
}
.gr-button-stop {
    background: rgba(248,81,73,.12) !important; color: var(--bad) !important;
    border: 1px solid rgba(248,81,73,.35) !important;
}

.gr-textbox, .gr-input, textarea, input[type="text"], input[type="number"],
.gr-dropdown, .gr-slider {
    background: var(--bg-2) !important; border: 1px solid var(--line) !important;
    color: var(--text-0) !important; border-radius: 8px !important;
    font-family: var(--mono) !important;
}
.gr-textbox:focus-within, .gr-input:focus-within { border-color: var(--accent) !important; }

/* Tabs */
.tab-nav button { font-family: var(--mono) !important; letter-spacing: .12em !important;
    text-transform: uppercase !important; font-size: 11px !important; }
.tab-nav button.selected { color: var(--accent-2) !important;
    border-bottom: 2px solid var(--accent-2) !important; }

/* Code/markdown */
.prose code, code {
    background: var(--bg-2) !important; border: 1px solid var(--line) !important;
    color: var(--accent-2) !important; padding: 1px 6px !important;
    border-radius: 6px !important; font-family: var(--mono) !important;
}
.prose pre {
    background: var(--bg-2) !important; border: 1px solid var(--line) !important;
    border-radius: 8px !important;
}

/* Chatbot polish */
.message { border-radius: 8px !important; border: 1px solid var(--line) !important; }
.message.user { background: var(--bg-2) !important; }
.message.bot  { background: var(--bg-1) !important; }

/* Subtle pulse for the brand logo */
@keyframes dc-pulse {
    0%,100% { box-shadow: 0 0 18px rgba(124,92,255,.5), inset 0 0 12px rgba(0,212,255,.35); }
    50%     { box-shadow: 0 0 26px rgba(124,92,255,.85), inset 0 0 14px rgba(0,212,255,.6); }
}
.dc-logo { animation: dc-pulse 4s ease-in-out infinite; }

/* Reward timeline list */
.dc-timeline { font-family: var(--mono); font-size: 12px; line-height: 1.7;
    color: var(--text-1); padding-left: 0; list-style: none; margin: 0; }
.dc-timeline li { display: flex; gap: 10px; align-items: center;
    border-bottom: 1px dashed rgba(255,255,255,.05); padding: 4px 0; }
.dc-timeline .step { color: var(--text-2); width: 36px; }
.dc-timeline .delta.pos { color: var(--good); }
.dc-timeline .delta.neg { color: var(--bad); }
.dc-timeline .label { color: var(--text-0); flex: 1; }
"""


# ── Gradio theme ─────────────────────────────────────────────────────────────

def _build_theme():
    import gradio as gr
    return gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#f0ecff", c100="#e1d8ff", c200="#c4b1ff", c300="#a78bff",
            c400="#8b6cff", c500="#7c5cff", c600="#6a48f5", c700="#5638d8",
            c800="#3f25a3", c900="#241466", c950="#150a3d",
        ),
        secondary_hue=gr.themes.Color(
            c50="#e6fbff", c100="#c2f3ff", c200="#85e6ff", c300="#3edaff",
            c400="#00d4ff", c500="#00b4d8", c600="#0096b8", c700="#007490",
            c800="#005268", c900="#003040", c950="#001a24",
        ),
        neutral_hue=gr.themes.Color(
            c50="#f0f3f6", c100="#d1d6dd", c200="#a8b3bf", c300="#6e7781",
            c400="#4a5260", c500="#2f3742", c600="#232a35", c700="#1c222b",
            c800="#161b22", c900="#0f1218", c950="#0a0c10",
        ),
        font=("ui-sans-serif", "Inter", "Segoe UI", "system-ui"),
        font_mono=("ui-monospace", "SFMono-Regular", "JetBrains Mono", "Menlo", "Consolas"),
    ).set(
        body_background_fill="*neutral_950",
        background_fill_primary="*neutral_900",
        background_fill_secondary="*neutral_800",
        body_text_color="*neutral_50",
        block_background_fill="*neutral_900",
        block_border_color="*neutral_700",
        block_border_width="1px",
        block_radius="10px",
        button_primary_background_fill="linear-gradient(180deg, #8b6cff, #6a48f5)",
        button_primary_background_fill_hover="linear-gradient(180deg, #9a7eff, #7c5cff)",
        button_primary_text_color="white",
        button_secondary_background_fill="*neutral_800",
        button_secondary_text_color="*neutral_50",
        slider_color="*primary_400",
        input_background_fill="*neutral_800",
        input_border_color="*neutral_700",
    )


# ── HTML helpers ─────────────────────────────────────────────────────────────

def _topbar_html() -> str:
    return """
    <div class="dc-topbar">
      <div class="dc-brand">
        <div class="dc-logo"></div>
        <div>
          <div class="dc-brand-name">DataClean&middot;Env</div>
          <div class="dc-brand-sub">openenv-compliant rl environment</div>
        </div>
      </div>
      <div style="display:flex; gap:8px; align-items:center;">
        <span class="dc-pill">v1.0</span>
        <span class="dc-pill dc-pill-good">live</span>
        <span class="dc-pill">PyTorch &times; Meta &times; HF</span>
      </div>
    </div>
    """


def _status_strip(step: int, budget: int, quality: float, reward: float, prov_ok: bool) -> str:
    q_class = "good" if quality >= 0.85 else ("warn" if quality >= 0.6 else "bad")
    r_class = "good" if reward >= 0 else "bad"
    p_label = "REPRODUCIBLE" if prov_ok else "DRIFT"
    p_class = "good" if prov_ok else "warn"
    return f"""
    <div class="dc-status">
      <div class="dc-stat">
        <div class="dc-stat-label">Step</div>
        <div class="dc-stat-value">{step:02d}</div>
      </div>
      <div class="dc-stat">
        <div class="dc-stat-label">Budget Left</div>
        <div class="dc-stat-value accent">{budget}</div>
      </div>
      <div class="dc-stat">
        <div class="dc-stat-label">Quality</div>
        <div class="dc-stat-value {q_class}">{quality:.3f}</div>
      </div>
      <div class="dc-stat">
        <div class="dc-stat-label">Total Reward</div>
        <div class="dc-stat-value {r_class}">{reward:+.3f}</div>
      </div>
      <div class="dc-stat">
        <div class="dc-stat-label">Provenance</div>
        <div class="dc-stat-value {p_class}">{p_label}</div>
      </div>
    </div>
    """


def _idle_status_strip() -> str:
    return _status_strip(0, 0, 0.0, 0.0, False).replace(">DRIFT<", ">IDLE<")


def _timeline_html(history: list[dict]) -> str:
    if not history:
        return '<div class="dc-card"><h3>Reward Timeline</h3>' \
               '<div style="color:var(--text-2); font-family:var(--mono); font-size:12px;">' \
               'no steps yet — apply an action to see the trace.</div></div>'
    rows = []
    for h in history[-15:]:
        cls = "pos" if h["reward"] >= 0 else "neg"
        breakdown = h.get("breakdown", {})
        major = max((k for k in breakdown if k != "step_penalty"),
                    key=lambda k: abs(breakdown[k]),
                    default="step")
        rows.append(
            f'<li><span class="step">#{h["step"]:02d}</span>'
            f'<span class="label">{major}</span>'
            f'<span class="delta {cls}">{h["reward"]:+.3f}</span>'
            f'<span class="step">q={h.get("quality", 0):.2f}</span></li>'
        )
    return ('<div class="dc-card"><h3>Reward Timeline</h3>'
            '<ul class="dc-timeline">' + "".join(rows) + '</ul></div>')


# ── Build app ────────────────────────────────────────────────────────────────

def build_gradio_app(env_factory: Callable) -> Any:
    """Construct and return the Gradio Blocks app. Mounted by server.py."""
    try:
        import gradio as gr
        from openai import OpenAI
    except ImportError:
        return None

    _ui_env = env_factory()

    # ── Callbacks ────────────────────────────────────────────────────────────

    def ui_reset(task_id: str, seed: int):
        obs = _ui_env.reset(task_id, seed=int(seed))
        quality = obs.quality_scores.get("overall", 0)
        return (
            obs_to_prompt(obs),
            _status_strip(0, obs.budget_remaining, quality, 0.0, False),
            _timeline_html([]),
            "Episode initialised. Inspect columns and pick your first move.",
            "{}",
        )

    def ui_step(action_type: str, column: str, strategy: str,
                target_dtype: str, clip_method: str, confidence: float):
        if _ui_env._state is None:
            return (
                "No active episode. Click Reset first.",
                _idle_status_strip(),
                _timeline_html([]),
                "No active episode.",
                "{}",
            )
        if _ui_env._state.done:
            quality = _ui_env._compute_quality_scores().get("overall", 0)
            return (
                "Episode finished. Hit Reset to start over.",
                _status_strip(_ui_env._state.step, 0, quality,
                              _ui_env._state.total_reward,
                              _ui_env.verify_provenance()),
                _timeline_html(_ui_env.get_reward_history()),
                "Episode terminal — start a new one.",
                "{}",
            )

        params: dict[str, Any] = {}
        col = (column or "").strip() or None

        if action_type == "fill_nulls" and strategy:
            params["strategy"] = strategy
        elif action_type == "fix_dtype" and target_dtype:
            params["target_dtype"] = target_dtype
        elif action_type == "clip_outliers" and clip_method:
            params["method"] = clip_method

        try:
            action = DataCleanAction(
                action_type=action_type,
                column=col,
                params=params,
                confidence=confidence,
            )
            result = _ui_env.step(action)
            obs = result.observation
            quality = obs.quality_scores.get("overall", 0)
            return (
                obs_to_prompt(obs),
                _status_strip(obs.step, obs.budget_remaining, quality,
                              _ui_env._state.total_reward,
                              _ui_env.verify_provenance()),
                _timeline_html(_ui_env.get_reward_history()),
                f"step {obs.step} | reward {result.reward:+.4f} | done={result.done}\n"
                f"{obs.last_action_result or ''}",
                json.dumps(result.reward_breakdown, indent=2),
            )
        except Exception as exc:
            return (
                str(exc),
                _idle_status_strip(),
                _timeline_html(_ui_env.get_reward_history()),
                f"Error: {exc}",
                "{}",
            )

    def ui_grade():
        if _ui_env._df is None:
            return "No active episode."
        score = _ui_env.grade()
        prov = _ui_env.verify_provenance()
        qs = _ui_env._compute_quality_scores()
        return json.dumps(
            {"score": score, "provenance_reproducible": prov, "quality_breakdown": qs},
            indent=2,
        )

    # ── Layout ───────────────────────────────────────────────────────────────

    # Gradio 6 moved `theme` / `css` from Blocks() → launch().
    # When mounted into FastAPI (no launch()), constructor args may be ignored.
    # Inject CSS explicitly so styling applies in both Gradio 4.x and 6.x.
    with gr.Blocks(
        title="DataClean-Env",
        analytics_enabled=False,
    ) as demo:
        gr.HTML(f"<style>{BRAND_CSS}</style>")
        # Theme is optional; keep best-effort for Gradio versions that support it.
        try:
            demo.theme = _build_theme()
        except Exception:
            pass
        gr.HTML(_topbar_html())

        # Live status (driven by callbacks)
        status_html = gr.HTML(_idle_status_strip())

        with gr.Tabs():
            # ──────────────────────────────────────────────────────────────
            with gr.Tab("Overview"):
                with gr.Accordion("Platform Navigation & Key Metrics", open=True):
                    gr.Markdown(
                        "**Top Bar (Always Visible)** — Live episode KPIs: current **Step**, "
                        "remaining **Budget**, aggregate **Quality** (0–1 preview from the dataframe), "
                        "cumulative **Total Reward** (RL training signal), and **Provenance** "
                        "(whether every successful op can be replayed from the raw table).\n\n"
                        "**Navigation Tabs**\n"
                        "- **Overview** — Platform motivation, tech stack, and navigation guide.\n"
                        "- **Manual Console** — Step-by-step human control (`reset → step → … → grade`) to simulate an RL rollout. "
                        "Reward breakdown explicitly details the causal shaping of each action.\n"
                        "- **Agent Copilot** — Connects a hosted LLM to autonomously issue JSON actions. "
                        "*(Note: Requires a valid HF Inference Token with the selected model enabled)*.\n"
                        "- **API** — Comprehensive REST endpoints that OpenEnv validators interact with. "
                        "Full OpenAPI documentation is automatically generated at `/docs`.\n\n"
                        "**Core Evaluation Metrics**\n"
                        "1. **Per-Step Reward** — The dense learning signal returned after every action (can be negative for destructive operations).\n"
                        "2. **Grade Snapshot (`POST /grader`)** — The deterministic, final task score evaluated against the current table state (0–1)."
                    )
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown(
                            "### A reinforcement-learning environment for tabular data engineering\n\n"
                            "DataClean-Env is the first OpenEnv-compliant benchmark that scores LLM agents on"
                            " the discipline of cleaning data — null imputation, dtype coercion, outlier"
                            " clipping, deduplication, provenance tracking, and **calibrated confidence**.\n\n"
                            "Three datasets, three difficulties, one strict grader."
                        )
                        gr.Markdown(
                            "<div class='dc-section-title'>What you can do here</div>"
                            "1. **Manual Console** &mdash; drive the env step-by-step yourself.  \n"
                            "2. **Agent Copilot** &mdash; let a hosted LLM propose JSON actions via the HF Router.  \n"
                            "3. **Reward + timeline** &mdash; inspect per-step shaping (dense RL signal).  \n"
                            "4. **API** &mdash; call `/reset`, `/step`, `/grader` directly (`/docs`).",
                            elem_classes=["prose"],
                        )
                    with gr.Column(scale=1):
                        gr.HTML(
                            "<div class='dc-card'>"
                            "<h3>Stack</h3>"
                            "<div style='font-family:var(--mono); font-size:12px; line-height:2; color:var(--text-1);'>"
                            "<div>runtime &nbsp; <code>FastAPI · Pydantic v2</code></div>"
                            "<div>data &nbsp; &nbsp; &nbsp;<code>pandas · numpy</code></div>"
                            "<div>model &nbsp; &nbsp;<code>Llama-3.2 · GRPO · TRL</code></div>"
                            "<div>spec &nbsp; &nbsp; &nbsp;<code>OpenEnv v0.2</code></div>"
                            "<div>deploy &nbsp; <code>HuggingFace Spaces (Docker)</code></div>"
                            "</div></div>"
                        )
                        gr.HTML(
                            "<div class='dc-card' style='margin-top:14px;'>"
                            "<h3>Reward Highlights</h3>"
                            "<div style='font-family:var(--mono); font-size:12px; line-height:2; color:var(--text-1);'>"
                            "<div><span style='color:var(--good);'>+0.10</span> &nbsp; null fix (scaled)</div>"
                            "<div><span style='color:var(--good);'>+0.12</span> &nbsp; remove duplicates</div>"
                            "<div><span style='color:var(--good);'>+0.04</span> &nbsp; calibrated confidence</div>"
                            "<div><span style='color:var(--good);'>+0.05</span> &nbsp; reproducible provenance</div>"
                            "<div><span style='color:var(--bad);'>-0.06</span> &nbsp; high-conf wrong move</div>"
                            "<div><span style='color:var(--bad);'>-0.10</span> &nbsp; early exit penalty</div>"
                            "</div></div>"
                        )

            # ──────────────────────────────────────────────────────────────
            with gr.Tab("Manual Console"):
                with gr.Accordion("Step-by-step: Workspace Controls", open=True):
                    gr.Markdown(
                        "1. **Dataset + Seed** — Select the synthetic corruption profile; "
                        "maintaining the same seed ensures a reproducible environment.\n"
                        "2. **Reset Episode** — Initializes the environment (`POST /reset`) and generates the "
                        "**Live Observation** state (column stats, corruption flags, and quality preview).\n"
                        "3. **Action Console** — Construct a single RL action payload: select `action_type`, target `column`, "
                        "parameters (strategy / dtype / clip method), and your **Confidence** interval.\n"
                        "4. **Execute Action** — Submits the payload to the Gym, returning the **Last action result**, "
                        "a detailed **Reward breakdown** (JSON), and updating the **Reward Timeline**.\n"
                        "5. **Official Grader** — The **Grade snapshot** evaluates the current DataFrame "
                        "against the deterministic validator (`POST /grader`), returning a final 0–1 score and provenance status.\n\n"
                        "**Tip:** The `Quality` metric in the top strip is a fast heuristic preview. The **Grade snapshot** is the "
                        "authoritative task score."
                    )
                with gr.Row():
                    task_dd = gr.Dropdown(
                        ["task_1", "task_2", "task_3"], value="task_1",
                        label="Dataset",
                    )
                    seed_num = gr.Number(value=42, label="Seed", precision=0)
                    reset_btn = gr.Button("🔄 Reset Environment", variant="primary")

                with gr.Accordion("Difficulty Tiers", open=False):
                    gr.Markdown(
                        "| Task | Tier | Rows | Highlights |\n"
                        "| --- | --- | --- | --- |\n"
                        "| `task_1` | Easy | 300 | 22% null ages, salary outliers, 30 dupes |\n"
                        "| `task_2` | Medium | 500 | 28% null qty, irrelevant `internal_hash` |\n"
                        "| `task_3` | Hard | 800 | physiologic bounds, severe type chaos |"
                    )

                with gr.Row():
                    with gr.Column(scale=2):
                        gr.HTML('<div class="dc-section-title">Live Observation</div>')
                        obs_box = gr.Textbox(
                            label="What you are seeing",
                            lines=18,
                            max_lines=28,
                            elem_classes=["dc-card"],
                            placeholder=(
                                "After Reset, this is the agent-facing state: per-column null rates, "
                                "dtype, corruption flags, duplicate rate, and a preview quality score."
                            ),
                        )
                        status_box = gr.Textbox(
                            label="Last action result (human-readable env feedback)",
                            lines=2,
                        )

                        gr.HTML('<div class="dc-section-title">Action Console (one RL step)</div>')
                        with gr.Row():
                            action_dd = gr.Dropdown(
                                ["fill_nulls", "remove_duplicates", "fix_dtype",
                                 "clip_outliers", "rename_column", "drop_column", "done"],
                                value="remove_duplicates", label="Action",
                            )
                            col_txt = gr.Textbox(
                                label="Column (blank if N/A)",
                                info="Required for all actions except remove_duplicates and done.",
                            )
                            conf_sl = gr.Slider(
                                0.0, 1.0, value=0.85, step=0.05,
                                label="Confidence",
                                info="≥0.75 on a correct move → small bonus; on a wrong move → penalty.",
                            )
                        with gr.Row():
                            strategy_dd = gr.Dropdown(
                                ["mean", "median", "mode", "constant", "ffill"],
                                value="median", label="fill_nulls strategy",
                            )
                            dtype_dd = gr.Dropdown(
                                ["float64", "int64", "str", "datetime64", "bool"],
                                value="float64", label="fix_dtype target",
                            )
                            clip_dd = gr.Dropdown(
                                ["iqr", "zscore", "percentile"],
                                value="iqr", label="clip_outliers method",
                            )
                        with gr.Row():
                            step_btn = gr.Button("⚡ Execute Action", variant="primary")

                        reward_box = gr.Code(
                            label="Reward breakdown (this step — RL shaping, JSON)",
                            language="json",
                            lines=8,
                        )

                        gr.HTML(
                            "<div class='dc-section-title'>Official grader — same as POST /grader</div>"
                        )
                        gr.Markdown(
                            "Click **Grade snapshot (official 0–1)** after any number of steps. "
                            "This does **not** advance the episode; it only evaluates the current dataframe "
                            "with the task’s deterministic grader (what judges should record)."
                        )
                        grade_btn = gr.Button("📊 Grade Snapshot (0-1 Score)", variant="primary")
                        grade_box = gr.Code(
                            label="Grade snapshot (JSON — score + provenance + quality axes)",
                            language="json",
                            lines=10,
                        )

                    with gr.Column(scale=1):
                        gr.HTML('<div class="dc-section-title">Reward Timeline</div>')
                        gr.Markdown(
                            "Each row is one env step: dominant component, signed reward delta, "
                            "and trailing quality. Useful to see whether the agent is earning dense signal."
                        )
                        timeline_html = gr.HTML(_timeline_html([]))
                        gr.HTML(
                            "<div class='dc-card' style='margin-top:14px;'>"
                            "<h3>Calibration Cheatsheet</h3>"
                            "<div style='font-family:var(--mono); font-size:12px; line-height:1.9; color:var(--text-1);'>"
                            "<div>0.85&ndash;0.95 &nbsp; you are sure</div>"
                            "<div>0.50&ndash;0.70 &nbsp; borderline / risky</div>"
                            "<div>conf &ge; .75 &amp; correct &rarr; <span style='color:var(--good);'>+0.04</span></div>"
                            "<div>conf &ge; .75 &amp; wrong &nbsp;&rarr; <span style='color:var(--bad);'>-0.06</span></div>"
                            "</div></div>"
                        )

                reset_btn.click(
                    ui_reset, [task_dd, seed_num],
                    [obs_box, status_html, timeline_html, status_box, reward_box],
                )
                step_btn.click(
                    ui_step,
                    [action_dd, col_txt, strategy_dd, dtype_dd, clip_dd, conf_sl],
                    [obs_box, status_html, timeline_html, status_box, reward_box],
                )
                grade_btn.click(ui_grade, [], [grade_box])

            # ──────────────────────────────────────────────────────────────
            with gr.Tab("Agent Copilot"):
                gr.Markdown(
                    "### Autonomous cleaning loop\n"
                    "The model receives the same **observation text** as your code would, replies with **JSON** "
                    "(`DataCleanAction`), and the environment **executes** it. The chat log shows "
                    "*think → propose → env feedback* for each step."
                )
                with gr.Accordion("HF Inference Router — models & errors (read this if a run aborts)", open=True):
                    gr.Markdown(
                        "**Why some model IDs fail with `model_not_supported`**  \n"
                        "The Router only routes to models that are **available for your account** and "
                        "**enabled on at least one Inference Provider** you have turned on. "
                        "If Hugging Face returns HTTP 400 with `model_not_supported`, that model is not "
                        "routable for you *right now* — it is not a bug in DataClean-Env.\n\n"
                        "**What to do**\n"
                        "- Pick another model from the dropdown (e.g. **Llama-3-70B** often works when smaller "
                        "Meta checkpoints are not exposed on your providers).\n"
                        "- On huggingface.co: **Settings → Inference Providers** (or the model’s **Deploy → "
                        "Inference Providers** page) and enable a provider that serves that checkpoint.\n"
                        "- You can still type any **custom** model id in the box if your account can route it.\n\n"
                        "**Note:** `Llama-3.2-3B-Instruct` is omitted from the default list because many accounts "
                        "see `model_not_supported` for it today; add it manually only if your Router supports it."
                    )
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Row():
                            copilot_model_dd = gr.Dropdown(
                                [
                                    "meta-llama/Meta-Llama-3-70B-Instruct",
                                    "meta-llama/Llama-3.2-1B-Instruct",
                                    "Qwen/Qwen2.5-7B-Instruct",
                                    "mistralai/Mixtral-8x7B-Instruct-v0.1",
                                ],
                                value="meta-llama/Meta-Llama-3-70B-Instruct",
                                label="Model (HF Router — type a custom id if needed)",
                                allow_custom_value=True,
                            )
                            copilot_token_txt = gr.Textbox(
                                label="HF Token (read scope)",
                                type="password",
                                info="Needs permission to call router.huggingface.co on your behalf.",
                            )
                        with gr.Row():
                            copilot_task_dd = gr.Dropdown(
                                ["task_1", "task_2", "task_3"], value="task_1",
                                label="Task",
                            )
                            copilot_file = gr.File(
                                label="Or upload a CSV (sandbox mode)",
                                file_types=[".csv"],
                            )
                        with gr.Row():
                            start_copilot_btn = gr.Button("🚀 Start Autonomous Run", variant="primary")
                            stop_copilot_btn = gr.Button("🛑 Stop Copilot", variant="stop")

                        try:
                            copilot_chatbot = gr.Chatbot(
                                label="Agent loop", height=520,
                                type="messages", avatar_images=(None, None),
                            )
                        except TypeError:
                            copilot_chatbot = gr.Chatbot(
                                label="Agent loop", height=520,
                            )

                    with gr.Column(scale=1):
                        gr.HTML(
                            "<div class='dc-card'>"
                            "<h3>Action Vocabulary</h3>"
                            "<div style='font-family:var(--mono); font-size:12px; line-height:1.9; color:var(--text-1);'>"
                            "<div><code>fill_nulls</code> &mdash; mean / median / mode / ffill</div>"
                            "<div><code>remove_duplicates</code></div>"
                            "<div><code>fix_dtype</code> &mdash; int / float / str / datetime</div>"
                            "<div><code>clip_outliers</code> &mdash; iqr / zscore / percentile</div>"
                            "<div><code>rename_column</code> · <code>drop_column</code></div>"
                            "<div><code>done</code> when quality &ge; 0.90</div>"
                            "</div></div>"
                        )
                        gr.HTML(
                            "<div class='dc-card' style='margin-top:14px;'>"
                            "<h3>Loop Phases</h3>"
                            "<div style='font-family:var(--mono); font-size:12px; line-height:2; color:var(--text-1);'>"
                            "<div>1. <span style='color:var(--accent-2);'>Observe</span> &mdash; column profile + quality</div>"
                            "<div>2. <span style='color:var(--accent-2);'>Decide</span> &mdash; emit JSON action</div>"
                            "<div>3. <span style='color:var(--accent-2);'>Execute</span> &mdash; env applies + grades</div>"
                            "<div>4. <span style='color:var(--accent-2);'>Iterate</span> &mdash; until done or budget</div>"
                            "</div></div>"
                        )

                # Copilot logic
                def run_copilot(model_id, hf_token, task_id, upload_file, history):
                    history = history or []
                    if not hf_token:
                        history.append({"role": "user",
                                        "content": f"start {task_id}"})
                        history.append({"role": "assistant",
                                        "content": "ERROR: please provide an HF token (read scope)."})
                        yield history
                        return

                    client = OpenAI(
                        base_url="https://router.huggingface.co/v1",
                        api_key=hf_token,
                    )

                    try:
                        if upload_file is not None:
                            df = pd.read_csv(upload_file.name)
                            obs = _ui_env.reset("custom", custom_df=df)
                            history.append({"role": "user",
                                            "content": f"uploaded csv ({len(df)} rows)"})
                        else:
                            obs = _ui_env.reset(task_id)
                            history.append({"role": "user",
                                            "content": f"start {task_id}"})
                        q0 = obs.quality_scores.get("overall", 0)
                        history.append({
                            "role": "assistant",
                            "content": f"initialised | shape {obs.n_rows}x{obs.n_cols} | quality {q0:.3f}",
                        })
                    except Exception as e:
                        history.append({"role": "user", "content": "reset"})
                        history.append({"role": "assistant", "content": f"reset error: {e}"})
                        yield history
                        return

                    yield history

                    for step in range(1, 21):
                        if _ui_env._state is None or _ui_env._state.done:
                            break

                        prompt = obs_to_prompt(obs)
                        history.append({"role": "assistant", "content": "thinking…"})
                        yield history

                        try:
                            completion = client.chat.completions.create(
                                model=model_id,
                                messages=[
                                    {"role": "system", "content": SYSTEM_PROMPT},
                                    {"role": "user", "content": prompt},
                                ],
                                temperature=0.1,
                                stream=False,
                            )
                            raw_reply = completion.choices[0].message.content or ""
                        except Exception as exc:
                            hint = ""
                            es = str(exc).lower()
                            if "model_not_supported" in es or "not supported" in es:
                                hint = (
                                    "\n\n**Fix:** choose a model your HF account can route (e.g. "
                                    "`meta-llama/Meta-Llama-3-70B-Instruct`), or enable an Inference Provider "
                                    "that serves this checkpoint under **Settings → Inference Providers**."
                                )
                            history[-1] = {
                                "role": "assistant",
                                "content": f"api error: {exc}{hint}\n\naborting.",
                            }
                            yield history
                            break

                        history[-1] = {
                            "role": "assistant",
                            "content": f"action proposed:\n```json\n{raw_reply}\n```",
                        }
                        yield history

                        try:
                            action_dict = parse_action(raw_reply)
                            action = DataCleanAction(
                                action_type=action_dict.get("action_type", "done"),
                                column=action_dict.get("column"),
                                params=action_dict.get("params", {}),
                                confidence=max(0.0, min(1.0, float(
                                    action_dict.get("confidence", 0.5)))),
                            )
                            result = _ui_env.step(action)
                            obs = result.observation
                            reward = result.reward
                            feedback = obs.last_action_result
                        except Exception as exc:
                            feedback = f"error: {exc}"
                            reward = -0.05

                        q = obs.quality_scores.get("overall", 0) if obs else 0
                        history.append({"role": "user", "content": "env executed"})
                        history.append({
                            "role": "assistant",
                            "content": (
                                f"{feedback}\n\n"
                                f"reward {reward:+.3f} · quality {q:.3f} · "
                                f"step {_ui_env._state.step}/{obs.budget_remaining + _ui_env._state.step}"
                            ),
                        })
                        yield history

                    final_score = _ui_env.grade()
                    prov = _ui_env.verify_provenance()
                    history.append({"role": "user", "content": "episode finished"})
                    history.append({
                        "role": "assistant",
                        "content": (
                            f"final score {final_score:.3f} · "
                            f"provenance {'reproducible' if prov else 'drift'} · "
                            f"steps {_ui_env._state.step}"
                        ),
                    })
                    yield history

                start_copilot_btn.click(
                    run_copilot,
                    inputs=[copilot_model_dd, copilot_token_txt,
                            copilot_task_dd, copilot_file, copilot_chatbot],
                    outputs=[copilot_chatbot],
                )
                copilot_file.change(lambda: [], outputs=[copilot_chatbot])
                copilot_task_dd.change(lambda: [], outputs=[copilot_chatbot])

            # ──────────────────────────────────────────────────────────────
            with gr.Tab("API"):
                gr.Markdown(
                    "### Programmatic access\n"
                    "These routes are what **OpenEnv validators** and external agents call. "
                    "Typical judge flow: `POST /reset` → repeated `POST /step` → `POST /grader`. "
                    "Open the live OpenAPI explorer at **`/docs`** on the same host."
                )
                with gr.Accordion("What each endpoint is for", open=False):
                    gr.Markdown(
                        "| Endpoint | Role |\n"
                        "| --- | --- |\n"
                        "| `GET /health` | Liveness — must return 200 for probes. |\n"
                        "| `POST /reset` | Start episode; returns `episode_id` + first observation. |\n"
                        "| `POST /step` | Apply one `DataCleanAction`; returns reward + new observation. |\n"
                        "| `GET /state` | Snapshot: step count, total reward, ops log, done flag. |\n"
                        "| `GET /tasks` | Lists tasks + JSON schema for actions + reward cheat-sheet. |\n"
                        "| `POST /grader` | **Official** 0–1 score on current table (+ provenance flag). |\n"
                        "| `GET /reward_history` | Per-step reward breakdown (RL debugging). |\n"
                        "| `GET /explain` | Natural-language trace of ops. |\n"
                        "| `GET /baseline` | Heuristic cleaner scores (no LLM lower bound). |\n"
                    )
                gr.HTML(
                    "<div class='dc-card'>"
                    "<h3>Endpoints</h3>"
                    "<div style='font-family:var(--mono); font-size:12px; line-height:2; color:var(--text-1);'>"
                    "<div><code>POST /reset</code> &nbsp; start episode</div>"
                    "<div><code>POST /step</code> &nbsp; apply action</div>"
                    "<div><code>GET&nbsp; /state</code> &nbsp; episode snapshot</div>"
                    "<div><code>GET&nbsp; /tasks</code> &nbsp; tasks + action schema</div>"
                    "<div><code>POST /grader</code> &nbsp; grade dataframe</div>"
                    "<div><code>GET&nbsp; /reward_history</code> &nbsp; per-step rewards</div>"
                    "<div><code>GET&nbsp; /explain</code> &nbsp; natural-language trace</div>"
                    "<div><code>GET&nbsp; /baseline</code> &nbsp; heuristic agent run</div>"
                    "<div><code>GET&nbsp; /health</code> &nbsp; liveness probe</div>"
                    "<div><code>GET&nbsp; /docs</code> &nbsp; OpenAPI explorer</div>"
                    "</div></div>"
                )
                gr.Code(
                    label="Quick example",
                    language="python",
                    value=(
                        "import httpx\n"
                        "url = 'http://localhost:7860'\n"
                        "r = httpx.post(f'{url}/reset', json={'task_id': 'task_1', 'seed': 42}).json()\n"
                        "ep = r['episode_id']\n"
                        "step = httpx.post(f'{url}/step', json={\n"
                        "    'episode_id': ep,\n"
                        "    'action': {'action_type': 'remove_duplicates', 'confidence': 0.9},\n"
                        "}).json()\n"
                        "print(step['reward'], step['observation']['quality_scores'])\n"
                    ),
                )

    return demo
