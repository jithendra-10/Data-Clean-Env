"""
inference.py — DataClean-Env Baseline Inference  v2.0
======================================================
MANDATORY env vars:
    API_BASE_URL  (default: https://router.huggingface.co/v1)
    MODEL_NAME    (default: meta-llama/Meta-Llama-3-70B-Instruct)
    HF_TOKEN      required

Usage:
    export API_BASE_URL=https://router.huggingface.co/v1
    export MODEL_NAME=meta-llama/Meta-Llama-3-70B-Instruct
    export HF_TOKEN=hf_...
    python inference.py
"""
import os, sys, json, time
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Meta-Llama-3-70B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")

if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set."); sys.exit(1)

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SEED=42; MAX_STEPS=15; TEMPERATURE=0.2; MAX_TOKENS=300; STEP_TIMEOUT=45

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataclean.env import DataCleanEnv
from dataclean.models import DataCleanAction
from dataclean.tasks import TASK_REGISTRY
from dataclean.utils import obs_to_prompt, parse_action, SYSTEM_PROMPT  # no circular import


# ── Structured logging (validator expects these) ──────────────────────────────

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward:.4f} "
          f"done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    rstr = ",".join(f"{r:.4f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} "
          f"score={score:.4f} rewards={rstr}", flush=True)


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(task_id: str, seed: int = SEED) -> dict:
    env = DataCleanEnv()
    obs = env.reset(task_id=task_id, seed=seed)
    print(f"\n  [{task_id}] {obs.n_rows}×{obs.n_cols} | quality={obs.quality_scores}")
    log_start(task=task_id, env="dataclean-env", model=MODEL_NAME)

    total_reward = 0.0; history = []; rewards_list = []

    for step in range(MAX_STEPS):
        obs_text = obs_to_prompt(obs)
        history.append({"role":"user","content":obs_text})
        if len(history) > 10: history = history[-10:]

        raw = ""
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role":"system","content":SYSTEM_PROMPT}, *history],
                temperature=TEMPERATURE, max_tokens=MAX_TOKENS, timeout=STEP_TIMEOUT,
            )
            raw = resp.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [step {step+1}] LLM error: {exc}")
            raw = '{"action_type":"done","column":null,"params":{},"confidence":0.5}'

        try:
            action_dict = parse_action(raw)
        except Exception:
            action_dict = {"action_type":"done","column":None,"params":{},"confidence":0.5}

        action_dict["confidence"] = max(0.0, min(1.0, float(action_dict.get("confidence",0.5))))

        print(f"    Step {step+1:02d} | {action_dict.get('action_type','?'):20s} "
              f"col={str(action_dict.get('column')):18s} conf={action_dict['confidence']:.2f}")

        step_reward=0.0; done=False; step_error=None
        try:
            result = env.step(DataCleanAction(
                action_type=action_dict.get("action_type","done"),
                column=action_dict.get("column"),
                params=action_dict.get("params",{}),
                confidence=action_dict["confidence"],
            ))
            step_reward=result.reward; obs=result.observation; done=result.done
        except Exception as exc:
            step_error=str(exc); done=True
            print(f"    Env error: {exc}")

        total_reward+=step_reward; rewards_list.append(step_reward)
        log_step(step+1, json.dumps(action_dict,separators=(',',':')),
                 step_reward, done, step_error)
        print(f"           reward={step_reward:+.4f} | quality={obs.quality_scores.get('overall',0):.3f}")
        history.append({"role":"assistant","content":raw})

        if done: break
        if obs.quality_scores.get("overall",0)>=0.99:
            r2=env.step(DataCleanAction(action_type="done",confidence=0.92))
            rewards_list.append(r2.reward); total_reward+=r2.reward
            log_step(step+2,'{"action_type":"done"}',r2.reward,True,None)
            break

    final_score   = env.grade()
    provenance_ok = env.verify_provenance()
    quality       = env._compute_quality_scores()
    success       = final_score >= 0.80
    log_end(success, env._state.step, final_score, rewards_list)

    print(f"\n  Score={final_score:.4f} | Reward={total_reward:.4f} | "
          f"Steps={env._state.step} | Provenance={provenance_ok}")

    return {
        "task_id":task_id,"seed":seed,"model":MODEL_NAME,
        "final_score":round(final_score,4),"total_reward":round(total_reward,4),
        "quality_scores":quality,"provenance_ok":provenance_ok,"steps_used":env._state.step,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("DataClean-Env — Baseline Inference  v2.0")
    print(f"Model    : {MODEL_NAME}")
    print(f"Endpoint : {API_BASE_URL}")
    print(f"Seed     : {SEED}")
    print("="*60)

    t0=time.time(); results=[]
    for task_id in TASK_REGISTRY:
        print(f"\nRunning {task_id}...")
        results.append(run_episode(task_id=task_id, seed=SEED))
        if time.time()-t0 > 1050:
            print("WARNING: 20-min limit approaching. Stopping."); break

    elapsed=time.time()-t0
    print(f"\n{'='*60}")
    print(f"RESULTS  (seed={SEED})")
    print(f"{'─'*60}")
    print(f"{'Task':<10} {'Score':>8} {'Reward':>10} {'Steps':>7} {'Prov':>6}")
    print(f"{'─'*60}")
    for r in results:
        print(f"{r['task_id']:<10} {r['final_score']:>8.4f} "
              f"{r['total_reward']:>10.4f} {r['steps_used']:>7d} "
              f"{'✓' if r['provenance_ok'] else '✗':>6}")
    if results:
        avg=sum(r["final_score"] for r in results)/len(results)
        print(f"{'─'*60}\n{'Average':<10} {avg:>8.4f}")
    print(f"{'='*60}")
    print(f"Runtime: {elapsed:.1f}s")

    out={"model":MODEL_NAME,"api_base_url":API_BASE_URL,"seed":SEED,
         "total_runtime_s":round(elapsed,1),"results":results}
    with open("baseline_scores.json","w") as f: json.dump(out,f,indent=2)
    print("Scores → baseline_scores.json")

if __name__=="__main__":
    main()
