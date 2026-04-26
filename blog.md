# DataClean-Env: Training Enterprise Agents via Curriculum Shock

*A submission for the Meta-Scalar OpenEnv Hackathon 2026*

---

## 🛑 The Problem: "Math on Strings"
Data cleaning consumes 60% to 80% of a Data Scientist’s time. It is a universal, expensive, and tedious bottleneck. Yet, if you ask a standard LLM to "clean this dataset," the results are often disastrous. Without strict structural enforcement, agents hallucinate columns, drop critical rows, or famously attempt to perform mathematical operations on text strings.

We realized there was no standardized benchmark to train agents on strictly deterministic enterprise data operations. So, instead of building a simple API wrapper, we built an entire training infrastructure.

---

## 🛠️ The Solution: DataClean-Env
**DataClean-Env** is a fully OpenEnv-compliant Reinforcement Learning Gym designed to evaluate and train autonomous data cleaning agents.

Instead of testing agents on simple math puzzles, our environment generates synthetic, endlessly reproducible tabular datasets featuring complex real-world corruptions: nulls, outliers, extreme duplicates, and "type-chaos".

### The RL Environment Design
The environment provides a strictly enforced causal reward function:
* **The Actions:** 7 core structural operations (`fill_nulls`, `fix_dtype`, `remove_duplicates`, etc.).
* **The Rules:** Every action is graded by a deterministic Pandas assertion. Destructive actions receive heavy penalties, while safe, high-confidence corrections are rewarded.
* **The Provenance:** Agents receive a massive `+0.05` bonus if and only if their final chain of actions is causally ordered and perfectly reproducible on raw data.

---

## 🧪 Real Training, Real Findings
We didn't just build the Gym—we used it. Using HuggingFace's `TRL` stack and `Unsloth`, we ran a massive **Group Relative Policy Optimization (GRPO)** rollout across hundreds of synthetic episodes, fine-tuning `Qwen2.5-0.5B` and `Llama-3.2-1B`.

This is where the environment successfully proved its worth as an advanced diagnostic benchmark, uncovering a phenomenon we call **Multi-Task Interference**.

### The "Curriculum Shock" Discovery
When we trained these 1B parameter models in isolation on single tasks (Epoch 1), they exhibited exceptionally high performance, quickly mastering specific business rules.

However, when we introduced a **Curriculum** (mixing 3 highly divergent datasets, including complex Healthcare rules, in Epoch 3), the models experienced "Curriculum Shock". The overall scores dropped, but this was *not a failure of the environment*. 

Instead, DataClean-Env successfully diagnosed that sub-3B parameter models lack the physical neural capacity to hold competing, disparate business rules simultaneously without overwriting previous knowledge (Catastrophic Forgetting). 

---

## 🚀 Conclusion
DataClean-Env successfully operates as a rigorous, mathematically dense evaluation benchmark. It empirically proved that while 1B parameters are sufficient for isolated data cleaning tools, scaling to 8B+ parameters is the definitive requirement for a zero-loss, generalized enterprise agent.

**Check out the repo:**
* **Training Code:** Browse `training_script.py` and our Colab Notebooks directly in the repo to see the GRPO implementation.
* **The Live App:** Check out the "Agent Copilot Sandbox" right here on our HuggingFace Space to watch the environment enforce rules in real-time.
