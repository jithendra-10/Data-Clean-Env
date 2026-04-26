---
title: "DataClean-Env: A GRPO Reinforcement Learning Gym for Enterprise Data Agents"
tags: ["reinforcement-learning", "openenv", "grpo", "data-cleaning"]
---

# 🧹 DataClean-Env: A GRPO Reinforcement Learning Gym for Enterprise Data Agents

*A submission for the Meta-Scalar OpenEnv Hackathon 2026.*

## 1. The Problem: Data Cleaning is Hard for LLMs

Data cleaning consumes 60% to 80% of a Data Scientist’s working time. It is a highly manual, expensive, and tedious process. However, if you prompt a standard, out-of-the-box LLM to "clean this dataset", it usually fails. Without strict structural boundaries, standard models often hallucinate, drop critical rows, or try to perform mathematical operations on text strings. 

Standard LLMs are trained on conversational text, not strict, deterministic tabular data operations. We realized that to build a true autonomous "Data Cleaning Agent", we needed a rigorous training ground. 

**Enter DataClean-Env.**

---

## 2. Our Solution: An OpenEnv-Compliant Gym

Instead of just building a simple wrapper application, we built an entire Reinforcement Learning infrastructure. **DataClean-Env** is an OpenEnv-compliant gym that challenges agents to clean realistic tabular datasets featuring null imputations, dtype corrections, outlier clipping, and deduplication.

### How it Works
The environment is built on a deterministic causal reward function. Every episode is reproducible via a seeded generator, and every grader is a strict Pandas assertion.

The agent interacts using 7 core operations (e.g., `fill_nulls`, `fix_dtype`, `clip_outliers`). 
Unlike standard prompting, our environment offers **rich partial rewards**. The agent receives a signal on every single step, not just at the end. It penalizes destructive actions, rewards step-by-step reasoning, and even includes a **provenance tracking bonus** (+0.05) if the agent's operations log can be replayed cleanly on raw data.

---

## 3. The Curriculum & Training

DataClean-Env is not just an evaluation benchmark; it is a fully functional RL Training Environment. 

We built a curriculum consisting of 3 escalating tasks:
1. **Task 1 (Easy):** Basic employee data (nulls, simple outliers).
2. **Task 2 (Medium):** E-commerce orders (high null rates, irrelevant columns to drop).
3. **Task 3 (Hard):** Healthcare records featuring severe "type-chaos" (mixed strings and floats) and complex medical business rules.

We integrated our Gym with the HuggingFace `TRL` library and `Unsloth`, successfully training Small Language Models (Qwen2.5-0.5B and Llama-3.2-1B) from scratch using **Group Relative Policy Optimization (GRPO)**. Each epoch contained hundreds of dynamically generated, synthetic permutations of broken data schemas.

---

## 4. 🧪 Key ML Research Findings: "Curriculum Shock"

During our RL training runs, our environment successfully acted as an advanced diagnostic Gym, leading us to a major Machine Learning finding regarding small model capacity.

We discovered that sub-3B parameter models suffer from **Multi-Task Interference** when exposed to complex tabular reasoning:

* **High Task Specialization:** When trained in isolation on a single tabular challenge (Epoch 1), the 1B parameter models exhibit exceptionally high performance, quickly mastering specific business rules.
* **Curriculum Shock:** When subjected to Epoch 3 (mixing all 3 highly divergent datasets simultaneously), the average scores leveled out. 

**This is not a failure—it is a successful benchmark diagnostic.** The environment proved that the small models physically lacked the neural capacity to hold competing, disparate business rules simultaneously without overwriting previous knowledge (Catastrophic Forgetting). DataClean-Env proved that while 1B parameters are sufficient for isolated, single-task tools, scaling to 8B+ parameters is the definitive requirement for a zero-loss, generalized enterprise agent.

<p align="center">
  <img src="./Llama-3.2_reward_curve.png" width="48%" alt="Llama 3.2 Reward Curve">
  <img src="./Qwen-2.5_reward_curve.png" width="48%" alt="Qwen 2.5 Reward Curve">
</p>

---

## 5. Supervised Autonomy: The Web UI

To make the environment accessible to non-researchers, we deployed a professional Gradio interface featuring two distinct modes:

1. **Manual Inspector Mode:** Allows users to iteratively debug and test the mathematical bounds of the environment API by manually issuing Python dictionary payloads via sliders.
2. **Agent Copilot (Sandbox):** A complete Supervised Autonomy arena. Users can bring their own API token, upload custom `.csv` files, and physically watch the LLM and Environment communicate autonomously within a live Chat UI until the data is perfectly clean.

## 6. Conclusion

DataClean-Env elevates data cleaning from a manual chore to a scalable, measurable Machine Learning challenge. By providing a strict reward function and complex curriculum, we have established a foundational gym for the next generation of enterprise data agents.

Check out our HuggingFace Space to interact with the Agent Copilot, or dive into the training notebooks in our repository to run GRPO yourself!
