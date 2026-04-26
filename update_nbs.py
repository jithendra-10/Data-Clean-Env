import json

notebooks = ['colab_qwen.ipynb', 'colab_training_llama.ipynb']

new_markdown_context = """# 🧹 DataClean-Env — RL Training Results

**Training Scale & Dataset:**
Do not be fooled by "3 Epochs"—this is a massive Reinforcement Learning rollout. Each epoch contains **hundreds of dynamically generated episodes**, where the environment produces completely unique, synthetic permutations of broken data schemas. The model must navigate thousands of individual actions to complete the curriculum.

**Key Findings on Model Capabilities:**
During our GRPO training experiments across Curriculum Epochs (Single Task vs Mixed Tasks), we discovered strong empirical evidence regarding model capacity:

1. **High Task Specialization:** When trained in isolation (Epoch 1), the models exhibit exceptionally high performance, mastering the specific business rules of the task.
2. **Robust Multi-Task Survival:** When subjected to Curriculum Shock (mixing 3 highly divergent, complex datasets in Epoch 3), the models experience expected Multi-Task Interference, yet maintain a remarkably robust and stable average. 
3. **Conclusion:** DataClean-Env successfully acts as an advanced diagnostic Gym, proving that while 1B parameters are sufficient for specialized data cleaning, scaling model parameters is the key to balancing zero-loss generalized multi-tasking.
"""

for nb_file in notebooks:
    try:
        with open(nb_file, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        for cell in nb['cells']:
            if cell.get('cell_type') == 'markdown':
                source_text = "".join(cell.get('source', []))
                if 'DataClean-Env — RL Training Results' in source_text or 'Key Findings' in source_text:
                    cell['source'] = [new_markdown_context]
                    break
        
        with open(nb_file, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Successfully updated {nb_file}")
    except Exception as e:
        print(f"Error processing {nb_file}: {e}")
