<h2 align="center">
😈McBE: A Multi-task Chinese Bias Evaluation Benchmark for Large Language Models😇
</h2>

<p align="center">
  <img alt="Static Badge" src="https://img.shields.io/badge/ACL-2025-green">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white">
</p>

<div align="center"style="font-family: charter; font-size: x-small;">
	Tian Lan<sup>1,2,3</sup>,</span>
	Xiangdong Su*<sup>1,2,3</sup>,</span>
	Xu Liu<sup>1,2,3</sup>,</span>
	Ruirui Wang<sup>1,2,3</sup>,</span>
	Ke Chang<sup>1,2,3</sup>,</span>
	Jiang Li<sup>1,2,3</sup>,</span>
	Guanglai Gao<sup>1,2,3</sup></span>
</div>
<br>
<div align="center">
    <sup>1</sup>College of Computer Science, Inner Mongolia University, China&emsp;<br>
    <sup>2</sup>National & Local Joint Engineering Research Center of Intelligent Information Processing Technology for Mongolian, China&emsp;<br>
    <sup>3</sup>Inner Mongolia Key Laboratory of Multilingual Artiffcial Intelligence Technology, China&emsp;<br>
    <br>
</div>

 \* corresponding author

<div>
  <img src="https://raw.githubusercontent.com/VelikayaScarlet/McBE/refs/heads/main/content/mcbe.png" alt="MCBE" />
</div>

Paper: [McBE: A Multi-task Chinese Bias Evaluation Benchmark for Large Language Models](https://arxiv.org/abs/2507.02088)

Dataset: https://huggingface.co/datasets/Velikaya/McBE

Code: https://github.com/VelikayaScarlet/McBE

<h2 align="center">
📜Abstract
</h2>

<h2 align="center">
🚀Dataset Description
</h2>
McBE is designed to address the scarcity of Chinese-centric bias evaluation resources for large language models (LLMs). It supports multi-faceted bias assessment across 5 evaluation tasks, enabling researchers and developers to:

Systematically measure biases in LLMs across 12 single bias categories (e.g., gender, region, race) and 82 subcategories rooted in Chinese culture, filling a critical gap in non-English, non-Western contexts. Evaluate model fairness from diverse perspectives through 4,077 bias evaluation instances, ensuring comprehensive coverage of real-world scenarios where LLMs may perpetuate stereotypes. Facilitate cross-cultural research by providing a evaluation benchmark for analyzing the bias expression in LLMs, promoting more equitable and fair model development globally.

Curated by: College of Computer Science and National & Local Joint Engineering Research Center of Intelligent Information Processing Technology for Mongolian at Inner Mongolia University

<h2 align="center">
🔬Dependencies
</h2>

```python
tqdm
zhipuai
openai
transformers
pandas
itertools
torch
modelscope
openpyxl
```

<h2 align="center">
💯How to Run a Evaluation?
</h2>

1. Open `utils.py` and fill in your GLM4-AIR API key on line 9. You can also use other LLMs to serve as LLM Judge.  
2. Open `load_model.py` and replace `model_dir` with the path to your models in lines 6–12.  
3. Open `eval.py` and update the path parameter to your local directory. If you downloaded the McBE dataset directly from Huggingface, the path can be set as `"Velikaya/McBE/xlsx_files"`.
4. Edit the categories list in eval.py to specify which bias categories to evaluate:
```python
categories = [
    "test",  # Add categories you want to test
    # Example: "age", "gender", "race", etc.
]
```

5. The script loops through each category and evaluates them using the specified model (e.g., "qwen2"). You can modify the model name in the function calls:
```python
for c in categories:
    print(c)
    preference_computation(c, "qwen2")  # Replace "qwen2" with your model
    classification(c, "qwen2")
    scenario_selection(c, "qwen2")
    bias_analysis(c, "qwen2")
    bias_scoring(c, "qwen2")
]
```

6.Run the `eval.py`


