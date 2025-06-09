# McBE
McBE: A Multi-task Chinese Bias Evaluation Benchmark for Large Language Models

Dataset: https://huggingface.co/datasets/Velikaya/McBE
Code: https://github.com/VelikayaScarlet/McBE

# Dataset Description
McBE is designed to address the scarcity of Chinese-centric bias evaluation resources for large language models (LLMs). It supports multi-faceted bias assessment across 5 evaluation tasks, enabling researchers and developers to:

Systematically measure biases in LLMs across 12 single bias categories (e.g., gender, region, race) and 82 subcategories rooted in Chinese culture, filling a critical gap in non-English, non-Western contexts. Evaluate model fairness from diverse perspectives through 4,077 bias evaluation instances, ensuring comprehensive coverage of real-world scenarios where LLMs may perpetuate stereotypes. Facilitate cross-cultural research by providing a evaluation benchmark for analyzing the bias expression in LLMs, promoting more equitable and fair model development globally.

Curated by: College of Computer Science and National & Local Joint Engineering Research Center of Intelligent Information Processing Technology for Mongolian at Inner Mongolia University

# Uses
## Direct Use
Evaluating Chinese Bias in Chinese and Multilingual LLMs.

## Out-of-Scope Use
Only applicable for bias assessment of Chinese LLMs and Multilingual LLMs.

Prohibited from generating or disseminating discriminatory content.

# Dataset Creation
## Curation Rationale
Although numerous studies have been dedicated to evaluating biases in LLMs, most of them face three limitations, as illustrated in Figure \ref{fig:limitations}. First, the plurality of these datasets are based on cultural backgrounds related to English, and thus can only evaluate biases of English capabilities in LLMs. They cannot measure the biases present in other cultural backgrounds. Second, existing evaluation benchmarks pay less attention to categories with regional and cultural characteristics. Additionally, other noteworthy categories also receive relatively scant consideration. Third, most previous works use Question-Answering or counterfactual-Inputting to evaluate LLMs, which cannot fully and comprehensively measure bias.

# Bias, Risks, and Limitations
The data may not cover all Chinese bias scenarios.

There are subjective differences in manual annotation, so we take the average of quantifiable values.

In the Preference Computation task, the NLL-based method relies on the predicted probability distribution. Consequently, this task can not be applied to black-box models where such information is not available. We hope future research will solve this issue.

