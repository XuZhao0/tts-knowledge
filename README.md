# Test-Time Scaling in Reasoning Models for Knowledge-Intensive Tasks
This repository contains the data and code for our paper: ["Test-time scaling in reasoning models is not effective for knowledge-intensive tasks yet"](http://arxiv.org/abs/2509.06861)  

## Overview
This repository contains the `benchmarks`, `code` and `model outputs` for all experiments. It includes:
1. `benchmarks/`: Datasets used in our experiments: SimpleQA and FRAMES.
2. `code`: Scripts to run experiments and evaluate model responses.
3. `outputs/`: Evaluated model outputs from 12 reasoning models with different thinking levels on 2 benchmarks.


## Benchmarks
- SimpleQA: 800 short, fact-seeking questions randomly sampled from [OpenAI SimpleQA](https://openai.com/index/introducing-simpleqa/)
- FRAMES: 824 complex fact-seeking questions from [frames-benchmark](https://huggingface.co/datasets/google/frames-benchmark)

## Installation
First, clone the repository:

```bash
git clone https://github.com/XuZhao0/tts-knowledge.git
cd tts-knowledge
```

Create and activate a new Python 3.10+ environment using `conda`. For example:

```bash
conda create -n tts_knowledge python=3.10
conda activate tts_knowledge
```

Install package requirements with the following command:

```bash
pip install -r requirements.txt
```

## Code

### 1. Proprietary models

To run experiments on proprietary models, you first need API keys for:
[OpenAI](https://platform.openai.com/docs/overview), [Anthropic Claude](https://docs.anthropic.com/en/api/overview), [Google Gemini](https://ai.google.dev/gemini-api/docs/models), [XAI Grok](https://docs.x.ai/docs/overview)

Run experiments on **OpenAI** closed-source models, use the following command. You can change the `--model` and `--effort` arguments to test different models and thinking levels. Replace `'your_openai_api_key'` with your actual OpenAI API key. More details can be found in the `openai_close_generate.py` script.

```bash
python openai_close_generate.py --model gpt-5-mini --effort low --api_key 'your_openai_api_key'
```

Run experiments on **XAI Grok** models:

```bash
python grok_generate.py --effort low --api_key 'your_xai_api_key'
```

Run experiments on **Gemini** models. You can adjust the `--thinking_budget` parameter to set different thinking levels. Disable the `thinking` mode by setting `--thinking_budget` to 0.

```bash
python gemini_generate.py --thinking_budget 512 --api_key 'your_google_api_key'
``` 

Run experiments on Anthropic **Claude** models. You can disable the `thinking` mode by setting `--thinking_budget` to 0.

```bash
python claude_generate.py --thinking_budget 1024 --api_key 'your_anthropic_api_key'
```

### 2. Open-source models
Run experiments on **gpt-oss** models, using the following command:

```bash
python gpt_oss_generate.py --effort low
```

Run experiments on **DeepSeek-R1-Distill** models with `budget forcing`, using the following command. You can adjust the `--extend_times` parameter to set different thinking levels. If it is set to 0, it means natural output without `budget forcing`.

```bash
python ds_r1_distill_exd.py --extend_times 2
```

Run experiments on **Qwen3** models (*thinking mode*) with `budget forcing`:

```bash
python qwen3_exd.py --extend_times 2
```

Run experiments on **Qwen3** models on *non-thinking mode*:

```bash
python qwen3_no_think.py
```

### 3. Evaluation
Evaluate model responses with the following command. We use `gpt-4o-mini` as the grader model. Replace `your_openai_api_key` with your OpenAI API key.

```bash
python evaluation.py --input_path '<model_response_file_path.jsonl>' --api_key 'your_openai_api_key'
```

## Model Outputs
We release evaluated outputs from 12 reasoning models with varying thinking levels (including *non-thinking* for several models). Each output includes: model response, reasoning trace (if applicable), and evaluation label (*correct* [A], *incorrect* [B], or *not attempted* [C]), and metadata such as token count.

## Citation
If you find our code or data useful, please cite:

```bibtex
@article{zhao2025testtimescalingreasoningmodels,
      title={Test-Time Scaling in Reasoning Models Is Not Effective for Knowledge-Intensive Tasks Yet}, 
      author={James Xu Zhao and Bryan Hooi and See-Kiong Ng},
      year={2025},
      eprint={2509.06861},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2509.06861}, 
}
```

## Contact
For questions or suggestions, feel free to contact: `xu.zhao@u.nus.edu`

## Acknowledgements
Our `budget forcing` implementation is adapted from [r1-overthinker](https://github.com/qunash/r1-overthinker). We thank the developers for open-sourcing their code.