# Test-Time Scaling in Reasoning Models for Knowledge-Intensive Tasks
This repository contains the data and code for the paper:
[Test-time scaling in reasoning models is not effective for knowledge-intensive tasks yet](http://arxiv.org/abs/2509.06861)

## Overview
We release the outputs of 12 reasoning models evaluated under different thinking levels on two knowledge-intensive benchmarks. We are still organizing the codebase for release. Please stay tuned for updates.

## Content
**Benchmarks:**
- SimpleQA: 800 short, fact-seeking questions, randomly sampled from [simple-evals](https://github.com/openai/simple-evals)
- FRAMES: 824 complex fact-seeking questions from [frames-benchmark](https://huggingface.co/datasets/google/frames-benchmark)

**Model Outputs**

12 reasoning models with differnt thinking levels on 2 benchmarks. Each output includes: model response, rasoning trace (if applicable), and evaluation label (correct, incorrect, or not attempted), and metadata such as token count.

## Citation
If you find our work or data useful, please cite:
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
For questions or suggestions, feel free to contact xu.zhao@u.nus.edu