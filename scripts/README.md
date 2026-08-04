# Runnable scripts

Run these entry points from the repository root. Use `python scripts/<script>.py --help` to see every option.

## Generation

| Script | Model family | Test-time scaling control |
| --- | --- | --- |
| `openai_close_generate.py` | OpenAI closed-source models | `--effort` |
| `grok_generate.py` | xAI Grok | `--effort` |
| `gemini_generate.py` | Google Gemini | `--thinking_budget` |
| `claude_generate.py` | Anthropic Claude | `--thinking_budget` |
| `gpt_oss_generate.py` | OpenAI gpt-oss | `--effort` |
| `ds_r1_distill_exd.py` | DeepSeek-R1-Distill | `--extend_times` |
| `qwen3_exd.py` | Qwen3 thinking mode | `--extend_times` |
| `qwen3_no_think.py` | Qwen3 non-thinking mode | disabled |

## Evaluation

`evaluation.py` grades a JSONL response file with an OpenAI grader model.

`tool.py` is an internal helper shared by the entry points; it is not intended to be run directly.
