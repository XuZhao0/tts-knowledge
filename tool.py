import json
from typing import Union

from openai import OpenAI
import anthropic
from google import genai

# ===== File Operations =====
def jsonlines_load(fname: str):
    with open(fname, 'r') as f:
        return [json.loads(line) for line in f]


def jsonlines_dump(fname: str, data: Union[dict, list]):
    try:
        with open(fname, 'a+') as f:
            if isinstance(data, dict):
                f.write(json.dumps(data)+'\n')
            elif isinstance(data, list):
                for d in data:
                    f.write(json.dumps(d)+'\n')

    except (FileNotFoundError, FileExistsError) as e:
        print(f'Error: {e}')
        print(f'Could not write to {fname}')

# ===== Load Client =====
def load_openai_client(api_key):
    client = OpenAI(
        api_key = api_key
    )
    return client

def load_xai_client(api_key):
    client = OpenAI(
        base_url="https://api.x.ai/v1",
        api_key=api_key
    )
    return client

def load_gemini_client(api_key):
    client = genai.Client(api_key=api_key)
    return client

def load_anthropic_client(api_key):
    client = anthropic.Anthropic(
        api_key = api_key
    )
    return client