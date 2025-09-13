"""This file is used to generate responses from Qwen3 models (non-thinking mode)."""
import argparse
import os
import time
import logging
from tool import *
from datetime import datetime
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

@torch.inference_mode
def reasoning_effort(tokenizer, model, task, args):

    question = """Give me the answer to the following question only when you are sure of it. \
Otherwise, say 'I don't know'. Put your answer on its own line after 'Answer:'.\n""" + task["problem"]
    
    current_tokens = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": question},
        ],
        add_generation_prompt=True, # add <|im_start|>assistant\n
        enable_thinking = False, # disable the thinking mode
        return_tensors="pt",
    )
    
    current_tokens = current_tokens.to(model.device)

    n_final_tokens = 0
    total_new_tokens = 0
    final_content = ""

    max_batch_tokens = args.max_batch_tokens
        
    while total_new_tokens < args.max_output_tokens:
                
        outputs = model.generate(
            input_ids=current_tokens,
            max_new_tokens = min(max_batch_tokens, args.max_output_tokens - total_new_tokens),
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k = args.top_k,
            min_p = args.min_p,
            repetition_penalty=args.repetition_penalty,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id = tokenizer.eos_token_id,
        )
                
        new_tokens = outputs.sequences[:, current_tokens.shape[1]:]
        
        total_new_tokens += new_tokens.shape[1]
        
        for i , token in enumerate(new_tokens[0]):
            token_id = token.item()
            token_str = tokenizer.decode([token_id])

            final_content += token_str
            n_final_tokens += 1
            
            if token_id == model.config.eos_token_id:
                task["response"] = final_content.strip()
                task["response_tokens"] = n_final_tokens                
                if args.verbose:
                    print("=="*20)
                    print("\nQuestion: \n", question)
                    print("\nGround truth:", task["answer"])
                    print("\nResponse: \n", final_content.strip())
                return task, question
        
        else:
            current_tokens = outputs.sequences

def main(args):
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map=args.device
    )

    dt_string = datetime.now().strftime("%m_%d_%H_%M")
    start_time = time.time()
    all_data = jsonlines_load(args.input_path)

    if args.end == -1:
        end_index = len(all_data)
    else:
        end_index = args.end
    
    saved_model_name = args.model_name.split("/")[-1]
    
    # create output directory if not exists
    saved_file_dir = f"{args.output_dir}/no_think"
    
    if not os.path.exists(saved_file_dir):
        os.makedirs(saved_file_dir)
    
    output_path = f"{saved_file_dir}/{saved_model_name}_no_think_s{args.start}_e{end_index}_{dt_string}.jsonl"
        
    logging.critical("*-"*40)
    logging.critical(f"Model: {args.model_name} || No-think || Start time: {dt_string}.")
    logging.critical(f"Input path: {args.input_path}")
    logging.critical(f"Ouput path: {output_path}")

    tasks = all_data[args.start:end_index]
    total_tasks_num = len(tasks)
    completed_tasks = 0
    prompt_demo = None
    
    with tqdm(total=total_tasks_num) as pbar:
        for i in range(total_tasks_num):
            
            retry_time = 0
            error_info = None
            max_retry_time = 5
            
            while retry_time < max_retry_time:
                try:
                    task, prompt_demo = reasoning_effort(tokenizer, model, tasks[i], args)
                    if i == 0:
                        print("=="*20)
                        print("\nPrompt: ", prompt_demo)
                        print("\nResponse: \n", task["response"])
                    jsonlines_dump(output_path, task)
                    break
                
                except Exception as e:
                    retry_time += 1
                    error_info = str(e)
                    torch.cuda.empty_cache()
                    time.sleep(3)
            
                    logging.warning(f"! Task: {i}, Retry time: {retry_time}/{max_retry_time} || Error: {e}")
  
            if retry_time == max_retry_time:
                if args.type == "math":
                    error_id = tasks[i]['unique_id']
                elif args.type == "simpleqa" or args.type == "frames":
                    error_id = tasks[i]['id']
                
                error_result = {
                    "id": error_id,
                    "error": error_info,
                }
                logging.warning(f"Task {i} failed after {max_retry_time} retries. Saving error result.")
                error_output_path = output_path.replace('.jsonl', '_error.jsonl')
                jsonlines_dump(error_output_path, error_result)

            completed_tasks += 1
            pbar.update(1)
            
            if i % 20 == 0:
                torch.cuda.empty_cache()
            
            elapsed_time = time.time() - start_time
            avg_time_per_task = elapsed_time / completed_tasks
            remaining_tasks = total_tasks_num - completed_tasks
            estimated_remaining_time = remaining_tasks * avg_time_per_task
            pbar.set_postfix({"task": i, "eta": f"{estimated_remaining_time:.2f}s"})
    
    # final log
    total_time = time.time() - start_time
    print(f"All tasks completed in {total_time:.2f} seconds.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, \
        default='benchmarks/simpleqa_800.jsonl')    
    parser.add_argument("--output_dir", type=str, default="model_responses")
    
    parser.add_argument("--start", type=int, default = 0)
    parser.add_argument("--end", type=int, default = -1)

    # Qwen/Qwen3-[*]B, 8B, 14B, (32B)
    parser.add_argument(
        "-m", "--model-name", default="Qwen/Qwen3-8B"
    )
    parser.add_argument("-mbt", "--max_batch_tokens", type=int, default = 520, help="max batch tokens")
    parser.add_argument("--max_output_tokens", type=int, default=20000, help="max output tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.8, help="top_p for sampling")
    parser.add_argument("--top_k", type=int, default=20, help="top_k for sampling")
    parser.add_argument("--min_p", type=float, default=0., help="min_p for sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="repetition penalty for sampling")
    
    parser.add_argument("-d", "--device", default="auto")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(args)
