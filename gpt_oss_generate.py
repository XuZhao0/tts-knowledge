"""This file is used to generate responses from gpt-oss models."""
import argparse
import torch
import os
import logging
import time
from tqdm import tqdm
from datetime import datetime
from tool import *
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

@torch.inference_mode
def inference_question(tokenizer, model, task, args):

    question = """Give me the answer to the following question only when you are sure of it. \
Otherwise, say 'I don't know'. Put your answer on its own line after 'Answer:'.\n""" + task["problem"]
  
    messages = [
        {"role": "user", "content": question},
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        reasoning_effort=args.effort,
    ).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=20000,
        temperature=args.temperature,
    )
    
    response = tokenizer.decode(outputs[0])
    
    # === Split model response into structured parts === 
    
    # Split the full response by the 'analysis' channel marker
    splitted_response = response.split("<|channel|>analysis<|message|>")
    
    prompt_part = splitted_response[0].strip()
    
    # Handle the analysis (thought) and final response parts
    analysis_final_response = splitted_response[-1]
    
    # Extract the "thought"
    analysis_part = analysis_final_response.split("<|channel|>final<|message|>")[0].strip()
    analysis_part = analysis_part.replace("<|end|><|start|>assistant","")
    # Extract the "response"
    final_response = analysis_final_response.split("<|channel|>final<|message|>")[-1].strip()
    final_response = final_response.replace("<|return|>","")
    
    length_analysis_part = len(tokenizer.encode(analysis_part))
    length_final_response = len(tokenizer.encode(final_response))
    
    task["response"] = final_response
    task["thought"] = analysis_part
    task["thinking_tokens"] = length_analysis_part
    task["response_tokens"] = length_final_response
    
    if args.verbose:
        print("=="*20)
        print("\nQuestion: \n", question)
        print("\nResponse: \n", task["response"])
        print("\nThinking content: \n", task["thought"])
        print("\n\n")
        
    return task, prompt_part

def main(args):
    
    dt_string = datetime.now().strftime("%m_%d_%H_%M")
    start_time = time.time()
    all_data = jsonlines_load(args.input_path)
    if args.end == -1:
        end_index = len(all_data)
    else:
        end_index = args.end
    
    saved_model_name = args.model_name.split("/")[-1]
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    output_path = f"{args.output_dir}/{saved_model_name}_{args.effort}_s{args.start}_e{end_index}_{dt_string}.jsonl"
    
    logging.critical("*-"*40)
    logging.critical(f"Start time: {dt_string}.")
    logging.critical(f"Model: {args.model_name} || Effort: {args.effort} || Temperature: {args.temperature}.")
    logging.critical(f"Input path: {args.input_path} || Output path: {output_path}.")
 
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    
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
                    task, prompt_demo = inference_question(tokenizer, model, tasks[i], args)
                    if i == 0:
                        print("=="*20)
                        print("\nPrompt: \n", prompt_demo)
                        print("\nThought: \n", task["thought"])
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
    
    total_time = time.time() - start_time
    print(f"All tasks completed in {total_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, \
        default='benchmarks/simpleqa_800.jsonl')
    parser.add_argument("--output_dir", type=str, default="model_responses")
    
    parser.add_argument("--start", type=int, default = 0)
    parser.add_argument("--end", type=int, default = -1)
    
    parser.add_argument(
        "-m", "--model-name", default="openai/gpt-oss-20b"
    )
    parser.add_argument("--effort", type=str, default="low", choices=["low", "medium", "high"])
    parser.add_argument("--temperature", type=float, default=0.7, help="temperature for sampling")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(args)