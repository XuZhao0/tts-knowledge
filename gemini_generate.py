"""This file is used to generate responses from Google Gemini models."""
import argparse
import logging
import time
import random
import os
import concurrent.futures
from datetime import datetime
from tqdm import tqdm
from tool import *
from google.genai import types

def generate_response(prompt, args):
    
    gemini_client = load_gemini_client(api_key=args.api_key)
    
    completion = gemini_client.models.generate_content(
        model= args.model,
        contents=prompt,
        config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=args.thinking_budget,
                                              include_thoughts=True)
        )
    )
    
    thought_summary = ""
    
    if args.thinking_budget != 0:
        for part in completion.candidates[0].content.parts:
            if not part.text:
                continue
            if part.thought:
                thought_summary += part.text
           
    return completion.text, completion, thought_summary

def process_task(task, args, max_retries=5):

    prompt = """Give me the answer to the following question only when you are sure of it. \
Otherwise, say 'I don't know'. Put your answer on its own line after 'Answer:'.\n""" + task['problem']
    
    retry_count = 0
    response = None
    thought_summary = ""
    
    while not response:
        try:
            response, completion, thought_summary = generate_response(
                prompt = prompt,
                args = args
            )
        except Exception as e:
            retry_count += 1
            logging.warning(f"!! Retry {retry_count}/{max_retries} for task {task['id']}|| Error: {e}")
            if retry_count >= max_retries:
                raise Exception("Max retries reached")
            time.sleep(2)
        
    if args.verbose:
        print(f"\nPrompt: {prompt}")
        print(f"\nResponse: {response}")
        print(f"\nThought Summary: {thought_summary}")
    
    task['response'] = response
    task['thought_summary'] = thought_summary
    task['reasoning_tokens'] = 0 if not completion.usage_metadata.thoughts_token_count \
        else completion.usage_metadata.thoughts_token_count
    task['response_tokens'] = completion.usage_metadata.candidates_token_count
    
    # sleep to avoid rate limit    
    time.sleep(random.uniform(0.2, 0.8))
    
    return task, completion

def main(args):
    dt_string = datetime.now().strftime("%m_%d_%H_%M")
    start_time = time.time()
    
    all_data = jsonlines_load(args.input_path)
    
    if args.end == -1:
        end_index = len(all_data)
    else:
        end_index = args.end
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    logging.critical(f"Thinking Budget: {args.thinking_budget} || Writing to: {args.output_dir}")    
    output_path = f"{args.output_dir}/{args.model}_tb{args.thinking_budget}_s{args.start}_e{end_index}_{dt_string}.jsonl"
    
    tasks = all_data[args.start:end_index] 
    total_tasks = len(tasks)
    completed_tasks = 0
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_task, task, args): task for task in tasks}
        print(f"Processing {total_tasks} tasks with {args.workers} workers...")
        # Progress bar with TQDM
        with tqdm(total=total_tasks) as pbar:
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]      
                try:
                    result, completion = future.result()
                    jsonlines_dump(output_path, result)
                except Exception as e:
                    logging.warning(f"!! Error: {e}")
                    error_result = {
                        "id": task['id'],
                        "error": str(e)
                    }
                    error_output_path = output_path.replace('.jsonl', '_error.jsonl')
                    jsonlines_dump(error_output_path, error_result)
                    
                completed_tasks += 1
                pbar.update(1)
                
                # Update ETA
                elapsed_time = time.time() - start_time
                avg_time_per_task = elapsed_time / completed_tasks
                remaining_tasks = total_tasks - completed_tasks
                estimated_remaining_time = remaining_tasks * avg_time_per_task
                pbar.set_postfix(eta=f"{estimated_remaining_time:.2f}s")

    # Final log
    total_time = time.time() - start_time
    print(f"All tasks completed in {total_time:.2f} seconds.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_path', type=str, \
        default='benchmarks/simpleqa_800.jsonl')
    parser.add_argument('--output_dir', type=str, default="model_responses")
    
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default = -1)
    
    parser.add_argument('--model', type=str, default='gemini-2.5-flash')
    parser.add_argument('--api_key', type=str, required=True) 
    parser.add_argument('-tb', '--thinking_budget', type=int, default=256,\
        help="Budget ranges from 0 to 24576. Setting to 0 disables thinking.")

    parser.add_argument("--workers", type=int, default = 2,\
        help="Number of workers for multi-processing")    
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    main(args)
