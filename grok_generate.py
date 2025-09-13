"""This file is used to generate responses from XAI Grok-3 models."""
import argparse
import logging
import time
import random
import os
import traceback
import concurrent.futures
from datetime import datetime
from tqdm import tqdm
from tool import *

def generate_responses(prompt, args):
    xai_client = load_xai_client(api_key=args.api_key)
    
    completion = xai_client.chat.completions.create(
        model= args.model,
        reasoning_effort= args.effort,
        messages = [
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=args.temperature
    )
    return completion.choices[0].message.content, completion

def process_task(task, args, max_retries=5):
    
    prompt = """Give me the answer to the following question only when you are sure of it. \
Otherwise, say 'I don't know'. Put your answer on its own line after 'Answer:'.\n""" + task['problem']

    for attempt in range(0, max_retries):
        try:
            response, completion = generate_responses(
                prompt = prompt,
                args = args
            )

            thought = completion.choices[0].message.reasoning_content
            if args.verbose:
                print('\nPrompt: ', prompt)
                print('\nResponse: ', response)
                print('\nThought: ', thought)
            
            task['response'] = response
            task['thought'] = thought
            task['completion_tokens'] = completion.usage.completion_tokens
            task['reasoning_tokens'] = completion.usage.completion_tokens_details.reasoning_tokens
            # sleep to avoid rate limit
            time.sleep(random.uniform(1.0, 2.0))
            return task, completion
        
        except Exception as e:
            logging.warning(f"Attempt {attempt}/{max_retries} failed: {e}")
            trace = traceback.format_exc()
            logging.warning(trace)
            with open('error_log.txt', 'a') as f:
                f.write("**"*20)
                f.write(f"output path: {args.output_dir}\n")
                f.write(f"Attempt {attempt}/{max_retries} failed: {e}\n")
                f.write(trace)
                f.write("\n")
            if attempt == max_retries:
                logging.error(f"Max retries reached for task ID {task.get('id', '[unknown]')}.")
                raise e
            time.sleep(random.uniform(2, 8))
    

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
    
    logging.critical(f"Writing to: {args.output_dir}")    
    logging.critical(f"Model: {args.model} with reasoning effort: {args.effort}")
    
    output_path = f"{args.output_dir}/{args.model}_{args.effort}_s{args.start}_e{end_index}_{dt_string}.jsonl"
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
                    continue
                    
                completed_tasks += 1
                pbar.update(1)
                
                # Update ETA
                elapsed_time = time.time() - start_time
                avg_time_per_task = elapsed_time / completed_tasks
                remaining_tasks = total_tasks - completed_tasks
                estimated_remaining_time = remaining_tasks * avg_time_per_task
                pbar.set_postfix(eta=f"{estimated_remaining_time:.2f}s")

    total_time = time.time() - start_time
    print(f"All tasks completed in {total_time:.2f} seconds.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, \
        default='benchmarks/simpleqa_800.jsonl')
    parser.add_argument('--output_dir', type=str, default="model_responses")
    
    parser.add_argument('--start', type=int, default = 0)
    parser.add_argument('--end', type=int, default = -1)
    
    parser.add_argument('--model', type=str, default='grok-3-mini')
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--effort', type=str, default='low',
        choices=["low", "high"],\
        help="Reasoning effort level for Grok-3. [low, high]")
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--workers", type=int, default = 2,\
        help="Number of workers for multi-processing")    
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    main(args)
