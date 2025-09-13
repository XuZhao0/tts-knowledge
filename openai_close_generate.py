"""This file is used to generate responses from OpenAI closed-source models."""
import concurrent.futures
import argparse
import logging
import time
import random
import traceback
import os
from datetime import datetime
from tqdm import tqdm
from tool import *

def generate_responses(prompt, args):
    openai_client = load_openai_client(api_key=args.api_key)
    
    # o4-mini and gpt-5-mini support "detailed" summary
    if args.model == "o4-mini" or args.model == "gpt-5-mini":
        summary_param = "detailed"
    else:
        summary_param = "auto"        
    
    completion = openai_client.responses.create(
        model= args.model,
        reasoning = {"effort": args.effort,
                     "summary": summary_param},
        input=[
            {"role": "user", "content": prompt}
        ]
    )
    return completion.output_text, completion

def process_task(task, args, max_retries=5):
    
    # For GPT-5 or GPT-5-mini, we do not explicitly ask for uncertainty handling, as it will cause the model to overly refuse to answer.
    if "gpt-5" in args.model:
        prompt = """Give me the answer to the following question. Put your answer on its own line after 'Answer:'.\n""" + task['problem']
    else:
        prompt = """Give me the answer to the following question only when you are sure of it. \
Otherwise, say 'I don't know'. Put your answer on its own line after 'Answer:'.\n""" + task['problem']

    for attempt in range(0, max_retries):
        try:
            response, completion = generate_responses(
                prompt = prompt,
                args = args
            ) 
            
            reasoning_summary = ""
            for item in completion.output:
                if item.type == "reasoning":
                    for summary in item.summary:
                        reasoning_summary += summary.text + "\n"
            
            reasoning_summary = reasoning_summary.strip()
            
            if args.verbose:
                print('prompt: ', prompt)
                print('\nresponse: ', response)
                print('\ncompletion: ', completion)
                print('\nreasoning_summary: ', reasoning_summary)
            
            task['response'] = response
            task['thought_summary'] = reasoning_summary
            task['completion_tokens'] = completion.usage.output_tokens
            task['reasoning_tokens'] = completion.usage.output_tokens_details.reasoning_tokens
            task['effort'] = completion.reasoning.effort if hasattr(completion, 'reasoning') else None
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
    
    parser.add_argument('--model', type=str, default='o3-mini')
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--effort', type=str, default='medium',
        choices=["minimal", "low", "medium", "high"],\
        help="Reasoning effort level for OpenAI reasoning models.\
            `minimal` is only supported for gpt-5 series models.")
    
    parser.add_argument("--workers", type=int, default = 2,\
        help="Number of workers for multi-processing")
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    main(args)
