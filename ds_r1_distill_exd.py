"""This file is used to generate responses from DeepSeek-R1-Distill models."""
import argparse
import os
import time
import logging
from datetime import datetime
from tqdm import tqdm
from tool import *

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

@torch.inference_mode
def reasoning_effort(tokenizer, model, task, args, exd_times):

# See the following commit for the tokenizer update in DeepSeek-R1-Distill models
# https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/commit/ca24ee48c0532a014ddea4c32437a6f4be981ab2    
    if "R1-Distill-Qwen-" in args.model_name or "R1-Distill-Llama-" in args.model_name:
        _, _start_think_token, end_think_token = tokenizer.encode("<think></think>")
    elif "R1-0528-Qwen3-8B" in args.model_name:
        logging.critical("Check tokenizer encode!")
        _start_think_token, end_think_token = tokenizer.encode("<think></think>")
    else:
        logging.critical("Check tokenizer encode for <think></think>.")
        _, _start_think_token, end_think_token = tokenizer.encode("<think></think>")

    question = """Give me the answer to the following question only when you are sure of it. \
Otherwise, say 'I don't know'. Put your answer on its own line after 'Answer:'.\n""" + task["problem"]
    
    current_tokens = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "<think>\n"},
        ],
        continue_final_message=True, # force the model to start with <think>\n
        
        # === !! You shouldn’t use add_generation_prompt and continue_final_message together. \===
        # === !! The former adds tokens that start a new message, while the latter removes end of sequence tokens.===
        return_tensors="pt",
    )
    
    current_tokens = current_tokens.to(model.device)
    
    n_thinking_tokens = 0
    n_final_tokens = 0
    extend_count = 0
    total_new_tokens = 0
    thinking_content = ""
    final_content = ""
    selected_replacement = []
    replacement_token_place = []
    is_thinking = True # we always start with <think>
    
    max_batch_tokens = args.max_batch_tokens
    
    while total_new_tokens < args.max_output_tokens:
                
        outputs = model.generate(
            input_ids=current_tokens,
            max_new_tokens = min(max_batch_tokens, args.max_output_tokens - total_new_tokens),
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
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
            
            # we force the model to think more, when it attempts to end the thinking phase
            if (
                token_id in (end_think_token, model.config.eos_token_id)
                and (exd_times == -1 or extend_count < exd_times)
            ): # the model attempts to end the thinking phase, but we want to force it to think more
                replacement = args.replacement
                selected_replacement.append(replacement)
               
                replacement_tokens = tokenizer.encode(replacement)[1:] # remove bos_token
                
                n_thinking_tokens += len(replacement_tokens)
                extend_count += 1
                thinking_content += replacement
                replacement_token_place.append(current_tokens.shape[1]+i)

                current_tokens = torch.cat(
                    [outputs.sequences[:, :current_tokens.shape[1]+i],
                     torch.tensor([replacement_tokens]).to(current_tokens.device)],
                    dim=1,
                )
                
                # set smaller max_batch_tokens to speed up the inference, after forcing the model to think more
                if max_batch_tokens >= 500:
                    max_batch_tokens = 400

                break
                
            token_str = tokenizer.decode([token_id])
                
            if token_id == end_think_token: #</think>
                is_thinking = False
                thinking_content += token_str
                continue
            
            if is_thinking:
                thinking_content += token_str
                n_thinking_tokens += 1
            else:
                final_content += token_str
                n_final_tokens += 1
            
            if token_id == model.config.eos_token_id:
                task["response"] = final_content.strip()
                task["thought"] = thinking_content.strip()
                task["thinking_tokens"] = n_thinking_tokens
                task["response_tokens"] = n_final_tokens
                task["append_wait_place"] = replacement_token_place
                
                if args.verbose:
                    print("=="*20)
                    print("\nQuestion: \n", question)
                    print("\nGround truth:", task["answer"])
                    print("\nThought: \n", thinking_content)
                    print("\nResponse: \n", final_content.strip())
                
                return task, question
        
        else:
            # if we reach here, it means we did not break the loop
            current_tokens = outputs.sequences

def main(args):
    exd_times = args.extend_times
    assert exd_times >= 0, "extend_times should be 0 or positive integer."
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map=args.device
    )

    dt_string = datetime.now().strftime("%m_%d_%H_%M")
    start_time = time.time()
    all_data = jsonlines_load(args.input_path)

    saved_replacement = args.replacement.strip()
    saved_replacement = saved_replacement.lower()

    if args.end == -1:
        end_index = len(all_data)
    else:
        end_index = args.end
    
    saved_model_name = args.model_name.split("/")[-1]
    
    # create output directory if not exists
    saved_file_dir = f"{args.output_dir}/exd{exd_times}"
    
    if not os.path.exists(saved_file_dir):
        os.makedirs(saved_file_dir)
    
    output_path = f"{saved_file_dir}/{saved_model_name}_exd{exd_times}_{saved_replacement}_s{args.start}_e{end_index}_{dt_string}.jsonl" 
    
    logging.critical("*-"*40)
    logging.critical(f"Model: {args.model_name} || Budget forcing token: {args.replacement}.")
    logging.critical(f"Input path: {args.input_path} || Ouput path: {output_path}.")
    logging.critical(f"Extend times: {exd_times} || Batch tokens: {args.max_batch_tokens}.")

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
                    task, prompt_demo = reasoning_effort(tokenizer, model, tasks[i], args, exd_times)
                    if i == 0:
                        print("=="*20)
                        print("\nPrompt: ", prompt_demo)
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

    # DeepSeek-R1-Distill-Qwen-[*]B, (1.5B), 7B, 14B, (32B)
    # DeepSeek-R1-Distill-Llama-[*]B, 8B, (70B)
    parser.add_argument(
        "-m", "--model-name", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    )
    parser.add_argument("-mbt", "--max_batch_tokens", type=int, default = 520, help="batch tokens for each inference")
    parser.add_argument("--max_output_tokens", type=int, default=20000, help="max output tokens")
    parser.add_argument("--temperature", type=float, default=0.6, help="temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="top_p for sampling")
    # for deepseek_r1_distill models, we use the default top_k = 50. 
    parser.add_argument("--top_k", type=int, default=50, help="top_k for sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="repetition penalty for sampling")
    
    parser.add_argument("-r", "--replacement", type=str, default="\nWait",\
        help="Budget forcing token. We use \nWait for our experiments.")
    parser.add_argument("-e","--extend_times", type=int, default= 2, help="number of times to extend the thinking phase,\
        -1 means unlimited")
    
    parser.add_argument("-d", "--device", default="auto")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(args)
