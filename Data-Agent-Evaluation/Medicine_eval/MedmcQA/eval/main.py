import os
import json
from setup import *
from utils import *
import argparse
import traceback
from config import PromptTemplates
from concurrent.futures import ThreadPoolExecutor, as_completed

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logging.getLogger('openai').setLevel(logging.WARNING)

# ========================================================
NEW_API_KEY = "sk-dummy"
NEW_BASE_URL = "http://172.96.160.199:3000/v1"
NEW_MODEL_NAME = "gpt-4o"
IS_FULL_EVAL = False 
REFER_MODEL_NAME = "gpt-4o"                                            
REFER_API_BASE_URL = "http://172.96.160.199:3000/v1"                  
REFER_API_KEY = "sk-dummy"  
# ========================================================

def init_file_if_needed(file_path, remove_cache):
    if os.path.exists(file_path) and remove_cache:
        os.remove(file_path)
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            pass


def write_to_file(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def zero_shot_ao(prompt_templates, input_sample, llm_agent, output, dataset, task, messages):
    # Construct the prompt
    options_num = len(input_sample.get('options', []))
    question = input_sample['question'].strip()

    if options_num > 0 and "{start}" in prompt_templates.zero_shot_ao_prompt:
        prompt = prompt_templates.zero_shot_ao_prompt.format(
            question=question,
            start=chr(65),
            end=chr(65 + options_num - 1)
        )
    else:
        prompt = prompt_templates.zero_shot_ao_prompt.format(question=question)
    output['prompt'] = prompt
    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": prompt}]
    })

    # Add images to messages if present
    images = input_sample.get('images', [])
    if images:
        for image in images:
            image_url = image if isinstance(image, str) else os.path.join("images", image.get('image_path', ''))
            messages[-1]["content"].append(llm_agent.image_content(image_url))

    # Get the response from the LLM agent
    response, conf = llm_agent.get_response(messages)
    response = response.strip()

    # Append the assistant's response to messages
    messages.append({"role": "assistant", "content": response})

    # Store prediction rationale in output
    output["prediction_rationale"] = response

    # Log the interaction
    logging.debug(f"【User】{prompt}\n\n")
    logging.debug(f"【Assistant】{response}\n\n")

    # Parse the answer
    prediction = answer_cleansing("zero_shot", prompt_templates.zero_shot_ao_trigger, response, dataset, task, input_sample)

    return response, prediction, conf, messages


def zero_shot_cot(prompt_templates, input_sample, llm_agent, output, dataset, task, messages):
    # Stage 1: Generate initial prompt and handle images
    prompt = prompt_templates.zero_shot_cot_prompt.format(question=input_sample['question'].strip())
    output['prompt'] = prompt
    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": prompt}],
    })

    # Add images to messages if present
    images = input_sample.get('images', [])
    if images:
        for image in images:
            image_url = image if isinstance(image, str) else os.path.join("images", image.get('image_path', ''))
            messages[-1]["content"].append(llm_agent.image_content(image_url))

    response, conf_1 = llm_agent.get_response(messages)
    response = response.strip()
    messages.append({"role": "assistant", "content": response})

    # Log the interaction
    logging.debug(f"【User】{prompt}\n\n")
    logging.debug(f"【Assistant】{response}\n\n")

    output['prediction_rationale'] = response

    # Stage 2: Handle options and triggers
    options_num = len(input_sample.get('options', []))
    if options_num > 0 and "{start}" in prompt_templates.zero_shot_ao_prompt:
        prompt = prompt_templates.zero_shot_cot_trigger.format(start=chr(65), end=chr(65 + options_num - 1))
    else:
        prompt = prompt_templates.zero_shot_cot_trigger

    messages.append({"role": "user", "content": prompt})

    response, conf_2 = llm_agent.get_response(messages)
    response = response.strip()
    messages.append({"role": "assistant", "content": response})

    # Log the interaction
    logging.debug(f"【User】{prompt}\n\n")
    logging.debug(f"【Assistant】{response}\n\n")

    # Parse the answer
    prediction = answer_cleansing("zero_shot", prompt_templates.zero_shot_cot_trigger, response, dataset, task, input_sample)

    conf = conf_1  # The confidence from stage 1 is used in the final return
    return response, prediction, conf, messages


def few_shot(prompt_templates, input_sample, llm_agent, output, dataset, task, prompting_type, messages):
    # Construct the prompt for the question
    prompt = prompt_templates.few_shot_prompt.format(question=input_sample["question"].strip())
    output["prompt"] = prompt

    # Add user prompt to messages
    messages.append(
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    )

    # Add images to messages if present
    images = input_sample.get('images', [])
    if images:
        for image in images:
            image_url = image if isinstance(image, str) else image.get('image_url', '')
            messages[-1]["content"].append(llm_agent.image_content(image_url))

    # Get the response from the LLM agent
    response, conf = llm_agent.get_response(messages)
    response = response.strip()

    # Add assistant's response to the messages
    messages.append({"role": "assistant", "content": response})

    # Log the interaction
    logging.debug(f"【User】{prompt}\n\n")
    logging.debug(f"【Assistant】{response}\n\n")

    # Set prediction rationale if prompting_type is "cot"
    if prompting_type == "cot":
        output['prediction_rationale'] = response

    # Parse the answer
    prediction = answer_cleansing("few_shot", prompt_templates.few_shot_trigger, response, dataset, task, input_sample)

    return response, prediction, conf, messages


def set_correctness(label, prediction, dataset, input_sample):
    """Helper function to determine correctness based on the method."""
    if len(prediction) == 0:
        return False

    if len(label) == 1 and label[0].replace('.', '', 1).replace('e', '', 1).isdigit():  # Math check
        label = float(label[0])
        prediction = float(prediction[0])
        return label == prediction
    
    elif len(label) == 1 and dataset in ["slake", "vqa-rad", "path-vqa"]:  # Open/Close QA
        label = general_postprocess(label[0])
        label = tokenize_and_lemmatize(label)
        extracted_item = {"label_list": label, "predict_list": prediction}
        metric = calculate_metrics(extracted_item)
        return metric['f1'] == 1.0
    
    elif len(label) == 1:  # Multiple Choice check
        return label[0].lower() == prediction[0].lower()
    
    elif len(label) > 1 and "options" in input_sample:  # Multiple Choice for multiple labels
        return len(label) == len(prediction) and all(l == p for l, p in zip(label, prediction))
    
    else:
        raise ValueError("Dataset is not properly defined ...")


def complete_item(args, task, llm_agent, prompt_templates, input_sample, demonstrations_messages, index):
    try:
        # Inference
        output = input_sample
        logging.debug("-------------------------------------------")
        logging.info(f"### Start Index: {index}")

        if args.method == "zero_shot":
            if args.prompting_type == "ao":
                messages = [{"role": "system", "content": prompt_templates.zero_shot_ao_system_role}]
                response, prediction, conf, messages = zero_shot_ao(prompt_templates, input_sample, llm_agent, output, args.dataset, task, messages)
            elif args.prompting_type == "cot":
                messages = [{"role": "system", "content": prompt_templates.zero_shot_system_role}]
                response, prediction, conf, messages = zero_shot_cot(prompt_templates, input_sample, llm_agent, output, args.dataset, task, messages)
        
        elif args.method == "few_shot":
            messages = [{"role": "system", "content": prompt_templates.few_shot_system_role}]
            messages.extend(demonstrations_messages)
            response, prediction, conf, messages = few_shot(prompt_templates, input_sample, llm_agent, output, args.dataset, task, args.prompting_type, messages)
        
        else:
            raise Exception('Method Error!')

        # Evaluation
        logging.debug(f"### Label: {input_sample['label']}")
        logging.debug(f"### Prediction: {prediction}")

        label = input_sample['label'] if isinstance(input_sample['label'], list) else [input_sample['label']]
        is_correct = set_correctness(label, prediction, args.dataset, input_sample)

        logging.info(f"### Completed Index: {index} [{is_correct}]")

        # Process and clean messages
        for msg in messages:
            if msg['role'] == "user" and isinstance(msg["content"], list):
                msg["content"] = [c for c in msg["content"] if c["type"] != "image_url"]
        
        # Prepare final output
        output['messages'] = messages
        output['response'] = response
        output['confidence'] = conf
        output['label'] = label
        output['prediction'] = prediction
        output['correct'] = is_correct

    except Exception as e:
        logging.error(f"Error: {e}")
        logging.error(traceback.format_exc())
        exit()

    return output, conf, index


def general_inference(args, task, llm_agent, prompt_templates):
    tmp_dir = os.path.join("outputs", args.output_dir, args.model, args.dataset, args.method, args.prompting_type)

    # Path Config
    task_suffix = f"_{task}" if task else ""
    demo_path = os.path.join(DEMO_PATH.format(dataset=args.dataset), f"{args.dataset}{task_suffix}_demonstrations.json")
    input_path = os.path.join(INPUT_PATH.format(dataset=args.dataset), f"{args.dataset}{task_suffix}_input.jsonl")
    tmp_path = os.path.join(tmp_dir, f"{args.dataset}{task_suffix}_output.jsonl")

    # File Initialization
    os.makedirs(tmp_dir, exist_ok=True)
    init_file_if_needed(tmp_path, args.remove_cache)

    # Load Demonstration
    demonstrations_messages = []
    if args.method == "few_shot":
        with open(demo_path, "r") as f:
            demonstrations = json.load(f)
        for demo in demonstrations:
            q = prompt_templates.few_shot_prompt.format(question=demo['question'].strip())
            if args.prompting_type == "cot":
                a = demo['label_rationale'].strip()
                if prompt_templates.few_shot_trigger.lower() not in a.lower():
                    a += f" {prompt_templates.few_shot_trigger} {demo['label'].strip()}."
            else:
                a = demo['label'].strip()
            demo_message = {
                "role": "user",
                "content": [{"type": "text", "text": q}],
            }
            if "images" in demo and demo['images']:
                for img in demo['images']:
                    demo_message["content"].append(llm_agent.image_content(img))
            demonstrations_messages.append(demo_message)
            demonstrations_messages.append({"role": "assistant", "content": a})

    # Load Inputs
    outputs = []
    with open(tmp_path, 'r', encoding='utf-8') as f:
        outputs = [json.loads(line) for line in f if line.strip()]
    with open(input_path, 'r', encoding='utf-8') as f:
        inputs = [json.loads(line) for line in f if line.strip()]

    start = len(outputs)
    end = min(args.max_samples if args.max_samples != -1 else len(inputs), len(inputs))

    assert end > 0

    batch_size = min(max(1, min(args.num_threads * 3, 500)), end - start)
    if start >= end:
        return

    for i in range(start, end, batch_size):
        batch_start, batch_end = i, min(i + batch_size, end)
        inputs_process = inputs[batch_start:batch_end]

        # Single Thread
        if args.num_threads == 1:
            for index, input_sample in enumerate(inputs_process):
                output, conf, _ = complete_item(args, task, llm_agent, prompt_templates, input_sample, demonstrations_messages, batch_start + index + 1)
                outputs.append(output)
            write_to_file(tmp_path, outputs)

        # Multi Thread
        else:
            num_threads = min(args.num_threads, len(inputs_process))  # Ensure num_threads is less than or equal to the batch size
            chunk_size = max(1, len(inputs_process) // num_threads)  # Ensure chunk_size is at least 1
            futures = []
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                for chunk_index in range(0, len(inputs_process), chunk_size):
                    chunk = inputs_process[chunk_index:chunk_index + chunk_size]
                    for index, item in enumerate(chunk):
                        futures.append(
                            executor.submit(
                                complete_item,
                                args,
                                task,
                                llm_agent,
                                prompt_templates,
                                item,
                                demonstrations_messages,
                                batch_start + chunk_index + index + 1
                            )
                        )

                results = [None] * len(inputs_process)
                for future in as_completed(futures):
                    output, conf, index = future.result()
                    results[index - batch_start - 1] = (output, conf)

                outputs.extend([result[0] for result in results])

            write_to_file(tmp_path, outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default= NEW_MODEL_NAME , type=str)
    parser.add_argument("--dataset", default='medxpertqa_sampled', type=str)
    parser.add_argument("--task", default="text", type=str)
    parser.add_argument("--output-dir", default='dev', type=str)
    parser.add_argument("--method", default='zero_shot', type=str)
    parser.add_argument("--prompting-type", default='cot', type=str)
    parser.add_argument("--remove-cache", default=False, action='store_true', help="remove cache")
    parser.add_argument("--max-samples", default=-1, type=int)
    parser.add_argument("--num-threads", default=1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    args = parser.parse_args()

    llm_agent, tasks = setup(args.model, args.dataset, args.method, args.prompting_type)
    llm_agent.temperature = args.temperature

    tasks = [args.task] if args.task else tasks

    prompt_templates = PromptTemplates().load_templates(args.dataset, args.model)

    if tasks:
        for task in tasks:
            general_inference(args, task, llm_agent, prompt_templates)
    else:
        general_inference(args, None, llm_agent, prompt_templates)
