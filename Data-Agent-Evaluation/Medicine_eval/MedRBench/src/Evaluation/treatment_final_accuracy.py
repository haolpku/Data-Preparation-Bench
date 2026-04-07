import os
import json
import argparse
import logging
import multiprocessing
from metrics.outcome_accuracy_eval import eval_accuracy_with_websearch
# ========================================================
NEW_API_KEY = "sk-dummy"
NEW_BASE_URL = "http://XXX/v1"
NEW_MODEL_NAME = "gpt-4o"
IS_FULL_EVAL = False 
REFER_MODEL_NAME = "gpt-4o"                                            
REFER_API_BASE_URL = "http://XXX/v1"                  
REFER_API_KEY = "sk-dummy"  
# ========================================================
os.environ['OPENAI_API_KEY'] = REFER_API_KEY
os.environ['OPENAI_BASE_URL'] = REFER_API_BASE_URL



# Configuration constants
NUM_WORKERS = 4  # Number of parallel worker processes
EVALUATION_MODEL = "gpt-4o"  # Language model used for evaluation

# Set up simple logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def evaluate_case(case_data, output_directory, model_name):
    """Evaluate a single case and save results."""
    logger.info(f'Evaluating case {case_data["id"]} for model {model_name}')

    try:
        # Extract case information components
        case_summary = case_data['generate_case']['case_summary']
        treatment_planning_analysis = case_data['generate_case']['treatment_planning_analysis']
        treatment_plan_results = case_data['generate_case']['treatment_plan_results']
        
        # Combine case information
        case_context = (f"Case Summary: {case_summary}\n"
                        f"Treatment Planning Analysis: {treatment_planning_analysis}\n"
                        f"Treatment Plan Results: {treatment_plan_results}")

        # Get ground truth and model prediction
        ground_truth = case_data['generate_case']['treatment_plan_results']
        # model_prediction_raw = case_data['results']['content']
        model_prediction_raw = case_data.get('result', {}).get('content', "")

        # Extract the answer part if it contains the specific format
        # model_prediction = model_prediction_raw.split('### Answer').split('### Answer:')[1].strip()
        if '### Answer:' in model_prediction_raw:
            model_prediction = model_prediction_raw.split('### Answer:')[1].strip()
        elif '### Answer' in model_prediction_raw:
            model_prediction = model_prediction_raw.split('### Answer')[1].strip()
        else:
            model_prediction = model_prediction_raw
        
        # Evaluate accuracy using imported function
        keywords, search_results, is_accurate = eval_accuracy_with_websearch(
            model_prediction, 
            ground_truth, 
            case_context,
            evaluation_model=EVALUATION_MODEL
        )

        # Store evaluation results
        case_data['search'] = {
            'keywords': keywords, 
            'info': search_results
        }   
        case_data['accuracy'] = is_accurate

        # Save results to file
        output_file = os.path.join(output_directory, f'{case_data["id"]}.json')
        with open(output_file, 'w', encoding="utf-8") as f:
            json.dump(case_data, f, ensure_ascii=False, indent=4)
            
    except Exception as e:
        logger.error(f'Error processing case {case_data["id"]}: {str(e)}')


def worker_process(task_queue):
    """Process evaluation tasks from a queue."""
    while not task_queue.empty():
        try:
            case_data, output_directory, model_name = task_queue.get()
            evaluate_case(case_data, output_directory, model_name)
        except Exception as e:
            logger.error(f"Worker error: {e}")


def main(model_name, patient_case_filepath, model_output_filepath, output_directory, use_parallel=True):
    """Orchestrate the evaluation process for a specific model."""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)


    with open(patient_case_filepath, 'r', encoding='utf-8') as f:
        patient_cases = json.load(f)
    
    with open(model_output_filepath, 'r', encoding='utf-8') as f:
        model_outputs = json.load(f)
    
    cases_to_evaluate = []
    completed_cases = os.listdir(output_directory)
    completed_case_ids = [name.split('.')[0] for name in completed_cases]
    

    for case_id in patient_cases.keys():

        if case_id in completed_case_ids:
            continue
        

        if case_id in model_outputs and 'result' in model_outputs[case_id]:
            case_data = patient_cases[case_id].copy()
            case_data['id'] = case_id

            case_data['result'] = model_outputs[case_id]['result']
            cases_to_evaluate.append(case_data)    
    
    logger.info(f'Total cases to evaluate: {len(cases_to_evaluate)}')
    
    if len(cases_to_evaluate) == 0:
        logger.warning("No cases found with 'result' key. Please check your model output JSON.")
        return   
    
    logger.info(f'Total cases to evaluate: {len(cases_to_evaluate)}')
    
    if use_parallel and len(cases_to_evaluate) > 1:
        # Parallel processing approach
        manager = multiprocessing.Manager()
        task_queue = manager.Queue()
        
        # Add all tasks to queue
        for case_data in cases_to_evaluate:
            task_queue.put((case_data, output_directory, model_name))

        # Create and start worker processes
        processes = []
        worker_count = min(NUM_WORKERS, len(cases_to_evaluate))
        logger.info(f"Starting {worker_count} worker processes")
        
        for _ in range(worker_count):
            process = multiprocessing.Process(target=worker_process, args=(task_queue,))
            process.start()
            processes.append(process)

        # Wait for all processes to complete
        for process in processes:
            process.join()
    else:
        # Sequential processing approach
        for case_data in cases_to_evaluate:
            evaluate_case(case_data, output_directory, model_name)
    
    logger.info(f"Evaluation completed for model {model_name}")


if __name__ == '__main__':
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Evaluate model accuracy on treatment planning tasks')
    parser.add_argument('--model', type=str, default=NEW_MODEL_NAME, 
                      help='Model to evaluate')
    parser.add_argument('--sequential', action='store_true', 
                      help='Run sequentially instead of using parallel processing')
    parser.add_argument('--output-dir', type=str, default=f'../../data/EvalResults/acc_results_treatment_{NEW_MODEL_NAME}',
                      help='Base directory for evaluation results')
    parser.add_argument('--patient-cases', type=str,
                      default='../../data/MedRBench/treatment_496_cases_with_rare_disease_165.json',
                      help='Path to patient cases file')
    parser.add_argument('--model-outputs', type=str,
                      default=f'../../data/InferenceResults/{NEW_MODEL_NAME}_oracle_treatment.json',
                      help='Path to model outputs file')
    
    args = parser.parse_args()
    
    # Define input and output file paths
    model_output_filepath = args.model_outputs
    patient_case_filepath = args.patient_cases
    output_directory = f'{args.output_dir}/{args.model}'
    
    # Run main evaluation process
    main(
        args.model, 
        patient_case_filepath, 
        model_output_filepath, 
        output_directory, 
        not args.sequential
    )