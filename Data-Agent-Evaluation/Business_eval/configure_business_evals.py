import re
import os
import openai

# ========================================================
NEW_API_KEY = "sk-dummy"
NEW_BASE_URL = "http://XXX/v1"
NEW_MODEL_NAME = "gpt-4o"
IS_FULL_EVAL = False                                                   
REFER_MODEL_NAME = "gpt-4o"                                            
REFER_API_BASE_URL = "http://XXX/v1"                  
REFER_API_KEY = "sk-dummy"  
# ========================================================

def test_openai_api(api_key, base_url, model_name):
    """
    Test OpenAI API connectivity by sending a simple message.
    Returns True if successful, False otherwise.
    """
    print(f"Testing API: model={model_name}, base_url={base_url}")
    try:
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Say 'Hello, world!'"}],
            max_tokens=10,
            temperature=0.0
        )
        content = response.choices[0].message.content
        print(f"  Success: Received response -> {content!r}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False

def update_file(file_path, patterns):
    if not os.path.exists(file_path):
        print(f" ERROR:can't find file: {file_path}")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f" Updated: {file_path}")

def run_configuration():
    print("=== Testing API connectivity ===")
    new_ok = test_openai_api(NEW_API_KEY, NEW_BASE_URL, NEW_MODEL_NAME)
    refer_ok = test_openai_api(REFER_API_KEY, REFER_API_BASE_URL, REFER_MODEL_NAME)
    if not (new_ok and refer_ok):
        print("ERROR: One or both API tests failed. Aborting configuration update.")
        return
    print("All API tests passed. Proceeding with file updates.\n")
    print(f" IS_FULL_EVAL: {IS_FULL_EVAL}")


    patterns = [
        (r'NEW_API_KEY\s*=\s*".*?"', f'NEW_API_KEY = "{NEW_API_KEY}"'),
        (r'NEW_BASE_URL\s*=\s*".*?"', f'NEW_BASE_URL = "{NEW_BASE_URL}"'),
        (r'NEW_MODEL_NAME\s*=\s*".*?"', f'NEW_MODEL_NAME = "{NEW_MODEL_NAME}"'),
        (r'IS_FULL_EVAL\s*=\s*.*', f'IS_FULL_EVAL = {IS_FULL_EVAL} '),
        (r'REFER_MODEL_NAME\s*=\s*".*?"', f'REFER_MODEL_NAME = "{REFER_MODEL_NAME}"'),
        (r'REFER_API_BASE_URL\s*=\s*".*?"', f'REFER_API_BASE_URL = "{REFER_API_BASE_URL}"'),
        (r'REFER_API_KEY\s*=\s*".*?"', f'REFER_API_KEY = "{REFER_API_KEY}"'),
    ]

    update_file("XFinBench/evaluate/load_models.py", patterns)
    update_file("XFinBench/evaluate/main.py", patterns)
    update_file("XFinBench/evaluate/rejudge.py", patterns)

    update_file("FinCDM/src/run_task_eval.py", patterns)
    update_file("FinCDM/src/referee_review.py", patterns)




    print("\n ALL FINISHED!")

if __name__ == "__main__":
    run_configuration()