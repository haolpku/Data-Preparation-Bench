import json
import os
import time

import tiktoken
from openai import OpenAI

API_URL = ""
API_KEY = ""
MODEL = "claude-opus-4-6"

# File path configuration
MD_FILE_PATH = "/Path/to/your/md/file.md"
RAW_RESPONSE_FILE_PATH = "/Path/to/your/raw_response.txt"
QA_JSONL_FILE_PATH = "/Path/to/your/file.jsonl"

# Runtime parameters selected by model name
MODEL_RUNTIME_CONFIG = {
    "gemini-3-pro-preview": {
        "SAFE_MAX_INPUT": 360000,
        "TEMPERATURE": 0.3,
        "MAX_OUTPUT_TOKENS": 128000,
    },
    "gpt-5.2": {
        "SAFE_MAX_INPUT": 270000,
        "TEMPERATURE": 0.3,
        "MAX_OUTPUT_TOKENS": 128000,
    },
    "claude-opus-4-6": {
        "SAFE_MAX_INPUT": 800000,
        "TEMPERATURE": 0.3,
        "MAX_OUTPUT_TOKENS": 128000,
    },
}

if MODEL not in MODEL_RUNTIME_CONFIG:
    raise ValueError(
        f"Unsupported MODEL: {MODEL}. Supported models: {', '.join(MODEL_RUNTIME_CONFIG.keys())}"
    )

SAFE_MAX_INPUT = MODEL_RUNTIME_CONFIG[MODEL]["SAFE_MAX_INPUT"]
TEMPERATURE = MODEL_RUNTIME_CONFIG[MODEL]["TEMPERATURE"]
MAX_OUTPUT_TOKENS = MODEL_RUNTIME_CONFIG[MODEL]["MAX_OUTPUT_TOKENS"]
REQUEST_TIMEOUT = 600  # Client timeout in seconds
MAX_RETRIES = 3

client = OpenAI(
    api_key=API_KEY,
    base_url=API_URL,
    timeout=REQUEST_TIMEOUT,
    max_retries=0,
)

SYSTEM_PROMPT = """You are an assistant that generates question-answer pairs from given text. Use the provided book excerpt to create QA pairs. Output only a JSON list where each item has \"question\" and \"answer\" fields."""

USER_PROMPT_TEMPLATE = """Book excerpt:
{book_content}

Generate question-answer pairs based on the above excerpt. Output as JSON list."""


def count_tokens(text: str, model: str = MODEL) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")  # Fallback encoding
    return len(encoding.encode(text))


def truncate_by_tokens(text: str, max_tokens: int, model: str = MODEL) -> str:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)


def read_md_file(file_path: str) -> str:
    with open(file_path, encoding="utf-8") as f:
        return f.read()


def call_llm_api(prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(MAX_RETRIES):
        try:
            print(f"Attempt {attempt + 1}/{MAX_RETRIES}, using streaming mode")
            stream = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_OUTPUT_TOKENS,
                stream=True,  # Streaming mode
                timeout=REQUEST_TIMEOUT,
            )
            collected_chunks = []
            for chunk in stream:
                # Some gateways may return errors inside stream chunks
                if hasattr(chunk, "error") and chunk.error:
                    raise Exception(f"Streaming response error: {chunk.error}")
                if chunk.choices and chunk.choices[0].delta.content:
                    collected_chunks.append(chunk.choices[0].delta.content)
            full_response = "".join(collected_chunks)
            if not full_response:
                raise Exception("Streaming response returned empty content")
            return full_response.strip()
        except Exception as e:
            print(f"API call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                sleep_time = 2**attempt
                print(f"Retrying in {sleep_time} seconds")
                time.sleep(sleep_time)
            else:
                raise


def parse_qa_response(response_text: str) -> list[dict[str, str]]:
    cleaned = response_text.strip()
    # Extract JSON fenced block when present
    if "```json" in cleaned:
        start = cleaned.find("```json") + 7
        end = cleaned.find("```", start)
        cleaned = cleaned[start:end].strip()
    elif "```" in cleaned:
        start = cleaned.find("```") + 3
        end = cleaned.find("```", start)
        cleaned = cleaned[start:end].strip()
    try:
        qa_list = json.loads(cleaned)
        if isinstance(qa_list, list) and all(
            "question" in item and "answer" in item for item in qa_list
        ):
            return qa_list

        print("Returned JSON format is invalid (missing question or answer field)")
        return []
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
        return []


def save_raw_response(response_text: str, file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(response_text)
    print(f"Raw response saved to: {file_path}")


def save_qa_pairs(qa_pairs: list[dict[str, str]], file_path: str):
    with open(file_path, "a", encoding="utf-8") as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")
    print(f"Appended {len(qa_pairs)} QA pairs to: {file_path}")


def main():
    qa_jsonl_path = QA_JSONL_FILE_PATH
    if qa_jsonl_path is None:
        base_name = os.path.splitext(os.path.basename(MD_FILE_PATH))[0]
        qa_jsonl_path = f"{base_name}_domain.jsonl"

    print(f"Reading source book: {MD_FILE_PATH}")
    full_content = read_md_file(MD_FILE_PATH)
    full_tokens = count_tokens(full_content)
    print(f"Total book tokens: {full_tokens:,}")

    system_tokens = count_tokens(SYSTEM_PROMPT)
    user_template_tokens = count_tokens(USER_PROMPT_TEMPLATE.format(book_content=""))
    fixed_overhead = system_tokens + user_template_tokens
    print(f"Fixed prompt overhead: {fixed_overhead:,} tokens")

    max_book_tokens = SAFE_MAX_INPUT - fixed_overhead
    if max_book_tokens <= 0:
        raise ValueError(f"SAFE_MAX_INPUT ({SAFE_MAX_INPUT}) is too small")
    print(f"Maximum allowed book tokens: {max_book_tokens:,}")

    if full_tokens > max_book_tokens:
        print(
            f"Book content exceeds limit, truncating to first {max_book_tokens:,} tokens"
        )
        book_content = truncate_by_tokens(full_content, max_book_tokens, MODEL)
        book_content += (
            f"\n\n[NOTE: Book content truncated to first {max_book_tokens} tokens.]"
        )
    else:
        print("Book content is within safe limit")
        book_content = full_content

    user_prompt = USER_PROMPT_TEMPLATE.format(book_content=book_content)
    final_prompt_tokens = count_tokens(user_prompt)
    print(f"User prompt tokens sent: {final_prompt_tokens:,}")

    print(
        f"Calling model {MODEL} (streaming), output cap {MAX_OUTPUT_TOKENS} tokens, timeout {REQUEST_TIMEOUT}s"
    )
    try:
        raw_response = call_llm_api(user_prompt)
    except Exception as e:
        print(f"API call failed after retries: {e}")
        save_raw_response(f"API call failed: {str(e)}", RAW_RESPONSE_FILE_PATH)
        return

    save_raw_response(raw_response, RAW_RESPONSE_FILE_PATH)
    qa_pairs = parse_qa_response(raw_response)

    if qa_pairs:
        save_qa_pairs(qa_pairs, qa_jsonl_path)
        print(f"Generated {len(qa_pairs)} QA pairs successfully")
    else:
        print("Failed to extract valid QA pairs, raw response was saved")
        error_entry = {
            "error": "parse_failed",
            "raw_response_preview": raw_response[:500],
        }
        with open(qa_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(error_entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        MD_FILE_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        RAW_RESPONSE_FILE_PATH = sys.argv[2]
    if len(sys.argv) > 3:
        QA_JSONL_FILE_PATH = sys.argv[3]

    print("=" * 50)
    print(f"API URL: {API_URL}")
    print(f"Model: {MODEL}")
    print(f"Max output tokens: {MAX_OUTPUT_TOKENS}")
    print(f"Request timeout: {REQUEST_TIMEOUT}s")
    print(f"MD file: {MD_FILE_PATH}")
    print(f"Raw response: {RAW_RESPONSE_FILE_PATH}")
    print(f"QA JSONL: {QA_JSONL_FILE_PATH}")
    print("=" * 50)

    if not os.path.exists(MD_FILE_PATH):
        print(f"File does not exist: {MD_FILE_PATH}")
        sys.exit(1)

    main()
