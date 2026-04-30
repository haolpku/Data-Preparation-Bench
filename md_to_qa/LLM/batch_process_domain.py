import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SOURCE_DIR = "/Path/to/your/md/directory"  # Source directory containing Markdown files
OUTPUT_BASE_DIR = "/Path/to/your/output/directory"  # Root output directory
SCRIPT_PATH = "script_domain.py"  # Path to processing script
LOG_FILE = os.path.join(OUTPUT_BASE_DIR, "processing_domain.log")  # Log file path

# Optional: continue processing remaining files when one file fails
CONTINUE_ON_ERROR = True


def setup_logging():
    """Create output directory and initialize log file."""
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    # Append if log file exists; create otherwise
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(
            f"Batch processing start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write(f"Source directory: {SOURCE_DIR}\n")
        f.write(f"Output directory: {OUTPUT_BASE_DIR}\n")
        f.write(f"{'='*60}\n\n")


def log_message(message, level="INFO"):
    """Write a log line and print it to stdout."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] [{level}] {message}"
    print(log_line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_line + "\n")


def process_md_file(md_path, output_subdir):
    """
    Process one Markdown file.
    md_path: full path to source Markdown file
    output_subdir: full path to per-file output subdirectory
    Returns (success, error_message)
    """
    # File base name without extension
    basename = os.path.splitext(os.path.basename(md_path))[0]

    # Create output subdirectory
    os.makedirs(output_subdir, exist_ok=True)

    # Output paths
    raw_response_path = os.path.join(output_subdir, "raw_response.txt")
    qa_jsonl_path = os.path.join(output_subdir, f"{basename}_domain.jsonl")

    # Build command
    cmd = [
        sys.executable,  # Use current Python interpreter
        SCRIPT_PATH,
        md_path,
        raw_response_path,
        qa_jsonl_path,
    ]

    log_message(f"Start processing: {md_path}")
    log_message(f"Output directory: {output_subdir}")

    start_time = time.time()
    try:
        # Execute child script and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1200,  # 10-minute timeout (adjust as needed)
        )
        elapsed = time.time() - start_time

        if result.returncode == 0:
            log_message(
                f"Processed successfully: {basename} (elapsed {elapsed:.2f} seconds)"
            )
            # Optional: store child stdout (for token stats, etc.)
            if result.stdout:
                log_message(f"Script output:\n{result.stdout}", level="DEBUG")
            return True, None

        error_msg = f"Return code {result.returncode}\nSTDERR: {result.stderr}\nSTDOUT: {result.stdout}"
        log_message(f"Processing failed: {basename} - {error_msg}", level="ERROR")
        return False, error_msg
    except subprocess.TimeoutExpired:
        log_message(
            f"Processing timeout: {basename} (exceeded 1200 seconds)", level="ERROR"
        )
        return False, "Timeout"
    except Exception as e:
        log_message(f"Processing exception: {basename} - {str(e)}", level="ERROR")
        return False, str(e)


def main():
    # Validate source directory
    if not os.path.isdir(SOURCE_DIR):
        print(f"Error: source directory does not exist: {SOURCE_DIR}")
        sys.exit(1)

    # Validate script file
    if not os.path.isfile(SCRIPT_PATH):
        print(f"Error: script file does not exist: {SCRIPT_PATH}")
        sys.exit(1)

    # Initialize logging
    setup_logging()

    # Collect all Markdown files
    md_files = list(Path(SOURCE_DIR).glob("*.md"))
    if not md_files:
        log_message(f"No .md files found in {SOURCE_DIR}", level="WARNING")
        return

    log_message(f"Found {len(md_files)} Markdown files")

    # Counters
    success_count = 0
    fail_count = 0

    # Process files one by one
    for i, md_path in enumerate(md_files, 1):
        log_message(f"\n--- Progress: {i}/{len(md_files)} ---")
        basename = md_path.stem  # File name without extension
        output_subdir = os.path.join(OUTPUT_BASE_DIR, basename)

        success, error = process_md_file(str(md_path), output_subdir)
        if success:
            success_count += 1
        else:
            fail_count += 1
            if not CONTINUE_ON_ERROR:
                log_message(
                    "CONTINUE_ON_ERROR is False, stopping further processing",
                    level="ERROR",
                )
                break

    log_message("\n" + "=" * 60)
    log_message("Batch processing completed")
    log_message(
        f"Success: {success_count}, Failed: {fail_count}, Total: {len(md_files)}"
    )
    log_message(f"Log saved to: {LOG_FILE}")
    log_message("=" * 60)


if __name__ == "__main__":
    main()
