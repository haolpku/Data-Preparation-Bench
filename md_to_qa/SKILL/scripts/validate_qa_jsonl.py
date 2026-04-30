#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from pathlib import Path

COMMON_REQ = ["sample_type", "source_file", "chunk_id"]
TYPE_REQ = {
    "concept_qa": ["question", "answer"],
    "process_qa": ["question", "reasoning", "answer"],
    "case_application": ["case", "question", "analysis", "answer"],
}
VALID_SAMPLE_TYPES = set(TYPE_REQ)
VALID_QUESTION_TYPES = {
    "definition",
    "function",
    "mechanism",
    "process",
    "comparison",
    "cause",
    "purpose",
    "rule",
    "constraint",
    "category",
    "enumeration",
    "condition",
    "exception",
    "consequence",
}
WS_RE = re.compile(r"\s+")

PLACEHOLDER_QUESTION_PATTERNS = [
    re.compile(r"这段内容的要点是什么"),
    re.compile(r"问题\s*[0-9]+"),
    re.compile(r"what are the key points of this"),
    re.compile(r"what is the main point of this chunk"),
    re.compile(r"what is this section about"),
]

SOURCE_ANCHORED_PATTERNS = [
    re.compile(r"^according to\b"),
    re.compile(r"\bbased on the (above|source|provided)\b"),
    re.compile(r"\bin this (chapter|section|book|passage|text)\b"),
    re.compile(r"\bfrom this (chapter|section|book|passage|text)\b"),
    re.compile(r"根据《"),
    re.compile(r"根据本[书文节章]"),
    re.compile(r"根据这段内容"),
]

META_PATTERNS = [
    re.compile(r"answer should"),
    re.compile(r"based on the source chunk"),
    re.compile(r"should summarize"),
    re.compile(r"^this section"),
    re.compile(r"^the passage"),
    re.compile(r"this question asks"),
    re.compile(r"first (identify|read|understand) the question"),
]

CITATION_LED_PATTERNS = [
    re.compile(r"^how does section\s+[0-9a-z()\.-]+"),
    re.compile(r"^what does section\s+[0-9a-z()\.-]+"),
    re.compile(r"\bsection\s+[0-9a-z()\.-]+\b"),
    re.compile(r"\bchapter\s+[0-9ivxlcdm]+\b"),
    re.compile(r"\bfigure\s+[0-9a-z()\.-]+\b"),
    re.compile(r"\barticle\s+[0-9a-z()\.-]+\b"),
]

GENERIC_REASONING_PATTERNS = [
    re.compile(r"first (identify|read|understand) the question"),
    re.compile(r"the question asks"),
    re.compile(r"according to the (passage|text|section|book)"),
    re.compile(r"based on the (passage|text|section|book|source)"),
    re.compile(r"therefore the correct answer is"),
]


def norm(s: str) -> str:
    return WS_RE.sub(" ", (s or "").strip().lower())


def matches_any(text: str, patterns) -> bool:
    text_n = norm(text)
    return any(p.search(text_n) for p in patterns)


def is_non_empty_list_of_strings(value):
    return isinstance(value, list) and value and all(str(x).strip() for x in value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_jsonl")
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    inp = Path(args.input_jsonl).resolve()
    report = Path(args.report).resolve()
    report.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    malformed = 0
    missing_required = 0
    empty_fields = 0
    invalid_sample_type = 0
    invalid_question_type = 0
    exact_items = Counter()
    norm_questions = Counter()
    norm_answers = Counter()
    issues = []
    placeholder_questions = 0
    source_anchored_texts = 0
    citation_led_questions = 0
    meta_answers_or_reasoning = 0
    invalid_reasoning_fields = 0
    generic_reasoning = 0
    sample_type_counts = Counter()

    with inp.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                malformed += 1
                issues.append({"line": line_no, "issue": "malformed_json"})
                continue

            missing = [k for k in COMMON_REQ if k not in obj]
            if missing:
                missing_required += 1
                issues.append(
                    {"line": line_no, "issue": "missing_required", "fields": missing}
                )
                continue

            sample_type = obj.get("sample_type")
            if sample_type not in VALID_SAMPLE_TYPES:
                invalid_sample_type += 1
                issues.append(
                    {
                        "line": line_no,
                        "issue": "invalid_sample_type",
                        "sample_type": sample_type,
                    }
                )
                continue
            sample_type_counts[sample_type] += 1

            type_missing = [k for k in TYPE_REQ[sample_type] if k not in obj]
            if type_missing:
                missing_required += 1
                issues.append(
                    {
                        "line": line_no,
                        "issue": "missing_type_required",
                        "fields": type_missing,
                    }
                )
                continue

            empties = [
                k
                for k in COMMON_REQ + TYPE_REQ[sample_type]
                if not str(obj.get(k, "")).strip()
            ]
            if empties:
                empty_fields += 1
                issues.append(
                    {"line": line_no, "issue": "empty_required", "fields": empties}
                )

            qtype = obj.get("question_type")
            if qtype and qtype not in VALID_QUESTION_TYPES:
                invalid_question_type += 1
                issues.append(
                    {
                        "line": line_no,
                        "issue": "invalid_question_type",
                        "question_type": qtype,
                    }
                )

            question = obj.get("question", "")
            answer = obj.get("answer", "")
            key_parts = [
                sample_type,
                norm(question),
                norm(answer),
                norm(obj.get("case", "")),
            ]
            exact_items[tuple(key_parts)] += 1
            if question:
                norm_questions[norm(question)] += 1
            if answer:
                norm_answers[norm(answer)] += 1

            if question and matches_any(question, PLACEHOLDER_QUESTION_PATTERNS):
                placeholder_questions += 1
                issues.append(
                    {
                        "line": line_no,
                        "issue": "placeholder_like_question",
                        "question": question[:200],
                    }
                )
            if question and matches_any(question, SOURCE_ANCHORED_PATTERNS):
                source_anchored_texts += 1
                issues.append(
                    {
                        "line": line_no,
                        "issue": "source_anchored_question",
                        "question": question[:200],
                    }
                )
            if question and matches_any(question, CITATION_LED_PATTERNS):
                citation_led_questions += 1
                issues.append(
                    {
                        "line": line_no,
                        "issue": "citation_led_question",
                        "question": question[:200],
                    }
                )
            if answer and (
                matches_any(answer, META_PATTERNS)
                or matches_any(answer, SOURCE_ANCHORED_PATTERNS)
            ):
                meta_answers_or_reasoning += 1
                issues.append(
                    {
                        "line": line_no,
                        "issue": "meta_like_answer",
                        "answer": answer[:200],
                    }
                )

            if sample_type == "process_qa":
                reasoning = obj.get("reasoning")
                if not is_non_empty_list_of_strings(reasoning):
                    invalid_reasoning_fields += 1
                    issues.append({"line": line_no, "issue": "invalid_reasoning_field"})
                else:
                    for step in reasoning:
                        if matches_any(step, META_PATTERNS) or matches_any(
                            step, SOURCE_ANCHORED_PATTERNS
                        ):
                            meta_answers_or_reasoning += 1
                            issues.append(
                                {
                                    "line": line_no,
                                    "issue": "meta_like_reasoning",
                                    "reasoning": step[:200],
                                }
                            )
                        if matches_any(step, GENERIC_REASONING_PATTERNS):
                            generic_reasoning += 1
                            issues.append(
                                {
                                    "line": line_no,
                                    "issue": "generic_reasoning_step",
                                    "reasoning": step[:200],
                                }
                            )
            elif sample_type == "case_application":
                analysis = obj.get("analysis")
                case_text = obj.get("case", "")
                if not is_non_empty_list_of_strings(analysis):
                    invalid_reasoning_fields += 1
                    issues.append({"line": line_no, "issue": "invalid_analysis_field"})
                else:
                    for step in analysis:
                        if matches_any(step, META_PATTERNS) or matches_any(
                            step, SOURCE_ANCHORED_PATTERNS
                        ):
                            meta_answers_or_reasoning += 1
                            issues.append(
                                {
                                    "line": line_no,
                                    "issue": "meta_like_analysis",
                                    "analysis": step[:200],
                                }
                            )
                        if matches_any(step, GENERIC_REASONING_PATTERNS):
                            generic_reasoning += 1
                            issues.append(
                                {
                                    "line": line_no,
                                    "issue": "generic_analysis_step",
                                    "analysis": step[:200],
                                }
                            )
                if case_text and matches_any(case_text, SOURCE_ANCHORED_PATTERNS):
                    source_anchored_texts += 1
                    issues.append(
                        {
                            "line": line_no,
                            "issue": "source_anchored_case",
                            "case": case_text[:200],
                        }
                    )

    dup_items = sum(1 for _, c in exact_items.items() if c > 1)
    dup_questions = sum(1 for _, c in norm_questions.items() if c > 1)
    repeated_answers = sum(1 for _, c in norm_answers.items() if c > 1)

    result = {
        "input": str(inp),
        "total_lines": total,
        "sample_type_counts": dict(sample_type_counts),
        "malformed_json": malformed,
        "missing_required": missing_required,
        "empty_required": empty_fields,
        "invalid_sample_type": invalid_sample_type,
        "invalid_question_type": invalid_question_type,
        "invalid_reasoning_or_analysis_fields": invalid_reasoning_fields,
        "duplicate_exact_items": dup_items,
        "duplicate_normalized_questions": dup_questions,
        "duplicate_normalized_answers": repeated_answers,
        "placeholder_like_questions": placeholder_questions,
        "source_anchored_texts": source_anchored_texts,
        "citation_led_questions": citation_led_questions,
        "meta_like_answers_or_reasoning": meta_answers_or_reasoning,
        "generic_reasoning_steps": generic_reasoning,
        "issues_preview": issues[:100],
    }
    report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
