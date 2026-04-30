# Quality rubric

Use this rubric before keeping a sample.

## Keep a sample only if all are true

1. The item teaches reusable domain knowledge or a reusable decision pattern.
2. The question stands alone without referring to the source text.
3. The answer is directly supported by the source chunk.
4. The wording is instructional rather than document-relative.
5. The item is materially distinct from nearby samples.
6. Any reasoning or analysis is grounded, concise, and not fake.

## Additional rubric for process_qa

Keep only if:

- the reasoning path is actually needed
- each step points to a rule, condition, comparison, exception, cause, or sequence
- the steps are short and factual
- the answer follows from the steps without hidden leaps

Reject if:

- the steps merely restate the answer
- the steps contain filler such as "first understand the question"
- the steps rely on facts not supported by the chunk

## Additional rubric for case_application

Keep only if:

- the case is solvable using source-grounded knowledge
- the case does not require outside domain facts
- the case is not just a wording variant of concept_qa
- the analysis clearly maps case facts to source rules or mechanisms

Reject if:

- the case is overly narrative
- the case invents domain context not in the source
- the analysis is generic and would fit almost any case

## Red flags

Reject or rewrite when you see:

- according to the text
- according to the section
- based on the source chunk
- this passage explains
- this question asks
- the answer is
- chapter or section references
- headings turned directly into questions
- identical answers repeated across many questions without good reason

# Reasoning patterns

Use these patterns to keep process supervision concise and grounded.

## Allowed reasoning_pattern values

- `rule_application`
- `condition_check`
- `exception_handling`
- `cause_effect`
- `comparison_resolution`
- `step_ordering`
- `category_assignment`
- `mechanism_trace`
- `constraint_evaluation`

## Pattern guidance

### rule_application

Use when the source states a rule and the question asks whether or how it applies.

Step shape:

1. identify the relevant stated condition or rule
2. compare the question facts against it
3. conclude the result

### condition_check

Use when multiple conditions must be verified.

Step shape:

1. name the necessary condition
2. state whether it is met
3. repeat for the material exception if one exists
4. conclude

### exception_handling

Use when a default rule changes under a stated exception.

### cause_effect

Use when the source explicitly links one factor to an outcome.

### comparison_resolution

Use when the source contrasts two categories, mechanisms, or outcomes and the question depends on that distinction.

### step_ordering

Use when sequence matters and the source clearly describes the order.

### category_assignment

Use when the question asks where an item belongs based on source criteria.

### mechanism_trace

Use when the source describes how one state leads to another through a mechanism.

### constraint_evaluation

Use when the source describes limits, boundaries, or conditions that block an action or outcome.

## Do not do this

Bad reasoning:

- "The question asks about..."
- "According to the passage..."
- "First read the source..."
- "Therefore the correct answer is..." as a standalone step

Good reasoning is brief, domain-specific, and grounded in source relations.

# Default supervision schema

Use one JSON object per line.

## Required common fields

Every record must include:

- `sample_type`
- `source_file`
- `chunk_id`

Allowed `sample_type` values:

- `concept_qa`
- `process_qa`
- `case_application`

## concept_qa

Use for atomic reusable knowledge.

```json
{
  "sample_type": "concept_qa",
  "question": "What is a buffer overflow?",
  "answer": "A buffer overflow is a condition in which a program writes more data into a memory buffer than the buffer can hold, causing adjacent memory to be overwritten.",
  "source_file": "security/memory.md",
  "chunk_id": "memory_0012_01",
  "question_type": "definition",
  "metadata": {
    "knowledge_point": "buffer overflow"
  }
}
```

## process_qa

Use for grounded reasoning patterns.

```json
{
  "sample_type": "process_qa",
  "question": "Why should input length be checked before writing into a fixed-size buffer?",
  "reasoning": [
    "A fixed-size buffer can hold only a limited amount of data.",
    "If the input is longer than the buffer, the write operation exceeds the buffer boundary.",
    "Exceeding the boundary can overwrite adjacent memory and create a buffer overflow."
  ],
  "answer": "Input length should be checked first to prevent writes that exceed the buffer capacity and corrupt adjacent memory.",
  "source_file": "security/memory.md",
  "chunk_id": "memory_0012_01",
  "question_type": "rule",
  "metadata": {
    "knowledge_points": ["buffer overflow", "bounds checking"],
    "reasoning_pattern": "cause_effect"
  }
}
```

## case_application

Use for source-grounded scenario transfer.

```json
{
  "sample_type": "case_application",
  "case": "A program stores usernames in a 16-byte buffer but accepts arbitrarily long strings without checking their length.",
  "question": "What risk does this design create, and why?",
  "analysis": [
    "The buffer has a fixed capacity of 16 bytes.",
    "The program accepts inputs that may be longer than that capacity.",
    "Writing an overly long username can exceed the buffer boundary and overwrite adjacent memory."
  ],
  "answer": "This design creates a buffer overflow risk because long inputs can exceed the buffer capacity and overwrite adjacent memory.",
  "source_file": "security/memory.md",
  "chunk_id": "memory_0012_01",
  "question_type": "condition",
  "metadata": {
    "knowledge_points": ["buffer overflow"],
    "task_form": "case_analysis"
  }
}
```

## Status schema

Use one final status line per processed chunk.

### kept

```json
{
  "chunk_id": "memory_0012_01",
  "source_file": "security/memory.md",
  "status": "kept",
  "skip_reason": "",
  "concept_count": 2,
  "process_count": 1,
  "case_count": 1,
  "total_sample_count": 4
}
```

### skipped

```json
{
  "chunk_id": "intro_0001_01",
  "source_file": "security/frontmatter.md",
  "status": "skipped",
  "skip_reason": "non_knowledge",
  "concept_count": 0,
  "process_count": 0,
  "case_count": 0,
  "total_sample_count": 0
}
```
