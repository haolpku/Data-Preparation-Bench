---
name: Text2QA
description: build concept, process, and case-application supervision datasets from markdown books or long markdown documents. use when generating training data from many .md files or precomputed chunk files and when full chunk coverage, resumable batch processing, status tracking, validation, and coverage auditing are required. use for book-to-sft pipelines where every chunk must end in exactly one final status and where answer-only qa is not sufficient because the dataset should also teach grounded reasoning patterns and rule application.
---

# Text2QA Builder

Build supervision datasets from markdown books or long markdown documents.

Books are knowledge sources only. The final dataset must teach reusable domain knowledge and how to apply it. Do not generate book-comprehension questions, citation-led questions, or document-structure questions.

Default behavior is full coverage, not sampling.

## Dataset objective

Compile book knowledge into three complementary supervision forms:

1. `concept_qa`: teach atomic reusable knowledge such as definitions, categories, rules, mechanisms, purposes, and constraints.
2. `process_qa`: teach concise, grounded reasoning patterns such as condition checking, rule application, causal explanation, comparison, exception handling, and step ordering.
3. `case_application`: teach knowledge transfer into realistic but source-grounded scenarios where the model must analyze a situation and apply the book's knowledge.

Use all three forms when supported. Do not force all three forms for every chunk.

## Inputs

Use either of these inputs:

1. A directory of markdown files.
2. One or more precomputed `*.chunks.jsonl` files.

If chunk files already exist, reuse them instead of re-splitting the source books.

## Required completion rule

A task is complete only when every chunk has exactly one of the following outcomes:

- at least one supervision record written to a batch JSONL file and a matching `kept` record written to `chunk_status.jsonl`, or
- a recorded `skipped` decision in `chunk_status.jsonl` with a non-empty `skip_reason`

Do not stop after producing a small sample unless the user explicitly asks for a sample.

Do not report the task as completed, finished, done, or ready until all of the following are true:

1. `check_coverage.py` reports `unprocessed_chunks = 0`
2. `sample_without_status_preview` is empty
3. `sample_status_mismatch_preview` is empty

Partial progress may be reported only as progress, never as completion.

## Output files

Use a resumable work layout like this:

```text
work/
  manifest.jsonl
  chunks/
    book_a.chunks.jsonl
  supervision_batches/
    batch_001.jsonl
    batch_002.jsonl
  chunk_status.jsonl
  supervision_merged.jsonl
  validation.json
  coverage.json
```

`chunk_status.jsonl` is required for full runs.

## Workflow

### 1. Prepare the corpus

If the user provides markdown files, run:

```bash
scripts/build_manifest.py <input_dir> --output work/manifest.jsonl
scripts/split_markdown_book.py <input_md> --output work/chunks/<name>.chunks.jsonl --source-root <input_dir>
```

If chunk files already exist, skip this step.

### 2. Process chunks in batches

Process chunk files sequentially in small batches.

Recommended batch size: 20 to 50 chunks.

Use:

```bash
scripts/next_unprocessed_chunks.py work/chunks/*.chunks.jsonl --status work/chunk_status.jsonl --limit 25 --output work/next_batch.jsonl
```

For each chunk in the batch, do the following in order:

1. Decide whether the chunk contains reusable knowledge.
2. If not, write a `skipped` record to `chunk_status.jsonl`.
3. If yes, identify all distinct reusable knowledge propositions in the chunk.
4. Identify proposition relations such as prerequisite, condition-result, cause-effect, contrast, sequence, category-membership, and exception-override.
5. Decide which sample types the chunk can support: `concept_qa`, `process_qa`, `case_application`.
6. Generate all strong, non-duplicative supervision records supported by the chunk.
7. Write exactly one `kept` status record for that chunk with per-type counts.

After finishing one batch, continue with the next unprocessed batch until no chunks remain.

### 3. Status and sample synchronization rule

After every processed batch:

1. write or append supervision records for the batch
2. write or append status records for the exact same processed chunk ids
3. only then merge supervision files
4. only then run validation and coverage auditing

Never leave supervision records without status records.

Never mark a chunk as `kept` unless at least one supervision record was actually written for that chunk.

Never leave a processed chunk without a status record.

## Triage: decide whether a chunk is knowledge-bearing

### Keep the chunk if it contains reusable knowledge such as:

- definitions
- functions of entities
- steps in a process
- mechanisms
- comparisons
- causes
- purposes
- rules and constraints
- categories
- enumerations
- conditions, exceptions, or consequences
- operational distinctions that teach reusable knowledge

### Skip the chunk if it is mainly:

- acknowledgements
- table of contents
- author lists
- navigation structure
- headings without body
- broken OCR fragments
- page furniture
- index-like lists with no teachable proposition
- pedagogy-only text such as exercises, study prompts, or self-check instructions without reusable knowledge
- pure transition text with no substantive proposition
- generic filler that does not teach a reusable fact, rule, distinction, mechanism, or application pattern

If a chunk contains no teachable knowledge, generate 0 samples and record a skip reason.

## Allowed skip reasons

Use only these values for `skip_reason`:

- navigation
- non_knowledge
- pedagogy
- heading_only
- low_information
- noisy
- broken_ocr
- duplicate_scope
- index_like

Do not invent new skip labels.

## Extract knowledge propositions

Before drafting samples, identify the knowledge taught by the chunk.

A knowledge proposition is a distinct reusable statement the model should learn.

Typical proposition types:

- definition
- function
- mechanism
- process_step
- comparison
- cause
- purpose
- rule
- constraint
- category
- enumeration
- condition
- exception
- consequence

If the chunk does not support clear propositions, skip it.

## Extract proposition relations

For each chunk kept for supervision, identify any proposition relations that are explicitly supported or can be derived in one grounded step from the chunk:

- prerequisite
- condition_result
- cause_effect
- exception_override
- contrast
- sequence
- category_member
- part_whole
- decision_rule

These relations determine whether the chunk can support process or case supervision. Do not fabricate relations not supported by the source.

## Canonicalize knowledge

Transform source statements into concept-level knowledge.

Remove:

- book-relative wording
- section references
- citation framing
- passage language
- chapter-led prompts
- instructional scaffolding such as “in this lesson” or “the following section explains”
- assessment framing such as “students should understand”

The samples must ask about the concept or application itself, not the document.

## Route each proposition into the right sample type

### Emit `concept_qa` when the chunk supports atomic knowledge such as:

- definitions
- categories
- purposes
- functions
- rules stated directly
- independent consequences
- concise mechanism descriptions

### Emit `process_qa` when the chunk supports concise grounded reasoning such as:

- applying a rule to stated conditions
- checking a decision path
- tracing a cause-effect link
- resolving a comparison
- ordering process steps
- handling exceptions or constraints
- explaining why one outcome follows and another does not

### Emit `case_application` when the chunk supports scenario reframing such as:

- a realistic situation can be described using only source-grounded concepts
- the answer requires applying one or more source rules or mechanisms
- the case can be solved without introducing external domain facts

Do not force process or case samples from chunks that only support atomic knowledge.

## Exhaustive proposition coverage rule

Do not impose a fixed upper limit on sample count per chunk.

The goal is to exhaust the chunk’s reusable knowledge propositions and supported reasoning patterns.

If a chunk teaches five distinct reusable propositions, generate supervision for all five.

If a chunk teaches ten distinct reusable propositions, generate supervision for all ten.

Do not stop early just because the chunk already has “enough” items.

However, exhaustiveness means exhausting distinct knowledge and reasoning patterns, not generating paraphrase variants.

Generate all distinct, supportable, reusable propositions and applications in the chunk, but do not ask multiple questions that test the same proposition with only wording changes.

Prefer proposition coverage over superficial sample count.

### What exhaustiveness means

Exhaustiveness includes:

- each distinct definition
- each independent function of an entity
- each rule or constraint
- each exception or condition that materially changes the concept
- each non-overlapping item in a meaningful category or enumeration when the items are teachable
- each comparison where the contrasted sides matter
- each process step only when the step is conceptually meaningful and reusable
- each grounded reasoning path where relation structure materially changes how the knowledge is applied

Exhaustiveness does not include:

- repeating the same fact in multiple phrasings
- turning every sentence into a separate item when several sentences express one proposition
- generating trivial heading-restatement questions
- fragmenting one clean proposition into many low-value samples
- wrapping a simple definition in fake multi-step reasoning

## Cross-chunk support rule

Default to generating supervision from the current chunk alone.

If adjacent chunks belong to the same concept and one chunk alone is insufficient for a clean conceptual or process sample, generate a sample anchored to the primary chunk and optionally record supporting chunk ids in metadata.

Do not merge distant chunks or broad chapter themes into one item.

## Prefer zero samples over weak samples

Skip the chunk instead of generating supervision when:

- the content is mainly structural or pedagogical
- the content is too generic to teach reusable knowledge
- the only possible questions would merely restate the heading
- all candidate items would be low-distinction paraphrases of the same proposition
- the chunk contains text but no clear, supportable conceptual takeaway
- a case would require too much invented context beyond the chunk
- the only possible reasoning is fake reasoning that merely restates the answer

Exhaustive coverage does not justify weak samples.

## How to write grounded reasoning

Reasoning in this dataset is external supervision, not hidden chain-of-thought.

Use short, explicit, domain-grounded reasoning steps that teach a reusable decision pattern. Keep them concise and factual.

Good reasoning characteristics:

- each step is justified by source knowledge
- steps identify the relevant condition, rule, comparison, exception, or causal link
- steps are brief and structured
- the final answer follows naturally from the steps

Bad reasoning characteristics:

- filler such as “first read the question” or “according to the passage”
- meta commentary such as “this question asks about”
- answer restatement disguised as steps
- invented facts not supported by the chunk
- long free-form essays

## Sample style

### Good `concept_qa`

Use when teaching reusable knowledge directly.

Question should stand alone.

Answer should:

- answer directly in the first clause
- be self-contained
- teach reusable knowledge
- use clean instructional language
- paraphrase the source unless exact wording is essential

### Good `process_qa`

Use when teaching how to reason with the knowledge.

Question should require applying a source-supported rule, condition, comparison, sequence, or exception.

Reasoning should:

- be 2 to 6 short steps
- name the relevant condition, rule, relation, or exception
- show the minimal grounded path from premises to conclusion

Answer should be brief and directly resolve the question.

### Good `case_application`

Use when the chunk supports scenario transfer without hallucination.

Case should:

- be realistic but generic
- use only source-grounded entities, conditions, rules, and mechanisms
- avoid unnecessary narrative decoration

Analysis should:

- identify which source knowledge applies
- compare the case facts against the relevant rule, mechanism, or exception
- reach the answer in a compact grounded path

Answer should resolve the case directly.

## Hard rules

### Do not generate source-anchored questions

Avoid phrases such as:

- according to the excerpt
- according to the passage
- according to the text
- according to the book
- according to the framework
- according to the model
- based on the above content
- based on the source chunk
- 根据本节
- 根据本文
- 根据这段内容

Questions must stand alone.

### Do not generate citation-led questions

Avoid questions framed around:

- section numbers
- chapter numbers
- book titles
- figure numbers
- statute citations

Ask about the concept instead.

### Do not generate meta answers or meta reasoning

Avoid answers or reasoning such as:

- the answer should summarize
- based on the source chunk
- this section mainly discusses
- the passage explains that
- first identify what the question is asking
- this problem tests whether

Answers and reasoning must provide knowledge, not instructions about answering.

### Do not create supervision from non-knowledge content

Never generate samples from:

- acknowledgements
- table of contents
- title-only sections
- advisor lists
- navigation headings
- index-only lists
- pedagogy-only scaffolding
- review questions copied from the source without conceptual rewriting

### Do not fabricate hidden reasoning

Never invent:

- external facts not supported by the chunk
- latent domain assumptions not present in the source
- extra diagnostic steps added only to sound smart
- cases that require outside knowledge to solve

If the chunk does not support a clean grounded reasoning path, emit `concept_qa` only.

## Question type schema

Use only these `question_type` values:

- definition
- function
- mechanism
- process
- comparison
- cause
- purpose
- rule
- constraint
- category
- enumeration
- condition
- exception
- consequence

Use singular labels exactly as written above.

## Sample schema

Write one JSON object per line.

Required fields for all sample types:

```json
{
  "sample_type": "concept_qa",
  "source_file": "...",
  "chunk_id": "..."
}
```

### `concept_qa`

```json
{
  "sample_type": "concept_qa",
  "question": "...",
  "answer": "...",
  "source_file": "...",
  "chunk_id": "...",
  "question_type": "definition",
  "metadata": {
    "knowledge_point": "...",
    "supporting_chunk_ids": []
  }
}
```

### `process_qa`

```json
{
  "sample_type": "process_qa",
  "question": "...",
  "reasoning": [
    "...",
    "..."
  ],
  "answer": "...",
  "source_file": "...",
  "chunk_id": "...",
  "question_type": "rule",
  "metadata": {
    "knowledge_points": ["..."],
    "reasoning_pattern": "rule_application",
    "supporting_chunk_ids": []
  }
}
```

### `case_application`

```json
{
  "sample_type": "case_application",
  "case": "...",
  "question": "...",
  "analysis": [
    "...",
    "..."
  ],
  "answer": "...",
  "source_file": "...",
  "chunk_id": "...",
  "question_type": "condition",
  "metadata": {
    "knowledge_points": ["..."],
    "task_form": "case_analysis",
    "supporting_chunk_ids": []
  }
}
```

Use `metadata.knowledge_point` or `metadata.knowledge_points` when it helps identify the canonical concept being taught.

Use `metadata.supporting_chunk_ids` only when adjacent chunks are genuinely needed.

## Chunk status schema

Write one JSON object per line to `chunk_status.jsonl`.

For kept chunks:

```json
{
  "chunk_id": "...",
  "source_file": "...",
  "status": "kept",
  "skip_reason": "",
  "concept_count": 2,
  "process_count": 1,
  "case_count": 1,
  "total_sample_count": 4
}
```

For skipped chunks:

```json
{
  "chunk_id": "...",
  "source_file": "...",
  "status": "skipped",
  "skip_reason": "navigation",
  "concept_count": 0,
  "process_count": 0,
  "case_count": 0,
  "total_sample_count": 0
}
```

There must be exactly one final status record per processed chunk.

## Validation and coverage audit

After each completed batch or after merging batches, run:

```bash
scripts/validate_qa_jsonl.py work/supervision_merged.jsonl --report work/validation.json
scripts/check_coverage.py work/chunks/*.chunks.jsonl --status work/chunk_status.jsonl --qa work/supervision_merged.jsonl --report work/coverage.json
```

If coverage reports any of the following, the run is not complete:

- `unprocessed_chunks > 0`
- non-empty `sample_without_status_preview`
- non-empty `sample_status_mismatch_preview`

If validation passes but coverage is incomplete, continue processing remaining chunks.

## Operating principle

The final dataset should read like a general domain-supervision corpus, not a book comprehension exercise.

The objective is exhaustive coverage of reusable knowledge propositions across the corpus, plus grounded reasoning patterns and case application whenever the source supports them, with zero tolerance for structural leakage, fake reasoning, status inconsistency, or paraphrase-only duplication.
