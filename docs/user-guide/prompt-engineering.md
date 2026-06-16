# Prompt Engineering

Prompt Engineering is the workflow in Loca for creating, editing, organizing, applying, and reusing
system prompts. It provides a structured interface for managing prompt templates that can be applied
to Text Generation, Document Q&A, extraction, summarization, coding, translation, and other local
model workflows.

## 🧭 Purpose

Prompt Engineering helps users standardize how the local model behaves across repeated tasks.
Instead of rewriting instructions manually, users can store reusable prompt templates, select them
by name, modify them, clone them, and apply them to the active generation workflow.

Prompt Engineering supports:

* reusable system instructions;
* task-specific prompt templates;
* prompt metadata management;
* XML-to-Markdown conversion;
* starter prompt generation;
* prompt reuse across Text Generation and Document Q&A;
* SQLite-backed prompt persistence;
* consistent local-model behavior across repeated workflows.

## 🧱 Workflow Position

Prompt Engineering sits between user intent and model execution.

```text id="qheavf"
Prompt Goal
  │
  ▼
Prompt Engineering UI
  │
  ├── Category
  ├── Task Type
  ├── Response Format
  ├── Preferred Language
  ├── Constraints
  └── Style
  │
  ▼
Prompt Template
  │
  ├── Save
  ├── Edit
  ├── Clone
  ├── Delete
  └── Apply
  │
  ▼
System Instructions
  │
  ▼
Text Generation or Document Q&A
```

The workflow is implemented primarily in `app.py` and stores prompt records in the SQLite `Prompts`
table.

## 🗃️ Prompt Storage Model

Prompt templates are persisted in SQLite so they can be reused across application sessions.

The prompt table supports records with fields such as:

| Field       | Purpose                                           |
| ----------- | ------------------------------------------------- |
| `PromptsId` | Internal SQLite primary key.                      |
| `Caption`   | User-facing prompt title shown in selectors.      |
| `Name`      | Internal or descriptive prompt name.              |
| `Text`      | Full prompt template text.                        |
| `Version`   | Optional version label.                           |
| `ID`        | Optional external or project-specific identifier. |

Prompt records are loaded into the Prompt Engineering interface and can be applied to shared
application state.

## 🧾 Prompt Metadata

Prompt Engineering uses metadata to structure prompt creation and reuse.

| Metadata Field  | Purpose                                                                                    |
| --------------- | ------------------------------------------------------------------------------------------ |
| Category        | Groups prompts by workflow or subject.                                                     |
| Task Type       | Aligns the prompt with Chat, Reasoning, Coding, Translation, Summarization, or Extraction. |
| Response Format | Guides whether output should be Markdown, JSON, plain text, or another format.             |
| Language        | Sets the preferred response or translation language.                                       |
| Goal            | Describes what the prompt should accomplish.                                               |
| Constraints     | Defines rules the model should follow.                                                     |
| Style           | Defines tone, level of detail, and formatting approach.                                    |

This metadata helps users create prompts that are specific enough to be useful but general enough to
reuse.

## 🧩 Supported Prompt Categories

Loca supports practical prompt categories for common local AI workflows.

| Category            | Best Use                                                             |
| ------------------- | -------------------------------------------------------------------- |
| General Chat        | General-purpose assistant behavior.                                  |
| Reasoning           | Structured analysis and careful conclusions.                         |
| Coding              | Editor-ready source code, debugging, refactoring, and documentation. |
| Translation         | Faithful translation into a selected target language.                |
| Summarization       | Concise summaries that preserve key facts.                           |
| Extraction          | Structured extraction from supplied content.                         |
| Document Extraction | Fact extraction from uploaded or retrieved document excerpts.        |
| OCR                 | Text extraction and cleanup workflows.                               |
| JSON Output         | Strict structured output generation.                                 |

Categories improve browsing and make it easier to choose the correct template for the current task.

## ⚙️ Task Types

Task types affect the instruction block used by Text Generation.

| Task Type     | Purpose                                             |
| ------------- | --------------------------------------------------- |
| Chat          | General conversational output.                      |
| Reasoning     | Analytical output with careful structure.           |
| Coding        | Code-focused output with implementation discipline. |
| Translation   | Language conversion while preserving meaning.       |
| Summarization | Condensed output that preserves key facts.          |
| Extraction    | Field-level extraction from supplied content.       |

A prompt template should clearly match its task type. For example, a Coding prompt should include
source-preservation rules, while an Extraction prompt should include missing-value behavior.

## 🧠 System Instructions

System instructions are the high-level behavioral rules supplied to the local model. Prompt
Engineering manages these instructions so users can apply them consistently.

Good system instructions should be:

* explicit;
* reusable;
* task-aligned;
* concise enough to fit within the context window;
* strict where output format matters;
* clear about missing information;
* careful about unsupported claims.

Example:

```text id="mqfj6t"
You are a Python documentation assistant. Convert comments into griffe-compatible Google-style docstrings. Preserve public signatures, executable behavior, imports, constants, and session-state keys. Do not add Returns sections for procedures that do not return meaningful values.
```

## 🧪 Example: Coding Prompt

Use this template for source-preserving code edits.

```text id="m6x4n8"
You are a senior Python source-preservation assistant.

Preserve all existing behavior, public function names, signatures, imports, constants, session-state keys, and control flow unless a change is explicitly requested.

When editing code:
- use type hints on new functions;
- preserve existing layout where practical;
- add Google-style docstrings to public functions;
- avoid changing runtime behavior;
- do not omit existing blocks;
- return editor-ready code.
```

Best used with:

| Setting         | Value                  |
| --------------- | ---------------------- |
| Category        | Coding                 |
| Task Type       | Coding                 |
| Response Format | Markdown or code block |
| Temperature     | `0.0` to `0.2`         |

## 🧪 Example: MkDocs Docstring Prompt

Use this template when converting Python source comments for MkDocs.

```text id="ouq4e8"
You are a Python documentation specialist.

Convert public module, class, function, method, and property comments into Google-style docstrings compatible with MkDocs, mkdocstrings, and griffe.

Rules:
- use Purpose, Args, Attributes, Returns, Raises, Notes, and Examples only when applicable;
- do not use underline-style section headings;
- do not document self or cls;
- do not add Returns sections to __init__ methods;
- do not add Returns sections for procedures that return None;
- preserve public signatures and executable behavior;
- treat griffe warnings as defects.
```

Best used with:

| Setting         | Value    |
| --------------- | -------- |
| Category        | Coding   |
| Task Type       | Coding   |
| Response Format | Markdown |
| Temperature     | `0.0`    |

## 🧪 Example: Extraction Prompt

Use this template for structured extraction from source material.

```text id="pf5t18"
Extract only facts explicitly present in the supplied content.

Rules:
- do not infer missing values;
- return "Not provided" for missing fields;
- preserve exact names, dates, paths, labels, and configuration values;
- do not add unsupported explanations;
- return the result as a Markdown table unless JSON is requested.
```

Best used with:

| Setting         | Value            |
| --------------- | ---------------- |
| Category        | Extraction       |
| Task Type       | Extraction       |
| Response Format | Markdown or JSON |
| Temperature     | `0.0` to `0.2`   |

## 🧪 Example: Document Q&A Prompt

Use this template for grounded answers from retrieved document excerpts.

```text id="go0h9e"
Answer using only the retrieved document excerpts.

Rules:
- do not use outside knowledge;
- if the excerpts do not contain the answer, say the excerpts do not provide enough information;
- preserve exact names, dates, paths, numbers, and labels;
- cite or identify the relevant excerpt when available;
- keep the answer concise and source-grounded.
```

Best used with:

| Setting                   | Value                       |
| ------------------------- | --------------------------- |
| Category                  | Document Extraction         |
| Task Type                 | Extraction or Summarization |
| Response Format           | Markdown                    |
| Require Grounding         | Enabled                     |
| Answer From Excerpts Only | Enabled                     |

## 🧪 Example: JSON Output Prompt

Use this template when valid JSON is required.

```text id="p4q0u0"
Return valid JSON only.

Rules:
- do not include Markdown fences;
- do not include prose before or after the JSON;
- use double quotes for all keys and string values;
- preserve exact source values;
- use null when a value is not provided;
- do not invent missing fields.
```

Best used with:

| Setting         | Value       |
| --------------- | ----------- |
| Category        | JSON Output |
| Task Type       | Extraction  |
| Response Format | JSON        |
| Temperature     | `0.0`       |

## 🔁 XML and Markdown Conversion

Prompt Engineering supports conversion between XML-like prompt blocks and Markdown headings.

Example XML-like input:

```xml id="wv5ecf"
<role>
You are a local documentation assistant.
</role>

<rules>
Preserve behavior. Use Google-style docstrings. Treat warnings as defects.
</rules>
```

Converted Markdown-style output:

```markdown id="jkdbxj"
## Role

You are a local documentation assistant.

## Rules

Preserve behavior. Use Google-style docstrings. Treat warnings as defects.
```

This is useful when converting structured prompts into a more readable format for editing and
documentation.

## 🧰 Applying Prompts

Prompt templates can be applied to shared application state.

| Apply Target    | Result                                                                                          |
| --------------- | ----------------------------------------------------------------------------------------------- |
| Text Generation | The prompt becomes the active system instruction for general model output.                      |
| Document Q&A    | The prompt becomes the active system instruction and can enforce grounded answering.            |
| Shared State    | Task type, response format, and language metadata can be synchronized with generation settings. |

Applying prompts avoids copy/paste errors and helps keep repeated workflows consistent.

## 🧬 Cloning Prompt Records

Cloning is useful when a prompt is mostly correct but needs a project-specific variation.

Use cloning when:

* creating a stricter version of an existing prompt;
* adapting a general prompt to a specific project;
* preserving an older version before making changes;
* testing different output rules;
* building prompt families for similar workflows.

Recommended naming pattern:

```text id="tdgq76"
MkDocs Docstring Conversion - Base
MkDocs Docstring Conversion - Strict Griffe
MkDocs Docstring Conversion - Logging Integration
MkDocs Docstring Conversion - Large File Workflow
```

## 🧾 Prompt Versioning

Prompt versioning helps preserve stable workflows.

Recommended version labels:

| Version Pattern   | Use                                          |
| ----------------- | -------------------------------------------- |
| `v1.0`            | Initial stable prompt.                       |
| `v1.1`            | Minor wording improvement.                   |
| `v2.0`            | Major behavior or format change.             |
| `project-name-v1` | Project-specific variation.                  |
| `strict-v1`       | More restrictive validation-oriented prompt. |

Version labels are especially useful for coding, documentation, extraction, and compliance-oriented
prompts.

## 🧭 Prompt Quality Checklist

A strong reusable prompt should answer these questions:

| Question                                 | Why It Matters                               |
| ---------------------------------------- | -------------------------------------------- |
| What role should the model play?         | Establishes behavior and domain focus.       |
| What task is being performed?            | Prevents generic output.                     |
| What input should be used?               | Controls grounding and context use.          |
| What output format is required?          | Reduces formatting corrections.              |
| What should happen when data is missing? | Prevents hallucinated values.                |
| What must not be changed?                | Protects source code and workflow integrity. |
| What validation is required?             | Improves reliability of technical output.    |

## ✅ Recommended Sequence

Use this sequence for reliable Prompt Engineering:

1. Open Prompt Engineering mode.
2. Select a prompt category.
3. Select a task type.
4. Choose the response format.
5. Enter or revise the prompt goal.
6. Add constraints that must always be followed.
7. Select the desired style.
8. Generate or draft the prompt text.
9. Review the prompt for ambiguity.
10. Save the prompt with a clear caption.
11. Apply the prompt to Text Generation or Document Q&A.
12. Test the prompt on a representative task.
13. Revise and version the prompt as needed.

## 🧭 Practical Defaults

A stable prompt setup for technical work is:

| Setting          | Recommended Value                                                |
| ---------------- | ---------------------------------------------------------------- |
| Category         | Coding or Extraction                                             |
| Task Type        | Coding, Summarization, or Extraction                             |
| Response Format  | Markdown unless strict JSON is required                          |
| Language         | English                                                          |
| Style            | Practical                                                        |
| Temperature      | `0.0` to `0.2`                                                   |
| Chat History     | Enabled for iterative prompt refinement                          |
| Document Context | Enabled only when the prompt should use specific source material |

## 🛠️ Troubleshooting

| Problem                         | Likely Cause                                   | Corrective Action                                   |
| ------------------------------- | ---------------------------------------------- | --------------------------------------------------- |
| Prompt produces generic answers | Goal is too vague                              | Add task-specific rules and examples.               |
| Output format is inconsistent   | Format instruction is weak                     | Add explicit response format rules.                 |
| Model invents missing values    | Missing-data rule is absent                    | Add “return Not provided” or `null` behavior.       |
| Prompt is too long              | Too many overlapping rules                     | Remove duplication and keep only enforceable rules. |
| Prompt ignores source material  | Grounding rule is weak                         | Add “use only supplied context” language.           |
| Coding output changes behavior  | Source-preservation rule is missing            | Add explicit preservation constraints.              |
| JSON output includes prose      | JSON-only rule is not strict enough            | Require valid JSON only and forbid Markdown fences. |
| Prompt is hard to reuse         | It contains one-off file names or task details | Move one-off details into the user input instead.   |

## 🚨 Exception Logging

Operational failures in the Prompt Engineering workflow should use the standard Loca logging
pattern.

```python id="ezg9ov"
except Exception as e:
    exception = Error( e )
    exception.module = 'app'
    exception.cause = 'PromptEngineering'
    exception.method = 'function_name( arg: type ) -> return_type'
    Logger( ).write( exception )
    raise exception
```

The method string should remain a stable signature only. It should not include prompt text, user
input, template contents, database row contents, secrets, or runtime object details.

## 🔗 Related API Pages

* [Application](../api/app.md)
* [Configuration](../api/config.md)
* [Logging](../api/boogr.md)
* [Text Generation](text-generation.md)
* [Document Q&A](document-qna.md)
* [Semantic Search](semantic-search.md)
* [Data Management](data-management.md)
