# Text Generation

The Text Generation workflow is the primary chat and completion surface in Loca. It lets the user
select a local model, configure runtime generation settings, provide system instructions, optionally
include document or semantic context, and generate responses through the local GGUF runtime.

## 🧭 Purpose

Text Generation provides the main local assistant workflow for Loca. It coordinates model selection,
prompt construction, generation controls, chat history, optional context injection, and response
rendering in the Streamlit interface.

This page explains how to use the Text Generation mode, how the prompt is assembled, and how runtime
settings affect model output.

## 🧱 Workflow Position

Text Generation sits at the center of the application.

```text
User Input
  │
  ▼
Text Generation UI
  │
  ├── System Instructions
  ├── Task Preset
  ├── Runtime Settings
  ├── Chat History
  ├── Document Context
  └── Semantic Context
  │
  ▼
Prompt Builder
  │
  ▼
Local GGUF Runtime
  │
  ▼
Generated Response
```

The workflow is implemented primarily in `app.py` and uses model, mode, and path information from
`config.py`.

## 🧠 Model Selection

Loca uses a central model registry to determine which local models are available and which modes
each model supports. The active model controls:

| Setting         | Description                                                                        |
| --------------- | ---------------------------------------------------------------------------------- |
| Model name      | The selected local model, such as Bro, Buddy, Boo, Gipity, Jimi, Leeroy, or Nisty. |
| Model path      | The configured GGUF file path used by `llama-cpp-python`.                          |
| Supported modes | The application modes exposed for the selected model.                              |
| Model family    | The model family or provider-style grouping.                                       |
| Base model      | The underlying base model name used for capability checks.                         |
| Chat template   | The expected prompt formatting style.                                              |

When the selected model changes, Loca synchronizes derived model state into Streamlit session state.
This prevents the interface from showing stale mode, path, or capability information.

## ⚙️ Runtime Settings

Text Generation uses several runtime controls to shape the model response.

| Setting        | Purpose                                                            |
| -------------- | ------------------------------------------------------------------ |
| Context Window | Controls how much prompt context the local model can process.      |
| CPU Threads    | Controls how many CPU threads are used by local inference.         |
| Max Tokens     | Limits the number of generated tokens.                             |
| Temperature    | Controls response randomness. Lower values are more deterministic. |
| Top-p          | Controls nucleus sampling. Lower values narrow token selection.    |
| Top-k          | Limits token candidates during sampling when supported.            |
| Repeat Penalty | Penalizes repeated tokens to reduce looping.                       |
| Repeat Window  | Controls how far back repetition penalties are considered.         |
| Random Seed    | Supports repeatable generations when fixed.                        |

Recommended starting point:

| Use Case        |    Temperature |            Top-p |       Max Tokens |
| --------------- | -------------: | ---------------: | ---------------: |
| Factual answer  | `0.0` to `0.2` | `0.90` to `0.95` |  `512` to `1024` |
| Code generation | `0.0` to `0.3` | `0.90` to `0.95` | `1024` to `2048` |
| Brainstorming   | `0.6` to `0.9` |  `0.95` to `1.0` | `1024` to `2048` |
| Summarization   | `0.0` to `0.3` | `0.90` to `0.95` |  `512` to `1536` |

## 🧾 System Instructions

System instructions define high-level behavior for the model. They can control tone, task framing,
output format, reasoning expectations, and response constraints.

Examples:

```text
You are a concise technical assistant. Answer with accurate, practical guidance and avoid unsupported claims.
```

```text
You are a Python documentation assistant. Produce Google-style docstrings compatible with MkDocs, mkdocstrings, and griffe.
```

```text
You are a local coding assistant. Preserve existing behavior, avoid changing public signatures, and return editor-ready code.
```

System instructions are included before user input when Loca builds the model prompt.

## 🧩 Task Presets

Text Generation supports task-oriented presets that modify the instruction block passed to the
model.

| Preset        | Purpose                                                  |
| ------------- | -------------------------------------------------------- |
| Chat          | General-purpose assistant behavior.                      |
| Reasoning     | More careful analytical answers.                         |
| Coding        | Editor-ready code generation, debugging, or refactoring. |
| Translation   | Faithful translation into a selected target language.    |
| Summarization | Concise summaries that preserve key facts.               |
| Extraction    | Structured extraction from supplied text.                |

The selected task preset is combined with the system instructions to form the effective prompt.

## 💬 Chat History

When chat history is enabled, prior user and assistant messages are included in the prompt. This
gives the local model continuity across turns.

Use chat history when:

* the current question depends on earlier conversation;
* the user is iterating on a code block;
* the assistant needs prior constraints;
* a multi-step workflow is underway.

Disable chat history when:

* the response should be independent;
* the prior conversation may pollute the answer;
* the context window is too small;
* the model starts repeating earlier content.

## 📄 Document Context

Text Generation can include document context from uploaded or processed content. When enabled,
selected document text is inserted into the prompt as supporting context.

Use document context for:

* summarizing uploaded material;
* answering questions from source documents;
* drafting documentation from project notes;
* grounding a response in local files.

Document context is separate from Document Q&A mode. Text Generation uses the available context
directly, while Document Q&A focuses on retrieval-grounded answering from extracted chunks.

## 🔎 Semantic Context

Semantic context uses embedding-based retrieval to locate relevant chunks from indexed content. When
enabled, Loca searches stored embeddings for content similar to the current user input and injects
the best matching chunks into the prompt.

The semantic workflow is useful when:

* the source material is too large to include directly;
* the user needs focused context from a document set;
* prior indexed content should support the answer;
* the model needs retrieval support before generating.

If no compatible embedder, embedding table, or vector data is available, Loca reports the limitation
and continues safely.

## 🧪 Example: Basic Chat

Use Text Generation for a simple local assistant exchange.

```text
System Instructions:
You are Loca, a practical local AI assistant. Be accurate, concise, and helpful.

Task Preset:
Chat

User Input:
Explain the difference between temperature and top-p in local LLM generation.
```

Expected behavior:

* Loca builds a prompt from the system instructions, task preset, and user input.
* The selected local model generates a response.
* The response is rendered in the Streamlit chat area.

## 🧪 Example: Coding Task

Use the Coding preset when requesting source code.

```text
System Instructions:
Produce correct, editor-ready Python code. Preserve existing public function signatures.

Task Preset:
Coding

User Input:
Create a function that validates a file path, confirms the file exists, and returns the resolved Path.
```

Recommended settings:

| Setting      |                                   Value |
| ------------ | --------------------------------------: |
| Temperature  |                          `0.0` to `0.2` |
| Top-p        |                        `0.90` to `0.95` |
| Max Tokens   |                        `1024` or higher |
| Chat History | Enabled when iterating on existing code |

## 🧪 Example: Summarization

Use the Summarization preset for concise output.

```text
System Instructions:
Summarize faithfully. Preserve names, dates, numbers, and conclusions.

Task Preset:
Summarization

User Input:
Summarize the following release notes into five bullets:
[paste release notes here]
```

Recommended settings:

| Setting     |            Value |
| ----------- | ---------------: |
| Temperature |   `0.0` to `0.3` |
| Top-p       | `0.90` to `0.95` |
| Max Tokens  |  `512` to `1536` |

## 🧪 Example: Extraction

Use the Extraction preset when the model should return facts from supplied content.

```text
System Instructions:
Extract only facts that are explicitly present. If a value is missing, return "Not provided."

Task Preset:
Extraction

Response Format:
JSON

User Input:
Extract the project name, primary model, database path, and logging table from the following text:
[paste source text here]
```

Recommended behavior:

* avoid invented values;
* preserve exact names;
* use stable field names;
* return valid JSON when JSON output is requested.

## 🧰 Effective Prompt Construction

Loca builds the final prompt from several sources.

```text
System Instructions
  +
Task Instruction Block
  +
Optional Semantic Context
  +
Optional Document Context
  +
Optional Chat History
  +
Current User Input
```

This layered prompt design allows the same model runtime to support general chat, coding, reasoning,
translation, summarization, extraction, and grounded context workflows.

## 🚨 Missing Model Handling

If the selected model path is missing or `llama-cpp-python` is unavailable, Loca returns a
user-facing diagnostic instead of failing with an unclear runtime error.

Common causes include:

| Cause                      | Fix                                                              |
| -------------------------- | ---------------------------------------------------------------- |
| Missing GGUF file          | Confirm the model file exists at the configured path.            |
| Empty model path           | Set the correct environment variable or update `config.py`.      |
| Missing `llama-cpp-python` | Install or repair the package in the active virtual environment. |
| Wrong selected model       | Select a model with a valid configured path.                     |

## ✅ Recommended Sequence

Use this sequence for reliable Text Generation results:

1. Select the local model.
2. Confirm the model path is configured.
3. Select `Text Generation` mode.
4. Set the task preset.
5. Enter or select system instructions.
6. Adjust runtime settings.
7. Enable chat history only when continuity is needed.
8. Enable document or semantic context only when grounding is needed.
9. Submit the user input.
10. Review the generated response.
11. Clear or revise context when switching tasks.

## 🧭 Practical Defaults

A stable default configuration for most technical work is:

| Setting          |                            Recommended Value |
| ---------------- | -------------------------------------------: |
| Task Preset      |                           `Chat` or `Coding` |
| Temperature      |                                        `0.1` |
| Top-p            |                                       `0.95` |
| Repeat Penalty   |                                        `1.1` |
| Max Tokens       |                                       `1024` |
| Chat History     |                                      Enabled |
| Document Context |                       Disabled unless needed |
| Semantic Context | Disabled unless indexed content is available |

## 🛠️ Troubleshooting

| Problem                         | Likely Cause                                       | Corrective Action                                            |
| ------------------------------- | -------------------------------------------------- | ------------------------------------------------------------ |
| Response is empty               | Prompt is empty or model failed to load            | Confirm input text and model path.                           |
| Response repeats itself         | Sampling settings are too loose                    | Increase repeat penalty or reduce temperature.               |
| Response ignores context        | Context not enabled or too much irrelevant context | Enable the correct context source and reduce unrelated text. |
| Model crashes during generation | Invalid runtime setting or unsupported argument    | Use conservative defaults and retry.                         |
| Output is too short             | Max tokens too low                                 | Increase max tokens.                                         |
| Output is too random            | Temperature too high                               | Lower temperature.                                           |
| Output is too rigid             | Temperature too low                                | Increase temperature slightly.                               |

## 🔗 Related API Pages

* [Application](../api/app.md)
* [Configuration](../api/config.md)
* [Logging](../api/boogr.md)
* [Document Q&A](document-qna.md)
* [Semantic Search](semantic-search.md)
* [Prompt Engineering](prompt-engineering.md)
