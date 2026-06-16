# Loca Architecture

Loca is a local AI application that provides a Streamlit interface for running local GGUF-based
language models, managing prompts, querying documents, performing semantic search, and storing
application data in SQLite. The application is organized around a single Streamlit runtime, a
centralized configuration contract, and a lightweight exception logging layer.

## 🧭 Purpose

The architecture is designed to keep local model execution, user-interface state, document
workflows, prompt management, and diagnostic logging coordinated through a small set of predictable
application layers.

Loca emphasizes:

* local-first inference through `llama-cpp-python`;
* configurable model routing through `config.py`;
* Streamlit session-state consistency across UI modes;
* document ingestion and retrieval workflows;
* SQLite-backed chat, prompt, document, and embedding storage;
* failure-safe exception logging through `boogr.py`;
* source-driven documentation through MkDocs, mkdocstrings, and Google-style docstrings.

## 🧱 Application Layers

Loca is organized into five primary layers:

| Layer                 | Primary File      | Responsibility                                                                                                                          |
| --------------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| User Interface        | `app.py`          | Renders Streamlit controls, tabs, selectors, upload panels, chat surfaces, and workflow-specific panels.                                |
| Runtime Orchestration | `app.py`          | Coordinates model loading, prompt construction, generation settings, streaming output, document context, and semantic retrieval.        |
| Configuration         | `config.py`       | Defines application constants, model registry entries, mode contracts, environment-variable helpers, resource paths, and logging paths. |
| Storage               | `app.py` / SQLite | Stores chat history, prompts, documents, chunks, embeddings, and application metadata.                                                  |
| Diagnostics           | `boogr.py`        | Wraps exceptions and writes structured error records to the configured SQLite exception database.                                       |

The application intentionally keeps its runtime contract explicit. Most workflows read from and
write to `st.session_state`, while longer-lived records are persisted to SQLite.

## 🧠 Local Model Runtime

Loca uses local model paths defined in the model registry. Each model entry contains the model path,
logo, supported modes, family, base model, chat template, size, and description.

The runtime flow is:

1. The user selects a model in the Streamlit interface.
2. `app.py` synchronizes the selected model name, path, supported modes, and model specification
   into session state.
3. The selected mode determines which workflow panel is rendered.
4. Runtime settings such as context window, CPU threads, temperature, top-p, repeat penalty, and max
   tokens are read from session state.
5. `llama-cpp-python` loads the selected GGUF model when available.
6. The application builds a prompt using system instructions, task instructions, document context,
   semantic context, and chat history.
7. The local model generates either a complete response or streamed output.

## 🔀 Model and Mode Contract

The model contract is centralized in `config.py`.

The core mode constants are:

| Constant        | Purpose                                                                              |
| --------------- | ------------------------------------------------------------------------------------ |
| `TEXT_MODE`     | General local text generation and chat.                                              |
| `IMAGE_MODE`    | Placeholder or gated image-capable workflow for models that advertise image support. |
| `AUDIO_MODE`    | Placeholder or gated audio-capable workflow for models that advertise audio support. |
| `DOCQNA_MODE`   | Document upload, extraction, chunking, retrieval, and grounded question answering.   |
| `SEMANTIC_MODE` | Semantic indexing and similarity search over uploaded or registered content.         |
| `PROMPT_MODE`   | Prompt template creation, editing, cloning, and reuse.                               |
| `DATA_MODE`     | Application data inspection and management.                                          |

Each model advertises its supported modes. The UI uses this registry to avoid exposing workflows
that are not available for the active model.

## 🗂️ Session-State Contract

Streamlit session state is the central runtime coordination mechanism. Loca initializes required
keys early so later UI panels can safely read from them without causing missing-key failures.

Important session-state groups include:

| Group               | Example Keys                                                                                           | Purpose                                                        |
| ------------------- | ------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------- |
| Model Selection     | `selected_model_name`, `selected_model_path`, `selected_model_modes`, `selected_model_spec`            | Tracks the active local model and its registry metadata.       |
| Generation Settings | `temperature`, `top_percent`, `top_k`, `max_tokens`, `repeat_penalty`, `context_window`, `cpu_threads` | Controls local model generation behavior.                      |
| Chat State          | `messages`, `use_chat_history`, `system_instructions`                                                  | Maintains conversational context and shared instruction text.  |
| Document Q&A        | `uploaded`, `active_docs`, `doc_bytes`, `docqna_vec_ready`, `retrieval_k`                              | Supports document upload, extraction, chunking, and retrieval. |
| Semantic Search     | `semantic_context_buffer`, `semantic_top_k`, `semantic_result_rows`, `semantic_index_chunk_count`      | Supports semantic indexing and similarity search.              |
| Prompt Engineering  | `prompt_category`, `prompt_task`, `pe_generated_template`, `active_prompt_caption`                     | Supports prompt template creation and reuse.                   |
| Data Management     | `dm_asset_sync_status`, `dm_asset_counts`, `dm_selected_asset_table`                                   | Supports database asset inspection and management.             |
| Capability State    | `active_model_capabilities`, `function_call_enabled`, `thinking_mode_enabled`, `coding_mode_enabled`   | Tracks model-gated advanced workflows.                         |

This structure allows the Streamlit app to rerun safely while preserving user choices, generated
content, uploaded document state, and workflow outputs.

## 📄 Document Q&A Pipeline

The Document Q&A workflow converts uploaded files into retrievable context for grounded local model
responses.

The pipeline is:

1. The user uploads supported documents.
2. The application extracts text using available parsers.
3. Text is normalized and split into overlapping chunks.
4. Chunks are stored in session state and optionally persisted to SQLite.
5. Embeddings are generated when an embedding backend is available.
6. Retrieval settings determine how many chunks are selected for a question.
7. Retrieved excerpts are inserted into the prompt as document context.
8. The local model answers using the supplied excerpts.

The retrieval profile can be adjusted by model. Smaller models receive narrower retrieval defaults
so prompts remain compact and less likely to exceed practical local runtime limits.

## 🔎 Semantic Search Pipeline

Semantic Search provides similarity-based retrieval over indexed text.

The workflow is:

1. Source documents are loaded or uploaded.
2. Text is chunked using the configured chunk size and overlap.
3. Chunks are embedded when an embedding model is available.
4. Embeddings are stored as vectors in SQLite or maintained in session state.
5. A query is embedded using the same embedding process.
6. Cosine similarity ranks the indexed chunks.
7. Matching chunks are returned to the user and may be reused as context.

The semantic workflow is designed to fail safely. When an embedder, embedding table, or compatible
vector is unavailable, the application reports the limitation instead of crashing the main Streamlit
workflow.

## 🧰 Prompt Engineering Workflow

The Prompt Engineering mode provides a structured interface for managing reusable prompts.

The workflow supports:

* browsing stored prompt templates;
* selecting templates by caption;
* editing prompt text and metadata;
* cloning existing prompt records;
* generating starter prompt drafts;
* applying prompt text to Text Generation;
* applying prompt text to Document Q&A;
* converting between XML-like prompt blocks and Markdown headings.

Prompt records are stored in the SQLite `Prompts` table. The application keeps prompt metadata
aligned with shared generation state so prompt templates can control task type, response format, and
preferred language.

## 🧪 Function Calling and Tool-Grounded Responses

Loca includes an app-mediated function-calling workflow for models that advertise function-calling
capability.

The function-calling flow is:

1. The user submits a task.
2. The selected model is prompted to produce a strict JSON tool-call object.
3. The application parses and normalizes the tool call.
4. The requested function name is checked against an allowlist.
5. The application executes only approved internal functions.
6. The tool result is passed back into the model for a grounded final answer.

This design prevents arbitrary model-generated function names from executing application code. The
model proposes a function call, but the application validates and controls execution.

## 🌐 Web Context Workflow

For models that advertise web-browsing capability, Loca includes a bounded web-context workflow.

The web workflow:

1. Validates the requested URL.
2. Allows only HTTP and HTTPS schemes.
3. Blocks local, loopback, private, reserved, multicast, and link-local hosts.
4. Optionally restricts the request to an allowed domain.
5. Fetches a bounded amount of public text.
6. Converts HTML to readable text when needed.
7. Sends the fetched context to Text Generation or returns it as a tool result.

This layer is intentionally constrained. It is designed for controlled grounding, not unrestricted
browsing.

## 🗃️ SQLite Storage

Loca uses SQLite for lightweight local persistence.

The database layer supports:

| Table                 | Purpose                                              |
| --------------------- | ---------------------------------------------------- |
| `chat_history`        | Stores persisted chat messages.                      |
| `embeddings`          | Stores text chunks and vector blobs for retrieval.   |
| `Prompts`             | Stores reusable prompt templates and metadata.       |
| `documents`           | Stores document-level metadata.                      |
| `document_chunks`     | Stores extracted document chunks.                    |
| `document_embeddings` | Stores document-specific embeddings where supported. |

SQLite keeps the application self-contained and suitable for local desktop use without requiring a
separate database server.

## 🚨 Exception Logging

Loca uses `boogr.py` for structured exception handling.

The diagnostic layer includes:

| Component         | Purpose                                                                                 |
| ----------------- | --------------------------------------------------------------------------------------- |
| `Error`           | Wraps an original exception with module, cause, method, traceback, and diagnostic text. |
| `Logger`          | Creates the exception table when needed and writes wrapped exception records to SQLite. |
| `config.LOG_PATH` | Defines the SQLite database path for exception records.                                 |
| `config.LOG_FILE` | Defines the exception table name.                                                       |

The logging pattern is:

```python
except Exception as e:
    exception = Error( e )
    exception.module = 'app'
    exception.cause = 'WorkflowName'
    exception.method = 'function_name( arg: type ) -> return_type'
    Logger( ).write( exception )
    raise exception
```

Method strings are stable signatures only. They should not include prompt text, uploaded file
contents, user input, dataframe contents, secrets, API keys, tokens, or full runtime data.

## 🧩 Documentation Architecture

Loca documentation is generated from the source code using MkDocs, Material for MkDocs,
mkdocstrings, and griffe-compatible Google-style docstrings.

The documentation architecture should include:

```text
docs/
├── index.md
├── architecture.md
├── development.md
├── user-guide/
│   ├── index.md
│   ├── text-generation.md
│   ├── document-qna.md
│   ├── semantic-search.md
│   ├── prompt-engineering.md
│   └── data-management.md
├── api/
│   ├── index.md
│   ├── app.md
│   ├── config.md
│   └── boogr.md
└── assets/
    ├── css/
    │   └── loca.css
    └── js/
        └── loca.js
```

The API pages should remain source-driven:

```markdown
# App API

::: app
```

```markdown
# Configuration API

::: config
```

```markdown
# Logging API

::: boogr
```

This keeps the API reference aligned with the actual source code and prevents manual documentation
drift.

## 🧭 End-to-End Workflow

The full application flow is:

```text
User
  │
  ▼
Streamlit UI
  │
  ├── Model and mode selection
  ├── Runtime settings
  ├── System instructions
  ├── Document upload
  ├── Prompt templates
  └── Data management controls
  │
  ▼
Session-State Contract
  │
  ├── Selected model metadata
  ├── Generation controls
  ├── Document state
  ├── Semantic search state
  ├── Prompt engineering state
  └── Capability flags
  │
  ▼
Runtime Orchestration
  │
  ├── Prompt construction
  ├── Context retrieval
  ├── Tool-call validation
  ├── Web-context validation
  └── Local model execution
  │
  ▼
Local GGUF Model
  │
  ▼
Generated Response
  │
  ├── Displayed in Streamlit
  ├── Stored in chat history when applicable
  └── Logged if runtime failure occurs
```

## ✅ Design Principles

Loca follows several practical design principles:

| Principle                 | Implementation                                                                    |
| ------------------------- | --------------------------------------------------------------------------------- |
| Local-first execution     | Models run from configured local GGUF paths.                                      |
| Explicit configuration    | Model registry, modes, logos, and paths are centralized in `config.py`.           |
| Safe reruns               | Session-state keys are initialized before use.                                    |
| Source-driven docs        | API documentation is generated directly from Google-style docstrings.             |
| Fail-safe diagnostics     | Logging failures do not mask original application failures.                       |
| Controlled tool execution | Function calls are parsed, normalized, allowlisted, and executed by the app.      |
| Bounded retrieval         | Document and semantic retrieval use configurable chunk and top-k limits.          |
| MkDocs compatibility      | Docstrings avoid griffe-hostile section formats and invalid return documentation. |

## 🔗 Related API Pages

* [Application](api/app.md)
* [Configuration](api/config.md)
* [Logging](api/boogr.md)
* [User Guide](user-guide/index.md)
* [Development Guide](development.md)
