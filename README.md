###### LocaLlama

![](https://github.com/is-leeroy-jenkins/LocaLlama/blob/main/resources/images/loca_project.png)

<p align="center">
  <a href="#-key-features">Features</a> ·
  <a href="#-application-modes">Modes</a> ·
  <a href="#-supported-models">Models</a> ·
  <a href="#-architecture">Architecture</a> ·
  <a href="#-repository-structure">Structure</a> ·
  <a href="#-installation--setup">Install</a> ·
  <a href="#-configuration">Configuration</a> ·
  <a href="#-text-generation">Text</a> ·
  <a href="#-images-api">Images</a> ·
  <a href="#-audio-api">Audio</a> ·
  <a href="#-document-qa">RAG</a> ·
  <a href="#-semantic-search">Search</a> ·
  <a href="#-prompt-engineering">Prompts</a> ·
  <a href="#-data-management">Data</a> ·
  <a href="#-requirements">Requirements</a>
</p>

LocaLlama is a local-first python application for GGUF-based text generation,
document-grounded retrieval, semantic search, prompt engineering, multimodal workflow staging,
function-calling experimentation, guarded web context ingestion, and SQLite-backed data management.
It is designed to run local language models through `llama-cpp-python` while giving analysts direct
control over model selection, inference parameters, prompt templates, document context, retrieval
behavior, semantic chunking, local data assets, and model-specific capability gates.

The application is organized around named local assistants such as Bro, Gipity, Buddy, Boo, Jimi,
Leeroy, and Nisty. Each assistant is configured through `config.py`, where its GGUF path, logo,
base model metadata, description, and allowed application modes are declared.

## 🎥 Demo

![](https://github.com/is-leeroy-jenkins/LocaLlama/blob/main/resources/images/loca-demo.gif)

## 🕸️ Streamlit Web

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit\&logoColor=white)](https://bro-py.streamlit.app/)

![](https://github.com/is-leeroy-jenkins/LocaLlama/blob/main/resources/images/Loca-streamlit.gif)

LocaLlama uses Streamlit to provide an interactive local-AI console with sidebar model selection,
mode selection, parameter controls, upload workflows, persisted chat, database tools, and status
visibility.

## ✨ Key Features

| Feature                     | Description                                                                                                                                                                       |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Local GGUF inference        | Runs configured local GGUF models through `llama-cpp-python` with context-window, CPU-thread, token, temperature, top-p, top-k, repeat-penalty, and seed controls.                |
| Multi-model registry        | Selects among Bro, Gipity, Buddy, Boo, Jimi, Leeroy, and Nisty through a centralized `MODEL_REGISTRY` in `config.py`.                                                             |
| Model-specific modes        | Exposes only the modes declared for the selected model, including Text Generation, Images API, Audio API, Document Q&A, Semantic Search, Prompt Engineering, and Data Management. |
| Text generation             | Supports chat, reasoning, coding, translation, summarization, extraction, system instructions, prompt templates, chat history, and document context.                              |
| Advanced model capabilities | Adds gated Thinking, Advanced Coding, and Function Calling controls for models that advertise those capabilities.                                                                 |
| Gipity tool workflows       | Provides app-mediated function-call generation, strict JSON parsing, allowlisted function execution, and guarded web-context ingestion for Gipity.                                |
| Images API mode             | Adds image upload, image preview, prompt capture, image-context buffering, and fail-closed adapter hooks for Jimi and Nisty image workflows.                                      |
| Audio API mode              | Adds audio upload, audio preview, transcription/analysis prompt capture, transcript buffering, and fail-closed adapter hooks for Jimi and Nisty audio workflows.                  |
| Persistent chat history     | Saves role/content chat messages to SQLite and reloads history on startup.                                                                                                        |
| System instructions         | Supports reusable system instructions, prompt templates, preset instructions, and XML-like delimiter conversion.                                                                  |
| Document Q&A                | Uploads PDFs, TXT files, and DOCX files and answers grounded questions over retrieved excerpts.                                                                                   |
| Buddy compact retrieval     | Allows Buddy to use Document Q&A and Semantic Search with smaller retrieval windows suitable for a compact 270M model.                                                            |
| Document actions            | Summarizes active documents, extracts key points, generates outlines, extracts entities, extracts tables, and compares active documents.                                          |
| Semantic search             | Builds a semantic index from uploaded files, queries indexed chunks, selects relevant chunks, and routes selected context into Text Generation or Document Q&A.                   |
| sqlite-vec support          | Uses sqlite-vec for document retrieval when available, with fallback cosine similarity over stored vectors.                                                                       |
| Prompt engineering          | Searches, pages, sorts, edits, clones, generates, and applies prompt templates stored in the local `Prompts` table.                                                               |
| Data management             | Imports Excel workbooks into SQLite, browses tables, runs CRUD, explores, filters, aggregates, visualizes, administers schema, and runs guarded read-only SQL.                    |
| AI asset governance         | Registers active documents, chunks, embeddings, uploaded image metadata, and local AI assets into SQLite governance tables.                                                       |
| Fixed status footer         | Displays active mode, model parameters, semantic state, context settings, document count, and runtime information.                                                                |

## 🧭 Application Modes

LocaLlama is built around a practical local-AI workflow:

* Select a local assistant model and load its configured GGUF path.
* Run instruction-following, reasoning, coding, extraction, translation, and summarization tasks.
* Upload PDF, TXT, or DOCX documents for local document question answering.
* Build semantic indexes from uploaded documents using sentence-transformer embeddings.
* Route retrieved document or semantic chunks into Text Generation.
* Stage image and audio workflows for models that advertise multimodal capability.
* Generate and validate app-mediated function calls.
* Fetch public web pages into bounded local context for Gipity-driven analysis.
* Store chat history, prompts, embeddings, document metadata, chunks, image metadata, and imported
  data in SQLite.
* Use Prompt Engineering mode to create, edit, clone, search, and apply prompt templates.
* Use Data Management mode to import Excel data, browse tables, run guarded SQL, profile data, and
  administer local AI-asset tables.

The current application mode contract is:

| Mode                   | Purpose                                                                                   | Major Controls / Outputs                                                                                                                                                                                                                                                  |
| ---------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Text Generation**    | Primary local chat and generation interface.                                              | Task presets, response format, chat history, document context, reasoning controls, coding controls, advanced model capabilities, function schema input, Gipity tools, web context, inference parameters, system instructions, prompt preview, and persisted chat history. |
| **Images API**         | Image upload, preview, prompt capture, image analysis staging, and image-context routing. | Image uploader, prompt box, model/runtime capability status, future image adapter hook, image response area, send-to-text context actions, and scoped image-state clearing.                                                                                               |
| **Audio API**          | Audio upload, preview, transcription/analysis staging, and audio-context routing.         | Audio uploader, audio preview, transcription/analysis prompt, model/runtime capability status, future audio adapter hook, transcript/response area, send-to-text context actions, and scoped audio-state clearing.                                                        |
| **Document Q&A**       | Retrieval-augmented question answering over uploaded documents.                           | PDF/TXT/DOCX upload, active document inventory, retrieval controls, chunk size/overlap, grounding toggles, sqlite-vec toggle, fallback cosine search, document actions, parsing controls, and retrieved chunk display.                                                    |
| **Semantic Search**    | Build and query a reusable semantic chunk index.                                          | Upload files, chunk size/overlap, clear/append behavior, top-k search, similarity threshold, selected chunks, send-to-text, send-to-Document-Q&A, save-as-context actions, diagnostics, and index deletion.                                                               |
| **Prompt Engineering** | Manage reusable prompt templates and cascade them into generation modes.                  | Search, category inference, sort, pagination, go-to-ID, prompt table, apply to Text Generation, apply to Document Q&A, clone, starter prompt generator, edit/create/delete.                                                                                               |
| **Data Management**    | Manage local SQLite data and AI asset metadata.                                           | Excel import, table browse, CRUD, profile/explore, filter, aggregate, visualize, schema admin, index creation, AI asset governance, and read-only SQL console.                                                                                                            |

## 🤖 Supported Models

Model availability is controlled by local environment-variable paths in `config.py`. A model appears
in
the application when its configured path is present and the selected registry entry advertises one
or more
supported modes.

| Assistant  | Base Model              | Size | Family  | Primary Role                                                                                                                                               | Configured Modes                                                                                           |
| ---------- | ----------------------- | ---- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **Bro**    | `gemma-3-4b-it`         | 4B   | Gemma   | Balanced local assistant for text, document, prompt, semantic, and data workflows.                                                                         | Text Generation, Document Q&A, Semantic Search, Prompt Engineering, Data Management                        |
| **Gipity** | `gpt-oss-20b`           | 21B  | GPT-OSS | Larger reasoning-oriented local assistant for text, document, semantic, function-calling, and guarded web-context workflows.                               | Text Generation, Document Q&A, Semantic Search, Prompt Engineering, Data Management                        |
| **Buddy**  | `gemma-3-270m-it`       | 0.3B | Gemma 3 | Compact local assistant using conservative retrieval defaults for Document Q&A and Semantic Search.                                                        | Text Generation, Document Q&A, Semantic Search, Prompt Engineering, Data Management                        |
| **Boo**    | `Phi-4-mini-instruct`   | 3.8B | Phi     | Lightweight reasoning assistant with strong text and document workflow support.                                                                            | Text Generation, Document Q&A, Semantic Search, Prompt Engineering, Data Management                        |
| **Jimi**   | `gemma-4-E4B-it`        | 4B   | Gemma   | Multimodal-capable local assistant for text, image, audio, document, semantic, prompt, data, thinking, coding, and function-call workflows.                | Text Generation, Images API, Audio API, Document Q&A, Semantic Search, Prompt Engineering, Data Management |
| **Leeroy** | `Llama-3.2-1B-Instruct` | 1B   | Llama   | Small instruction-tuned text assistant for dialogue, summarization, retrieval, and prompt workflows.                                                       | Text Generation, Document Q&A, Semantic Search, Prompt Engineering, Data Management                        |
| **Nisty**  | `gemma-4-E4B-it`        | 4B   | Gemma   | Governance/document-oriented multimodal assistant for text, image, audio, document, semantic, prompt, data, thinking, coding, and function-call workflows. | Text Generation, Images API, Audio API, Document Q&A, Semantic Search, Prompt Engineering, Data Management |

## 🏛 Architecture

```text
Streamlit UI
    │
    ├── Sidebar Model Registry
    │       ├── config.py MODEL_REGISTRY
    │       ├── environment-variable GGUF paths
    │       ├── selected model state
    │       ├── selected mode state
    │       └── model capability gates
    │
    ├── Text Generation
    │       ├── llama-cpp-python
    │       ├── selected local GGUF model
    │       ├── chat_history table
    │       ├── Prompts table
    │       ├── system instructions
    │       ├── task-specific instruction block
    │       ├── thinking / coding / function-calling prompt extensions
    │       └── Gipity tool and web-context controls
    │
    ├── Images API
    │       ├── image upload
    │       ├── image preview
    │       ├── model capability gate
    │       ├── runtime adapter gate
    │       ├── optional analyze_image_with_model adapter
    │       └── image context routing into Text Generation
    │
    ├── Audio API
    │       ├── audio upload
    │       ├── audio preview
    │       ├── model capability gate
    │       ├── runtime adapter gate
    │       ├── optional analyze_audio_with_model adapter
    │       └── audio context routing into Text Generation
    │
    ├── Document Q&A
    │       ├── uploaded PDFs / TXT / DOCX
    │       ├── PyMuPDF / text extraction
    │       ├── sentence-transformers embeddings
    │       ├── sqlite-vec vector table when available
    │       ├── cosine fallback search
    │       └── compact Buddy retrieval profile when Buddy is selected
    │
    ├── Semantic Search
    │       ├── chunked uploaded documents
    │       ├── embeddings table
    │       ├── vector diagnostics
    │       ├── selected semantic context
    │       └── context routing into Text Generation / Document Q&A
    │
    ├── Prompt Engineering
    │       ├── Prompts table
    │       ├── template metadata
    │       ├── starter prompt generation
    │       └── system-instruction cascade
    │
    ├── Gipity Tools
    │       ├── strict JSON function-call generation
    │       ├── JSON extraction and normalization
    │       ├── allowlisted function execution
    │       ├── guarded public HTTP/HTTPS web fetch
    │       └── tool-grounded final answer generation
    │
    └── Data Management
            ├── Excel import
            ├── CRUD / profiling / visualization
            ├── guarded SQL console
            └── AI asset governance tables
```

## 🗂 Repository Structure

```text
LocaLlama/
├─ app.py                         # Main Streamlit application
├─ config.py                      # App constants, modes, model registry, paths, logos, help text
├─ requirements.txt               # Python dependencies
├─ resources/
│  └─ images/
│     ├─ loca_project.png
│     ├─ LocaLlama-streamlit.gif
│     ├─ loca-llama_logo.png
│     ├─ bro_logo.png
│     ├─ gipity_logo.png
│     ├─ buddy_logo.png
│     ├─ boo_logo.png
│     ├─ jimi_logo.png
│     ├─ leeroy_logo.png
│     └─ nisty_logo.png
├─ stores/
│  └─ sqlite/
│     └─ loca.db                  # Chat history, prompts, embeddings, documents, chunks, images
├─ models/
│  └─ *.gguf                      # Optional local model directory depending on environment variables
└─ README.md
```

## ⚙️ System Requirements

| Requirement      |                                        Minimum |                                                             Recommended |
| ---------------- | ---------------------------------------------: | ----------------------------------------------------------------------: |
| Operating system |          Windows 10/11 64-bit, Linux, or macOS |                                              Windows 11 64-bit or Linux |
| Python           |                                           3.10 |                                                                    3.11 |
| RAM              |                                           8 GB |                                                           16 GB or more |
| CPU              |                                 Modern x64 CPU |                                    AVX2-capable CPU with multiple cores |
| Storage          |                                    5-7 GB free |           10+ GB free for models, SQLite assets, and uploaded documents |
| GPU              |                                   Not required |                 Optional; current app is CPU-oriented through llama.cpp |
| Network          | Not required for local text/document workflows | Required only for Gipity web-context fetches or package/model downloads |

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/is-leeroy-jenkins/LocaLlama.git
cd LocaLlama
```

### 2️⃣ Create and Activate a Virtual Environment

#### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Windows Command Prompt

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 📥 Local Model Setup

LocaLlama expects local GGUF model paths to be supplied through environment variables and consumed
by
`config.py`.

| Environment Variable | Assistant |
| -------------------- | --------- |
| `BRO_LLM_PATH`       | Bro       |
| `GIPITY_LLM_PATH`    | Gipity    |
| `BUDDY_LLM_PATH`     | Buddy     |
| `BOO_LLM_PATH`       | Boo       |
| `JIMI_LLM_PATH`      | Jimi      |
| `LEEROY_LLM_PATH`    | Leeroy    |
| `NISTY_LLM_PATH`     | Nisty     |

Example on Windows PowerShell:

```powershell
$env:BRO_LLM_PATH="C:\Users\you\models\bro-3-4b-it-Q4_K_M.gguf"
$env:GIPITY_LLM_PATH="C:\Users\you\models\gpt-oss-20b-Q4_K_M.gguf"
$env:BUDDY_LLM_PATH="C:\Users\you\models\buddy-gemma-3-270m-it-Q4_K_M.gguf"
$env:BOO_LLM_PATH="C:\Users\you\models\boo-phi-4-mini-instruct-Q4_K_M.gguf"
$env:JIMI_LLM_PATH="C:\Users\you\models\jimi-gemma-4-e4b-it-Q4_K_M.gguf"
$env:LEEROY_LLM_PATH="C:\Users\you\models\leeroy-llama-3.2-1b-instruct-Q4_K_M.gguf"
$env:NISTY_LLM_PATH="C:\Users\you\models\nisty-gemma-4-e4b-it-Q4_K_M.gguf"
```

Restart the terminal or IDE after changing environment variables.

## ▶️ Running LocaLlama

Run Streamlit through Python so the correct virtual environment is used.

```bash
python -m streamlit run app.py
```

The application opens in wide layout, renders the LocaLlama subtitle, and exposes the model and mode
selectors in the sidebar.

## 🔧 Configuration

LocaLlama reads runtime configuration from `config.py` and Streamlit session state.

| Configuration Item             | Purpose                                                                                            |
| ------------------------------ | -------------------------------------------------------------------------------------------------- |
| `cfg.APP_TITLE`                | Streamlit browser/page title.                                                                      |
| `cfg.APP_SUBTITLE`             | Startup caption displayed under page configuration.                                                |
| `cfg.BASE_DIR`                 | Application base directory.                                                                        |
| `cfg.DB_PATH`                  | SQLite database path used for chat, prompts, embeddings, documents, chunks, and images.            |
| `cfg.DEFAULT_CTX`              | Default model context window.                                                                      |
| `cfg.CORES`                    | Maximum CPU thread count exposed in the UI.                                                        |
| `cfg.FAVICON`                  | Streamlit page icon.                                                                               |
| `cfg.LOGO`                     | Default application logo.                                                                          |
| `cfg.BRO_LOGO`                 | Bro model logo.                                                                                    |
| `cfg.GIPITY_LOGO`              | Gipity model logo.                                                                                 |
| `cfg.BUDDY_LOGO`               | Buddy model logo.                                                                                  |
| `cfg.BOO_LOGO`                 | Boo model logo.                                                                                    |
| `cfg.JIMI_LOGO`                | Jimi model logo.                                                                                   |
| `cfg.LEEROY_LOGO`              | Leeroy model logo.                                                                                 |
| `cfg.NISTY_LOGO`               | Nisty model logo.                                                                                  |
| `cfg.TEXT_MODE`                | Text Generation mode label.                                                                        |
| `cfg.IMAGE_MODE`               | Images API mode label.                                                                             |
| `cfg.AUDIO_MODE`               | Audio API mode label.                                                                              |
| `cfg.DOCQNA_MODE`              | Document Q&A mode label.                                                                           |
| `cfg.SEMANTIC_MODE`            | Semantic Search mode label.                                                                        |
| `cfg.PROMPT_MODE`              | Prompt Engineering mode label.                                                                     |
| `cfg.DATA_MODE`                | Data Management mode label.                                                                        |
| `cfg.MODES`                    | Application mode list.                                                                             |
| `cfg.MODEL_REGISTRY`           | Assistant model metadata, model paths, logos, base model names, descriptions, and supported modes. |
| `cfg.AUDIO_API`                | Help text for Audio API mode.                                                                      |
| `cfg.IMAGES_API`               | Help text for Images API mode.                                                                     |
| `cfg.XML_BLOCK_PATTERN`        | XML-like prompt delimiter pattern used by conversion utilities.                                    |
| `cfg.MARKDOWN_HEADING_PATTERN` | Markdown heading pattern used by prompt conversion utilities.                                      |
| `cfg.BLUE_DIVIDER`             | Shared divider styling.                                                                            |

### Optional Runtime Capability Flags

Image and audio modes are model-gated and runtime-gated. The mode may be visible for Jimi or Nisty,
but actual multimodal inference requires an image/audio-capable local adapter to be wired into
`app.py`.

Optional config flags may be added later:

```python
IMAGE_RUNTIME_AVAILABLE = True
AUDIO_RUNTIME_AVAILABLE = True
WEB_RUNTIME_AVAILABLE = True
MULTIMODAL_RUNTIME_NAME = 'custom-local-adapter'
```

Without those runtime flags and adapter functions, Image Mode and Audio Mode fail closed with a
clear
status message instead of raising a runtime error.

## 💬 Text Generation

Text Generation is the primary local chat interface. It uses chat history, system instructions,
task-specific prompt blocks, optional semantic context, optional document context, and
model-specific
capability gates to construct a local model prompt.

| Control Group                 | Controls                                                                                                                                                                                         |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Task Preset                   | Chat, Reasoning, Coding, Translation, Summarization, Extraction.                                                                                                                                 |
| Response Format               | Plain Text, Markdown, Bullet Summary, JSON.                                                                                                                                                      |
| Conversation Context          | Use Conversation History, Use Document Context.                                                                                                                                                  |
| Reasoning Controls            | Reasoning Depth, Answer Only, Use Self-Check, Prefer Deterministic Reasoning.                                                                                                                    |
| Coding Controls               | Code Language, Coding Task, Include Comments, Use Editor Format, Emit Fenced Code, Translation Target Language.                                                                                  |
| Advanced Model Capabilities   | Enable Thinking, Thinking Effort, Reasoning Summary, Enable Advanced Coding, Include Test Strategy, Explain Implementation, Enable Function Calling, Function Call Prompt, Function Schema JSON. |
| Gipity Tools and Web Browsing | Generate function-call JSON, execute allowlisted tools, fetch public web context, send web context to Text Generation, generate tool-grounded final answer.                                      |
| Response Controls             | Temperature, Top-P, Top-K, Use Grounding.                                                                                                                                                        |
| Inference Settings            | Repeat Window, Repeat Penalty, Presence Penalty, Frequency Penalty.                                                                                                                              |
| Context Controls              | Context Window, CPU Threads, Max Tokens, Random Seed.                                                                                                                                            |
| System Instructions           | Free-text instruction area, template selector, clear, XML-like conversion, Markdown conversion, preset application, effective prompt preview.                                                    |

### Advanced Text Capabilities

| Capability         | Models              | Behavior                                                                                          |
| ------------------ | ------------------- | ------------------------------------------------------------------------------------------------- |
| Thinking           | Jimi, Nisty         | Adds model-gated thinking-effort instructions while avoiding exposure of hidden chain-of-thought. |
| Advanced Coding    | Jimi, Nisty         | Adds code-generation, test-strategy, and implementation-explanation controls.                     |
| Function Calling   | Jimi, Nisty, Gipity | Allows strict JSON function-call generation against user-supplied or default schemas.             |
| Gipity Web Context | Gipity              | Fetches public HTTP/HTTPS web pages into bounded context for model-grounded answers.              |

## 🖼️ Images API

Images API stages image analysis workflows for models configured with image capability, currently
Jimi
and Nisty. The mode is designed to be safe even when the local runtime is still text-only.

| Component            | Description                                                                                                   |
| -------------------- | ------------------------------------------------------------------------------------------------------------- |
| Model gate           | Only models configured for `IMAGE_MODE` and recognized by the capability layer expose active image workflows. |
| Runtime gate         | The app checks whether an image-capable runtime adapter is configured before attempting inference.            |
| Upload               | Accepts PNG, JPG, JPEG, WEBP, and BMP files.                                                                  |
| Preview              | Renders the uploaded image in Streamlit and displays file size.                                               |
| Prompt               | Captures user instructions for image captioning, description, extraction, or analysis.                        |
| Adapter hook         | Calls `analyze_image_with_model` only if that callable exists in `app.py`.                                    |
| Context routing      | Can send image analysis text into shared Text Generation document context.                                    |
| Fail-closed behavior | If no adapter is present, the app displays a status message instead of crashing.                              |

Optional adapter signatures:

```python
def analyze_image_with_model( model_path: str, model_name: str, image_bytes: bytes,
	image_name: str, prompt: str ) -> str:
```

or:

```python
def analyze_image_with_model( image_bytes: bytes, prompt: str ) -> str:
```

## 🎧 Audio API

Audio API stages transcription, translation, summarization, and audio-analysis workflows for models
configured with audio capability, currently Jimi and Nisty. The mode is safe to expose before a true
audio
runtime adapter is wired.

| Component            | Description                                                                                                   |
| -------------------- | ------------------------------------------------------------------------------------------------------------- |
| Model gate           | Only models configured for `AUDIO_MODE` and recognized by the capability layer expose active audio workflows. |
| Runtime gate         | The app checks whether an audio-capable runtime adapter is configured before attempting inference.            |
| Upload               | Accepts WAV, MP3, M4A, FLAC, and OGG files.                                                                   |
| Preview              | Renders an audio player and displays file metadata.                                                           |
| Prompt               | Captures transcription, translation, summarization, or analysis instructions.                                 |
| Adapter hook         | Calls `analyze_audio_with_model` only if that callable exists in `app.py`.                                    |
| Transcript state     | Stores audio output in `audio_transcript`, `audio_response`, and `audio_context_buffer`.                      |
| Context routing      | Can send transcript or audio analysis text into shared Text Generation document context.                      |
| Fail-closed behavior | If no adapter is present, the app displays a status message instead of crashing.                              |

Optional adapter signatures:

```python
def analyze_audio_with_model( model_path: str, model_name: str, audio_bytes: bytes,
	audio_name: str, prompt: str ) -> str:
```

or:

```python
def analyze_audio_with_model( audio_bytes: bytes, prompt: str ) -> str:
```

## 📚 Document Q&A

Document Q&A provides local retrieval-augmented answering over uploaded files. The app accepts PDF,
TXT, and DOCX uploads, extracts text, chunks documents, indexes chunks, retrieves relevant excerpts,
and routes document-grounded prompts through the same local generation pipeline.

| Component                 | Description                                                                                                                                   |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Document Loader           | Upload one or more PDF, TXT, or DOCX documents and select active documents.                                                                   |
| Preview                   | Renders PDFs with Streamlit where possible or shows extracted text previews.                                                                  |
| Active Document Inventory | Displays file name, byte size, text length, chunk count, and loaded state.                                                                    |
| Retrieval Controls        | Chunks to retrieve, chunk size, chunk overlap, show retrieved chunks.                                                                         |
| Grounding Controls        | Require grounding and answer from excerpts only.                                                                                              |
| Search Backend            | Prefer sqlite-vec, with optional fallback cosine search.                                                                                      |
| Document Actions          | Answer Question, Summarize Active Document, Extract Key Points, Generate Outline, Extract Entities, Extract Tables, Compare Active Documents. |
| Parsing Controls          | Enable OCR, prefer native PDF text, include page markers, show diagnostics.                                                                   |
| Buddy Profile             | Applies compact retrieval defaults when Buddy is selected to reduce prompt size and improve stability.                                        |

### Buddy Compact Retrieval Profile

Buddy is configured for Document Q&A and Semantic Search, but it uses a smaller retrieval profile
because
it is a compact 270M model.

| Setting                   | Buddy Default | Purpose                                                 |
| ------------------------- | ------------: | ------------------------------------------------------- |
| Retrieved chunks          |             3 | Keeps prompt context compact.                           |
| Retrieval chunk size      |           800 | Reduces excerpt length.                                 |
| Retrieval chunk overlap   |           120 | Preserves continuity without unnecessary context bloat. |
| Semantic top-k            |             4 | Limits semantic handoff size.                           |
| Semantic chunk size       |           800 | Keeps indexed chunks compact.                           |
| Semantic chunk overlap    |           120 | Maintains retrieval continuity.                         |
| Grounding required        |          True | Reduces unsupported answer generation.                  |
| Answer from excerpts only |          True | Keeps document answers tied to retrieved text.          |

## 🔍 Semantic Search

Semantic Search builds a reusable semantic index from uploaded PDF, TXT, or DOCX files. It stores
chunk vectors in the local SQLite `embeddings` table and allows ranked semantic retrieval.

| Section        | Controls / Outputs                                                                                                      |
| -------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Index Builder  | Upload files, chunk size, chunk overlap, clear existing index, append to existing index, show diagnostics, build index. |
| Diagnostics    | Indexed documents, indexed chunks, vector dimension, semantic status.                                                   |
| Semantic Query | Query text, top-k results, minimum similarity, group by document.                                                       |
| Results        | Selectable rows with rank, score, chunk text, and length.                                                               |
| Actions        | Send selected chunks to Text Generation, send selected chunks to Document Q&A, save selected chunks as prompt context.  |
| Maintenance    | Delete index, recompute diagnostics, clear query results.                                                               |
| Safety         | Guards database read/write failures, missing vector tables, malformed vectors, and embedding-dimension mismatches.      |

Semantic Search is app-level retrieval infrastructure. The selected LLM does not perform embedding
generation directly. Instead, embeddings are created by the configured embedding layer, and the
selected
assistant model reads or reasons over retrieved chunks.

## 🧰 Function Calling and Web Browsing

Function Calling and Web Browsing are app-mediated workflows. The model can request a function call,
but LocaLlama validates, normalizes, and executes only allowlisted functions.

| Capability                 | Description                                                                                                             |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Function-call generation   | Asks the selected model to produce one strict JSON object using the shape `{"name":"function_name","arguments":{...}}`. |
| JSON extraction            | Extracts valid JSON even if the model accidentally wraps output in markdown or prose.                                   |
| Function normalization     | Converts model output into the app contract.                                                                            |
| Allowlisted execution      | Executes only functions explicitly allowed by the application.                                                          |
| Tool-grounded final answer | Feeds the validated tool result back into the model for a grounded final response.                                      |
| Web fetch                  | Fetches public HTTP/HTTPS pages into bounded readable text context.                                                     |
| Private-network guard      | Blocks localhost, private IPs, loopback, link-local, reserved, multicast, and non-HTTP schemes.                         |
| Optional domain guard      | Restricts URL fetches to a user-specified allowed domain when provided.                                                 |

Allowlisted functions:

| Function           | Purpose                                                        |
| ------------------ | -------------------------------------------------------------- |
| `summarize_text`   | Deterministically summarizes supplied text into bullet points. |
| `extract_keywords` | Extracts simple frequency-ranked keywords from supplied text.  |
| `web_browse_url`   | Fetches public web text for model grounding.                   |

Example function-call JSON:

```json
{
	"name": "web_browse_url",
	"arguments": {
		"url": "https://example.com",
		"prompt": "Summarize the page in five bullets.",
		"allowed_domain": "example.com",
		"max_chars": 12000
	}
}
```

## 📝 Prompt Engineering

Prompt Engineering manages reusable prompt templates in SQLite. It can search, sort, page, edit,
clone, generate, and cascade prompts into other modes.

| Capability         | Description                                                                                                                                        |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Search and filter  | Search captions, names, and prompt text; infer categories from prompt content.                                                                     |
| Pagination         | Browse prompt records in pages.                                                                                                                    |
| Go to ID           | Jump directly to a prompt by primary key.                                                                                                          |
| Category inference | Classifies prompt records into General Chat, Reasoning, Coding, Translation, Summarization, Extraction, Document Extraction, OCR, and JSON Output. |
| Prompt actions     | Apply to Text Generation, apply to Document Q&A, clone as new template, generate starter prompt.                                                   |
| Prompt generator   | Drafts a prompt template from task type, response format, language, goal, constraints, and style.                                                  |
| Edit surface       | Create, update, delete, and clear prompt records.                                                                                                  |
| Cascade            | Optionally cascade selected prompts into shared System Instructions and task settings.                                                             |

## 🗄️ Data Management

Data Management operates on the local SQLite database and exposes both general table operations and
AI asset governance workflows.

| Tab              | Purpose                                                                                                                                                                                  |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **📥 Import**    | Import Excel workbooks into SQLite tables and register active AI assets.                                                                                                                 |
| **🗂 Browse**    | Browse selected SQLite tables.                                                                                                                                                           |
| **💉 CRUD**      | Insert, update, and delete rows with type-aware widgets.                                                                                                                                 |
| **📊 Explore**   | Page through table records.                                                                                                                                                              |
| **🔎 Filter**    | Filter table rows by column substring.                                                                                                                                                   |
| **🧮 Aggregate** | Compute SUM, AVG, and COUNT over numeric columns.                                                                                                                                        |
| **📈 Visualize** | Render Plotly histograms for numeric columns.                                                                                                                                            |
| **⚙ Admin**      | Refresh asset counts, rebuild active document asset rows, purge orphaned AI assets, profile tables, drop tables, create indexes, create custom tables, inspect schema, and alter tables. |
| **🧠 SQL**       | Run guarded read-only SQL queries and export results to CSV.                                                                                                                             |

### Local SQLite Tables

The application initializes core local tables when the database is created.

| Table                 | Purpose                                 |
| --------------------- | --------------------------------------- |
| `chat_history`        | Persistent local chat history.          |
| `embeddings`          | Semantic-search chunk vectors.          |
| `Prompts`             | Prompt templates and metadata.          |
| `documents`           | Registered document metadata.           |
| `document_chunks`     | Registered document chunks.             |
| `document_embeddings` | Registered document embedding metadata. |
| `images`              | Registered uploaded image metadata.     |

## 📊 Status Footer

LocaLlama renders a fixed footer showing the current operating state.

| Footer Item                   | Description                                      |
| ----------------------------- | ------------------------------------------------ |
| Mode                          | Current application mode.                        |
| Model                         | Selected assistant model.                        |
| Temp / Top-P / Top-K          | Active generation sampling settings.             |
| Frequency / Presence / Repeat | Active penalty controls.                         |
| Repeat Window                 | Repetition penalty window.                       |
| Max Tokens                    | Active maximum generation token setting.         |
| Context                       | Active context window.                           |
| Threads                       | Active CPU thread count.                         |
| Semantic                      | Whether semantic context is enabled.             |
| Docs                          | Number of shared basic document context entries. |

## 📦 Requirements

The table below reflects the active imports and runtime features used by the current application.
Use
`requirements.txt` as the installation source of truth when version pins are present.

| Requirement           | Package / Import        | Purpose                                                                                                               | Used By                                                            |
| --------------------- | ----------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Python                | `python>=3.10`          | Runtime for modern type hints and Streamlit execution.                                                                | Entire application                                                 |
| Streamlit             | `streamlit`             | Web UI framework, chat UI, file upload, dataframes, tabs, expanders, metrics, and session state.                      | All modes                                                          |
| llama-cpp-python      | `llama_cpp`             | Local GGUF model loading and inference.                                                                               | Text Generation and Document Q&A generation path                   |
| NumPy                 | `numpy`                 | Vector math, cosine similarity, embedding arrays, decoded vector blobs.                                               | Document Q&A, Semantic Search                                      |
| Pandas                | `pandas`                | Dataframes, Excel import, SQL query results, prompt tables, asset inventory, visualization source data.               | Prompt Engineering, Data Management, Document Q&A, Semantic Search |
| Plotly Express        | `plotly.express`        | Interactive visualizations over SQLite table data.                                                                    | Data Management Visualize tab                                      |
| SQLite                | `sqlite3`               | Local persistence for chat history, prompts, embeddings, documents, chunks, images, imported tables, and SQL console. | All persistence workflows                                          |
| sqlite-vec            | `sqlite_vec`            | Optional vector-table backend for document retrieval.                                                                 | Document Q&A vector search                                         |
| sentence-transformers | `sentence_transformers` | Local embedding model loading through `SentenceTransformer('all-MiniLM-L6-v2')`.                                      | Document Q&A and Semantic Search                                   |
| PyMuPDF               | `fitz` / `pymupdf`      | Native PDF text extraction and PDF preview support where available.                                                   | Document Q&A document parsing                                      |
| OpenPyXL              | `openpyxl`              | Excel workbook reading through pandas.                                                                                | Data Management Import                                             |
| python-docx           | `python-docx`           | DOCX text extraction when supported by document parsing helpers.                                                      | Document Q&A uploads                                               |
| Pillow                | `pillow`                | Image metadata and image-handling support.                                                                            | Images API and Data Management image registration                  |
| urllib                | Python standard library | Public HTTP/HTTPS fetches for guarded web context.                                                                    | Gipity Tools and Web Browsing                                      |
| ipaddress             | Python standard library | Blocks private, local, loopback, reserved, and link-local web targets.                                                | Gipity Tools and Web Browsing                                      |
| socket                | Python standard library | Resolves hostnames before web fetch safety checks.                                                                    | Gipity Tools and Web Browsing                                      |
| json                  | Python standard library | Parses function-call JSON.                                                                                            | Function Calling                                                   |
| regex / re            | `re` / optional `regex` | Prompt conversion, SQL safety checks, identifier sanitization, HTML cleanup, and text normalization.                  | Utilities, Prompt Engineering, Data Management, Web Browsing       |
| hashlib               | Python standard library | Stable fingerprints for documents, chunks, and uploaded image metadata.                                               | Document Q&A and AI asset governance                               |
| pathlib               | Python standard library | Model path and filesystem path handling.                                                                              | Model loading and local paths                                      |
| base64                | Python standard library | Image/base64 helper support.                                                                                          | UI/image utilities                                                 |

## 🔒 Privacy, Safety, and Design Philosophy

| Principle                       | Implementation                                                                                 |
| ------------------------------- | ---------------------------------------------------------------------------------------------- |
| Local-first inference           | Text generation runs through local GGUF models when configured paths are available.            |
| Local persistence               | Chat history, prompts, embeddings, documents, chunks, and image metadata are stored in SQLite. |
| Inspectable retrieval           | Retrieved document and semantic chunks can be shown before or after answers.                   |
| Grounding controls              | Document Q&A can require grounding and answer only from excerpts.                              |
| Compact-model safety            | Buddy uses smaller retrieval defaults to avoid overloading a 270M model.                       |
| SQL safety                      | SQL console blocks mutation statements and permits guarded read-only query forms.              |
| Tool safety                     | Function Calling executes only allowlisted local functions.                                    |
| Web safety                      | Web browsing permits public HTTP/HTTPS only and blocks private/local network targets.          |
| Multimodal fail-closed behavior | Image and Audio modes display runtime-status messages unless actual adapters are wired.        |
| Operational transparency        | Footer summarizes active mode and generation parameters.                                       |

## 🧬 Related Applications

| Application | Role                                                                                       |
| ----------- | ------------------------------------------------------------------------------------------ |
| Leeroy      | Small instruction assistant for local dialogue and retrieval workflows.                    |
| Bro         | Balanced local instruction and reasoning assistant.                                        |
| Gipity      | Larger reasoning-oriented assistant with function-calling and web-context workflows.       |
| Buddy       | Compact local assistant for lightweight text and conservative retrieval.                   |
| Boo         | Lightweight reasoning assistant based on Phi-style local inference.                        |
| Jimi        | Multimodal-capable local assistant for text, image, audio, coding, and thinking workflows. |
| Nisty       | Governance/document-oriented multimodal assistant.                                         |
| Chonky      | Text-processing, tokenization, embeddings, and vector-persistence pipeline.                |

## 📜 License

This application is provided for personal, research, and open-source use. Refer to the project and
model repositories for application and model-specific licensing terms.

## 📜 License

This application is provided for personal, research, and open-source use. Refer to the project and
model repositories for application and model-specific licensing terms.

