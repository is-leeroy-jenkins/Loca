###### Bro

![](https://github.com/is-leeroy-jenkins/Bro/blob/main/resources/images/bro_project.png)

<p align="center">
  <a href="#-key-features">Features</a> ·
  <a href="#-application-modes">Modes</a> ·
  <a href="#-architecture">Architecture</a> ·
  <a href="#-repository-structure">Structure</a> ·
  <a href="#-installation--setup">Install</a> ·
  <a href="#-configuration">Configuration</a> ·
  <a href="#-text-generation">AI</a> ·
  <a href="#-document-qa">RAG</a> ·
  <a href="#-semantic-search">Search</a> ·
  <a href="#-prompt-engineering">Prompts</a> ·
  <a href="#-data-management">Data</a> ·
  <a href="#-requirements">Requirements</a> ·
</p>


Bro is a local-first Streamlit application for text generation, document-grounded retrieval,
semantic search, prompt engineering, and SQLite-backed data management. It is designed to run a
GGUF language model through `llama-cpp-python` while giving analysts direct control over inference
parameters, prompt templates, document context, retrieval behavior, semantic chunking, and local
application data.

## 🎥 Demo

![](https://github.com/is-leeroy-jenkins/Bro/blob/main/resources/images/bro-demo.gif)



## 🕸️ Streamlit (Web)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://bro-py.streamlit.app/)
![](https://github.com/is-leeroy-jenkins/Bro/blob/main/resources/images/Bro-streamlit.gif)
- A Python framework to build dynamic, interactive web applications.


## 🧱 Databricks
[![Bro](https://img.shields.io/badge/Databricks-Bro-FF3621?logo=databricks&logoColor=white)](https://dbc-a0c21f80-7bb3.cloud.databricks.com/browse/folders/3169291152440505?o=7474645703081351)
- A data engineering, analytics, and artificial intelligence collaborative workspace
- Codebase
  

## 🧠 Custom LLM

[![](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/leeroy-jankins/bro)
- Fine-tuned
- Post-trained
  
## ✨ Key Features

| Feature                 | Description                                                                                                                                                |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Local GGUF inference    | Runs a configured GGUF model through `llama-cpp-python` with context-window, CPU-thread, token, temperature, top-p, top-k, and repeat-penalty controls.    |
| Text generation         | Chat, reasoning, coding, translation, summarization, and extraction presets with configurable response format.                                             |
| Persistent chat history | Saves role/content chat messages to SQLite and reloads history on startup.                                                                                 |
| System instructions     | Supports reusable system instructions, prompt templates, preset instructions, and XML ↔ Markdown conversion.                                               |
| Document Q&A            | Upload PDFs, TXT files, or DOCX files and ask grounded questions over retrieved excerpts.                                                                  |
| Document actions        | Summarize active documents, extract key points, generate outlines, extract entities, extract tables, and compare active documents.                         |
| Semantic search         | Build a semantic index from uploaded files, query indexed chunks, select relevant chunks, and route selected context into Text Generation or Document Q&A. |
| sqlite-vec support      | Uses sqlite-vec for document Q&A vector search when available, with fallback cosine similarity over stored vectors.                                        |
| Prompt engineering      | Search, page, sort, edit, clone, generate, and apply prompt templates stored in the local `Prompts` table.                                                 |
| Data management         | Import Excel workbooks into SQLite, browse tables, run CRUD, explore, filter, aggregate, visualize, administer schema, and run guarded read-only SQL.      |
| AI asset governance     | Register active documents, chunks, embeddings, and uploaded image metadata into local governance tables.                                                   |
| Fixed status footer     | Displays active mode, model parameters, semantic state, context settings, and document count.                                                              |

## 🧭 Application Modes

Bro is built around a practical local-AI workflow:

* Run instruction-following and reasoning tasks with a local GGUF model.
* Upload PDF, TXT, or DOCX documents for local document question answering.
* Build a semantic index from uploaded documents using sentence-transformers.
* Store chat history, prompts, embeddings, document metadata, chunks, and image metadata in SQLite.
* Use Prompt Engineering mode to create, edit, clone, search, and apply prompt templates.
* Use Data Management mode to import Excel data, browse tables, run guarded SQL, profile data, and
  administer local AI-asset tables.
  
The current `app.py` exposes these sidebar modes through `cfg.MODES`.

| Mode                   | Purpose                                                                  | Major Controls / Outputs                                                                                                                                                                                           |
| ---------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Text Generation**    | Primary local chat and generation interface.                             | Task presets, response format, chat history, document context, reasoning controls, coding controls, inference parameters, context window, CPU threads, system instructions, prompt preview, chat history.          |
| **Document Q&A**       | Retrieval-augmented question answering over uploaded documents.          | PDF/TXT/DOCX upload, active document inventory, retrieval controls, chunk size/overlap, grounding toggles, sqlite-vec toggle, fallback cosine search, document actions, parsing controls, retrieved chunk display. |
| **Semantic Search**    | Build and query a reusable semantic chunk index.                         | Upload files, chunk size/overlap, clear/append behavior, top-k search, similarity threshold, selected chunks, send-to-text, send-to-Document-Q&A, save-as-context actions.                                         |
| **Prompt Engineering** | Manage reusable prompt templates and cascade them into generation modes. | Search, category inference, sort, pagination, go-to-ID, prompt table, apply to Text Generation, apply to Document Q&A, clone, starter prompt generator, edit/create/delete.                                        |
| **Data Management**    | Manage local SQLite data and AI asset metadata.                          | Excel import, table browse, CRUD, profile/explore, filter, aggregate, visualize, schema admin, index creation, AI asset governance, read-only SQL console.                                                         |

## 🏛 Architecture

```text
Streamlit UI
    │
    ├── Text Generation
    │       ├── llama-cpp-python
    │       ├── local GGUF model
    │       ├── chat_history table
    │       └── Prompts table
    │
    ├── Document Q&A
    │       ├── uploaded PDFs / TXT / DOCX
    │       ├── PyMuPDF / text extraction
    │       ├── sentence-transformers embeddings
    │       ├── sqlite-vec vector table when available
    │       └── cosine fallback search
    │
    ├── Semantic Search
    │       ├── chunked uploaded documents
    │       ├── embeddings table
    │       ├── selected semantic context
    │       └── context routing into Text Generation / Document Q&A
    │
    ├── Prompt Engineering
    │       ├── Prompts table
    │       ├── template metadata
    │       ├── starter prompt generation
    │       └── system-instruction cascade
    │
    └── Data Management
            ├── Excel import
            ├── CRUD / profiling / visualization
            ├── guarded SQL console
            └── AI asset governance tables
```

## 🗂 Repository Structure

```text
bro/
├─ app.py                     # Main Streamlit application
├─ config.py                  # Model path, app constants, modes, labels, defaults, and styling
├─ requirements.txt           # Python dependencies
├─ resources/
│  └─ images/
│     ├─ bro_project.png
│     ├─ Bro-streamlit.gif
│     └─ bro_logo.png
├─ stores/
│  └─ sqlite/
│     └─ bro.db               # Chat history, prompts, embeddings, documents, chunks, images
├─ models/
│  └─ bro-3-1b-it-Q4_K_M.gguf # Optional/local model location depending on config
└─ README.md
```

## ⚙️ System Requirements

| Requirement      |                               Minimum |                                                   Recommended |
| ---------------- | ------------------------------------: | ------------------------------------------------------------: |
| Operating system | Windows 10/11 64-bit, Linux, or macOS |                                    Windows 11 64-bit or Linux |
| Python           |                                  3.10 |                                                          3.11 |
| RAM              |                                  8 GB |                                                 16 GB or more |
| CPU              |                        Modern x64 CPU |                          AVX2-capable CPU with multiple cores |
| Storage          |                           5–7 GB free | 10+ GB free for models, SQLite assets, and uploaded documents |
| GPU              |                          Not required |       Optional; current app is CPU-oriented through llama.cpp |

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/is-leeroy-jenkins/Bro.git
cd Bro
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

## 📥 Download Lil Bro

1. Open the Hugging Face model repository:

   [![](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/leeroy-jankins/bro)

2. Download the GGUF model file used by your configuration. The README historically references:

   ```text
   bro-3-1b-it-Q4_K_M.gguf
   ```

3. Place the file at the path expected by `cfg.MODEL_PATH`, or update the configuration to point to
   the model location.

4. Confirm that the model exists before launching Bro. The application checks model availability
   through `Path(cfg.MODEL_PATH).exists()`.

## ▶️ Running Bro

Run Streamlit through Python so the correct virtual environment is used.

```bash
python -m streamlit run app.py
```

The application opens in wide layout, renders the Bro subtitle, and exposes the mode selector in the
sidebar under **⚙️ Application Mode**.

## 🔧 Configuration

Bro reads runtime configuration from `config.py` and session state. The most important values are
listed below.

| Configuration Item           | Purpose                                                                                 |
| ---------------------------- | --------------------------------------------------------------------------------------- |
| `cfg.MODEL_PATH`             | Path to the local GGUF model loaded by llama.cpp.                                       |
| `cfg.DEFAULT_CTX`            | Default model context window.                                                           |
| `cfg.CORES`                  | Maximum CPU thread count exposed in the UI.                                             |
| `cfg.DB_PATH`                | SQLite database path used for chat, prompts, embeddings, documents, chunks, and images. |
| `cfg.FAVICON`                | Streamlit page icon.                                                                    |
| `cfg.LOGO`                   | Sidebar logo.                                                                           |
| `cfg.APP_SUBTITLE`           | Startup caption displayed below the page setup.                                         |
| `cfg.MODES`                  | Sidebar application mode list.                                                          |
| `cfg.BLUE_DIVIDER`           | Shared divider styling.                                                                 |
| `cfg.XML_BLOCK_PATTERN`      | XML-like prompt delimiter pattern used by conversion utilities.                         |
| `cfg.TEXT_GENERATION`        | Help text for Text Generation mode.                                                     |
| `cfg.RETRIEVAL_AUGMENTATION` | Help text for Document Q&A mode.                                                        |
| `cfg.SEMANTIC_SEARCH`        | Help text for Semantic Search mode.                                                     |
| `cfg.PROMPT_ENGINEERING`     | Help text for Prompt Engineering mode.                                                  |
| `cfg.DATA_MANAGEMENT`        | Help text for Data Management mode.                                                     |

### Optional Environment Variable Pattern

If `config.py` resolves `cfg.MODEL_PATH` from an environment variable, use a value similar to this:

```powershell
$env:BRO_LLM_PATH="C:\Users\you\models\bro-3-1b-it-Q4_K_M.gguf"
```

Restart the terminal or IDE after changing environment variables.

## 💬 Text Generation

Text Generation is the primary local chat interface. It uses chat history, system instructions,
task-specific prompt blocks, optional semantic context, and optional document context to construct a
llama.cpp-compatible prompt.

| Control Group        | Controls                                                                                                                       |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Task Preset          | Chat, Reasoning, Coding, Translation, Summarization, Extraction.                                                               |
| Response Format      | Plain Text, Markdown, Bullet Summary, JSON.                                                                                    |
| Conversation Context | Use Conversation History, Use Document Context.                                                                                |
| Reasoning Controls   | Reasoning Depth, Answer Only, Use Self-Check, Prefer Deterministic Reasoning.                                                  |
| Coding Controls      | Code Language, Coding Task, Include Comments, Use Editor Format, Emit Fenced Code, Translation Target Language.                |
| Response Controls    | Temperature, Top-P, Top-K, Use Grounding.                                                                                      |
| Inference Settings   | Repeat Window, Repeat Penalty, Presence Penalty, Frequency Penalty.                                                            |
| Context Controls     | Context Window, CPU Threads, Max Tokens, Random Seed.                                                                          |
| System Instructions  | Free-text instruction area, template selector, clear, XML ↔ Markdown conversion, preset application, effective prompt preview. |

## 📚 Document Q&A

Document Q&A provides local retrieval-augmented answering over uploaded files. The app accepts PDF,
TXT, and DOCX uploads, extracts text, chunks documents, indexes chunks, retrieves relevant excerpts,
and routes document-grounded prompts through the same local generation pipeline.

| Component                 | Description                                                                                                                                   |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Document Loader           | Upload one or more PDF, TXT, or DOCX documents and select active documents.                                                                   |
| Preview                   | Renders PDFs with `st.pdf` when possible or shows extracted text previews.                                                                    |
| Active Document Inventory | Displays file name, byte size, text length, chunk count, and loaded state.                                                                    |
| Retrieval Controls        | Chunks to retrieve, chunk size, chunk overlap, show retrieved chunks.                                                                         |
| Grounding Controls        | Require grounding and answer from excerpts only.                                                                                              |
| Search Backend            | Prefer sqlite-vec, with optional fallback cosine search.                                                                                      |
| Document Actions          | Answer Question, Summarize Active Document, Extract Key Points, Generate Outline, Extract Entities, Extract Tables, Compare Active Documents. |
| Parsing Controls          | Enable OCR, prefer native PDF text, include page markers, show diagnostics.                                                                   |

## 🔍 Semantic Search

Semantic Search builds a reusable semantic index from uploaded PDF, TXT, or DOCX files. It stores
chunk vectors in the local SQLite `embeddings` table and allows ranked semantic retrieval.

| Section        | Controls / Outputs                                                                                                      |
| -------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Index Builder  | Upload files, chunk size, chunk overlap, clear existing index, append to existing index, show diagnostics, build index. |
| Diagnostics    | Indexed documents, indexed chunks, vector dimension.                                                                    |
| Semantic Query | Query text, top-k results, minimum similarity, group by document.                                                       |
| Results        | Selectable rows with rank, score, chunk text, and length.                                                               |
| Actions        | Send selected chunks to Text Generation, send selected chunks to Document Q&A, save selected chunks as prompt context.  |
| Maintenance    | Delete index, recompute diagnostics, clear query results.                                                               |

## 📝 Prompt Engineering

Prompt Engineering manages reusable prompt templates in SQLite. It can search, sort, page, edit,
clone, generate, and cascade prompts into other modes.

| Capability         | Description                                                                                                                                        |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Search and filter  | Search captions, names, and prompt text; infer categories from prompt content.                                                                     |
| Pagination         | Browse prompt records 10 at a time with previous/next controls.                                                                                    |
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

The application initializes these core tables when the database is created.

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

Bro renders a fixed footer showing the current operating state.

| Footer Item                   | Description                                      |
| ----------------------------- | ------------------------------------------------ |
| Mode                          | Current application mode.                        |
| Temp / Top-P / Top-K          | Active generation sampling settings.             |
| Frequency / Presence / Repeat | Active penalty controls.                         |
| Repeat Window                 | Repetition penalty window.                       |
| Max Tokens                    | Active maximum generation token setting.         |
| Context                       | Active context window.                           |
| Threads                       | Active CPU thread count.                         |
| Semantic                      | Whether semantic context is enabled.             |
| Docs                          | Number of shared basic document context entries. |

## 📦 Requirements

The table below reflects the active imports and runtime features used by the current `app.py`. Use
`requirements.txt` as the installation source of truth when version pins are present.

| Requirement           | Package / Import        | Purpose                                                                                                               | Used By                                                             |
| --------------------- | ----------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Python                | `python>=3.10`          | Runtime for modern type hints and Streamlit execution.                                                                | Entire application.                                                 |
| Streamlit             | `streamlit`             | Web UI framework, chat UI, file upload, dataframes, tabs, expanders, metrics, and session state.                      | All modes.                                                          |
| llama-cpp-python      | `llama_cpp`             | Local GGUF model loading and inference.                                                                               | Text Generation and Document Q&A generation path.                   |
| NumPy                 | `numpy`                 | Vector math, cosine similarity, embedding arrays, decoded vector blobs.                                               | Document Q&A, Semantic Search.                                      |
| Pandas                | `pandas`                | Dataframes, Excel import, SQL query results, prompt tables, asset inventory, visualization source data.               | Prompt Engineering, Data Management, Document Q&A, Semantic Search. |
| Plotly Express        | `plotly.express`        | Interactive visualizations over SQLite table data.                                                                    | Data Management Visualize tab.                                      |
| SQLite                | `sqlite3`               | Local persistence for chat history, prompts, embeddings, documents, chunks, images, imported tables, and SQL console. | All persistence workflows.                                          |
| sqlite-vec            | `sqlite_vec`            | Optional vector-table backend for document retrieval.                                                                 | Document Q&A vector search.                                         |
| sentence-transformers | `sentence_transformers` | Local embedding model loading through `SentenceTransformer('all-MiniLM-L6-v2')`.                                      | Document Q&A and Semantic Search.                                   |
| PyMuPDF               | `fitz` / `pymupdf`      | Native PDF text extraction and PDF preview support where available.                                                   | Document Q&A document parsing.                                      |
| OpenPyXL              | `openpyxl`              | Excel workbook reading through pandas.                                                                                | Data Management Import.                                             |
| python-docx           | `python-docx`           | DOCX text extraction when supported by document parsing helpers.                                                      | Document Q&A uploads.                                               |
| Pillow                | `pillow`                | Image metadata/handling support for uploaded images.                                                                  | Data Management image registration.                                 |
| Requests / HTTPX      | `requests`, `httpx`     | Supporting HTTP client dependencies for provider or package internals.                                                | Package support.                                                    |
| pydantic              | `pydantic`              | Structured validation used by modern SDKs and dependency stack.                                                       | Dependency support.                                                 |
| typing-extensions     | `typing_extensions`     | Backported typing support.                                                                                            | Dependency support.                                                 |
| regex / re            | `re` / optional `regex` | Prompt conversion, SQL safety checks, identifier sanitization, and text normalization.                                | Utilities, Prompt Engineering, Data Management.                     |
| hashlib               | Python standard library | Stable fingerprints for documents, chunks, and uploaded image metadata.                                               | Document Q&A and AI asset governance.                               |
| pathlib               | Python standard library | Model path and filesystem path handling.                                                                              | Model loading and local paths.                                      |
| base64                | Python standard library | Image/base64 helper support.                                                                                          | UI/image utilities.                                                 |

## 🔒 Privacy & Design Philosophy

| Principle                | Implementation                                                                                 |
| ------------------------ | ---------------------------------------------------------------------------------------------- |
| Local-first inference    | Text generation runs through a local GGUF model when `cfg.MODEL_PATH` is available.            |
| Local persistence        | Chat history, prompts, embeddings, documents, chunks, and image metadata are stored in SQLite. |
| Inspectable retrieval    | Retrieved document chunks can be shown before or after answers.                                |
| Grounding controls       | Document Q&A can require grounding and answer only from excerpts.                              |
| SQL safety               | SQL console blocks mutation statements and permits read-only query forms.                      |
| Operational transparency | Footer summarizes active mode and generation parameters.                                       |

## 🧬 Related Applications

| Application | Role                                                                        |
| ----------- | --------------------------------------------------------------------------- |
| Leeroy      | Entry-level instruction assistant.                                          |
| Bro         | Local, balanced instruction and reasoning assistant.                        |
| Gipity      | Larger multimodal/OpenAI-centered workflow application.                     |
| Chonky      | Text-processing, tokenization, embeddings, and vector-persistence pipeline. |

## 📜 License

This application is provided for personal, research, and open-source use. Refer to the project and
model repositories for application and model-specific licensing terms.

