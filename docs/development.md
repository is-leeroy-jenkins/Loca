# Development

This page documents the development, validation, documentation, and deployment workflow for Loca. It
is intended for maintainers who update the Streamlit application, modify the local model registry,
add document workflows, adjust SQLite persistence, or maintain the MkDocs documentation site.

## 🧭 Purpose

The development workflow protects Loca from regressions while keeping the documentation
source-driven and MkDocs-compatible.

Development work should preserve:

* local model runtime behavior;
* Streamlit session-state keys;
* model and mode registry contracts;
* SQLite schema compatibility;
* document extraction and retrieval behavior;
* prompt engineering persistence;
* exception logging through `boogr.py`;
* MkDocs API rendering through Google-style docstrings.

## 🧱 Project Structure

A typical Loca project structure should resemble:

```text
Loca/
├── app.py
├── config.py
├── boogr.py
├── mkdocs.yml
├── requirements.txt
├── resources/
│   └── images/
│       ├── favicon.ico
│       └── loca-llama_logo.png
├── stores/
│   └── sqlite/
│       └── loca.db
├── logging/
│   └── Exceptions.db
└── docs/
    ├── index.md
    ├── architecture.md
    ├── development.md
    ├── images/
    │   └── loca-architecture-dark.png
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

The Python source files are the source of truth for API documentation. The Markdown pages explain
workflows, architecture, usage, and maintenance.

## ⚙️ Environment Setup

Create and activate a virtual environment before installing dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the application and documentation dependencies.

```powershell
pip install streamlit pandas numpy plotly mkdocs mkdocs-material mkdocstrings[python] pymdown-extensions
```

Install optional document and local model dependencies as needed.

```powershell
pip install python-docx PyMuPDF llama-cpp-python
```

Use `llama-cpp-python` only when local GGUF inference is required in the active environment.

## 🧠 Model Registry Development

Model configuration is centralized in `config.py`.

When adding or modifying a model, update the model registry entry with:

| Field           | Purpose                                                |
| --------------- | ------------------------------------------------------ |
| `path`          | GGUF model path, usually from an environment variable. |
| `logo`          | Logo path used by Streamlit branding.                  |
| `modes`         | Supported application modes.                           |
| `family`        | Model family label.                                    |
| `model_name`    | User-facing model name.                                |
| `size`          | Model size label.                                      |
| `base_model`    | Base model identifier used for capability checks.      |
| `chat_template` | Prompt template style.                                 |
| `description`   | Short user-facing description.                         |

Use environment variables for model paths. Do not hard-code local machine paths into `config.py`.

Example:

```python
BRO_LLM_PATH = os.getenv( 'BRO_LLM_PATH', '' )
```

## 🔀 Mode Contract Development

The mode contract is defined through constants in `config.py`.

```python
TEXT_MODE = 'Text Generation'
IMAGE_MODE = 'Images API'
AUDIO_MODE = 'Audio API'
DOCQNA_MODE = 'Document Q&A'
SEMANTIC_MODE = 'Semantic Search'
PROMPT_MODE = 'Prompt Engineering'
DATA_MODE = 'Data Management'
```

When adding a new mode:

1. Add a mode constant.
2. Add the mode to the appropriate model registry entries.
3. Initialize required session-state keys before the UI reads them.
4. Add the mode panel in `app.py`.
5. Add user-guide documentation.
6. Add or update API documentation if new functions are introduced.
7. Run the full validation workflow.

Do not reuse existing session-state keys for unrelated data. Each workflow should have a clear state
contract.

## 🗂️ Session-State Development Rules

Streamlit reruns the script frequently. Session-state keys must be initialized before they are read.

Required practices:

| Rule                                                    | Reason                                               |
| ------------------------------------------------------- | ---------------------------------------------------- |
| Initialize keys early                                   | Prevents missing-key failures during reruns.         |
| Use workflow-specific prefixes                          | Prevents accidental state collisions.                |
| Preserve widget-owned keys                              | Avoids Streamlit widget state errors.                |
| Do not clear upstream state unexpectedly                | Prevents unrelated workflows from losing data.       |
| Keep model-derived state separate from user-owned state | Prevents model changes from overwriting user inputs. |

Examples of useful prefixes:

| Prefix        | Workflow             |
| ------------- | -------------------- |
| `docqna_`     | Document Q&A         |
| `semantic_`   | Semantic Search      |
| `pe_`         | Prompt Engineering   |
| `dm_`         | Data Management      |
| `image_`      | Image workflow       |
| `audio_`      | Audio workflow       |
| `web_browse_` | Web context workflow |

## 📄 Document Workflow Development

Document-related development should preserve safe fallback behavior.

When changing document extraction or retrieval:

1. Confirm optional dependency checks remain intact.
2. Confirm unsupported file types fail safely.
3. Confirm extracted text is not silently discarded.
4. Confirm chunk size and overlap controls are respected.
5. Confirm document fingerprints remain stable.
6. Confirm retrieval diagnostics still show useful state.
7. Confirm Document Q&A and Semantic Search do not overwrite each other’s state.

Optional dependency behavior should be explicit. If `python-docx` or `PyMuPDF` is missing, the
application should report the unavailable capability rather than crash.

## 🔎 Retrieval Development

Retrieval development affects Document Q&A, Semantic Search, and Text Generation context injection.

Preserve these behaviors:

| Behavior            | Development Rule                                    |
| ------------------- | --------------------------------------------------- |
| Chunking            | Validate chunk size and overlap before use.         |
| Embedding           | Confirm vector dimensions before comparing vectors. |
| Similarity          | Avoid comparing incompatible vector shapes.         |
| Fallback retrieval  | Keep fallback behavior explicit and diagnostic.     |
| Context injection   | Keep retrieved context bounded.                     |
| Small model support | Use compact retrieval profiles where appropriate.   |

When changing retrieval defaults, test with both short and long documents.

## 🧰 Prompt Engineering Development

Prompt Engineering uses SQLite-backed prompt records and shared system-instruction state.

When modifying prompt features:

1. Preserve the `Prompts` table contract.
2. Preserve prompt selector behavior.
3. Preserve template application to shared system instructions.
4. Preserve XML-to-Markdown conversion behavior.
5. Preserve prompt metadata synchronization.
6. Avoid overwriting user-edited prompt text during reruns.
7. Confirm saved prompts reappear after app restart.

Prompt text can be long and user-authored. Do not write prompt text into exception method strings or
diagnostic identifiers.

## 🗃️ SQLite Development

Loca uses SQLite for local persistence. Database initialization should be idempotent.

Core development rules:

| Rule                                      | Reason                                               |
| ----------------------------------------- | ---------------------------------------------------- |
| Use `CREATE TABLE IF NOT EXISTS`          | Prevents startup failures when tables already exist. |
| Keep schema changes backward-aware        | Protects existing local databases.                   |
| Use parameterized SQL                     | Prevents malformed queries and unsafe interpolation. |
| Keep connections bounded                  | Avoids long-lived database locks.                    |
| Commit write operations                   | Ensures records persist.                             |
| Do not store secrets in diagnostic tables | Protects local configuration data.                   |

Before changing schema, inspect dependent functions and user-guide pages.

## 🚨 Exception Logging Development

Loca uses `boogr.py` for structured exception logging.

Use this pattern in operational exception handlers:

```python
except Exception as e:
    exception = Error( e )
    exception.module = 'app'
    exception.cause = 'WorkflowName'
    exception.method = 'function_name( arg: type ) -> return_type'
    Logger( ).write( exception )
    raise exception
```

For failure-safe helpers that intentionally return a default value, log only when diagnostics are
valuable. Do not turn every small fallback into a raised exception.

Never include the following in `exception.method`:

* prompt text;
* user input;
* uploaded document text;
* dataframe contents;
* embedding vectors;
* file contents;
* API keys;
* tokens;
* full local paths;
* object memory addresses.

The method string should be a stable signature only.

## 🧾 Documentation Comment Standard

All public Python modules, classes, functions, methods, and properties should use Google-style
docstrings compatible with MkDocs, mkdocstrings, and griffe.

Use these section names only when applicable:

```text
Purpose:
Args:
Attributes:
Returns:
Raises:
Notes:
Examples:
```

Do not use underline-style sections:

```text
Purpose:
--------
Parameters:
-----------
Returns:
--------
```

Do not document `self` or `cls`.

Do not add `Returns:` sections to `__init__`.

Do not add `Returns:` sections to functions that return no meaningful value.

Correct:

```python
def get_model_path( model_name: str ) -> str:
    """Return the GGUF path for a selected local model.

    Purpose:
        Resolves the configured model path from the central model registry so model loading,
        UI diagnostics, and selected-model session state use one authoritative path source.

    Args:
        model_name: Selected local model name.

    Returns:
        str: Resolved GGUF model path.
    """
```

Incorrect:

```python
def get_model_path( model_name: str ) -> str:
    """
    Purpose:
    --------
    Gets model path.

    Parameters:
    -----------
    model_name : str

    Returns:
    --------
    str
    """
```

## 🧪 Source Validation

Run source validation after editing Python files.

```powershell
python -m py_compile .\app.py
python -m py_compile .\config.py
python -m py_compile .\boogr.py
```

Then run a project-wide compile check.

```powershell
python -m compileall .
```

Optional import checks:

```powershell
python -c "import config; print('config ok')"
python -c "import boogr; print('boogr ok')"
```

Be careful importing `app.py` directly because Streamlit applications may execute top-level UI code
during import.

## 🧩 MkDocs API Pages

API pages should be source-driven and minimal.

`docs/api/app.md`:

```markdown
# App API

::: app
```

`docs/api/config.md`:

```markdown
# Configuration API

::: config
```

`docs/api/boogr.md`:

```markdown
# Logging API

::: boogr
```

Manual prose belongs in architecture and user-guide pages. API pages should let mkdocstrings render
the source docstrings.

## 🏗️ MkDocs Build

Build the documentation site from the project root.

```powershell
mkdocs build
```

Serve the site locally.

```powershell
mkdocs serve
```

Then open the local server URL shown in the terminal.

## 🔬 MkDocs Warning Triage

Treat griffe and MkDocs warnings as defects.

| Warning                                  | Likely Cause                                                    | Fix                                                                                |
| ---------------------------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Page exists but is not in `nav`          | Markdown file not listed in `mkdocs.yml`                        | Add the page to `nav` or remove the unused file.                                   |
| Nav page not found                       | `mkdocs.yml` references a missing file                          | Create the file or remove the nav entry.                                           |
| Failed to get `name: description` pair   | Malformed `Args:` or `Attributes:` entry                        | Rewrite entries as `name: Description.`                                            |
| No type or annotation for returned value | Return section lacks explicit type or function lacks annotation | Add a return annotation or explicit return type in `Returns:`.                     |
| Import failure during API build          | mkdocstrings cannot import the module                           | Run direct import checks and fix missing dependencies or top-level runtime errors. |
| Broken image link                        | Image path does not exist under `docs/`                         | Move the image or correct the Markdown link.                                       |

Do not hide API pages to suppress warnings. Repair the source docstrings or import issue.

## 🎨 CSS and JavaScript Development

The documentation site uses custom assets:

```text
docs/assets/css/loca.css
docs/assets/js/loca.js
```

The CSS provides:

* dark-mode theme overrides;
* dark blue header and tab bar;
* wider documentation content;
* styled tables;
* styled API reference blocks;
* code block styling;
* image and diagram styling;
* print styling;
* responsive layout adjustments.

The JavaScript provides:

* API search/filter tools;
* expand/collapse controls;
* heading copy links;
* page copy and print buttons;
* table filters;
* code labels;
* long-code expand/collapse;
* reading progress;
* scroll-to-top control.

After changing CSS or JavaScript, run:

```powershell
mkdocs serve
```

Then hard-refresh the browser to avoid stale cached assets.

## 🖼️ Diagram Assets

Architecture and class-map diagrams should be stored under:

```text
docs/images/
```

Recommended files:

```text
docs/images/loca-architecture-dark.png
docs/images/loca-class-map-dark.png
```

Reference the architecture diagram from `architecture.md`:

```markdown
![Loca Architecture](images/loca-architecture-dark.png)
```

Use relative paths from the Markdown file location.

For `docs/architecture.md`, use:

```markdown
![Loca Architecture](images/loca-architecture-dark.png)
```

For a page under `docs/user-guide/`, use:

```markdown
![Loca Architecture](../images/loca-architecture-dark.png)
```

## ✅ Development Checklist

Before committing changes, confirm:

| Check                                                       | Status   |
| ----------------------------------------------------------- | -------- |
| Python files compile                                        | Required |
| Public docstrings are Google-style                          | Required |
| No underline-style docstring sections remain                | Required |
| No `Returns: None` sections exist for procedures            | Required |
| Logging pattern uses stable method signatures               | Required |
| No prompt text or file content is logged as method metadata | Required |
| Session-state keys are initialized before use               | Required |
| Model registry entries use environment-based paths          | Required |
| API pages exist for documented modules                      | Required |
| Every docs page is listed in `mkdocs.yml`                   | Required |
| Images referenced by Markdown exist under `docs/`           | Required |
| CSS and JavaScript paths exist                              | Required |
| `mkdocs build` succeeds                                     | Required |

## 🚀 GitHub Pages Deployment

After the documentation builds locally, deploy with:

```powershell
mkdocs gh-deploy --force
```

This publishes the generated site to the repository’s `gh-pages` branch.

In GitHub repository settings:

1. Open **Settings**.
2. Open **Pages**.
3. Set the source to **Deploy from a branch**.
4. Select the `gh-pages` branch.
5. Select `/ (root)`.
6. Save the setting.

The repository README can link to the published documentation site after GitHub Pages is active.

## 🛠️ Common Development Problems

| Problem                               | Likely Cause                                                           | Fix                                                     |
| ------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------- |
| Streamlit widget state error          | Widget-owned key modified after widget creation                        | Move state mutation before widget creation.             |
| Model mode resets unexpectedly        | Derived model state overwrites user-owned selection                    | Separate selected state from derived state.             |
| API page fails to render              | Module import fails during mkdocstrings build                          | Fix import dependency or top-level runtime execution.   |
| Logo does not appear                  | Path is outside `docs/` for MkDocs or missing at runtime for Streamlit | Use correct path for the target environment.            |
| Griffe warnings persist               | Docstring section is still malformed                                   | Repair the exact function named in the warning.         |
| Document Q&A loses files              | Session-state key was cleared during rerun                             | Preserve upload state unless user explicitly clears it. |
| Semantic Search returns stale results | Old index was appended instead of cleared                              | Clear or rebuild the semantic index.                    |
| Prompt selector is empty              | `Prompts` table is empty or captions are missing                       | Save a prompt with a valid caption.                     |
| Exception log is empty                | No exceptions have been logged or path/table differs                   | Confirm `LOG_PATH` and `LOG_FILE`.                      |

## 🔗 Related Pages

* [Architecture](architecture.md)
* [Text Generation](user-guide/text-generation.md)
* [Document Q&A](user-guide/document-qna.md)
* [Semantic Search](user-guide/semantic-search.md)
* [Prompt Engineering](user-guide/prompt-engineering.md)
* [Data Management](user-guide/data-management.md)
* [App API](api/app.md)
* [Configuration](api/config.md)
* [Logging](api/boogr.md)
