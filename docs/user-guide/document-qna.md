# Document Q&A

Document Q&A is the grounded question-answering workflow in Loca. It allows the user to upload
documents, extract readable text, split that text into retrievable chunks, and ask questions that
are answered using document excerpts as context.

## 🧭 Purpose

Document Q&A helps the local model answer questions from supplied files instead of relying only on
general model knowledge. The workflow is designed for local document review, summarization,
extraction, policy analysis, technical documentation, and source-grounded question answering.

Document Q&A emphasizes:

* local document processing;
* controlled text extraction;
* configurable chunking;
* optional vector retrieval;
* grounded answers from retrieved excerpts;
* diagnostic visibility when extraction or retrieval is unavailable;
* safe fallback behavior when optional dependencies are missing.

## 🧱 Workflow Position

Document Q&A sits between raw document upload and local model generation.

```text id="ffhg3l"
Uploaded Documents
  │
  ▼
Text Extraction
  │
  ├── PDF extraction
  ├── DOCX extraction
  └── Plain text extraction
  │
  ▼
Chunking
  │
  ├── Chunk size
  └── Chunk overlap
  │
  ▼
Retrieval
  │
  ├── SQLite vector retrieval when available
  └── Similarity fallback when enabled
  │
  ▼
Grounded Prompt
  │
  ▼
Local Model Answer
```

The workflow is implemented primarily in `app.py` and uses configuration values from `config.py`.

## 📄 Supported Document Flow

The Document Q&A workflow starts with uploaded files. The application keeps uploaded file state in
Streamlit session state and uses available extraction libraries when they are installed.

| File Type               | Extraction Method                  | Dependency                       |
| ----------------------- | ---------------------------------- | -------------------------------- |
| PDF                     | Native PDF text extraction         | `PyMuPDF` / `fitz`               |
| DOCX                    | Word document paragraph extraction | `python-docx`                    |
| TXT / text-like content | Direct text decoding               | Python standard library fallback |

When an optional dependency is unavailable, Loca should fail safely and report the limitation
instead of breaking the Streamlit app.

## 🧾 Document Inventory

Document Q&A maintains document-related state so uploaded files, extracted text, chunk counts, and
retrieval diagnostics can survive Streamlit reruns.

Important state values include:

| Session-State Key       | Purpose                                                              |
| ----------------------- | -------------------------------------------------------------------- |
| `uploaded`              | Tracks uploaded file objects.                                        |
| `active_docs`           | Tracks active document text or document metadata.                    |
| `doc_bytes`             | Stores uploaded document bytes.                                      |
| `doc_source`            | Tracks whether documents came from local upload or another source.   |
| `docqna_vec_ready`      | Indicates whether vector retrieval is ready.                         |
| `docqna_fingerprint`    | Tracks the active document-set fingerprint.                          |
| `docqna_chunk_count`    | Stores the number of prepared chunks.                                |
| `docqna_fallback_rows`  | Stores fallback retrieval rows when vector retrieval is unavailable. |
| `docqna_last_retrieval` | Stores the most recent retrieved chunks.                             |
| `docqna_inventory_rows` | Stores document inventory rows for display.                          |

This state contract prevents uploaded documents and retrieval results from being lost during normal
Streamlit reruns.

## ✂️ Chunking Strategy

After extraction, document text is split into overlapping chunks. Chunking controls how much context
is retrieved and how much continuity exists between adjacent excerpts.

| Setting                   | Purpose                                          |
| ------------------------- | ------------------------------------------------ |
| `retrieval_chunk_size`    | Maximum character length of each chunk.          |
| `retrieval_chunk_overlap` | Number of overlapping characters between chunks. |
| `retrieval_k`             | Number of chunks retrieved for a question.       |

A larger chunk size provides more surrounding context. A smaller chunk size produces more targeted
excerpts. Overlap helps preserve meaning across chunk boundaries.

Recommended defaults:

| Use Case                |       Chunk Size |        Overlap | Retrieval K |
| ----------------------- | ---------------: | -------------: | ----------: |
| Short policies or memos |  `800` to `1200` | `100` to `200` |  `3` to `5` |
| Long technical files    | `1200` to `1800` | `200` to `300` |  `5` to `8` |
| Small local models      |   `600` to `900` |  `80` to `150` |  `2` to `4` |
| Precise extraction      |  `600` to `1000` | `100` to `200` |  `3` to `6` |

## 🧠 Model-Safe Retrieval Profiles

Loca can apply model-safe retrieval defaults based on the selected model. Smaller models should
receive narrower context windows, while larger models can handle more retrieved material.

The retrieval profile may control:

| Profile Setting             | Purpose                                                          |
| --------------------------- | ---------------------------------------------------------------- |
| `retrieval_k`               | Limits the number of document excerpts inserted into the prompt. |
| `retrieval_chunk_size`      | Controls excerpt size for Document Q&A.                          |
| `retrieval_chunk_overlap`   | Preserves context across chunk boundaries.                       |
| `require_grounding`         | Requires answers to be grounded in retrieved excerpts.           |
| `answer_from_excerpts_only` | Prevents the model from relying on outside knowledge.            |
| `show_retrieved_chunks`     | Displays the source excerpts used for answering.                 |
| `prefer_sqlite_vec`         | Prefers SQLite vector retrieval when available.                  |
| `allow_similarity_fallback` | Allows fallback retrieval when vector search is unavailable.     |

This keeps retrieved context proportional to the selected model’s practical capacity.

## 🔎 Retrieval Behavior

Document Q&A supports two retrieval paths.

| Retrieval Path          | Description                                                                     |
| ----------------------- | ------------------------------------------------------------------------------- |
| SQLite vector retrieval | Uses stored embeddings when compatible vectors are available.                   |
| Similarity fallback     | Uses simpler matching logic when vector retrieval is unavailable or incomplete. |

The application should prefer vector retrieval when the embeddings table, vector dimensions, and
embedder are available. If those components are missing, fallback retrieval can still provide useful
document excerpts.

## 🧾 Grounding Controls

Grounding controls determine how strictly the model must answer from retrieved document excerpts.

| Control                     | Recommended Use                                             |
| --------------------------- | ----------------------------------------------------------- |
| `require_grounding`         | Enable when the answer must be based on the document.       |
| `answer_from_excerpts_only` | Enable for compliance, policy, audit, and extraction tasks. |
| `show_retrieved_chunks`     | Enable when the user needs to inspect evidence.             |
| `allow_similarity_fallback` | Enable when vector retrieval is not guaranteed.             |

For most official document-review workflows, use both `require_grounding` and
`answer_from_excerpts_only`.

## 🧪 Example: Ask a Question from a PDF

Use this workflow when reviewing a report, policy, guide, or exported documentation file.

```text id="3p7pe8"
1. Open Document Q&A mode.
2. Upload the PDF.
3. Confirm text extraction succeeded.
4. Keep Require Grounding enabled.
5. Keep Answer From Excerpts Only enabled.
6. Ask a question.
```

Example question:

```text id="vy7roe"
What are the major system components described in this document?
```

Expected behavior:

* Loca extracts text from the PDF.
* The document is chunked.
* Relevant chunks are retrieved.
* The model answers using retrieved excerpts.
* Retrieved chunks are shown when diagnostics are enabled.

## 🧪 Example: Extract Facts from a Document

Use Document Q&A for controlled extraction.

```text id="zgj52j"
Question:
Extract the project name, database path, logging table, supported modes, and model registry names from the document.

Instructions:
Return a Markdown table. Use "Not provided" when a value is missing.
```

Recommended settings:

| Setting                   | Value          |
| ------------------------- | -------------- |
| Require Grounding         | Enabled        |
| Answer From Excerpts Only | Enabled        |
| Show Retrieved Chunks     | Enabled        |
| Retrieval K               | `5` to `8`     |
| Temperature               | `0.0` to `0.2` |

## 🧪 Example: Summarize a Document

Use Document Q&A to create a grounded summary.

```text id="980cda"
Question:
Summarize this document in five bullets. Preserve names, dates, file paths, and configuration values exactly.
```

Recommended settings:

| Setting                   | Value            |
| ------------------------- | ---------------- |
| Require Grounding         | Enabled          |
| Answer From Excerpts Only | Enabled          |
| Retrieval K               | `6` to `8`       |
| Chunk Size                | `1200` to `1800` |
| Temperature               | `0.0` to `0.3`   |

For long documents, ask for section-level summaries first, then generate the final summary from
those results.

## 🧪 Example: Build Documentation from Source Notes

Document Q&A can support documentation generation when the uploaded document contains project notes,
source descriptions, README material, or architecture descriptions.

```text id="md5wra"
Question:
Using only the uploaded document, draft a MkDocs user-guide page for the Text Generation workflow. Use header-level icons and include practical examples.
```

Recommended settings:

| Setting                   | Value   |
| ------------------------- | ------- |
| Require Grounding         | Enabled |
| Answer From Excerpts Only | Enabled |
| Show Retrieved Chunks     | Enabled |
| Retrieval K               | `6`     |
| Temperature               | `0.2`   |

## 🔬 Diagnostics

Document Q&A includes diagnostics to help confirm whether the document pipeline is working.

Useful diagnostics include:

| Diagnostic              | Meaning                                                                |
| ----------------------- | ---------------------------------------------------------------------- |
| Uploaded document count | Confirms the app received the files.                                   |
| Extracted text length   | Confirms text was readable.                                            |
| Chunk count             | Confirms chunking succeeded.                                           |
| Retrieval rows          | Shows which chunks were selected.                                      |
| Vector readiness        | Confirms whether embedding-based retrieval is available.               |
| Fallback rows           | Shows fallback retrieval results when vector retrieval is unavailable. |
| Inventory rows          | Shows registered document metadata.                                    |

Enable diagnostics when validating a new file type, debugging extraction, or confirming that answers
are grounded.

## 🚨 Missing Dependency Handling

Some document features depend on optional libraries.

| Dependency         | Used For         | Failure Behavior                                       |
| ------------------ | ---------------- | ------------------------------------------------------ |
| `python-docx`      | DOCX extraction  | DOCX extraction is unavailable when missing.           |
| `PyMuPDF` / `fitz` | PDF extraction   | Native PDF extraction is unavailable when missing.     |
| Embedding backend  | Vector retrieval | Semantic/vector retrieval is unavailable when missing. |
| SQLite             | Persistence      | Persistent document and embedding storage may fail.    |

The application should clearly report unavailable features and continue running wherever possible.

## 🛡️ Grounded Answering Rules

For reliable Document Q&A, use conservative prompt rules.

Recommended system instruction:

```text id="14rg8t"
Answer using only the retrieved document excerpts. If the excerpts do not contain the answer, say that the document excerpts do not provide enough information. Do not invent missing facts.
```

Recommended extraction instruction:

```text id="xj6q8i"
Extract only values that are explicitly present in the retrieved excerpts. Preserve exact names, dates, file paths, configuration values, and labels.
```

Recommended summarization instruction:

```text id="11u85e"
Summarize only the provided excerpts. Preserve key names, dates, numbers, paths, and conclusions. Identify uncertainty when the excerpts are incomplete.
```

## ✅ Recommended Sequence

Use this sequence for reliable Document Q&A results:

1. Select a model with Document Q&A support.
2. Open Document Q&A mode.
3. Upload the document.
4. Confirm extraction succeeded.
5. Confirm chunk count is greater than zero.
6. Enable grounding controls.
7. Enable retrieved-chunk display for verification.
8. Ask a focused question.
9. Review the retrieved excerpts.
10. Revise chunk size, overlap, or retrieval count if the answer lacks context.
11. Disable unrelated documents when switching topics.

## 🧭 Practical Defaults

A stable default setup is:

| Setting                        | Recommended Value |
| ------------------------------ | ----------------: |
| Retrieval K                    |               `6` |
| Chunk Size                     |            `1200` |
| Chunk Overlap                  |             `200` |
| Require Grounding              |           Enabled |
| Answer From Excerpts Only      |           Enabled |
| Show Retrieved Chunks          |           Enabled |
| Prefer SQLite Vector Retrieval |           Enabled |
| Allow Similarity Fallback      |           Enabled |
| Temperature                    |    `0.0` to `0.2` |

For smaller local models, reduce retrieval count and chunk size.

## 🛠️ Troubleshooting

| Problem                         | Likely Cause                                         | Corrective Action                                                      |
| ------------------------------- | ---------------------------------------------------- | ---------------------------------------------------------------------- |
| Uploaded document has no text   | Scanned PDF or unsupported encoding                  | Use OCR outside the native text path or provide a text-based document. |
| DOCX does not extract           | `python-docx` is unavailable                         | Install `python-docx` in the active environment.                       |
| PDF does not extract            | `PyMuPDF` is unavailable or PDF has no embedded text | Install `PyMuPDF` or use OCR.                                          |
| Retrieved chunks are irrelevant | Chunk size or query is too broad                     | Ask a narrower question or reduce chunk size.                          |
| Answer ignores the document     | Grounding controls are off                           | Enable Require Grounding and Answer From Excerpts Only.                |
| Too much context is inserted    | Retrieval K or chunk size is too high                | Reduce retrieval count or chunk size.                                  |
| Model gives incomplete answer   | Retrieved chunks lack the needed section             | Increase retrieval count or ask a more targeted question.              |
| Vector retrieval fails          | Embedder or vectors are unavailable                  | Enable similarity fallback or rebuild embeddings.                      |

## 🧩 Relationship to Semantic Search

Document Q&A and Semantic Search are related but not identical.

| Workflow                              | Best Use                                                             |
| ------------------------------------- | -------------------------------------------------------------------- |
| Document Q&A                          | Ask questions and generate answers from retrieved document excerpts. |
| Semantic Search                       | Find and inspect similar chunks across indexed content.              |
| Text Generation with Document Context | Use document text directly as general model context.                 |

Use Document Q&A when the final output should be an answer. Use Semantic Search when the primary
goal is to find relevant passages.

## 🚨 Exception Logging

Operational failures in the Document Q&A pipeline should use the standard Loca logging pattern.

```python id="4lhbre"
except Exception as e:
    exception = Error( e )
    exception.module = 'app'
    exception.cause = 'DocumentQnA'
    exception.method = 'function_name( arg: type ) -> return_type'
    Logger( ).write( exception )
    raise exception
```

The method string should remain a stable signature. It should not include document text, uploaded
file contents, prompts, full file paths, or user data.

## 🔗 Related API Pages

* [Application](../api/app.md)
* [Configuration](../api/config.md)
* [Logging](../api/boogr.md)
* [Text Generation](text-generation.md)
* [Semantic Search](semantic-search.md)
* [Prompt Engineering](prompt-engineering.md)
