# Data Management

Data Management is the workflow in Loca for inspecting, registering, synchronizing, and maintaining
local application assets stored in SQLite. It supports visibility into documents, chunks,
embeddings, prompts, chat history, and other persisted records used by the local AI workflows.

## 🧭 Purpose

Data Management provides operational visibility into Loca’s local storage layer. It helps users
confirm that uploaded documents, extracted chunks, embeddings, prompt templates, and chat records
are available to the application and aligned with the current workflow.

Data Management supports:

* SQLite asset inspection;
* document and image registration controls;
* prompt and document inventory review;
* table-level counts;
* document chunk verification;
* embedding availability checks;
* synchronization status reporting;
* troubleshooting for retrieval and persistence workflows.

## 🧱 Workflow Position

Data Management sits below the user-facing model workflows and above the SQLite storage layer.

```text id="e42f7c"
Text Generation
Document Q&A
Semantic Search
Prompt Engineering
  │
  ▼
Application State
  │
  ▼
Data Management
  │
  ├── Asset Counts
  ├── Table Inspection
  ├── Document Registry
  ├── Chunk Registry
  ├── Embedding Registry
  └── Prompt Registry
  │
  ▼
SQLite Database
```

The workflow is implemented primarily in `app.py` and uses database paths and table names defined in
`config.py`.

## 🗃️ Local Storage Model

Loca uses SQLite for lightweight local persistence. This keeps the application self-contained and
suitable for desktop, local-model, and project-specific workflows.

Core storage areas include:

| Storage Area     | Purpose                                                    |
| ---------------- | ---------------------------------------------------------- |
| Chat history     | Stores persisted user and assistant messages.              |
| Prompt templates | Stores reusable system prompts and prompt metadata.        |
| Documents        | Stores document-level metadata.                            |
| Document chunks  | Stores extracted chunks used for retrieval.                |
| Embeddings       | Stores vector data for semantic retrieval when available.  |
| Exception logs   | Stores structured error records through the logging layer. |

Data Management helps confirm that these storage areas are populated correctly.

## 📊 Asset Counts

Asset counts provide a quick view of what the database contains.

Typical asset counts include:

| Asset                | Meaning                                                                          |
| -------------------- | -------------------------------------------------------------------------------- |
| Documents            | Number of registered source documents.                                           |
| Document Chunks      | Number of extracted text chunks.                                                 |
| Document Embeddings  | Number of chunk embeddings available for retrieval.                              |
| Prompts              | Number of saved prompt templates.                                                |
| Chat History Records | Number of persisted chat messages.                                               |
| Exception Records    | Number of logged application failures, when the exception database is inspected. |

Counts are useful for confirming that ingestion, prompt creation, and retrieval preparation have
completed successfully.

## 🧾 Data Management Session State

The Data Management workflow uses Streamlit session-state values to preserve active selections and
synchronization status across reruns.

Important state values include:

| Session-State Key             | Purpose                                                         |
| ----------------------------- | --------------------------------------------------------------- |
| `dm_asset_sync_status`        | Stores the most recent synchronization or registration message. |
| `dm_asset_counts`             | Stores current table or asset counts.                           |
| `dm_selected_asset_table`     | Tracks the active table or asset view selected by the user.     |
| `dm_register_uploaded_docs`   | Controls whether uploaded documents should be registered.       |
| `dm_register_uploaded_images` | Controls whether uploaded images should be registered.          |

These values allow the interface to show stable status information after uploads, refreshes, and
table changes.

## 📄 Document Registration

Document registration records metadata about uploaded or processed files. It helps the application
know which files exist, how much text was extracted, how many chunks were created, and whether a
document has already been processed.

A document record may include:

| Field         | Purpose                                                        |
| ------------- | -------------------------------------------------------------- |
| `DocumentId`  | Internal database identifier.                                  |
| `Name`        | Document name.                                                 |
| `Type`        | Document or file type.                                         |
| `SizeBytes`   | File size in bytes.                                            |
| `Source`      | Source location or workflow.                                   |
| `Fingerprint` | Stable identifier used to detect duplicate or changed content. |
| `TextLength`  | Length of extracted text.                                      |
| `ChunkCount`  | Number of chunks produced from the document.                   |
| `CreatedOn`   | Timestamp for the registered document record.                  |

Document registration supports repeatable retrieval workflows and prevents the user from guessing
whether a file has been processed.

## ✂️ Chunk Registry

The chunk registry stores extracted document sections used by Document Q&A and Semantic Search.

A chunk record may include:

| Field          | Purpose                                   |
| -------------- | ----------------------------------------- |
| `ChunkId`      | Internal chunk identifier.                |
| `DocumentName` | Source document name.                     |
| `ChunkIndex`   | Sequence number within the document.      |
| `ChunkText`    | Extracted text for the chunk.             |
| `ChunkLength`  | Character length of the chunk.            |
| `Fingerprint`  | Source fingerprint used for traceability. |
| `CreatedOn`    | Timestamp for the chunk record.           |

Chunk records are important because retrieval quality depends on clean text, useful chunk sizes, and
complete source coverage.

## 🧠 Embedding Registry

Embeddings support semantic retrieval. When available, each chunk can be represented as a vector and
stored for similarity search.

Embedding records may include:

| Field                       | Purpose                                             |
| --------------------------- | --------------------------------------------------- |
| Chunk reference             | Associates the vector with source text.             |
| Vector blob                 | Stores numeric embedding data.                      |
| Model or dimension metadata | Helps confirm vector compatibility.                 |
| Fingerprint                 | Links the embedding to the source document version. |

The embedding registry should be rebuilt when:

* source text changes;
* chunking settings change;
* the embedding model changes;
* vector dimensions do not match;
* semantic results become inconsistent.

## 🧰 Prompt Registry

Prompt records support the Prompt Engineering workflow. Data Management can help verify that
templates are saved, visible, and available to selector controls.

Prompt records typically include:

| Field       | Purpose                                 |
| ----------- | --------------------------------------- |
| `PromptsId` | Internal prompt identifier.             |
| `Caption`   | User-facing prompt name.                |
| `Name`      | Internal prompt name or category label. |
| `Text`      | Full prompt template text.              |
| `Version`   | Prompt version label.                   |
| `ID`        | Optional external identifier.           |

If a prompt does not appear in the UI, confirm that the `Caption` field is populated and that the
prompt table exists.

## 💬 Chat History

Chat history records preserve prior interactions when persistence is enabled. This allows users to
resume context or inspect earlier user and assistant exchanges.

Chat history is useful for:

* reviewing prior prompts;
* tracing generated outputs;
* debugging repeated model behavior;
* verifying what context may have influenced a response;
* clearing stale messages when changing tasks.

Clear chat history when switching projects, changing source documents, or troubleshooting unexpected
model behavior.

## 🚨 Exception Data

Loca’s exception logging layer writes structured error records to the configured SQLite exception
database. This data is separate from normal application workflow tables unless the same database
path is configured.

Exception records may include:

| Field     | Purpose                                    |
| --------- | ------------------------------------------ |
| `created` | Timestamp of the logged exception.         |
| `cause`   | Logical component or workflow that failed. |
| `module`  | Source module where the failure occurred.  |
| `method`  | Stable function or method signature.       |
| `message` | Exception message.                         |
| `info`    | Diagnostic text.                           |
| `trace`   | Traceback text.                            |

Exception data is useful for debugging document processing, model loading, retrieval, prompt
persistence, and database failures.

## 🔄 Synchronization

Synchronization means aligning Streamlit session-state assets with persisted SQLite records.

Synchronization may include:

* registering uploaded documents;
* refreshing asset counts;
* rebuilding table inventory rows;
* confirming document chunk records;
* confirming prompt records;
* checking embedding availability;
* updating status messages after a write operation.

A synchronization status message should clearly indicate what changed and whether the action
completed successfully.

## 🧪 Example: Inspect Stored Assets

Use Data Management to confirm that local assets exist.

```text id="yztjdu"
1. Open Data Management mode.
2. Refresh asset counts.
3. Review document count.
4. Review chunk count.
5. Review prompt count.
6. Select a table for inspection.
7. Confirm expected records are present.
```

Expected outcome:

* counts are populated;
* selected table records are visible;
* document and prompt records match prior uploads or saves.

## 🧪 Example: Confirm Document Q&A Readiness

Use Data Management after uploading documents.

```text id="m99a34"
1. Upload a document in Document Q&A.
2. Confirm extraction succeeded.
3. Open Data Management.
4. Check document count.
5. Check chunk count.
6. Confirm the document name appears in the document table.
7. Confirm chunks exist for the document.
```

If the document exists but has zero chunks, extraction or chunking likely failed.

## 🧪 Example: Confirm Semantic Search Readiness

Use Data Management after building a semantic index.

```text id="39h0m8"
1. Build or refresh the semantic index.
2. Open Data Management.
3. Review embedding-related counts.
4. Confirm chunk records exist.
5. Confirm vectors are present when vector retrieval is expected.
6. Return to Semantic Search and run a focused query.
```

If chunks exist but embeddings do not, semantic search may require fallback retrieval or index
rebuilding.

## 🧪 Example: Verify Prompt Templates

Use Data Management to confirm saved prompts.

```text id="3t2sih"
1. Save a prompt in Prompt Engineering.
2. Open Data Management.
3. Select the prompt table.
4. Confirm the prompt caption appears.
5. Confirm the prompt text is populated.
6. Return to Prompt Engineering and load the template.
```

If the prompt text exists but the prompt does not appear in a selector, verify that the caption
field is populated.

## 🧭 Recommended Data Hygiene

For reliable local workflows:

| Practice                                | Reason                                               |
| --------------------------------------- | ---------------------------------------------------- |
| Use clear document names                | Makes retrieval results easier to trace.             |
| Rebuild embeddings after source changes | Prevents stale vector matches.                       |
| Clear unrelated chat history            | Reduces prompt contamination.                        |
| Version important prompts               | Makes reusable workflows easier to maintain.         |
| Review chunk counts                     | Confirms document extraction and chunking succeeded. |
| Use fingerprints                        | Helps detect duplicate or changed files.             |
| Inspect exception logs                  | Speeds debugging of failed workflows.                |

## ✅ Recommended Sequence

Use this sequence for Data Management:

1. Open Data Management mode.
2. Refresh or synchronize asset counts.
3. Select the relevant asset table.
4. Confirm expected records are present.
5. Check document and chunk counts after uploads.
6. Check embeddings after semantic indexing.
7. Check prompt records after saving templates.
8. Review status messages for failed writes or missing data.
9. Inspect exception logs when workflows fail.
10. Clear or rebuild stale data when switching projects.

## 🧭 Practical Defaults

A stable Data Management setup is:

| Setting                     | Recommended Value                            |
| --------------------------- | -------------------------------------------- |
| Selected Table              | Start with `documents`                       |
| Register Uploaded Documents | Enabled when building Document Q&A assets    |
| Register Uploaded Images    | Enabled only when image workflows are active |
| Refresh Counts              | After every ingestion or save operation      |
| Review Chunks               | After document extraction                    |
| Review Embeddings           | After semantic index creation                |
| Review Prompts              | After prompt edits or imports                |

## 🛠️ Troubleshooting

| Problem                                | Likely Cause                                         | Corrective Action                                          |
| -------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------------- |
| Asset counts are empty                 | Database has not been initialized or populated       | Run the relevant workflow first and refresh counts.        |
| Document appears without chunks        | Extraction or chunking failed                        | Reprocess the document and inspect extraction diagnostics. |
| Chunks exist but semantic search fails | Embeddings are missing or incompatible               | Rebuild the semantic index.                                |
| Prompt does not appear in selector     | Missing or blank caption                             | Confirm the prompt record has a populated `Caption`.       |
| Chat history is stale                  | Prior messages are still persisted                   | Clear chat history before starting a new task.             |
| Database write fails                   | Path, permissions, or schema issue                   | Confirm database path and inspect exception logs.          |
| Results reference old documents        | Old records were not cleared                         | Clear or rebuild the relevant tables for the project.      |
| Exception table is missing             | No exception has been logged or logging path differs | Confirm `LOG_PATH` and `LOG_FILE`.                         |

## 🚨 Exception Logging

Operational failures in Data Management should use the standard Loca logging pattern.

```python id="xliv88"
except Exception as e:
    exception = Error( e )
    exception.module = 'app'
    exception.cause = 'DataManagement'
    exception.method = 'function_name( arg: type ) -> return_type'
    Logger( ).write( exception )
    raise exception
```

The method string should remain a stable signature only. It should not include database row
contents, prompt text, document text, file contents, full file paths, vector values, secrets, or
runtime object details.

## 🔗 Related API Pages

* [Application](../api/app.md)
* [Configuration](../api/config.md)
* [Logging](../api/boogr.md)
* [Text Generation](text-generation.md)
* [Document Q&A](document-qna.md)
* [Semantic Search](semantic-search.md)
* [Prompt Engineering](prompt-engineering.md)
