# Semantic Search

Semantic Search is the retrieval workflow in Loca for finding relevant document chunks by meaning
rather than by exact keyword match. It supports local document review, source discovery, contextual
search, and retrieval preparation for grounded model workflows.

## 🧭 Purpose

Semantic Search helps users locate relevant passages across uploaded or indexed text. Instead of
matching only literal words, the workflow compares the meaning of the user query against stored text
chunks and returns the most similar results.

Semantic Search is useful when:

* a document set is too large to inspect manually;
* the user does not know the exact wording used in the source material;
* related passages need to be found across multiple files;
* Document Q&A needs better source discovery;
* Text Generation needs focused context from indexed content.

## 🧱 Workflow Position

Semantic Search sits between document ingestion and answer generation. It is primarily a retrieval
and inspection workflow.

```text id="1kx3oc"
Documents or Text
  │
  ▼
Text Extraction
  │
  ▼
Chunking
  │
  ▼
Embedding
  │
  ▼
Vector Store or Session-State Index
  │
  ▼
User Query
  │
  ▼
Similarity Ranking
  │
  ▼
Relevant Chunks
```

The output of Semantic Search can be reviewed directly or reused as context for Text Generation and
Document Q&A.

## 🧩 Relationship to Other Modes

Semantic Search overlaps with Document Q&A and Text Generation, but each mode has a different
purpose.

| Mode               | Primary Goal                                                     | Output                           |
| ------------------ | ---------------------------------------------------------------- | -------------------------------- |
| Text Generation    | Generate a response from prompt, settings, and optional context. | Assistant response.              |
| Document Q&A       | Answer a question using retrieved document excerpts.             | Grounded answer.                 |
| Semantic Search    | Find relevant chunks by meaning.                                 | Ranked excerpts and diagnostics. |
| Prompt Engineering | Create, manage, and apply reusable prompts.                      | Prompt templates.                |
| Data Management    | Inspect and manage stored application assets.                    | Database and asset views.        |

Use Semantic Search when the task is discovery. Use Document Q&A when the task is answering. Use
Text Generation when the task is drafting, explaining, or transforming content.

## 📄 Source Preparation

Semantic Search begins with source text. The source may come from uploaded documents, extracted
document content, stored chunks, or active context buffers.

Typical source preparation includes:

1. Load or upload source material.
2. Extract readable text.
3. Normalize text where appropriate.
4. Split text into chunks.
5. Store chunk metadata.
6. Generate embeddings when an embedding backend is available.
7. Make the index available for search.

The quality of retrieval depends heavily on the quality of extracted text and chunk boundaries.

## ✂️ Chunking Strategy

Semantic Search uses chunking to split long source text into searchable units.

| Setting                   | Purpose                                                             |
| ------------------------- | ------------------------------------------------------------------- |
| `semantic_chunk_size`     | Controls the maximum size of each searchable chunk.                 |
| `semantic_chunk_overlap`  | Preserves continuity across neighboring chunks.                     |
| `semantic_top_k`          | Controls how many matching chunks are returned.                     |
| `semantic_min_similarity` | Filters out weak matches below the configured similarity threshold. |

Recommended defaults:

| Use Case                 |       Chunk Size |        Overlap |        Top K | Min Similarity |
| ------------------------ | ---------------: | -------------: | -----------: | -------------: |
| General document search  |           `1200` |          `200` |          `8` |          `0.0` |
| Small local models       |            `800` |          `120` |          `4` |         `0.05` |
| Precise source discovery |  `700` to `1000` | `100` to `180` |   `5` to `8` |         `0.05` |
| Broad research scan      | `1200` to `1800` | `200` to `300` | `10` to `15` |          `0.0` |

Smaller chunks improve precision. Larger chunks preserve more surrounding context.

## 🧠 Embedding and Similarity

Semantic Search uses vector representations of text when an embedding backend is available.

The general process is:

1. Convert each text chunk into an embedding vector.
2. Convert the user query into an embedding vector.
3. Compare the query vector to each chunk vector.
4. Rank chunks by similarity.
5. Return the highest-scoring chunks.

Cosine similarity is commonly used to compare normalized vector direction. A higher score indicates
a closer semantic relationship between the query and the chunk.

## 🗃️ Storage Model

Semantic Search may use Streamlit session state and SQLite-backed storage depending on the active
workflow.

Important state values include:

| Session-State Key            | Purpose                                                        |
| ---------------------------- | -------------------------------------------------------------- |
| `semantic_context_buffer`    | Holds active semantic source text or prepared context.         |
| `semantic_chunk_size`        | Controls semantic chunk length.                                |
| `semantic_chunk_overlap`     | Controls chunk overlap.                                        |
| `semantic_top_k`             | Controls number of returned results.                           |
| `semantic_min_similarity`    | Controls minimum accepted similarity score.                    |
| `semantic_group_by_document` | Groups results by document when enabled.                       |
| `semantic_clear_existing`    | Clears existing semantic context before indexing new material. |
| `semantic_append_existing`   | Appends new material to the current semantic index.            |
| `semantic_show_diagnostics`  | Displays diagnostic details for indexing and retrieval.        |
| `semantic_uploaded_names`    | Tracks uploaded document names.                                |
| `semantic_result_rows`       | Stores ranked semantic search results.                         |
| `semantic_selected_rows`     | Stores selected result rows.                                   |
| `semantic_index_chunk_count` | Tracks indexed chunk count.                                    |
| `semantic_index_dim`         | Tracks embedding vector dimension.                             |
| `semantic_index_doc_count`   | Tracks indexed document count.                                 |
| `semantic_last_query`        | Stores the most recent semantic query.                         |

These values allow the UI to preserve indexing state, display diagnostics, and coordinate retrieval
results across Streamlit reruns.

## 🔎 Search Behavior

Semantic Search returns chunks that are similar to the query. The search does not require exact word
overlap.

Example:

```text id="48evmm"
Query:
How does the app handle missing local model files?
```

This may retrieve chunks containing language such as:

```text id="gcu18b"
The selected local GGUF model is unavailable.
Verify that the file exists or update the model path in config.py.
```

Even if the chunk does not use the exact phrase “missing local model files,” semantic similarity can
locate the relevant passage.

## 🧪 Example: Search Project Documentation

Use Semantic Search to locate relevant project sections.

```text id="d6x0qc"
Search Query:
Where does the application define model paths and supported modes?
```

Expected relevant sources may include:

* model registry documentation;
* configuration API details;
* `config.py` descriptions;
* model/mode contract sections.

Recommended settings:

| Setting           |                                 Value |
| ----------------- | ------------------------------------: |
| Chunk Size        |                                `1200` |
| Chunk Overlap     |                                 `200` |
| Top K             |                                   `8` |
| Min Similarity    |                                 `0.0` |
| Group By Document | Enabled when searching multiple files |

## 🧪 Example: Find Error-Handling References

Use Semantic Search to inspect diagnostic behavior.

```text id="h35r1h"
Search Query:
How are exceptions logged to SQLite?
```

Expected relevant sources may include:

* `boogr.Error`;
* `boogr.Logger`;
* `config.LOG_PATH`;
* `config.LOG_FILE`;
* exception logging examples;
* architecture documentation.

Recommended settings:

| Setting        |           Value |
| -------------- | --------------: |
| Chunk Size     | `900` to `1200` |
| Chunk Overlap  |  `150` to `200` |
| Top K          |             `6` |
| Min Similarity |          `0.05` |

## 🧪 Example: Prepare Context for Text Generation

Semantic Search can identify excerpts that should be copied or sent into Text Generation.

```text id="cocm3s"
Search Query:
What is the recommended sequence for Document Q&A?
```

After reviewing results, selected chunks can be used as context for a Text Generation prompt such
as:

```text id="iwjo34"
Using the selected semantic search excerpts, draft a concise user-guide checklist for Document Q&A.
```

This is useful when building documentation pages, release notes, or technical summaries from larger
source material.

## 🧪 Example: Compare Similar Sections

Semantic Search can help locate related sections across files.

```text id="vsm0yy"
Search Query:
session state keys used for document retrieval
```

Potentially relevant sections may include:

* Document Q&A session-state contract;
* Semantic Search session-state contract;
* Text Generation context injection;
* model-safe retrieval defaults.

Use this approach when consolidating duplicated documentation or tracing data flow across the
application.

## 🧰 Result Review

Semantic Search results should be reviewed before using them as grounding context.

A useful result row should include:

| Field       | Purpose                                                     |
| ----------- | ----------------------------------------------------------- |
| Document    | Identifies the source document or file.                     |
| Chunk       | Shows the retrieved passage.                                |
| Similarity  | Indicates relative match strength.                          |
| Chunk Index | Helps locate the passage in the source sequence.            |
| Text Length | Helps evaluate whether the chunk is too short or too large. |

High-ranked results are not automatically authoritative. They are candidates for review.

## 📊 Similarity Thresholds

Similarity scores are relative to the embedding model and source content. Use thresholds
conservatively.

| Threshold Pattern     | Effect                                                     |
| --------------------- | ---------------------------------------------------------- |
| Low or zero threshold | Returns more results and supports broad discovery.         |
| Moderate threshold    | Filters weak matches while keeping useful context.         |
| High threshold        | Returns only close matches but may miss relevant passages. |

For initial searches, start with a low threshold. Increase the threshold only after confirming that
enough relevant content is being returned.

## 🧭 Grouping Results

When searching multiple documents, grouping by document can make results easier to inspect.

Use `semantic_group_by_document` when:

* multiple files are indexed;
* results are dominated by one large document;
* each document needs representation;
* the user wants a document-by-document comparison.

Disable grouping when:

* the goal is simply to find the highest-scoring chunks;
* all content belongs to one source;
* strict ranking is more important than document coverage.

## 🔄 Clear vs. Append Behavior

Semantic Search supports two common indexing behaviors.

| Behavior        | Use When                                         |
| --------------- | ------------------------------------------------ |
| Clear Existing  | Starting a new search set or switching projects. |
| Append Existing | Adding more files to the current search set.     |

Use clear behavior when moving to a new topic. Use append behavior when building a larger index from
related files.

## 🧾 Diagnostics

Semantic diagnostics help confirm that indexing and retrieval are working.

Useful diagnostics include:

| Diagnostic             | Meaning                                             |
| ---------------------- | --------------------------------------------------- |
| Indexed chunk count    | Confirms source text was split and indexed.         |
| Embedding dimension    | Confirms vector shape and compatibility.            |
| Indexed document count | Confirms how many source documents are represented. |
| Uploaded names         | Confirms which files contributed content.           |
| Last query             | Confirms the active retrieval request.              |
| Result rows            | Confirms ranked search output.                      |
| Selected rows          | Confirms which results were selected for reuse.     |

Enable diagnostics when validating a new source set, troubleshooting retrieval quality, or preparing
reliable context for another workflow.

## 🚨 Missing Embedder Handling

Semantic Search depends on an available embedding backend. If the embedder is missing or
incompatible, the application should report that semantic retrieval is unavailable rather than
failing silently.

Common causes include:

| Problem                   | Likely Cause                                           | Corrective Action                            |
| ------------------------- | ------------------------------------------------------ | -------------------------------------------- |
| No semantic results       | No indexed chunks                                      | Upload or prepare source text first.         |
| Embedder unavailable      | Embedding dependency is missing                        | Install or configure the embedding backend.  |
| Vector dimension mismatch | Query and stored vectors came from different embedders | Rebuild the index with one embedding model.  |
| Empty chunks              | Source extraction failed or text was blank             | Confirm document extraction before indexing. |
| Weak results              | Query is too broad or chunks are too large             | Narrow the query or reduce chunk size.       |

## 🛡️ Grounding and Evidence Use

Semantic Search finds passages. It does not automatically prove that an answer is correct.

For evidence-sensitive workflows:

1. Search for relevant chunks.
2. Inspect the returned text.
3. Select the strongest excerpts.
4. Use Document Q&A or Text Generation with grounding instructions.
5. Require the model to answer only from selected excerpts.
6. State when the excerpts are incomplete.

Recommended grounding instruction:

```text id="80cfhi"
Use only the selected semantic search excerpts. If the excerpts do not contain enough information, say that the excerpts are insufficient.
```

## ✅ Recommended Sequence

Use this sequence for reliable Semantic Search results:

1. Select a model and mode that supports Semantic Search.
2. Open Semantic Search mode.
3. Upload or prepare source text.
4. Choose whether to clear or append existing indexed content.
5. Set chunk size and overlap.
6. Build or refresh the semantic index.
7. Enter a focused search query.
8. Review ranked results.
9. Adjust top-k or minimum similarity if needed.
10. Select useful chunks for downstream use.
11. Send selected context to Text Generation or use Document Q&A for grounded answers.

## 🧭 Practical Defaults

A stable starting configuration is:

| Setting                |                   Recommended Value |
| ---------------------- | ----------------------------------: |
| Semantic Chunk Size    |                              `1200` |
| Semantic Chunk Overlap |                               `200` |
| Semantic Top K         |                                 `8` |
| Minimum Similarity     |                               `0.0` |
| Group By Document      | Disabled for single-document search |
| Clear Existing         |       Enabled when switching topics |
| Append Existing        |      Enabled only for related files |
| Show Diagnostics       |                             Enabled |

For smaller local models, reduce chunk size and top-k before sending results into generation.

## 🛠️ Troubleshooting

| Problem                                     | Likely Cause                                  | Corrective Action                            |
| ------------------------------------------- | --------------------------------------------- | -------------------------------------------- |
| Results are irrelevant                      | Query is too broad                            | Use a more specific query with domain terms. |
| Important section is missing                | Top K is too low                              | Increase `semantic_top_k`.                   |
| Results are too noisy                       | Threshold is too low                          | Increase `semantic_min_similarity`.          |
| Results lack surrounding context            | Chunk size is too small                       | Increase chunk size or overlap.              |
| Results are too broad                       | Chunk size is too large                       | Reduce chunk size.                           |
| Index disappears after rerun                | State was cleared or source was not persisted | Rebuild or persist the index.                |
| Search fails after changing embedder        | Vector dimensions changed                     | Rebuild all embeddings.                      |
| Multiple documents are unevenly represented | One document dominates ranking                | Enable grouping by document.                 |

## 🚨 Exception Logging

Operational failures in Semantic Search should use the standard Loca logging pattern.

```python id="5adn7t"
except Exception as e:
    exception = Error( e )
    exception.module = 'app'
    exception.cause = 'SemanticSearch'
    exception.method = 'function_name( arg: type ) -> return_type'
    Logger( ).write( exception )
    raise exception
```

The method string should remain a stable signature only. It should not include query text, document
text, embedding values, dataframe contents, file contents, secrets, or runtime object details.

## 🔗 Related API Pages

* [Application](../api/app.md)
* [Configuration](../api/config.md)
* [Logging](../api/boogr.md)
* [Text Generation](text-generation.md)
* [Document Q&A](document-qna.md)
* [Prompt Engineering](prompt-engineering.md)
* [Data Management](data-management.md)
