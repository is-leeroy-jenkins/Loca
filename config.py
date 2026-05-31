'''
  ******************************************************************************************
      Assembly:                Bro
      Filename:                config.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="config.py" company="Terry D. Eppler">

	     config.py
	     Copyright ©  2022  Terry Eppler

     Permission is hereby granted, free of charge, to any person obtaining a copy
     of this software and associated documentation files (the “Software”),
     to deal in the Software without restriction,
     including without limitation the rights to use,
     copy, modify, merge, publish, distribute, sublicense,
     and/or sell copies of the Software,
     and to permit persons to whom the Software is furnished to do so,
     subject to the following conditions:

     The above copyright notice and this permission notice shall be included in all
     copies or substantial portions of the Software.

     THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
     INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
     FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
     ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
     DEALINGS IN THE SOFTWARE.

     You can contact me at:  terryeppler@gmail.com or eppler.terry@epa.gov

  </copyright>
  <summary>
    config.py
  </summary>
  ******************************************************************************************
'''
import os
import re
import multiprocessing

# ---------- DEFINITIONS -------------------

BRO_LLM_PATH = os.getenv( 'BRO_LLM_PATH' )
BLUE_DIVIDER = "<div style='height:2px;align:left;background:#0078FC;margin:20px 0 20px 0;'></div>"
APP_TITLE = 'Bro'
APP_SUBTITLE = 'Local AI based on Gemma 3'
BASE_DIR = os.path.dirname( os.path.abspath( __file__ ) )
DB_PATH = 'stores/sqlite/bro.db'
MODEL_PATH = r'C:\Users\terry\source\llm\bro\bro-3-4b-it-qat-Q4_K_M.gguf'
DEFAULT_CTX = 4096
CORES = multiprocessing.cpu_count( )
FAVICON = r'resources/images/favicon.ico'
LOGO = r'resources/bro_logo.png'
XML_BLOCK_PATTERN = re.compile( r"<(?P<tag>[a-zA-Z0-9_:-]+)>(?P<body>.*?)</\1>", re.DOTALL )
MARKDOWN_HEADING_PATTERN = re.compile( r"^##\s+(?P<title>.+?)\s*$" )
MODES = [ 'Text Generation', 'Document Q&A', 'Semantic Search',
          'Prompt Engineering', 'Data Management' ]

# ---------- DEFINITIONS -------------------

SYSTEM_INSTRUCTIONS = r'''Optional. Gives the model high-level instructions on how it should behave while
		generating a response, including tone, goals, and examples of correct responses. Any
		instructions provided this way will take priority over a prompt in the input parameter.'''

TEMPERATURE = r'''Optional. A number between 0 and 2. Higher values like 0.8 will make the output
		more random, while lower values like 0.2 will make it more focused and deterministic'''

TOP_P = r'''Optional. The maximum cumulative probability of tokens to consider when sampling.
		The model uses combined Top-k and Top-p (nucleus) sampling. Tokens are sorted based on
		their assigned probabilities so that only the most likely tokens are considered.
		Top-k sampling directly limits the maximum number of tokens to consider,
		while Nucleus sampling limits the number of tokens based on the cumulative probability.'''

TOP_K = r'''Optional. The maximum number of tokens to consider when sampling. Gemini models use
		Top-p (nucleus) sampling or a combination of Top-k and nucleus sampling. Top-k sampling considers
		the set of topK most probable tokens. Models running with nucleus sampling don't allow topK setting.
		Note: The default value varies by Model and is specified by theModel.top_p attribute returned
		from the getModel function. An empty topK attribute indicates that the model doesn't apply
		top-k sampling and doesn't allow setting topK on requests.'''

PRESENCE_PENALTY = r'''Optional. Presence penalty applied to the next token's logprobs
		if the token has already been seen in the response. This penalty is binary on/off
		and not dependant on the number of times the token is used (after the first).'''

FREQUENCY_PENALTY = r'''Optional. Frequency penalty applied to the next token's logprobs,
		multiplied by the number of times each token has been seen in the respponse so far.
		A positive penalty will discourage the use of tokens that have already been used,
		proportional to the number of times the token has been used: The more a token is used,
		the more difficult it is for the model to use that token again increasing
		the vocabulary of responses.'''

REPEAT_PENALTY = '''Penalizes repeated tokens to reduce looping and redundant responses.'''

CONTEXT_WINDOW = '''The context window is the maximum length of input a large language model (LLM)
		can consider at once. In the development and maturation of LLM technology expanding the
		context window has been a major goal. The length of a context window is
		measured in tokens. '''

REPEAT_WINDOW = '''"Prompt repetition" is a technique where repeating the entire input prompt
		(e.g., <prompt><prompt>) improves LLM performance on non-reasoning tasks by 21-97%.
		This method allows models to better focus their attention, particularly when references
		at the end of a prompt need to connect back to information at the beginning'''

CPU_CORES = '''Number of CPU threads used for inference; higher values improve speed but increase CPU
            "usage." '''

MAX_TOKENS = '''Maximum number of tokens generated per response.'''

SEED = '''Set to a fixed value for reproducible outputs; use -1 for a random seed each run.'''

PROMPT_ENGINEERING = r'''Prompt engineering is the process of writing effective instructions
		for a model, such that it consistently generates content that meets your requirements.
		Because the content generated from a model is non-deterministic, prompting to get your
		desired output is a mix of art and science. However, you can apply techniques and
		best practices to get good results consistently.
		'''

TEXT_GENERATION = r'''Use a large language model to produce coherent, context-aware natural language
		output in response to user prompts, system instructions, or retrieved document context.
		When a user submits a request—whether it is a general inquiry, a structured analytical task,
		or a document-grounded question—Bro constructs a prompt that may include system directives,
		conversation history, and optionally retrieved content from its vector store. The underlying
		model then generates text according to configurable parameters such as temperature,
		maximum tokens, and response format. This capability enables Bro to function as
		a conversational assistant, analytical explainer, summarizer, drafting tool, and reasoning engine,
		producing structured or narrative outputs tailored to the user’s workflow. '''

DATA_MANAGEMENT = r'''Structured handling, organization, processing of
		user-provided data in a self-contained SQLite Database. It allows uploading of files, extracting and
		normalizing their content, chunking text for semantic processing, generating embeddings,
		storing metadata, and enabling controlled retrieval for downstream features such as Document Q&A
		and Data Analysis. Beyond ingestion, it includes version awareness, indexing, schema inspection
		(where applicable), and the ability to manage or remove stored assets safely. Document
		Management provides the foundational infrastructure that transforms raw files into structured,
		searchable, and model-ready assets, ensuring that Bro’s intelligence features operate
		on reliable, well-governed data rather than unmanaged documents.  '''

RETRIEVAL_AUGMENTATION = '''Retrieval-Augmented Generation (RAG) improves LLM accuracy and relevance
		by fetching up-to-date, external data—such as documents, databases, or web results—and feeding
		it into the prompt before generating a response. It reduces hallucinations and eliminates the
		need to retrain models for new information.'''

SEMANTIC_SEARCH = '''LLM semantic search uses Large Language Models and embedding vectors to retrieve
		information based on conceptual meaning and user intent, rather than strict keyword matching.
		By converting documents and queries into numerical vector embeddings stored in a database,
		systems can find contextually relevant information, enabling more accurate, conversational,
		and nuanced search experiences, often used in RAG (Retrieval-Augmented Generation) systems.'''

USE_CHAT_HISTORY = '''When enabled, Bro includes prior user and assistant turns when constructing
		the current prompt. This helps preserve conversational continuity, allows follow-up
		questions to reference earlier context, and makes multi-turn interactions feel coherent.
		Disable it when you want each request to be handled as a fresh, isolated prompt.'''

USE_DOCUMENT_CONTEXT = '''When enabled, Bro appends shared document context stored in session state
		to the prompt. This is useful when you want generation to be influenced by previously
		selected excerpts, semantic-search results, or other document-derived context beyond
		the live user message. Disable it for purely standalone generation.'''

ANSWER_ONLY = '''When enabled, Bro instructs the model to return the answer directly with minimal
		prefatory narration. This is useful for concise responses, direct question answering,
		and structured workflows where extra explanation is undesirable. Disable it when you
		want fuller reasoning, framing, or narrative context in the response.'''

USE_SELF_CHECK = '''When enabled, Bro instructs the model to internally verify its conclusion before
		responding. This can improve care and consistency for reasoning-heavy tasks, though it
		may slightly increase response latency or verbosity depending on the prompt.'''

DETERMINISTIC_REASONING = '''When enabled, Bro biases the model toward stable, conservative reasoning
		and reduced variation across similar prompts. This is useful when you want less creative
		drift and more repeatable analytical behavior. It complements, but does not replace,
		temperature and sampling controls.'''

CODING_INCLUDE_COMMENTS = '''When enabled, Bro asks the model to include documentation comments
		and useful inline comments in generated code where appropriate. This is helpful for
		readability, maintainability, and teaching scenarios. Disable it when you want cleaner,
		minimal code with less commentary.'''

CODING_EDITOR_FORMAT = '''When enabled, Bro instructs the model to format code as editor-ready
		source rather than pseudo-code or conversational fragments. This is useful when the
		output is intended to be copied directly into an IDE, notebook, or source file.'''

CODING_FENCED_OUTPUT = '''When enabled, Bro wraps generated code in fenced Markdown code blocks.
		This improves readability in the UI and preserves formatting for copy/paste. Disable it
		when you prefer raw source text without Markdown fences.'''

USE_GROUNDING = '''When enabled, Bro indicates that responses should remain anchored to available
		context rather than drifting into unsupported generalization. In text generation this is
		a soft behavioral instruction; in document-oriented workflows, grounding is reinforced
		through retrieved evidence.'''

SHOW_RETRIEVED_CHUNKS = '''When enabled, Bro displays the document chunks retrieved for the current
		Document Q&A request. This makes retrieval behavior transparent, helps with debugging,
		and lets users inspect exactly what evidence informed the answer. Disable it for a
		cleaner chat experience.'''

REQUIRE_GROUNDING = '''When enabled, Document Q&A instructs the model to ground its answer in the
		retrieved document excerpts. This reduces unsupported claims and keeps responses tied to
		the active evidence base rather than general background knowledge.'''

ANSWER_FROM_EXCERPTS_ONLY = '''When enabled, Bro tells the model to answer only from the retrieved
		excerpts and to say clearly when the evidence is insufficient. This is useful when you
		want strict retrieval-based answering and minimal hallucination risk. Disable it when
		you are willing to allow broader model inference beyond the excerpts.'''

USE_SQLITE_VEC = '''When enabled, Bro attempts to use the sqlite-vec virtual table for vector
		retrieval in Document Q&A. This can provide fast nearest-neighbor lookup over document
		embeddings. Disable it if sqlite-vec is unavailable or if you want to force fallback
		retrieval behavior.'''

FALLBACK_COSINE_SEARCH = '''When enabled, Bro falls back to in-memory cosine-similarity search if
		sqlite-vec retrieval is unavailable or fails. This improves robustness and keeps Document
		Q&A usable even when vector-table support is not available, though it may be slower on
		larger document sets.'''

ENABLE_OCR = '''When enabled, Bro is permitted to use OCR-oriented parsing behavior for documents
		when native text extraction is inadequate. This is most useful for scanned PDFs or image-
		like documents where embedded text is missing or poor.'''

PREFER_NATIVE_PDF_TEXT = '''When enabled, Bro prioritizes native text extraction from PDFs before
		considering other parsing approaches. This is generally faster and cleaner for digital PDFs
		with embedded text. Disable it when native extraction is unreliable for the document set.'''

INCLUDE_PAGE_MARKERS = '''When enabled, Bro inserts page markers such as [Page N] into extracted
		document text. This helps preserve page locality, improves traceability during retrieval,
		and can make downstream answers easier to verify against the source document.'''

SHOW_DOC_PARSE_DIAGNOSTICS = '''When enabled, Bro displays document parsing and indexing diagnostics
		such as chunk size, overlap, vector readiness, and chunk counts. This is useful for
		debugging ingestion and retrieval behavior during development or evaluation.'''

SEMANTIC_CLEAR_EXISTING = '''When enabled, building a semantic index will clear the existing
		embeddings table before inserting new chunks. Use this when you want a fresh semantic
		search corpus rather than a cumulative one.'''

SEMANTIC_APPEND_EXISTING = '''When enabled, Bro appends new semantic chunks to the existing
		embeddings table instead of replacing prior content. This is useful when you want to
		accumulate multiple document sets into one searchable semantic index.'''

SEMANTIC_SHOW_DIAGNOSTICS = '''When enabled, Semantic Search displays index diagnostics such as
		document count, chunk count, and vector dimension. This helps validate indexing behavior
		and troubleshoot embedding workflows.'''

SEMANTIC_GROUP_BY_DOCUMENT = '''When enabled, Semantic Search is intended to group ranked results
		by source document rather than treating all chunks as one flat result set. In the current
		implementation this is primarily a UI intent flag and will be most useful once document-
		level grouping metadata is fully surfaced in the embeddings workflow.'''