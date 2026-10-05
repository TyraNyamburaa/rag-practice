# Research-Paper RAG for Deep Learning in Skin Lesion Detection

## What is this?

This is a Retrieval-Augmented Generation (RAG) system that lets you ask questions about what published
open-access research papers say on a specific topic, and get back answers that are grounded in those
papers with citations. The topic here is deep learning for skin lesion (melanoma) detection: which
datasets the studies use, which model architectures they employ, what metrics they report, and what
limitations the authors admit.

The whole thing is built as a single Jupyter notebook (`rag-skin-lesion-research.ipynb`) that walks through
the full RAG pipeline in eight tasks, from collecting documents all the way through to an evaluation set
and an interactive command-line interface.

This was written as a coursework project. It is a prototype, not a production system, and it explicitly
includes guardrails so it never passes itself off as medical advice.

## The pipeline, step by step

The notebook is organised into eight tasks. Here is what each one does.

**Task 1 -- Document collection.** Searches the Europe PMC REST API for open-access papers on deep
learning and skin lesions published between 2019 and 2026. For each paper that has a PMCID and an
acceptable Creative Commons licence (CC BY, CC BY-SA, CC BY-NC, or CC0, but not NoDerivatives), it
downloads the full-text JATS XML and converts it to plain text with section headings preserved
(Abstract, Methods, Results, etc.). The result is 40 `.txt` files in `rag/data/raw/`, plus a
`collection_log.json` that records the source URL, licence, title, year, DOI, and file size for every
document.

**Task 2 -- Cleaning.** Normalises Unicode ligatures and odd spaces, removes empty brackets left behind
by stripped citation markers, fixes spaces before punctuation, collapses extra whitespace, and drops
exact-duplicate documents by content hash. The output goes into `rag/data/clean/`. Section headings and
all numbers, units, and abbreviations are intentionally left untouched, because those are what users
will ask about later.

**Task 3 -- Chunking and metadata.** Compares two chunking strategies: a fixed-size character window
with overlap, and a section-aware paragraph packing approach. The section-aware strategy is chosen
because it keeps chunks inside one logical section and prefixes each chunk with its section name, which
matters for questions like "what limitations do the authors report?" The chunk size is set to 750
characters (roughly 170-200 tokens) so it fits within the embedding model's context window.

**Task 4 -- Embeddings.** Uses the `all-MiniLM-L6-v2` sentence-transformers model (384 dimensions, runs
locally on CPU) to embed every chunk. Before embedding, it counts tokens in each chunk and reports any
that would be silently truncated by the model's 256-token limit. It also does a quick sanity check by
finding the most and least similar pairs of chunks across different papers.

**Task 5 -- Vector storage and retrieval.** Stores all chunk embeddings in ChromaDB, a local
open-source vector database, using cosine distance (configured explicitly since ChromaDB defaults to
L2). The database is persisted to disk as `./chroma_db`, and the notebook demonstrates reloading it
without re-embedding. A `retrieve()` helper embeds a query and returns the top-k most similar chunks
along with similarity scores and metadata.

**Task 6 -- LLM integration and prompting.** Connects to the OpenAI API using `gpt-4o-mini` (key read
from a `.env` file). It compares a naive prompt against a carefully designed final prompt that uses
role-setting, structured XML context blocks, explicit abstention rules, few-shot examples, and an
injection defense. Temperature is set to 0.1 for factual accuracy.

**Task 7 -- Interactive interface and guardrails.** Two layers of protection before the LLM is called:
first, a relevance threshold that declines to answer if the top retrieved chunk is below a similarity
cut-off (calibrated by printing top scores for in-scope and out-of-scope queries); second, the prompt's
own abstention rule. The result is a command-line interface where you type questions and get cited,
sourced answers.

**Task 8 -- Evaluation.** Runs a 10-question test set (6 single-document factual, 2 multi-document
synthesis, 2 out-of-scope) and produces a comparison table checking whether the right chunk was
retrieved and whether the answer is faithful to the cited sources.

## Project structure

```
rag-practice/
  README.md                       this file
  rag/
    rag-skin-lesion-research.ipynb  the main notebook (run this)
    requirements.txt                pinned Python dependencies
    .env                            OpenAI API key (not committed)
    .gitignore                      ignores venv, .env, __pycache__
    data/
      raw/                          collected papers as plain text (40 files)
          collection_log.json        metadata for every collected paper
      clean/                        cleaned, de-duplicated papers (40 files)
    chroma_db/                      persisted ChromaDB vector database (created on first run)
```

## Who is this for

A student or junior researcher who wants to quickly get cited answers to questions like "which datasets
do these papers use?" or "what limitations do the authors report?" without reading 40 papers end to end.
It is also a worked example of a complete RAG pipeline, showing every design decision and trade-off
along the way, rather than hiding behind a framework.

## What questions can it answer

Examples of in-scope questions:

- Which public datasets (such as ISIC or HAM10000) do the studies use for training and evaluation?
- Which pretrained CNN or Vision Transformer architectures do the studies report using?
- What limitations do the authors report about generalising across skin tones or imaging devices?
- What accuracy or AUC figures do specific papers report?

What it should NOT answer:

- "What is the recommended treatment for stage III melanoma?" The corpus is about machine-learning
  detection methods, not clinical treatment guidance. The system is designed to abstain from this.
- Any question whose topic is not covered by the 40 collected papers.

## How to run it

You need Python 3.9+ and a free Europe PMC account is not required (the API is open), but you do need
an OpenAI API key for the LLM step.

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r rag/requirements.txt

# 3. Set your OpenAI API key
#    Option A: export it in your shell
export OPENAI_API_KEY="your-key-here"
#    Option B: create rag/.env with OPENAI_API_KEY=your-key-here

# 4. Run the notebook
jupyter notebook rag/rag-skin-lesion-research.ipynb
```

Run the cells from top to bottom. Europe PMC asks API users to identify themselves, so make sure to
update the contact email in the `HEADERS` dict in Task 1 if you are reusing this for a different project.

The first run will download the `all-MiniLM-L6-v2` embedding model from Hugging Face (cached
afterwards) and will call the OpenAI API to generate answers, which incurs cost. Re-running the
notebook will reuse the saved ChromaDB database and skip re-embedding.

## Key design choices and why

**Embedding model: `all-MiniLM-L6-v2`.** It is open-source, small enough to run on a laptop CPU, and a
well-tested general-purpose model for semantic search. The 384-dimension output and 256-token context
window are verified (not assumed) in Task 4.

**Chunk size: 750 characters with no overlap.** This is large enough to hold roughly one idea but small
enough to fit the embedding model's token limit. Overlap is omitted because chunks already end at
sentence and paragraph boundaries, so overlap would mainly add duplicate text to the index.

**Section-aware chunking.** The fixed-window strategy was compared against it and found to cut sentences
in half and mix text from different sections (for example, the end of Methods and the start of Results)
in the same chunk. Section-aware chunks stay inside one section and carry its name, which helps
questions about limitations land on the Discussion section text.

**Cosine distance in ChromaDB.** This is the standard choice for text embeddings because it compares
direction (meaning) rather than vector length. It is set explicitly because ChromaDB defaults to squared
L2.

**Low temperature (0.1).** This is a factual, citation-grounded task, so we want the model to add as
little of its own variation as possible.

## Guardrails

This project is deliberately careful about not giving medical advice and not making up answers:

1. **Relevance threshold.** Before the LLM is called, the best retrieved chunk's similarity score is
   checked against a threshold (default 0.30, calibrated on real data). If it is below the threshold,
   the system declines without calling the LLM at all.

2. **Explicit abstention rule in the prompt.** If the retrieved sources do not contain enough
   information, the LLM is instructed to reply with a fixed sentence: "The retrieved papers don't cover
   this."

3. **Out-of-scope questions are tested.** The test set includes questions like "What is the recommended
   treatment for stage III melanoma?" and "What is the capital city of Kenya?" to verify the system
   abstains when the topic is not covered by the corpus.

4. **No medical advice disclaimer.** Every CLI response is appended with a disclaimer stating this is a
   coursework prototype that summarises research, not medical advice.

5. **Source-attributed answers.** Every answer from the final prompt includes a `SOURCES:` line listing
   which numbered chunks were used, so claims can be traced back to their origin.

6. **Prompt injection defense.** Text inside the context block is treated strictly as data to quote and
   cite, never as instructions to follow.

## Evaluation

Task 8 includes a 10-question evaluation set defined before running, so the test is not biased by seeing
results first. The set covers:

- Six single-document factual questions (dataset used, architecture, metrics, limitations, sample size).
- Two multi-document synthesis questions (comparing datasets across studies, common limitations).
- Two out-of-scope questions that the system must abstain from.

The evaluation checks whether the right source documents were retrieved and whether the generated
answer is faithful to the cited sources. Questions are written with placeholder topics and empty
`expected_files` lists that you fill in by inspecting your own corpus first, to keep the evaluation
honest.

## Dependencies

The main libraries used are:

- `requests` for calling the Europe PMC API.
- `pandas` for logging and evaluation tables.
- `sentence-transformers` for the embedding model.
- `chromadb` for the local vector database.
- `openai` for the LLM API.
- `python-dotenv` for loading the API key from `.env`.

A full pinned list is in `rag/requirements.txt`.

## Notes and caveats

- This downloads papers at the time the notebook is run, so the set of 40 documents may differ slightly
  if the Europe PMC index has changed. The collection log records exactly what was fetched.
- The OpenAI API key is stored in a `.env` file that is git-ignored and should not be committed.
- The ChromaDB database is persisted to `rag/chroma_db` and will be reused on subsequent runs unless the
  collection is explicitly deleted.
- All papers are open-access and the licence for each is recorded individually in the collection log.
  NoDerivatives articles are filtered out because the pipeline splits and transforms the text.
