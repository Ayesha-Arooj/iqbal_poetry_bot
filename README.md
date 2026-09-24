# Allama Iqbal Poetry Bot

A hybrid retrieval chatbot that lets you ask questions about themes in
Allama Muhammad Iqbal's poetry and get back real, verbatim shers
(couplets) — Urdu text, English translation, and Latin
transliteration — plus a short AI-generated explanation of how they
relate to your question.

Retrieval combines **semantic search** (FAISS + multilingual sentence
embeddings) and **keyword search** (BM25), fused with **Reciprocal
Rank Fusion**. Generation uses **Qwen3.5-0.8B** running locally via
Hugging Face `transformers`.

The whole project now lives in a **single notebook**
(`iqbal_poetry_bot.ipynb`), run top to bottom. There's a
`REBUILD_INDEX` toggle near the top so you only pay the cost of
rebuilding the dataset/embeddings/FAISS index when your source poems
actually change — day-to-day, you can skip straight to loading and
chatting.

**Data source:** the source poems (`poems/*.yaml`) are from
[AzeemGhumman/iqbal-demystified-dataset](https://github.com/AzeemGhumman/iqbal-demystified-dataset).

---

## How it works

```
poems/*.yaml
     │
     ▼
Build dataset            (skipped if REBUILD_INDEX = False)
     →  iqbal_shers.json
     │
     ▼
Build embeddings + FAISS index   (skipped if REBUILD_INDEX = False)
     →  iqbal_shers.faiss
     →  iqbal_shers_embeddings.npy
     │
     ▼
Load dataset, FAISS index, BM25   (always runs)
     │
     ▼
Load Qwen3.5-0.8B + generation config
     │
     ▼
hybrid_search() + ask_iqbal_bot() definitions
     │
     ▼
Interactive chat loop
```

Set `REBUILD_INDEX = True` the first time you run the notebook, or
any time you edit the source YAML poems. Leave it `False` on later
runs to skip straight to loading the already-built JSON/FAISS files
and start chatting faster.

---

## Pipeline stages

### 1. Build the dataset

Reads every `*.yaml` file in `poems_folder`, extracts each sher
(couplet) with its poem title and ID, and writes a single flat JSON
file (`sher_output_file`). This JSON is the source of truth for
everything downstream — the embeddings, the BM25 index, and the exact
text shown to users in chat.

Cleanup applied here:
- Skips empty or whitespace-only shers.
- Skips very short or placeholder entries (e.g. `"asd"`, `"test"`,
  `"tbd"`, `"n/a"`, or anything under 10 characters) that shouldn't be
  in the corpus.
- Falls back to `"Untitled"` when a poem has no heading, instead of an
  empty title.
- Reports how many entries were skipped so you can sanity-check your
  source data.

**Known limitation:** the placeholder denylist is a fixed set of
strings. It won't catch every kind of stub or junk text in the source
YAMLs — periodically `grep` your raw files for other obvious
placeholders and extend the list as needed.

### 2. Build embeddings + FAISS index

Encodes every sher's text with
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` and
builds a FAISS `IndexFlatIP` index.

**Important:** embeddings are encoded with `normalize_embeddings=True`.
This is required for `IndexFlatIP` to behave as true cosine
similarity — without normalization, inner-product scores are skewed
by each vector's magnitude rather than its direction, which quietly
degrades search relevance. If you ever add a new encoding step
elsewhere in the notebook, keep this flag consistent or the FAISS
scores stop being meaningful.

This step is skipped when `REBUILD_INDEX = False`. If you ever see a
size-mismatch warning when loading, it means the JSON changed but the
index wasn't rebuilt — set `REBUILD_INDEX = True` and run again.

### 3. Load, hybrid search, and chat

- Loads the JSON dataset and FAISS index and builds BM25 in memory —
  this always runs, regardless of the rebuild toggle.
- Loads the Qwen model and fixes up its generation config.
- **Retrieval:** `hybrid_search()` runs FAISS and BM25 independently,
  then fuses their rankings with **Reciprocal Rank Fusion (RRF)**
  rather than summing raw scores. Raw BM25 scores and raw FAISS scores
  live on incompatible scales, so summing them directly lets whichever
  method produces bigger numbers dominate the ranking regardless of
  actual relevance. RRF instead scores each candidate by
  `1 / (rrf_k + rank)` from each method and sums *that*, which only
  depends on rank position — a standard, scale-independent way to
  combine two retrieval methods.
- **Display:** the retrieved poetry is printed to the user **exactly
  as stored in the JSON** — the LLM never touches this text. This
  guarantees the Urdu verse shown is always authentic and never
  paraphrased or altered by generation.
- **Generation:** the LLM is only asked to *explain* how the
  already-displayed entries relate to the question — explicitly
  instructed not to re-quote, translate, or invent verses. This keeps
  the one part of the pipeline prone to hallucination (the LLM) away
  from the one part that must be exact (the poetry itself).
- Qwen3.5 emits a `<think>...</think>` reasoning block before its real
  answer; `strip_think()` removes it so only the final explanation is
  shown.
- The final section is the interactive `input()` chat loop.

---

## Setup

```bash
pip install faiss-cpu rank_bm25 sentence-transformers transformers pyyaml
```

Expected file layout (adjust paths at the top of the notebook if
yours differs):

```
/content/drive/MyDrive/project/
├── poems/                       # source *.yaml files
└── output/
    ├── iqbal_shers.json
    ├── iqbal_shers.faiss
    └── iqbal_shers_embeddings.npy
```

**First run:** set `REBUILD_INDEX = True`, then Run All.

**Later runs (no changes to source poems):** set
`REBUILD_INDEX = False`, then Run All — this skips the dataset and
index-building steps and loads the existing files directly, so you
reach the chat loop much faster.

---

## Known limitations

- **Generation is slow without a GPU.** Qwen3.5-0.8B on CPU can take
  a couple of minutes per response, especially with the `<think>`
  reasoning block eating into the token budget. Check `!nvidia-smi`
  in Colab and switch to a GPU runtime if available
  (Runtime → Change runtime type). Lowering `max_new_tokens` (currently
  600) also helps.
- **Small models hallucinate under interpretive pressure.** Retrieval
  has been solid in testing, but the 0.8B explanation model has, on
  occasion, invented meanings, mistranslated phrases, or produced
  circular reasoning — even though it's explicitly instructed not to
  fabricate. This happens unpredictably (not on every query), which
  makes it hard to catch in advance. Treat the "Explanation" section
  as AI interpretation, not a scholarly source — the "Retrieved
  Poetry" section above it is the only part guaranteed to be authentic.
  If explanation quality matters a lot, try a larger Qwen variant
  (1.8B/3B+) — instruction-following on "don't invent details"
  improves noticeably with model size.
- **Retrieval can be sensitive to exact query phrasing.** Near-identical
  queries (e.g. "poetry about books" vs. "any poetry about books")
  have produced different top-3 results in testing. If this matters,
  try raising `candidate_k` (currently 50) and/or lowering `rrf_k`
  (currently 60) in `hybrid_search()` — more candidates and a smaller
  RRF constant both make top-ranked matches count for relatively more,
  which tends to stabilize results across rephrasings.
- **BM25 tokenization is naive** (`.split()` on whitespace). It works
  reasonably because each sher's `text` field includes an English
  translation alongside the Urdu, but it isn't stemmed or
  punctuation-aware. Good enough for this project's scale; worth
  revisiting if search quality needs to improve further.

## Troubleshooting

- **`Both max_new_tokens and max_length seem to have been set` warning:**
  cosmetic only (`max_new_tokens` always takes precedence), silenced by
  deleting `generation_config.max_length` rather than setting it to
  `None` — setting it to `None` doesn't clear the model's baked-in
  default of 20, it just gets shadowed.
- **FAISS index size doesn't match JSON entry count:** set
  `REBUILD_INDEX = True` and re-run — this happens when the dataset
  changed but the index wasn't rebuilt.
- **No response printed after a query:** most likely the model is
  still generating (especially on CPU) rather than stuck — errors are
  caught and printed explicitly in the chat loop, so silence usually
  means "still working."
