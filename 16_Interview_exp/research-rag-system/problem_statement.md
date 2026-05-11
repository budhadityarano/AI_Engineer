# HackerRank-Style Interview Problem: Research Paper RAG System

## Context

You are building a Retrieval-Augmented Generation (RAG) system for a team of researchers.
Researchers upload PDF papers to the system. When they search for a topic, the system
returns the most relevant AND most recent papers.

The system has two layers of ranking:
1. **Semantic similarity** — how closely a paper chunk matches the query (via embeddings)
2. **Freshness** — how recent the paper is (derived from the year in the filename)

The interviewer wants to see that you can design a clean pipeline, handle edge cases,
and understand the tradeoffs between relevance and recency.

---

## System Overview

```
PDFs  →  extract text  →  chunk text  →  embed chunks  →  FAISS index
                                                               ↓
User query  →  embed query  →  search FAISS  →  rerank by freshness  →  results
```

---

## The 8 Functions You Must Implement

All functions live in a single file. You do NOT need real OpenAI keys or FAISS installed
to write these — focus on the logic. The `get_embedding()` and FAISS calls are stubs
you can treat as black boxes.

---

### 1. `get_embedding(text: str) -> list[float]`

Call the OpenAI `text-embedding-3-small` model on the input text.
Return the embedding as a list of floats.
Wrap the API call in a try/except and re-raise on error.

**Hint:** `client.embeddings.create(model=..., input=text)` → `.data[0].embedding`

---

### 2. `add_documents_to_store(chunks: list[dict], index, metadata_store: list) -> None`

Each chunk dict has keys: `text`, `title`, `source`, `freshness` (float 0–1), `chunk_id` (int).

- Embed each chunk's `text` using `get_embedding()`
- Collect embeddings into a float32 numpy matrix
- Call `index.add(matrix)` once
- Append each chunk dict to `metadata_store`

**Key constraint:** position `i` in `metadata_store` must always match vector `i` in the FAISS index.

---

### 3. `search_store(query: str, index, metadata_store: list, top_k: int = 5) -> list[dict]`

- Embed the query using `get_embedding()`
- Call `index.search(query_vector, top_k)` — returns `(distances, indices)`
- Skip indices equal to `-1` (FAISS sentinel for "no result")
- For each valid result, copy the metadata dict and add a `similarity_score` key:
  ```
  similarity_score = 1 / (1 + distance)   # converts L2 distance to 0–1 score
  ```
- Return the list sorted by `similarity_score` descending

---

### 4. `extract_text_from_pdf(pdf_path: str) -> str`

Use PyMuPDF (`import fitz`) to open the PDF, iterate over pages, and call `page.get_text()`.
Concatenate all pages with `"\n"` and return the result.
Handle `FileNotFoundError` and any general exception — return `""` on error.

---

### 5. `improved_chunker(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]`

**This is the key function the interviewer cares about most.**

Do NOT do naive fixed-size character slicing.

Algorithm:
1. Split the text on paragraph boundaries: `re.split(r"\n\n+", text)`
2. For each paragraph:
   - If it fits within `chunk_size`, treat it as one unit
   - If it's too large, split it further at sentence boundaries (`". "`, `"? "`, `"! "`)
3. Accumulate units into a `current_chunk` string
4. When adding the next unit would exceed `chunk_size`:
   - Save `current_chunk` to the results list
   - Start a new chunk, but **prepend the last `overlap` characters** of the previous chunk
     to maintain context continuity across the boundary
5. After the loop, don't forget to flush the final `current_chunk`
6. Strip and skip empty strings

Return a `list[str]` of non-empty chunk strings.

**Edge cases to handle:**
- Empty or whitespace-only input → return `[]`
- A single sentence shorter than `chunk_size` → return `[that_sentence]`
- Paragraphs much longer than `chunk_size` → must be further split at sentence level

---

### 6. `compute_freshness(filename: str, all_filenames: list[str]) -> float`

Extract a 4-digit year from a filename using regex.

- Pattern: `r"\b(19|20)\d{2}\b"` — matches years like 1990–2029
- If no year found in a filename, default to `2000`

Collect years from ALL filenames in `all_filenames`.
Normalize using min-max:
```
freshness = (year - min_year) / (max_year - min_year)
```
If all years are equal (or only one file), return `1.0`.

Return a float in [0.0, 1.0].

**Example:**
```
files = ["2021_paper.pdf", "2023_paper.pdf", "2024_paper.pdf"]
compute_freshness("2021_paper.pdf", files)  →  0.0
compute_freshness("2023_paper.pdf", files)  →  0.667
compute_freshness("2024_paper.pdf", files)  →  1.0
```

---

### 7. `rerank_with_freshness(results: list[dict], use_freshness: bool, freshness_weight: float = 0.3) -> list[dict]`

If `use_freshness` is `False`: return `results` unchanged (don't modify dicts).

If `use_freshness` is `True`:
- For each result dict, compute:
  ```
  final_score = (1 - freshness_weight) * similarity_score + freshness_weight * freshness
  ```
- Add `final_score` key to the dict
- Return sorted by `final_score` descending

---

### 8. `retrieve(query: str, index, metadata_store: list, use_freshness: bool = False, top_k: int = 5) -> list[dict]`

The top-level orchestrator. Simply:
1. Call `search_store(query, index, metadata_store, top_k)`
2. Call `rerank_with_freshness(results, use_freshness)`
3. Return the result

This is the function an external caller (e.g., a UI) would invoke.

---

## What the Interviewer is Evaluating

| Function | Skill being tested |
|----------|--------------------|
| `get_embedding` | API usage, error handling |
| `add_documents_to_store` | Index/metadata alignment, batch embedding |
| `search_store` | Vector search mechanics, score formula, sentinel handling |
| `extract_text_from_pdf` | Library usage, defensive coding |
| `improved_chunker` | **Core** — semantic splitting, overlap logic, edge cases |
| `compute_freshness` | Regex, normalization, edge case (equal years) |
| `rerank_with_freshness` | Hybrid scoring design, conditional mutation |
| `retrieve` | Clean orchestration, separation of concerns |

---

## Tips

- Start with `improved_chunker` — it has the most logic and is worth the most
- `compute_freshness` is a pure function — test it mentally with 3 filenames before coding
- For `search_store`, remember that L2 distance 0 = perfect match → score should be 1.0
- The `retrieve` function should be ~3 lines; if it's longer, you're doing too much there
- All functions are independently testable — the interviewer will unit-test each one
