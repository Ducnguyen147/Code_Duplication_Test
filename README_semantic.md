0) What the script does (in one minute)

Semantic channel (embeddings): encodes each file with a Sentence‑Transformers model (default: CodeBERT cd‑ft). Cosine between L2‑normalized vectors ≈ semantic similarity.

Structural channels (optional):

AST bag‑of‑features with Tree‑sitter (node types, parent→child edges, depth buckets), with TF‑IDF + optional centering to reduce generic syntax effects.

Lexical vector made from token n‑grams (default Python tokens with identifier anonymization) or raw character n‑grams.

Local fingerprints (optional but powerful on small code): token k‑grams with optional winnowing (MOSS) to select stable anchors; the script computes Jaccard‑like similarity, total overlap, and longest matching run to gate pairs before final scoring—these are the same coverage‑style metrics popularized by MOSS and used in Dolos. 
Semantic Scholar

1) The score you see

When --fusion late, a pair (i, j) passes through:

AISS prefilter on the embedding space: keep only neighbors with embed‑cosine ≥ --prefilter-threshold.

Structural gate(s): require ast_cos ≥ --min-ast-sim and/or lex_cos ≥ --min-lex-sim. (If both AST and Lex are off, gate is bypassed.)

Fingerprint gate (if enabled): require
fp_sim ≥ --min-fp-sim and fp_total ≥ --min-fp-total and fp_longest ≥ --min-fp-longest.
(Pairs with embed_cos ≥ --embed-superpass bypass the fp gate—useful for deep Type‑4 similarities.)

Final score
s = (w_embed·embed_cos + w_ast·max(ast_cos,0) + w_lex·max(lex_cos,0)) / (w_embed + w_ast + w_lex)
unless --allow-negative-structure is set, structural cosines are clipped at 0 (they can only help, not hurt).

2) Arguments, grouped and explained
A) Inputs & model

--dir PATH
Root folder to scan (recursively).

--extensions .py .java ...
Filters files by suffix. If omitted, all files under --dir are read.

--model NAME (default: mchochlov/codebert-base-cd-ft)
Sentence‑Transformers model for embeddings.

--batch-size INT (default: 16)
Embedding batch size; raise if you have GPU memory.

B) Fusion & thresholds

--fusion {late, concat} (default: late)
late = prefilter on embeddings, then re‑score with structure and fp gates.
concat = build one big vector and search once (fast, less controllable).

--prefilter-threshold FLOAT (default: 0.75)
Embedding cosine cut for the FAISS prefilter (before gates).
Lower this for tiny datasets so near‑misses aren’t thrown away too early.

--threshold FLOAT (default: 0.85)
Final fused‑score cut; pairs below are not reported.

C) Channel weights

--w-embed FLOAT (default: 1.0)
Weight of the semantic channel.

--w-ast FLOAT (default: 0.6)
Weight of AST structural channel (if active).

--w-lex FLOAT (default: 0.1)
Weight of lexical vector channel (if active).
Rule of thumb: start high on embeddings; add 0.2–0.6 AST for Type‑1/2/3 robustness; keep lex small (0.05–0.2) to avoid over‑rewarding boilerplate.

D) AST options (Tree‑sitter)

--no-ast
Disable AST channel completely.

--ast-dim INT (default: 2048)
Hash space for AST features; 2–8k are typical.

--ast-tfidf / --ast-no-tfidf (default: TF‑IDF ON)
Apply TF‑IDF to down‑weight ubiquitous syntax tokens.

--ast-stop-topk INT (default: 0)
Drop the k most frequent AST dims (by total count) before TF‑IDF.
Helpful on large sets (e.g., 32–64) to remove ultra‑common scaffolding.

--ast-center / --ast-no-center (default: OFF)
Mean‑center AST vectors after TF‑IDF; good on large N, but can be noisy on very small corpora.

E) Lexical vector options

--no-lex
Disable lexical vector channel.

--lex-mode {py-token, char} (default: py-token)
py-token = Python tokenizer with identifier anonymization (NAME → ID, NUMBER → NUM) and punctuation/keywords kept—similar to classic token‑based plagiarism detectors. 
arXiv

char = raw character n‑grams (language‑agnostic, more boilerplate‑prone).

--lex-n INT (default: 3)
N‑gram length (token or character). For tokens, 3–5 works well.

--lex-dim INT (default: 4096)
Hash space for lexical features.

F) Fingerprint (MOSS/Dolos‑style) gate

--no-fp
Disable fingerprint gating entirely (not recommended for tiny files).

--fp-k INT (default: 5)
Token k‑gram size used to form fingerprints. For short functions, use smaller values (3–4) to avoid “zero fingerprints”.

--fp-w INT (default: 4)
Winnowing window; 0 keeps all k‑grams. Standard winnowing picks the minimum hash per sliding window to create a stable, position‑aware subset (MOSS).

--min-fp-sim FLOAT (default: 0.12)
Minimum Jaccard‑like similarity of fingerprint multisets.

--min-fp-total INT (default: 4)
Minimum count of intersecting fingerprints (coverage).

--min-fp-longest INT (default: 2)
Minimum length of the longest common contiguous run of fingerprints (robust “long match” evidence).

These three gates mirror the similarity / total overlap / longest fragment metrics used by Dolos to balance recall and precision on short programs.