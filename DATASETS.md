# Datasets

Every dataset below was checked for availability, license and label provenance on
2026-10-09 before use. Raw files are downloaded by `judgment-calls prepare` into
`data/raw/` (git-ignored); the fixed samples used for every number in this repo are
committed in `data/samples/` together with `manifest.json` (seed `20261009`).

## Used

### 1. Faithfulness / hallucination detection: RAGTruth

| | |
|---|---|
| Source | <https://github.com/ParticleMedia/RAGTruth> (`dataset/response.jsonl`, `dataset/source_info.jsonl`) |
| License | MIT (repository license, checked via the GitHub API). The underlying source texts come from MS MARCO (QA), Yelp Open Dataset (data-to-text) and CNN/Daily Mail (summarization), each under its own terms; this repo uses them for non-commercial research evaluation only. |
| Paper | Niu et al., *RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models*, ACL 2024 |
| How labels were made | Human annotators marked hallucinated spans (word level, with types such as evident/subtle conflict or baseless information) in responses written by six LLMs to retrieval-augmented prompts. |
| Our label | `hallucination = true` iff the response has at least one annotated span. Only the official `test` split, and only responses with `quality == "good"` (incorrect refusals and truncations removed): 2,675 responses, 35% hallucinated. |
| Our sample | 200 responses, 100 per label, spread evenly over four length buckets where available (0–700 / 700–1,400 / 1,400–2,100 / ≥2,100 approx. tokens: 56 / 55 / 55 / 34). |
| Judge input | `state = {"response": ..., "source": <the full generation prompt incl. reference material>}`. The response comes first so that judges that read only a prefix still see the text being judged. |

### 2. Rubric scoring: SummEval

| | |
|---|---|
| Source | Hugging Face `mteb/summeval` (`data/test-*.parquet`), a copy of the SummEval expert annotations |
| License | MIT (dataset card; the original Yale-LILY/SummEval repository is also MIT). Articles are CNN/Daily Mail news text. |
| Paper | Fabbri et al., *SummEval: Re-evaluating Summarization Evaluation*, TACL 2021 |
| How labels were made | 1,600 summaries (100 CNN/DM test articles × 16 summarization systems) were rated 1–5 on coherence, consistency, fluency and relevance by three expert annotators; the `mteb` copy stores the per-summary mean of the expert ratings. |
| Our label | Mean expert rating per dimension, shifted to the 0–4 level-index scale. QWK and kappa use the rounded mean. Human–human agreement cannot be computed from this copy (individual ratings are not included). |
| Our sample | 200 summaries stratified by the rounded mean over the four dimensions (0–4: 6 / 43 / 51 / 50 / 50, i.e. all low-quality summaries that exist plus an even split of the rest). Articles are short (max ~800 tokens), so this task has no long-input bucket. |
| Judge input | `state = {"summary": ..., "article": ...}`; one call answers all four `score` questions. |

### 3. Pairwise preference with ties: MT-Bench human judgments

| | |
|---|---|
| Source | Hugging Face `lmsys/mt_bench_human_judgments` (`human` split) |
| License | CC BY 4.0 (dataset card) |
| Paper | Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, NeurIPS 2023 Datasets & Benchmarks |
| How labels were made | 3,355 pairwise votes (A / B / tie) by expert annotators (65 distinct annotator ids in the file) comparing answers of six models to the 80 MT-Bench questions, for turn 1 and turn 2. |
| Our label | Votes are pooled per (question, model pair, turn) after putting the two models in a canonical order; the label is the strict majority. 202 of 1,814 comparisons have no strict majority and are dropped, leaving 1,612. |
| Human ceiling | On the 961 comparisons with several votes: pairwise human–human agreement is **66% with ties** (2,368 vote pairs) and **83% without ties** (1,550 pairs), in line with the original paper. |
| Our sample | 150 comparisons, 50 per label (A / B / tie), spread over the four length buckets (42 / 41 / 38 / 29). Reduced from the planned 200 to fit the budget (each item is judged twice). |
| Judge input | Turn 1: `{"user_message", "answer_A", "answer_B"}`. Turn 2: `{"first_user_message", "first_answer_A", "first_answer_B", "follow_up_user_message", "final_answer_A", "final_answer_B"}`. Every item is also judged with all `_A`/`_B` fields swapped. |

## Considered and not used

- **LLM-AggreFact** (`lytang/LLM-AggreFact`): the preferred faithfulness benchmark, but it is
  *gated* on Hugging Face (anonymous download returns HTTP 401; access requires accepting terms
  with an account) and licensed **CC BY-ND 4.0**, which makes committing a derived sample awkward.
  RAGTruth is openly downloadable, MIT-licensed and human-annotated, so it was used instead.
- **A fresh, author-written contamination-check set**: planned as optional, not built. The API key
  ran out of credit (see README), so it could not have been evaluated anyway.

## Contamination risk

All three datasets have been public since 2021–2024 and are widely used to evaluate LLM judges, so
any of the evaluated models may have seen the texts and possibly the labels during training. We
have no way to measure this for closed models. The risk applies to both judge families, so it
matters less for the *relative* comparison than for absolute numbers, but it is not zero: a
model trained specifically for judging (the decision models) could have been tuned on exactly
these benchmarks. Treat absolute scores as optimistic.
