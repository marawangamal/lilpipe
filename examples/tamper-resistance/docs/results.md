## Clean upstream SEAM reproduction

The previous SEAM models, evaluations, figures, and dataset caches were removed
before this reproduction. `examples/seam` is the pristine
[marawangamal/seam fork](https://github.com/marawangamal/seam), pinned to upstream
commit `fa7224f8754585786c850f327b08f32a6976420e`. The previous adapted source is
preserved under `examples/seam-v1`.

The two defenses share the original upstream dataset construction, tokenization,
and trainer:

| Defense | Training duration | Effective batch size | LR | α | β | ε |
|---|---|---:|---|---:|---:|---:|
| Codebase config | 1 epoch (1,000 steps) | 8 | 2e-5 | 1 | 0.01 | 0.001 |
| Paper config | 500 steps | 8 | 2e-5 | 1 | 0.01 | 0.001 |

The codebase run uses the fork's unchanged `config/train.yaml`. The paper run uses
[the paper config](../configs/stages-beavertails/0_alignment/llama2-7b-beavertail--align-fft-seam--upstream-paper-config/selected.yml),
following [Appendix A](https://arxiv.org/html/2505.12186#A1).

For both defenses and their attacks, harmful completions use
`Question: {prompt}\nAnswer:\n{answer}`. Refusals are read directly from the fresh
training JSON and use `Question: {prompt}\nAnswer: {refusal}`. No formatting or
masking changes are applied.

Each defended model receives the authors' five attack learning rates
(2e-5, 5e-5, 8e-5, 1e-4, 2e-4), using `config/eval.yaml`: 1,000 examples,
batch size 4, one epoch (250 steps), AdamW, cosine scheduling, 10% warmup, and
weight decay 0.01. Pre/post utility evaluation is performed separately for the
saved models. An external adapter persists final weights and JSON results;
it calls the original training method without replacing its implementation.
GPT API judging is skipped. Harmfulness uses the authors' DeBERTa evaluator;
capability is the mean of MMLU, TruthfulQA MC1, ARC-Easy normalized accuracy,
and HellaSwag normalized accuracy, all zero-shot.
Harmfulness generation retains the pretrained sampling defaults: temperature 0.6,
top-p 0.9, and 50 new tokens. The standalone upstream evaluation does not
explicitly seed its random generators.

Fresh dataset downloads were prepared and verified offline under
`$SCRATCH/lilpipe/seam-clean/`. Cached pretrained Llama-2 and DeBERTa weights
are reused. All packages pinned in upstream `requirements.txt` match the
execution environment.

[Submission and source provenance](../artifacts/tamia/runs/seam-clean-reproduction/jobs.json),
[cache preparation](../artifacts/tamia/runs/seam-clean-reproduction/preparation.json),
and [deletion inventory](../artifacts/tamia/runs/seam-clean-reproduction/cleanup.json)
record the exact inputs and jobs. Results and the comparison figure will be
added after both job chains complete.
