# Experiments

## 1. Method comparison

Compare online DPO, online GD, and weight steering. Let \(R_H\) and \(R_V\)
be the hidden- and visible-test rewards.

### Online DPO (formerly `rdiff`)

Both rewards use the same prompt and rollout:

$$
J_{\mathrm{online\text{-}dpo}}
=\mathbb{E}_{x,\,y\sim\pi_\theta(\cdot\mid x)}
\left[R_H(x,y)-R_V(x,y)+2R_H(x,y)R_V(x,y)\right].
$$

### Online GD

The rewards use independent prompts and rollouts:

$$
J_{\mathrm{online\text{-}gd}}
=\mathbb{E}_{\substack{x_a,x_b\sim\mathcal D \\
                        y_a\sim\pi_\theta(\cdot\mid x_a),\,
                        y_b\sim\pi_\theta(\cdot\mid x_b)}}
\left[
R_H(x_a,y_a)-R_V(x_b,y_b)
+2R_H(x_a,y_a)R_V(x_b,y_b)
\right].
$$

### Weight steering

Train separate models on \(R_H\) and \(R_V\), starting from \(\theta_0\), then
subtract their task vectors:

$$
\theta_{\mathrm{ws}}(\alpha)
=\theta_0+\alpha(\theta_H-\theta_V).
$$

This experiment asks whether combining directions after training performs
better than optimizing a combined reward online.

## 2. Weight-steering strength

Compare two ways to obtain a stronger steered model:

1. Increase the steering coefficient \(\alpha\).
2. Train \(\theta_H\) and \(\theta_V\) for more steps.

Shorter training may keep task-vector arithmetic in a more linear regime and
reduce capability degradation.

## 3. Cosine similarity across seeds

This experiment compares independently trained honesty, non-sycophancy, and
non-cheating directions for seeds 42–46. It forms the aligned directions

$$
(\Delta W_{\mathrm{Honest}}-\Delta W_\mathrm{Dishonest}),\quad
(\Delta W_{\mathrm{Non\text{-}Sycophantic}}-\Delta W_\mathrm{Sycophantic}),\quad
(\Delta W_{\mathrm{Non\text{-}Cheat}}-\Delta W_\mathrm{Cheat}).
$$

Submit or inspect the standalone, evaluation-free pipeline from this directory:

```bash
lilpipe configs/experiments/pipeline-cosine-similarity-across-seeds.yml --dry-run
lilpipe configs/experiments/pipeline-cosine-similarity-across-seeds.yml
```

The pipeline trains all 30 seeded adapters from the canonical behavior configs.
After training, run the analysis separately:

```bash
python scripts/analysis/compute_cosine_similarity_gen_results.py
python scripts/analysis/compute_cosine_similarity_plot_results.py \
  artifacts/analysis/cosine-similarity-across-seeds/results.json
```

It writes `results.json`, `cosine-similarity.pdf`, and
`layerwise-norms.pdf` under
`artifacts/analysis/cosine-similarity-across-seeds/`.
