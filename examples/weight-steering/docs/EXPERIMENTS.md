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
