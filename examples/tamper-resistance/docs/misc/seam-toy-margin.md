# SEAM toy experiment

## Setup

The model is an MLP with 12 input features, one 24-unit `tanh` hidden layer,
and three output logits:

$$
z(x) = W_2\tanh(W_1x+b_1)+b_2 \in \mathbb{R}^3.
$$

Classes 0 and 1 are the two utility classes. Class 2 represents the harmful
behavior. The utility set contains 512 synthetic examples with binary labels
$y\in\{0,1\}$. Its twelfth feature is always zero, and its label is determined
by a random linear classifier over the other 11 features.

The defended set is a balanced subset of 64 utility examples: 32 from each
utility class. We copy these examples and set their twelfth feature to 5, which
acts as a trigger. Their safe labels remain their original utility labels,
$y\in\{0,1\}$.

Initial training uses both the utility examples and the triggered examples with
their safe labels. We then copy this same trained model to produce the Base and
each SEAM run.

## Defense and relearning

SEAM defense combines three components:

- $L_{\mathrm{adv}}$ increases the loss of predicting harmful class 2, including
  an auxiliary loss obtained by applying the output layer directly to the
  hidden-layer preactivation (before `tanh`).
- $L_{\mathrm{align}}$ preserves each triggered example's original safe label.
- $g_{\mathrm{sd}}$ is the finite-difference self-destruction gradient computed
  from the harmful-task and utility-task gradients.

The installed defense gradient is

$$
g_{\mathrm{defense}}
= g_{\mathrm{adv}} + \alpha g_{\mathrm{align}} + \beta g_{\mathrm{sd}},
$$

with $\alpha=1$. We sweep $\beta\in\{0.1,0.5,1,5\}$.

During relearning, the attacker reuses exactly the same 64 triggered examples,
but changes every target to harmful class 2. Harmful accuracy is therefore

$$
\frac{1}{64}\sum_i
\mathbf{1}\!\left[\arg\max_{c\in\{0,1,2\}}z_c(x_i)=2\right].
$$

Utility accuracy is measured on the original 512 untriggered binary examples.

## Why larger $\beta$ can make initial relearning easier

For a triggered example, define the harmful margin as

$$
m(x)=z_2(x)-\max\left(z_0(x),z_1(x)\right).
$$

A negative margin means the model predicts a safe class, zero is the decision
boundary, and a positive margin means it predicts harmful class 2. Immediately
after defense, all SEAM models have 0% harmful accuracy, but their confidence
differs:

| $\beta$ | Harmful-target cross-entropy | Mean harmful margin |
| ---: | ---: | ---: |
| 0.1 | 77.9 | -77.9 |
| 0.5 | 63.7 | -63.7 |
| 1 | 54.7 | -54.7 |
| 5 | 12.7 | -12.7 |

The $\beta=5$ model is therefore much closer to the harmful decision boundary
than the $\beta=0.1$ model, despite both having 0% harmful accuracy. It takes
less relearning to make $z_2$ the largest logit.

This is not contradictory: $\beta$ weights the term intended to make future
harmful relearning damage utility. It does not directly guarantee greater
resistance to harmful relearning. In this toy setup, a large $\beta$ can let
$g_{\mathrm{sd}}$ dominate $g_{\mathrm{adv}}$, trading weaker initial resistance
for greater subsequent utility damage.
