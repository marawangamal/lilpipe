# Cone projection from inner products

Let (V=[v_1,\ldots,v_N]) contain the cone generators and let (x) be the
target. The distance to their positive cone is

$$
\min_{\alpha\geq0}\|x-V\alpha\|_2.
$$

Expanding the squared objective gives

$$
\|x-V\alpha\|_2^2
=x^\top x-2\alpha^\top V^\top x+\alpha^\top V^\top V\alpha.
$$

Define (G=V^\top V) and (b=V^\top x). Since (x^\top x) is constant, we
solve the bound-constrained quadratic problem

$$
\alpha^\star=\arg\min_{\alpha\geq0}
\left(\alpha^\top G\alpha-2\alpha^\top b\right),
$$

then report

$$
d=\sqrt{x^\top x-2(\alpha^\star)^\top b
+(\alpha^\star)^\top G\alpha^\star}.
$$

Thus the dense LoRA update vectors are never materialized: the low-rank
inner-product routine only needs to compute (G), (b), and (x^\top x).
