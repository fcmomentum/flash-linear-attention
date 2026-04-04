# Centered DeltaNet and the Best Nonnegative Approximation

This note derives the approximation error for a nonnegative-feature surrogate to centered DeltaNet.

## Setup

Let

$$
\phi(x) \in \mathbb{R}_{\ge 0}^r,
\qquad
\nu = \mathbb{E}[\phi(x)] \in \mathbb{R}_{\ge 0}^r.
$$

The exact centered feature is

$$
\phi_c(x) = \phi(x) - \nu,
$$

and the exact centered kernel is

$$
K_c(q, k) = (\phi(q) - \nu)^\top (\phi(k) - \nu).
$$

Expanding,

$$
K_c(q, k)
= \phi(q)^\top \phi(k)
- \nu^\top \phi(q)
- \nu^\top \phi(k)
+ \|\nu\|^2.
$$

Define

$$
m(x) := \nu^\top \phi(x),
\qquad
c := \|\nu\|^2.
$$

Then

$$
K_c(q, k) = \phi(q)^\top \phi(k) - m(q) - m(k) + c.
$$

## One-Channel Nonnegative Approximation

To preserve nonnegative runtime features, keep the write/read feature as \(\phi(x)\), and subtract only the dominant mean-direction component through one auxiliary state.

Choose

$$
\lambda(x) := \frac{\nu^\top \phi(x)}{\|\nu\|^2 + \varepsilon}.
$$

Ignoring the small \(\varepsilon\) for the algebra, the induced approximate kernel is

$$
\tilde K_1(q, k) := \phi(q)^\top \phi(k) - \lambda(q)\lambda(k)\|\nu\|^2.
$$

Since \(\lambda(x)\|\nu\|^2 = \nu^\top \phi(x) = m(x)\), this can also be written as

$$
\tilde K_1(q, k)
= \phi(q)^\top \phi(k)
- \frac{m(q)m(k)}{\|\nu\|^2}.
$$

This removes the rank-1 component along the mean direction, while leaving the actual feature coefficients nonnegative.

Nonnegativity is preserved because the runtime feature vectors themselves are still the original nonnegative features \(\phi(q), \phi(k) \ge 0\). The subtraction happens only through the scalar correction term \(\lambda(q)\lambda(k)\|\nu\|^2\), or equivalently through an auxiliary value-space channel such as \(\lambda(x)b_t\). So the model never writes or reads with signed feature coordinates; it only debiases the final interaction by removing one rank-1 mean-mode contribution.

## Approximation Error

The kernel error is

$$
E_1(q, k) := \tilde K_1(q, k) - K_c(q, k).
$$

Substituting the two expressions,

$$
E_1(q, k)
=
\left[
\phi(q)^\top \phi(k)
- \frac{m(q)m(k)}{\|\nu\|^2}
\right]
-
\left[
\phi(q)^\top \phi(k) - m(q) - m(k) + \|\nu\|^2
\right].
$$

So

$$
E_1(q, k)
=
m(q) + m(k) - \|\nu\|^2
- \frac{m(q)m(k)}{\|\nu\|^2}.
$$

This factors cleanly:

$$
E_1(q, k)
=
-\frac{(m(q) - \|\nu\|^2)(m(k) - \|\nu\|^2)}{\|\nu\|^2}.
$$

Since

$$
m(x) - \|\nu\|^2
=
\nu^\top(\phi(x) - \nu)
=
\nu^\top \phi_c(x),
$$

the error can also be written as

$$
E_1(q, k)
=
-\frac{(\nu^\top \phi_c(q))(\nu^\top \phi_c(k))}{\|\nu\|^2}.
$$

## Interpretation

This shows exactly what the one-channel nonnegative approximation misses:

- It is exact on the subspace orthogonal to the mean direction \(\nu\).
- The residual error is a rank-1 term along the centered projection onto \(\nu\).
- Therefore the approximation is good when the centered fluctuations have small projection on \(\nu\).

Equivalently, if we decompose

$$
\phi(x) = \lambda(x)\nu + r(x),
\qquad
\nu^\top r(x) = 0,
$$

then the one-channel correction keeps the \(r(q)^\top r(k)\) interaction and drops only the remaining centered self-correlation along the \(\nu\) direction.

## Multi-Channel Approximation

If one mean direction is not enough, choose nonnegative anchor directions

$$
u_1, \dots, u_m \in \mathbb{R}_{\ge 0}^r
$$

with coefficients \(\lambda_j(x)\). The induced approximate kernel is

$$
\tilde K_m(q, k)
=
\phi(q)^\top \phi(k)
- \sum_{j=1}^m w_j \lambda_j(q)\lambda_j(k).
$$

If the anchors span the dominant biased subspace, then the error becomes

$$
E_m(q, k) = \tilde K_m(q, k) - K_c(q, k),
$$

which is the part of the centered kernel still living outside the chosen debiasing span.

In operator language:

- exact centering removes the full affine mean shift
- the nonnegative approximation removes only a low-rank projection of that shift

## Recurrence-Level Approximation

A practical one-channel recurrence is

$$
\hat v_t = S_{t-1}\phi(k_t) - \lambda(k_t)b_{t-1},
$$
$$
\delta_t = v_t - \hat v_t,
$$
$$
S_t = \alpha_t S_{t-1} + \beta_t \delta_t \phi(k_t)^\top,
$$
$$
b_t = \alpha_t b_{t-1} + \beta_t \delta_t,
$$
$$
y_t = S_t\phi(q_t) - \lambda(q_t)b_t.
$$

This is not exactly equal to the kernel formula above, because the online state dynamics also matter. But at the kernel level, its intended approximation target is \(\tilde K_1\), and the irreducible one-channel error is exactly

$$
E_1(q, k)
=
-\frac{(\nu^\top \phi_c(q))(\nu^\top \phi_c(k))}{\|\nu\|^2}.
$$

## When One Channel Is Enough

The one-channel approximation is strong when:

- the feature mean \(\nu\) dominates the kernel bias
- most centered variation is orthogonal to \(\nu\)
- the empirical kernel has one large spurious top eigenmode

It is weak when:

- the biased subspace has rank greater than one
- the feature covariance is strongly anisotropic near the mean direction
- multiple heads or layers have distinct mean directions that cannot be captured by one \(\nu\)

## Practical Takeaway

The best cheap nonnegative approximation to centered DeltaNet is a low-rank debiasing correction, not exact centering.

For one auxiliary channel, the exact kernel-level error is

$$
E_1(q, k)
=
-\frac{(\nu^\top \phi_c(q))(\nu^\top \phi_c(k))}{\|\nu\|^2}.
$$

So the approximation fails only through the centered component that still lies along the mean direction.

## Unified Projection View: XSA vs. Centered DeltaNet

Both Exclusive Self Attention (XSA) and centered DeltaNet can be written as "remove a projection," but the projection acts in different spaces.

### XSA

Let the standard attention output at token \(i\) be

$$
y_i = \sum_j a_{ij} v_j.
$$

Then the XSA output can be written explicitly as

$$
z_i = y_i - \frac{y_i^\top v_i}{\|v_i\|_2^2 + \varepsilon} v_i.
$$

Equivalently,

$$
z_i = (I - P_{v_i}) y_i,
\qquad
P_{v_i} := \frac{v_i v_i^\top}{\|v_i\|_2^2 + \varepsilon}.
$$

This guarantees

$$
z_i^\top v_i = 0.
$$

So XSA removes the component of the attention output aligned with the token's own value direction. This is:

- token-dependent
- value-dependent
- an output-space projection

### Rank-1 Centered DeltaNet Approximation

The one-channel nonnegative approximation can be written as a projected kernel:

$$
\tilde K_1(q, k) = \phi(q)^\top (I - P_\nu) \phi(k),
\qquad
P_\nu := \frac{\nu \nu^\top}{\|\nu\|^2}.
$$

Expanding,

$$
\phi(q)^\top (I - P_\nu) \phi(k)
= \phi(q)^\top \phi(k) - \frac{(\nu^\top \phi(q))(\nu^\top \phi(k))}{\|\nu\|^2}.
$$

So this approximation removes the rank-1 component along the global mean direction \(\nu\). This is:

- token-independent once \(\nu\) is fixed
- feature-dependent
- a kernel-space projection

### Exact Centered DeltaNet

The exact centered kernel is

$$
K_c(q, k) = (\phi(q) - \nu)^\top (\phi(k) - \nu).
$$

This is not just an orthogonal projection, because centering is an affine operation rather than a purely linear one.

### Bottom-Line Comparison

- XSA removes a self-aligned direction in output/value space.
- Rank-1 centered DeltaNet removes a mean direction in feature/kernel space.
- Exact centered DeltaNet performs affine centering in feature space.
- The one-channel nonnegative DeltaNet approximation is the closest projection-style analogue of XSA.

## Making DeltaNet More Like XSA

If the goal is to make DeltaNet behave more like XSA, the key change is this:

- centered DeltaNet removes a global mean direction in feature space
- XSA removes a token-dependent self direction in value space

So an XSA-like DeltaNet should stop trying to subtract a fixed mean direction \(\nu\), and instead project the prediction and readout away from the current token's own value direction.

### XSA-Like DeltaNet Readout

Let the standard DeltaNet readout be

$$
y_t = S_t \psi_t,
\qquad
\psi_t := \phi(q_t).
$$

Define the token-specific projector onto the current value direction:

$$
P_{v_t} := \frac{v_t v_t^\top}{\|v_t\|^2 + \varepsilon}.
$$

Then the XSA-like readout is

$$
y_t^{\mathrm{XDL}} := (I - P_{v_t}) S_t \psi_t.
$$

This is the closest analogue of XSA in DeltaNet form: the memory readout is forced to be orthogonal to the token's own value direction.

### XSA-Like Prediction

The same idea can be applied to the prediction used to form the residual:

$$
\hat v_t := (I - P_{v_t}) S_{t-1} \phi(k_t).
$$

Then the innovation becomes

$$
\delta_t := v_t - \hat v_t.
$$

The update remains

$$
S_t = \alpha_t S_{t-1} + \beta_t \, \delta_t \, \phi(k_t)^\top.
$$

### Full XSA-Like DeltaNet

A direct analogue of XSA is therefore

$$
P_{v_t} := \frac{v_t v_t^\top}{\|v_t\|^2 + \varepsilon},
$$
$$
\hat v_t = (I - P_{v_t}) S_{t-1} \phi(k_t),
$$
$$
\delta_t = v_t - \hat v_t,
$$
$$
S_t = \alpha_t S_{t-1} + \beta_t \, \delta_t \, \phi(k_t)^\top,
$$
$$
y_t = (I - P_{v_t}) S_t \phi(q_t).
$$

### Important Caveat

This is no longer a centered-kernel method.

- It does not remove the global DC bias caused by positive feature maps.
- It removes a token-specific self-aligned direction in output space.
- It is therefore conceptually much closer to XSA than to exact centered DeltaNet.

### Hybrid Version

If both issues matter, the natural hybrid is:

1. debias the kernel using centered or approximately centered features
2. apply an XSA-style output projection at prediction/readout time

For example,

$$
\hat v_t = (I - P_{v_t}) S_{t-1} \phi_c(k_t),
$$
$$
S_t = \alpha_t S_{t-1} + \beta_t \, \delta_t \, \phi_c(k_t)^\top,
$$
$$
y_t = (I - P_{v_t}) S_t \phi_c(q_t).
$$

This separates the two mechanisms cleanly:

- feature centering removes the global mean-mode bias
- output projection removes token-specific self-copy directions

### Practical Takeaway

To make centered DeltaNet "just like XSA," replace the global mean-direction subtraction with a token-dependent projection in value space. Strictly speaking, that produces an exclusive DeltaNet rather than a centered DeltaNet.
