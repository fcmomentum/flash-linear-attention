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

## How the Rank-1 DC Removal Would Map to a Triton Kernel

The recurrent rank-1 DC-removal update is

$$
\hat v_t = S_{t-1} \phi(k_t) - \lambda_k(t) b_{t-1},
$$
$$
\delta_t = \beta_t (v_t - \hat v_t),
$$
$$
S_t = \alpha_t S_{t-1} + \delta_t \phi(k_t)^\top,
$$
$$
b_t = \alpha_t b_{t-1} + \delta_t,
$$
$$
y_t = S_t \phi(q_t) - \lambda_q(t) b_t.
$$

Here:

- \(S_t \in \mathbb{R}^{K \times V}\) is the usual DeltaNet state
- \(b_t \in \mathbb{R}^{V}\) is the extra bias state
- \(\lambda_k(t), \lambda_q(t)\) are scalar coefficients per token and head

### Fused Recurrent Triton Kernel

For the fused recurrent kernel, the implementation is conceptually straightforward.

At each time step, the Triton kernel already holds one tile of the recurrent state \(S_t\) in SRAM/registers. To support rank-1 DC removal, it additionally keeps one vector tile of \(b_t\).

The per-step kernel logic becomes:

$$
\text{pred}_t = S_{t-1} \phi(k_t) - \lambda_k(t) b_{t-1},
$$
$$
\delta_t = \beta_t (v_t - \text{pred}_t),
$$
$$
S_t = \alpha_t S_{t-1} + \delta_t \phi(k_t)^\top,
$$
$$
b_t = \alpha_t b_{t-1} + \delta_t,
$$
$$
y_t = S_t \phi(q_t) - \lambda_q(t) b_t.
$$

So compared to the standard fused recurrent kernel, the only additional state is one \(V\)-vector per head, and the only additional arithmetic is:

- one subtraction using \(\lambda_k(t) b_{t-1}\) in prediction
- one recurrence update for \(b_t\)
- one subtraction using \(\lambda_q(t) b_t\) in readout

This is a good fit for Triton because it preserves the same time-serial structure as the existing recurrent kernel.

### Why the Chunked Kernel Is Harder

The chunked Gated DeltaNet kernel is not a simple time loop over the recurrence. It relies on a chunk factorization that rewrites the recurrence into two parts:

1. an intra-chunk triangular solve / scan
2. an inter-chunk state carry

This works cleanly for the standard DeltaNet update because the hidden state contribution can be represented entirely through the \(K \times V\) matrix state and the derived chunk quantities built from \(k\), \(\beta\), and the decay factors.

The rank-1 DC-removal recurrence breaks that exact structure in two ways.

First, the prediction is no longer purely

$$
S_{t-1} \phi(k_t),
$$

but instead

$$
S_{t-1} \phi(k_t) - \lambda_k(t) b_{t-1}.
$$

So the carry state is no longer just \(S_t\); it is the coupled pair

$$
(S_t, b_t).
$$

Second, the extra state \(b_t\) is updated with coefficient \(1\) in the write path,

$$
b_t = \alpha_t b_{t-1} + \delta_t,
$$

while the matrix state is updated with feature coefficient \(\phi(k_t)\),

$$
S_t = \alpha_t S_{t-1} + \delta_t \phi(k_t)^\top.
$$

That mismatch is exactly why the nonnegative approximation is cheap in the recurrent form but awkward in the chunked factorization. The chunk algorithm would need a matched reformulation that carries both:

- the usual feature-weighted cumulative quantities for \(S_t\)
- an additional unweighted cumulative quantity for \(b_t\)
- the cross-term induced by \(\lambda_k(t) b_{t-1}\) inside the innovation

### What a Chunked Triton Implementation Would Need

A correct chunked implementation would therefore need to augment the chunk summary with an additional bias-state channel.

At a minimum, each chunk would need to propagate:

- the final matrix state contribution for \(S\)
- the final vector state contribution for \(b\)
- the intra-chunk correction induced by the \(\lambda_k(t) b_{t-1}\) term

Operationally, that means the existing WY-style / chunk-scan preprocessing is no longer sufficient by itself. One needs a new factorization for the coupled recurrence.

A useful way to think about it is that the recurrent state becomes an augmented object

$$
\mathcal{H}_t = (S_t, b_t),
$$

but the write coefficients are asymmetric:

- \(S_t\) writes with \(\phi(k_t)\)
- \(b_t\) writes with the constant coefficient \(1\)
- prediction reads \(b_t\) with token-dependent coefficient \(\lambda_k(t)\)
- output reads \(b_t\) with token-dependent coefficient \(\lambda_q(t)\)

So a chunked kernel must derive a block recurrence for this augmented state, rather than trying to reuse the standard chunk algebra unchanged.

### Practical Recommendation

In implementation order, the sensible path is:

1. implement and validate the fused recurrent Triton kernel first
2. keep chunk mode on a reference fallback until the augmented chunk factorization is derived cleanly
3. only then build a chunked Triton kernel for the coupled \((S_t, b_t)\) recurrence

This is why the current prototype is naturally a recurrent reference path rather than a drop-in chunk-kernel modification. The recurrent kernel matches the math directly; the chunked kernel requires a new derivation.

## What the Proper Chunked Version Would Need

A proper chunked implementation must not treat the rank-1 DC-removal term as a small patch on top of the existing chunk algebra. The correct object to chunk is the coupled recurrence

$$
\mathcal{H}_t = (S_t, b_t),
$$

with updates

$$
\hat v_t = S_{t-1} \phi(k_t) - \lambda_k(t) b_{t-1},
$$
$$
\delta_t = \beta_t (v_t - \hat v_t),
$$
$$
S_t = \alpha_t S_{t-1} + \delta_t \phi(k_t)^\top,
$$
$$
b_t = \alpha_t b_{t-1} + \delta_t.
$$

The key point is that this can be written as a linear recurrence in an augmented state, but not in the same reduced form used by the current chunked DeltaNet kernel.

### Augmented Linear View

For each output channel, define the augmented state

$$
\widetilde{h}_t =
\begin{bmatrix}
\operatorname{vec}(S_t) \\
b_t
\end{bmatrix}.
$$

Then each step is an affine linear map driven by token-dependent coefficients \(\phi(k_t)\), \(\lambda_k(t)\), \(\beta_t\), and \(\alpha_t\). The important consequence is that chunk composition is still possible in principle:

- each chunk induces a linear map from incoming augmented state to outgoing augmented state
- each chunk also induces a local readout contribution for outputs inside the chunk

So the right chunked formulation is a block scan over augmented states, not a reuse of the existing scalar/matrix chunk summary.

### What Must Be Summarized Per Chunk

For a chunk spanning timesteps \(t \in [c_0, c_1]\), a correct summary needs at least:

- a matrix-to-matrix carry for how incoming \(S\) contributes to outgoing \(S\)
- a vector-to-matrix carry for how incoming \(b\) contributes to outgoing \(S\)
- a matrix-to-vector carry for how incoming \(S\) contributes to outgoing \(b\)
- a vector-to-vector carry for how incoming \(b\) contributes to outgoing \(b\)
- an inhomogeneous term coming from the values \(v_t\)

In other words, the chunk transfer is block-structured:

$$
\begin{bmatrix}
S_{\mathrm{out}} \\
b_{\mathrm{out}}
\end{bmatrix}
=
\begin{bmatrix}
A_{SS} & A_{Sb} \\
A_{bS} & A_{bb}
\end{bmatrix}
\begin{bmatrix}
S_{\mathrm{in}} \\
b_{\mathrm{in}}
\end{bmatrix}
+
\begin{bmatrix}
c_S \\
c_b
\end{bmatrix}.
$$

The standard chunked DeltaNet effectively only needs the \(A_{SS}\) block plus its associated inhomogeneous term. Rank-1 DC removal requires all four blocks.

### Why This Is the Matched Chunk Form

The mismatch comes from the innovation term

$$
\delta_t = \beta_t \bigl(v_t - S_{t-1}\phi(k_t) + \lambda_k(t) b_{t-1}\bigr).
$$

Substituting this into the updates gives:

$$
S_t = \alpha_t S_{t-1} + \beta_t v_t \phi(k_t)^\top - \beta_t (S_{t-1}\phi(k_t)) \phi(k_t)^\top + \beta_t \lambda_k(t) b_{t-1} \phi(k_t)^\top,
$$

$$
b_t = \alpha_t b_{t-1} + \beta_t v_t - \beta_t S_{t-1}\phi(k_t) + \beta_t \lambda_k(t) b_{t-1}.
$$

This makes the coupling explicit:

- incoming \(S\) affects outgoing \(S\)
- incoming \(b\) affects outgoing \(S\)
- incoming \(S\) affects outgoing \(b\)
- incoming \(b\) affects outgoing \(b\)

That is exactly the block transfer structure above.

### Practical Chunk Construction

A practical chunk kernel would therefore proceed in three stages.

1. Intra-chunk local solve

Compute all purely local terms within the chunk, including the local outputs and the inhomogeneous contribution \((c_S, c_b)\).

2. Chunk transfer extraction

Build the block transfer operator

$$
T_c =
\begin{bmatrix}
A_{SS}^{(c)} & A_{Sb}^{(c)} \\
A_{bS}^{(c)} & A_{bb}^{(c)}
\end{bmatrix}
$$

for each chunk.

3. Inter-chunk scan

Perform a prefix scan over chunk transfers, composing them exactly as affine block maps, then feed the resulting incoming state into each chunk's local solve.

This is the proper chunked analogue of the recurrent implementation.

### What Should Stay Low-Rank

Even though the full transfer is block-structured, the new coupling introduced by DC removal is still low-rank in the feature space.

In particular:

- the extra state \(b_t\) is only a \(V\)-vector
- the coupling from \(b\) into \(S\) always appears through \(\phi(k_t)^\top\)
- the coupling from \(S\) into \(b\) always appears through \(\phi(k_t)\)

So a good chunk derivation should exploit that low-rank structure rather than materializing a dense augmented operator.

### The Right Engineering Goal

So the proper chunked version is not:

- "reuse the current chunk kernel and add one extra subtraction"

It is:

- "derive the chunk transfer operator for the augmented low-rank recurrence and implement that operator efficiently"

That is the mathematically matched chunked version of rank-1 DC removal.

## Actual Chunked Derivation for the Augmented Recurrence

This section writes the rank-1 DC-removal recurrence in a form that can actually be chunked.

We start from

$$
\hat v_t = S_{t-1} \phi_t - \lambda_t b_{t-1},
$$
$$
\delta_t = \beta_t (v_t - \hat v_t),
$$
$$
S_t = \alpha_t S_{t-1} + \delta_t \phi_t^\top,
$$
$$
b_t = \alpha_t b_{t-1} + \delta_t,
$$

where for brevity we write

$$
\phi_t := \phi(k_t),
\qquad
\lambda_t := \lambda_k(t).
$$

## 1. Expand the Coupling Explicitly

Substitute the innovation into the two state updates:

$$
S_t
= \alpha_t S_{t-1}
+ \beta_t v_t \phi_t^\top
- \beta_t (S_{t-1}\phi_t) \phi_t^\top
+ \beta_t \lambda_t b_{t-1} \phi_t^\top,
$$

$$
b_t
= \alpha_t b_{t-1}
+ \beta_t v_t
- \beta_t S_{t-1}\phi_t
+ \beta_t \lambda_t b_{t-1}.
$$

So the incoming pair \((S_{t-1}, b_{t-1})\) is mapped to \((S_t, b_t)\) by a linear operator plus an inhomogeneous term from \(v_t\).

## 2. Per-Step Block Operator

Define the per-step linear map

$$
\mathcal{T}_t:
\begin{bmatrix}
S \\
b
\end{bmatrix}
\mapsto
\begin{bmatrix}
\alpha_t S - \beta_t (S\phi_t) \phi_t^\top + \beta_t \lambda_t b \phi_t^\top \\
\alpha_t b - \beta_t S\phi_t + \beta_t \lambda_t b
\end{bmatrix}.
$$

Define also the value-driven inhomogeneous term

$$
\mathcal{c}_t(v_t)
=
\begin{bmatrix}
\beta_t v_t \phi_t^\top \\
\beta_t v_t
\end{bmatrix}.
$$

Then the recurrence is exactly

$$
\begin{bmatrix}
S_t \\
b_t
\end{bmatrix}
=
\mathcal{T}_t
\begin{bmatrix}
S_{t-1} \\
b_{t-1}
\end{bmatrix}
+
\mathcal{c}_t(v_t).
$$

This is the object that must be chunked.

## 3. Matrix-Friendly Decomposition of the Step Operator

It is useful to separate the action on \(S\) and \(b\).

For any incoming \((S,b)\),

$$
S' = \mathcal{T}^{SS}_t(S) + \mathcal{T}^{Sb}_t(b),
$$
$$
b' = \mathcal{T}^{bS}_t(S) + \mathcal{T}^{bb}_t(b),
$$

with

$$
\mathcal{T}^{SS}_t(S) = \alpha_t S - \beta_t (S\phi_t) \phi_t^\top,
$$
$$
\mathcal{T}^{Sb}_t(b) = \beta_t \lambda_t b \phi_t^\top,
$$
$$
\mathcal{T}^{bS}_t(S) = -\beta_t S\phi_t,
$$
$$
\mathcal{T}^{bb}_t(b) = (\alpha_t + \beta_t \lambda_t) b.
$$

So the step operator already has the four-block structure

$$
\mathcal{T}_t =
\begin{bmatrix}
\mathcal{T}^{SS}_t & \mathcal{T}^{Sb}_t \\
\mathcal{T}^{bS}_t & \mathcal{T}^{bb}_t
\end{bmatrix}.
$$

## 4. Chunk Transfer as a Product of Step Operators

Consider a chunk with local indices \(j = 1, \dots, C\). Let the incoming state be \((S_0, b_0)\), and let \((S_j, b_j)\) denote the state after local step \(j\).

Then

$$
\begin{bmatrix}
S_C \\
b_C
\end{bmatrix}
=
\mathcal{T}_C \mathcal{T}_{C-1} \cdots \mathcal{T}_1
\begin{bmatrix}
S_0 \\
b_0
\end{bmatrix}
+
\sum_{m=1}^C
\left(
\mathcal{T}_C \mathcal{T}_{C-1} \cdots \mathcal{T}_{m+1}
\right)
\mathcal{c}_m(v_m).
$$

This is the exact chunk summary. So for each chunk we need:

$$
\mathcal{A}_{\mathrm{chunk}} := \mathcal{T}_C \mathcal{T}_{C-1} \cdots \mathcal{T}_1,
$$
$$
\mathcal{c}_{\mathrm{chunk}} :=
\sum_{m=1}^C
\left(
\mathcal{T}_C \mathcal{T}_{C-1} \cdots \mathcal{T}_{m+1}
\right)
\mathcal{c}_m(v_m).
$$

Then the chunk output state is

$$
\begin{bmatrix}
S_{\mathrm{out}} \\
b_{\mathrm{out}}
\end{bmatrix}
=
\mathcal{A}_{\mathrm{chunk}}
\begin{bmatrix}
S_{\mathrm{in}} \\
b_{\mathrm{in}}
\end{bmatrix}
+
\mathcal{c}_{\mathrm{chunk}}.
$$

## 5. Block Form of the Chunk Transfer

Write the chunk transfer in blocks:

$$
\mathcal{A}_{\mathrm{chunk}} =
\begin{bmatrix}
A_{SS} & A_{Sb} \\
A_{bS} & A_{bb}
\end{bmatrix},
$$
$$
\mathcal{c}_{\mathrm{chunk}} =
\begin{bmatrix}
c_S \\
c_b
\end{bmatrix}.
$$

Then

$$
S_{\mathrm{out}} = A_{SS}(S_{\mathrm{in}}) + A_{Sb}(b_{\mathrm{in}}) + c_S,
$$
$$
b_{\mathrm{out}} = A_{bS}(S_{\mathrm{in}}) + A_{bb}(b_{\mathrm{in}}) + c_b.
$$

This is the precise meaning of the four-block summary described earlier.

## 6. Composition Rule Across Chunks

Suppose chunk 1 has transfer \((\mathcal{A}_1, \mathcal{c}_1)\) and chunk 2 has transfer \((\mathcal{A}_2, \mathcal{c}_2)\). Then their composition is

$$
(\mathcal{A}_2, \mathcal{c}_2) \circ (\mathcal{A}_1, \mathcal{c}_1)
=
(\mathcal{A}_2 \mathcal{A}_1,\; \mathcal{A}_2 \mathcal{c}_1 + \mathcal{c}_2).
$$

This is the associative law needed for a chunk-level scan.

So the correct inter-chunk scan is not over the original DeltaNet summaries, but over affine block operators of this form.

## 7. Local Readout Inside a Chunk

For a token inside the chunk, the output is

$$
y_t = S_t \psi_t - \lambda_q(t) b_t,
$$

where

$$
\psi_t := \phi(q_t)
$$

or the scaled version used by the implementation.

Inside a chunk, we can decompose the output into:

- a contribution from the incoming chunk state \((S_{\mathrm{in}}, b_{\mathrm{in}})\)
- a purely local contribution from values inside the chunk

That is,

$$
y_t = R_t
\begin{bmatrix}
S_{\mathrm{in}} \\
b_{\mathrm{in}}
\end{bmatrix}
+ d_t,
$$

for some token-local readout operator \(R_t\) and token-local inhomogeneous term \(d_t\).

This is exactly analogous to the usual chunked DeltaNet decomposition, except that the readout now depends on the augmented state.

## 8. Where the Existing Chunk Derivation Stops Matching

The current chunked DeltaNet derivation exploits the fact that the only state is the matrix \(S_t\), and both prediction and write couple through the same feature vector \(\phi_t\).

Rank-1 DC removal destroys that simplification because:

$$
\text{prediction uses } (S_{t-1}, b_{t-1}) \text{ through } (\phi_t, \lambda_t),
$$

while

$$
\text{writing uses } \delta_t \text{ into } (S_t, b_t) \text{ through } (\phi_t, 1).
$$

So the write and read coefficients are no longer matched by a single feature map. That is why a new chunk derivation is required.

## 9. The Low-Rank Opportunity

Even though the augmented transfer is larger, the new terms are still structured.

Specifically:

$$
\mathcal{T}^{Sb}_t(b) = \beta_t \lambda_t b \phi_t^\top
$$

is rank-1 in feature space, and

$$
\mathcal{T}^{bS}_t(S) = -\beta_t S\phi_t
$$

depends on \(S\) only through the projection \(S\phi_t\).

So a good chunk implementation should never materialize the full dense block operator. It should instead propagate only the low-rank factors required to apply these blocks.

## 10. Engineering Interpretation

A proper chunked kernel therefore needs three derived objects per chunk:

1. the standard feature-space carry used by the original DeltaNet chunk kernel
2. an additional bias-state carry for \(b\)
3. the cross-coupling summaries that map incoming \(S\) into outgoing \(b\), and incoming \(b\) into outgoing \(S\)

Once these are derived, the inter-chunk scan is well-defined and associative.

That is the actual derivation target for a mathematically correct chunked rank-1 DC-removal kernel.
