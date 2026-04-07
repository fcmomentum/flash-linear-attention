Absolutely — here's the **Phased Delta GatedNet** written out in the *same mathematical style* as the paper you linked (the "Delta Gated Network / Mamba" paper). That means:

* concise recurrence formulas
* clear definitions of gates and state updates
* position/indexed sequences
* and separation into *signal* and *write* components

---

# 📌 Phased Delta GatedNet — Full Formulas (Paper‑Style)

We use the following symbols:

* $x_t \in \mathbb{R}^{d_v}$: input/value at time $t$
* $k_t \in \mathbb{R}^{d_k}$: key (or query) vector at time $t$
* $h_t \in \mathbb{C}^D$: complex hidden state with $D$ phased channels
* $\alpha_t \in \mathbb{R}_{\ge 0}^D$: retention (decay) gate
* $\psi_t \in \mathbb{R}^D$: write/gain gate
* $\phi_t \in \mathbb{R}^D$: phase rotation gate
* $\beta_t \in \mathbb{R}_{\ge 0}^D$: write strength gating
* $W_q,W_k,W_v$: linear projections
* $\Delta_t$: step size / time increment

We structure the model *exactly like the paper's gating/SSM style*, but with **phased complex channels**.

---

## 1) Projections

We project inputs into key, value, and write space:

$$
\begin{aligned}
q_t &= W_q x_t, \\
k_t &= W_k x_t, \\
v_t &= W_v x_t.
\end{aligned}
$$

Each of $q_t,k_t,v_t \in \mathbb{R}^{d_k}$.

---

## 2) Phased Delta GatedNet Recurrence

We define the *complex hidden state* update with decay and phase rotation:

$$
\boxed{
h_t = (\alpha_t \odot e^{i,\phi_t}) \odot h_{t-1}
+ (\beta_t \odot e^{i,\psi_t}) \odot v_t
  }
$$

where:

* $\alpha_t = \exp(-\lambda_t,\Delta_t)\in[0,1]^D$ (decay magnitudes),
* $\phi_t\in\mathbb{R}^D$ (phase rotation per channel),
* $\beta_t\in[0,1]^D$ (write strength),
* $\psi_t\in\mathbb{R}^D$ (write phase shift),
* and $\odot$ is elementwise product.

Expanded per channel $m=1,\ldots,D$:

$$
h_t^{(m)} =
\alpha_t^{(m)} e^{i,\phi_t^{(m)}},h_{t-1}^{(m)}
+
\beta_t^{(m)},e^{i,\psi_t^{(m)}},v_t^{(m)}.
$$

This update is analogous to the paper's gated recurrence, except we have *complex rotations* instead of purely real retention.

---

## 3) Unrolled Hidden State

By unrolling over time, we get:

$$
\boxed{
h_T^{(m)}
=
\sum_{\tau=1}^T
\left(
\beta_\tau^{(m)},e^{i\psi_\tau^{(m)}}
;\prod_{u=\tau+1}^T
\left(
\alpha_u^{(m)} e^{i\phi_u^{(m)}}
\right)
\right)
v_\tau^{(m)}.
}
$$

* $\prod_{u=\tau+1}^T \alpha_u^{(m)}$ gives exponential decay
* $\prod_{u=\tau+1}^T e^{i\phi_u^{(m)}} = e^{i\sum_{u=\tau+1}^T \phi_u^{(m)}}$ gives phase evolution

This is the *complex Laplace/Fourier-style memory* across time, channel‑wise.

---

## 4) Querying / Readout

Given a query key $q_T$, we define phase‑aligned readout:

$$
\hat y_T = \sum_{m=1}^D w^{(m)}, e^{-i,\theta_T^{(m)}}, h_T^{(m)}
$$

where $\theta_T^{(m)}$ is a **demodulation phase** depending on the intended relative position (lag) or query location.

Equivalently, in inner‑product form:

$$
\hat y_T = \left( \sum_{m=1}^D w^{(m)} e^{-i\theta_T^{(m)}}, h_T^{(m)} \right).
$$

This matches the retrieval style in the paper (inner product with read weights), but adapted for complex phased channels.

---

## 5) Effective Memory Kernel

Define the complex **memory kernel** per channel:

$$
\kappa^{(m)}(T,\tau)
=
\beta_\tau^{(m)},
\exp!\left(
i\sum_{u=\tau+1}^T\phi_u^{(m)}
\right)
\prod_{u=\tau+1}^T \alpha_u^{(m)}.
$$

Then

$$
h_T^{(m)} = \sum_{\tau=1}^T \kappa^{(m)}(T,\tau),v_\tau^{(m)}.
$$

This is exactly the complex analog of the *Green's function / path‑ordered exponential* in SSM/Laplace formulations.

---

## 6) Summary in Paper Style

### Phased Delta GatedNet Update

$$
h_t =
(\alpha_t\odot e^{i\phi_t})\odot h_{t-1}
+
(\beta_t\odot e^{i\psi_t})\odot v_t
$$

### Unrolled Memory

$$
h_T = \sum_{\tau=1}^T
\left(
(\beta_\tau\odot e^{i\psi_\tau})
;\odot;
\prod_{u=\tau+1}^T
(\alpha_u\odot e^{i\phi_u})
\right)
\odot v_\tau
$$

### Retrieval

$$
\hat y_T = \sum_{m=1}^D w^{(m)} e^{-i,\theta_T^{(m)}},h_T^{(m)}.
$$

---

# 🧠 Interpretation (Paper Alignment)

* $\alpha_t$ plays the role of **forgetting/retention gate**
* $\beta_t$ plays the role of **write gate**
* $\phi_t, \psi_t, \theta_t$ give **phase indexing**
* Complex exponentials create **oscillatory memory traces**
* This parallels Laplace/Fourier kernels but retains *gated control* as in the original paper

---

If you want the **signal/noise SNR expression for this exact phased model**, I can also write it in the same paper's notation — with explicit $d_k$ and $D$ dependencies spelled out.

---

# Design Note: Training-Quality Kernelization

The current implementation status is split into three pieces:

1. A reference phased recurrence exists and is differentiable.
2. A fused recurrent Triton forward path exists for the RoPE-style phased update.
3. A true chunk Triton training path does **not** exist yet.

That last point is the blocker for training-quality performance.

## 1. Why Chunk Is Harder Than Fused Recurrent

For the plain Gated DeltaNet chunk rule, the kernelization works because each chunk can be summarized through the usual WY-style decomposition:

$$
S_t = \alpha_t S_{t-1} + \delta_t k_t^\top,
\qquad
\delta_t = \beta_t \left(v_t - S_{t-1} k_t\right).
$$

Inside a chunk, this leads to the familiar lower-triangular system over token interactions, and the implementation can factor the chunk through the intermediate objects

$$
A,\; w,\; u,\; h,\; o.
$$

The phased recurrence changes the state transition itself:

$$
S_t = \alpha_t R(\phi_t) S_{t-1} + \delta_t k_t^\top,
\qquad
\delta_t = \beta_t \left(R(\psi_t) v_t - S_{t-1} k_t\right),
$$

with RoPE-style real rotations $R(\cdot)$ applied only on the designated phase channels.

So the phase is not just an output decoration. It changes the linear operator that maps one state to the next.

## 2. Why the Existing Chunk Derivation Cannot Be Reused As-Is

The current chunk kernels assume that the cross-token coupling inside a chunk is driven only by:

1. scalar decays $\alpha_t$ / $g_t$,
2. feature vectors $k_t$,
3. write strengths $\beta_t$.

With phase, each step also applies a token-dependent rotation on the value/state subspace. That means:

1. the state carried between chunk positions is no longer updated by a pure scalar decay,
2. the effective contribution of an earlier write at time $\tau$ to a later time $t$ includes a product of rotations,
3. the chunk-local triangular system is now expressed in a rotated basis that changes with token position.

In particular, the current `A = beta * K K^T` style derivation is no longer the right internal object, because the value-space transport between two positions is not identity; it is the accumulated rotation between them.

## 3. The Right Way To Re-Derive Chunk Phase Kernels

The clean way to recover a chunk factorization is to move to a transformed frame.

Define the cumulative phase transport

$$
P_t := \prod_{u=1}^t R(\phi_u).
$$

Then define a transported state

$$
\widetilde S_t := P_t^{-1} S_t.
$$

Substituting into the phased recurrence gives

$$
\widetilde S_t
=
\alpha_t \widetilde S_{t-1}
+
\widetilde \delta_t k_t^\top,
$$

where the write term is expressed in the transported frame:

$$
\widetilde \delta_t
=
\beta_t \left(P_t^{-1} R(\psi_t) v_t - \widetilde S_{t-1} k_t\right).
$$

This is the key observation:

1. in the transported frame, the state-to-state transition returns to the ordinary scalar-decay DeltaNet form;
2. all phase complexity is pushed into the transported values
   $$
   \widetilde v_t := P_t^{-1} R(\psi_t) v_t;
   $$
3. the readout must also be transported consistently:
   $$
   y_t = P_t \left(\widetilde S_t q_t\right)
   $$
   or, more generally, use the appropriate demodulated read vector in the transported frame.

This suggests a chunk strategy:

1. compute cumulative phase transport per token,
2. rotate values into the transported frame before the chunk solve,
3. run the existing DeltaNet chunk algebra in that transported frame,
4. rotate outputs back at readout,
5. propagate chunk boundary states in transported coordinates.

## 4. Consequences for the Triton Implementation

A real chunk Triton phase implementation therefore needs updates in all of these places:

1. `chunk_fwd.py`
   The chunk-local triangular solve must consume transported values rather than raw values.
2. `common/chunk_delta_h.py`
   The hidden-state propagation across chunk boundaries must carry the transported state, not the raw rotated state.
3. `common/chunk_o.py`
   The readout must apply the inverse transport required to map chunk-frame outputs back to the model frame.
4. backward kernels
   Gradients must flow through cumulative phase transport, which means the backward pass must differentiate both the transport and the transported-frame DeltaNet algebra.

This is why a correct Triton chunk backward is a re-derivation task, not a small patch.

## 5. Practical Implementation Plan

The practical path to training-quality chunk kernels is:

1. Keep the dense recurrent reference as the correctness oracle.
2. Implement a transported-frame chunk forward in Python first.
3. Verify that transported-frame chunk forward matches the dense phased recurrence.
4. Port that transported-frame chunk forward to Triton.
5. Implement the transported-frame backward, either by:
   1. saving transported intermediates, or
   2. recomputing transport and chunk-local quantities in backward.
6. Replace the temporary dense fallback only after the transported-frame chunk forward/backward matches the reference.

## 6. Current Recommendation

Until the transported-frame chunk derivation is implemented, phase-enabled training should be treated as:

1. correct through the dense recurrent fallback,
2. partially accelerated for fused recurrent forward,
3. not yet training-quality from a chunk-kernel performance standpoint.
