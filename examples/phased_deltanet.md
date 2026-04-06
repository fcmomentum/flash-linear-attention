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
