# Phased Gated DeltaNet with RoPE-Style Memory Addressing

This note replaces the earlier "complex hidden state" formulation.

If the goal is a **phased Gated DeltaNet based on the Gated Delta Networks paper**
([arXiv:2412.06464](https://arxiv.org/abs/2412.06464)), then the phase should
live in the **query/key addressing space**, not in a standalone complex value
state. The memory must remain the usual DeltaNet matrix state

$$
S_t \in \mathbb{R}^{d_k \times d_v},
$$

and the readout from each stored memory should contain a RoPE-style relative
term

$$
q_t^\top R(\Delta_{t,\tau}) k_\tau.
$$

That is the right analogue of Transformer RoPE for a delta-rule memory.

---

## 1. Base Gated DeltaNet

Using the paper's notation and structure, the scalar-gated delta rule is

$$
\boxed{
S_t = \alpha_t S_{t-1} + k_t \delta_t^\top
}
$$

with

$$
\boxed{
\delta_t = \beta_t \left(v_t - S_{t-1}^\top k_t\right).
}
$$

Here:

* $q_t, k_t \in \mathbb{R}^{d_k}$
* $v_t, \delta_t \in \mathbb{R}^{d_v}$
* $S_t \in \mathbb{R}^{d_k \times d_v}$
* $\alpha_t \in (0,1]$ is the forget / retention gate
* $\beta_t \in [0,1]$ is the write gate

The readout is

$$
\boxed{
o_t = S_t^\top q_t
}
$$

or equivalently $o_t = q_t^\top S_t$.

---

## 2. Correct RoPE-Style Phased Extension

To get a per-memory factor of the form

$$
q_t^\top R(\Delta_{t,\tau}) k_\tau,
$$

we should apply positional rotation to the **query and key vectors**:

$$
\bar q_t = R(\theta_t) q_t, \qquad \bar k_t = R(\theta_t) k_t,
$$

where $R(\theta_t)$ is the usual block-diagonal RoPE rotation acting on a paired
subspace of $\mathbb{R}^{d_k}$.

Then the phased gated delta rule is simply

$$
\boxed{
S_t = \alpha_t S_{t-1} + \bar k_t \delta_t^\top
}
$$

with

$$
\boxed{
\delta_t = \beta_t \left(v_t - S_{t-1}^\top \bar k_t\right)
}
$$

and readout

$$
\boxed{
o_t = S_t^\top \bar q_t.
}
$$

This is the cleanest phased extension of Gated DeltaNet because it preserves the
paper's state geometry and delta update rule exactly.

---

## 3. Why This Gives the Desired RoPE Term

Unrolling the recurrence gives

$$
S_t
=
\sum_{\tau=1}^t
\left(
\prod_{u=\tau+1}^t \alpha_u
\right)
\bar k_\tau \delta_\tau^\top.
$$

Therefore

$$
\begin{aligned}
o_t
&= S_t^\top \bar q_t \\
&=
\sum_{\tau=1}^t
\left(
\prod_{u=\tau+1}^t \alpha_u
\right)
(\bar q_t^\top \bar k_\tau)\,\delta_\tau.
\end{aligned}
$$

Now substitute the RoPE-rotated query and key:

$$
\bar q_t^\top \bar k_\tau
=
q_t^\top R(\theta_t)^\top R(\theta_\tau) k_\tau
=
q_t^\top R(\theta_\tau - \theta_t) k_\tau.
$$

So the output becomes

$$
\boxed{
o_t
=
\sum_{\tau=1}^t
\left(
\prod_{u=\tau+1}^t \alpha_u
\right)
\Bigl[q_t^\top R(\theta_\tau - \theta_t) k_\tau\Bigr]\,
\delta_\tau.
}
$$

This is exactly the desired "for each memory" term:

$$
q_t^\top R(\Delta_{t,\tau}) k_\tau,
$$

with $\Delta_{t,\tau} = \theta_\tau - \theta_t$.

For standard fixed-frequency RoPE, if $\theta_t$ is built from the token
position $p_t$, then this is the same relative-position mechanism used in
Transformer attention.

---

## 4. What Was Wrong with the Previous Note

The previous draft wrote a recurrence of the form

$$
h_t = (\alpha_t e^{i\phi_t}) \odot h_{t-1} + (\beta_t e^{i\psi_t}) \odot v_t
$$

with a direct demodulation readout from $h_t$.

That is **not** the right mathematical analogue of the Gated DeltaNet paper for
two reasons:

1. it replaces the matrix memory $S_t \in \mathbb{R}^{d_k \times d_v}$ with an
   unrelated vector/complex state;
2. it puts phase on the value channels, so it does **not** expose the desired
   per-memory address term $q_t^\top R(\Delta) k_\tau$.

If the requirement is specifically "RoPE-like relative addressing when reading
memory", then the phase has to enter through the **query-key interaction**.

---

## 5. Implementation Consequence

If this model is what we actually want, then phase channels should live in the
**key/query dimension** $d_k$, not the value dimension $d_v$.

The practical recipe is:

1. project to $q_t, k_t, v_t$ as usual;
2. apply RoPE to $q_t$ and $k_t$ only:
   $$
   \bar q_t = R(\theta_t) q_t,\qquad \bar k_t = R(\theta_t) k_t;
   $$
3. run the ordinary gated delta rule on $(\bar q_t, \bar k_t, v_t)$;
4. keep the state as $S_t \in \mathbb{R}^{d_k \times d_v}$.

In other words, the chunk/recurrent algebra from Gated DeltaNet stays the same;
the only change is that the addresses are RoPE-rotated before the write and
read operations.

---

## 6. Final Formula Set

### Projections

$$
q_t = W_q x_t,\qquad
k_t = W_k x_t,\qquad
v_t = W_v x_t.
$$

### RoPE Addressing

$$
\bar q_t = R(\theta_t) q_t,\qquad
\bar k_t = R(\theta_t) k_t.
$$

### Phased Gated Delta Rule

$$
S_t = \alpha_t S_{t-1} + \bar k_t \delta_t^\top,
$$

$$
\delta_t = \beta_t \left(v_t - S_{t-1}^\top \bar k_t\right).
$$

### Readout

$$
o_t = S_t^\top \bar q_t.
$$

### Per-memory Expansion

$$
o_t
=
\sum_{\tau=1}^t
\left(
\prod_{u=\tau+1}^t \alpha_u
\right)
\Bigl[q_t^\top R(\theta_\tau - \theta_t) k_\tau\Bigr]\,
\delta_\tau.
$$

That is the phased Gated DeltaNet formulation that is faithful to the paper and
has the RoPE-style readout term you asked for.

---

## 7. Retrieval SNR: Phased vs. Baseline DeltaNet

This section follows the **retrieval SNR** viewpoint from
*Understanding Transformer from the Perspective of Associative Memory*
([arXiv:2505.19488](https://arxiv.org/abs/2505.19488)).

The paper defines retrieval in the form

$$
\text{retrieval} = c\,v_i + r,
$$

where $c$ is the signal coefficient and $r$ is the interference term. Instead of
using the paper's inverse-SNR notation, we work directly with

$$
\boxed{
\mathrm{SNR}
:=
\frac{c^2 \,\mathbb{E}\|v_i\|^2}{\mathbb{E}\|r\|^2}.
}
$$

Larger $\mathrm{SNR}$ means better retrieval.

### 7.1 Baseline DeltaNet as an Associative Memory

For the baseline gated DeltaNet readout,

$$
o_t
=
\sum_{\tau=1}^t
a_{t,\tau}\,
\bigl(q_t^\top k_\tau\bigr)\,
\delta_\tau,
\qquad
a_{t,\tau}:=\prod_{u=\tau+1}^t \alpha_u.
$$

Suppose we want to retrieve the memory written at time $i$. Then

$$
o_t
=
\underbrace{
a_{t,i}(q_t^\top k_i)\,\delta_i
}_{\text{signal}}
+
\underbrace{
\sum_{\tau\neq i}
a_{t,\tau}(q_t^\top k_\tau)\,\delta_\tau
}_{\text{noise}}.
$$

So the baseline signal coefficient is

$$
c_{\mathrm{base}} = a_{t,i}(q_t^\top k_i),
$$

and the noise is

$$
r_{\mathrm{base}}
=
\sum_{\tau\neq i}
a_{t,\tau}(q_t^\top k_\tau)\,\delta_\tau.
$$

Under the same isotropic-random-key assumptions used in the paper,

$$
\mathbb{E}\bigl[(q_t^\top k_\tau)^2\bigr] \asymp \frac{1}{d_k}
\qquad (\tau\neq i),
$$

so if $\|\delta_\tau\|^2$ is roughly comparable across memories, then

$$
\boxed{
\mathrm{SNR}_{\mathrm{base}}
\asymp
\frac{a_{t,i}^2 (q_t^\top k_i)^2}
{
\frac{1}{d_k}
\sum_{\tau\neq i} a_{t,\tau}^2
}.
}
$$

If decay is weak over the relevant window, this reduces to the usual linear
associative-memory scaling

$$
\mathrm{SNR}_{\mathrm{base}}
\asymp
\frac{d_k}{N_{\mathrm{eff}}},
\qquad
N_{\mathrm{eff}} := \sum_{\tau\neq i} a_{t,\tau}^2.
$$

So baseline DeltaNet inherits the same key-dimension bottleneck: retrieval gets
worse as too many memories compete in the same $d_k$-dimensional address space.

### 7.2 Phased DeltaNet

For the phased model derived above,

$$
o_t
=
\sum_{\tau=1}^t
a_{t,\tau}\,
\Bigl[q_t^\top R(\theta_\tau-\theta_t)k_\tau\Bigr]\,
\delta_\tau.
$$

Therefore the target memory at time $i$ has

$$
c_{\mathrm{phase}}
=
a_{t,i}\,
q_t^\top R(\theta_i-\theta_t)k_i,
$$

and the interference is

$$
r_{\mathrm{phase}}
=
\sum_{\tau\neq i}
a_{t,\tau}\,
\Bigl[q_t^\top R(\theta_\tau-\theta_t)k_\tau\Bigr]\,
\delta_\tau.
$$

So

$$
\boxed{
\mathrm{SNR}_{\mathrm{phase}}
=
\frac{
a_{t,i}^2
\Bigl(q_t^\top R(\theta_i-\theta_t)k_i\Bigr)^2
\mathbb{E}\|\delta_i\|^2
}{
\mathbb{E}\left\|
\sum_{\tau\neq i}
a_{t,\tau}
\Bigl[q_t^\top R(\theta_\tau-\theta_t)k_\tau\Bigr]
\delta_\tau
\right\|^2
}.
}
$$

This is exactly the same SNR template as the paper's kernelized analysis, with
the effective kernel

$$
\boxed{
\kappa_{\mathrm{phase}}\bigl((q_t,t),(k_\tau,\tau)\bigr)
=
q_t^\top R(\theta_\tau-\theta_t)k_\tau.
}
$$

### 7.3 What Changes Relative to Baseline

The key fact is:

* for completely unrelated isotropic random keys, multiplying by an orthogonal
  rotation does **not** change the second moment of a random inner product.

To turn that into a statement about the **SNR denominator**, we also need the
usual decoupling assumptions used in this kind of retrieval analysis:

1. $q_t$ has fixed norm;
2. each non-target key $k_\tau$ is independent and isotropic;
3. the write vectors $\delta_\tau$ are independent across $\tau$ and are
   uncorrelated with the addressing coefficients.

Then, writing

$$
c_\tau^{\mathrm{phase}}
:=
a_{t,\tau}\,q_t^\top R(\theta_\tau-\theta_t)k_\tau,
$$

the phased noise term is

$$
r_{\mathrm{phase}}
=
\sum_{\tau\neq i} c_\tau^{\mathrm{phase}}\,\delta_\tau,
$$

and its power satisfies

$$
\mathbb{E}\|r_{\mathrm{phase}}\|^2
\approx
\sum_{\tau\neq i}
\mathbb{E}\Bigl[(c_\tau^{\mathrm{phase}})^2\Bigr]\,
\mathbb{E}\|\delta_\tau\|^2,
$$

because the cross terms vanish under the independence assumptions.

Now if the stored keys $k_\tau$ are independent isotropic random vectors, then

$$
\mathbb{E}\Bigl[
\bigl(q_t^\top R(\theta_\tau-\theta_t)k_\tau\bigr)^2
\Bigr]
\asymp
\mathbb{E}\Bigl[
\bigl(q_t^\top k_\tau\bigr)^2
\Bigr]
\asymp
\frac{1}{d_k}.
$$

Hence the denominator is unchanged to first order:

$$
\mathbb{E}\|r_{\mathrm{phase}}\|^2
\asymp
\mathbb{E}\|r_{\mathrm{base}}\|^2.
$$

So in the fully random-key regime,

$$
\boxed{
\mathrm{SNR}_{\mathrm{phase}}
\asymp
\mathrm{SNR}_{\mathrm{base}}.
}
$$

This is an important point: **phase does not magically improve SNR for random
memories.**

### 7.4 Where Phase Actually Helps

Phase helps when interference is caused by **address collisions with similar or
repeated content keys across different positions**.

Suppose multiple memories share nearly the same content key $k$, but occur at
different positions. Then:

* baseline DeltaNet uses the same address score $q_t^\top k$ for all of them;
* phased DeltaNet replaces that with
  $q_t^\top R(\theta_\tau-\theta_t)k$,
  so memories at different relative positions receive different scores.

If the target was stored at position $i$, then for another copy at position
$\tau\neq i$, the interference is weighted by

$$
q_t^\top R(\theta_\tau-\theta_t)k
$$

instead of

$$
q_t^\top k.
$$

To compare against the target memory at position $i$, assume the query is
aligned with that target address:

$$
\bar q_t \approx \bar k_i
\qquad\Longleftrightarrow\qquad
q_t \approx R(\theta_i-\theta_t)k.
$$

Then the interference coefficient for a duplicate key at position $\tau$ is

$$
q_t^\top R(\theta_\tau-\theta_t)k
\approx
k^\top R(\theta_i-\theta_t)^\top R(\theta_\tau-\theta_t)k
=
k^\top R(\theta_\tau-\theta_i)k.
$$

Writing

$$
\rho(\Delta)
:=
\mathbb{E}_{k}\bigl[k^\top R(\Delta)k\bigr],
$$

the average same-content match at relative offset $\Delta$ is controlled by
$\rho(\Delta)$, which decays or oscillates away from $\Delta=0$ under RoPE.
Hence duplicate memories at the wrong relative position are suppressed by
approximately $\rho(\Delta)^2$ in the noise power. Neglecting cross-covariances
among different $\delta_\tau$, we obtain

$$
\mathbb{E}\|r_{\mathrm{phase}}\|^2
\approx
\sum_{\tau\neq i}
a_{t,\tau}^2\,
\rho(\theta_\tau-\theta_i)^2\,
\mathbb{E}\|\delta_\tau\|^2.
$$

So in a repeated-key setting, a useful approximation is

$$
\boxed{
\mathrm{SNR}_{\mathrm{phase}}
\asymp
\frac{
a_{t,i}^2\,\mathbb{E}\|\delta_i\|^2
}
{
\sum_{\tau\neq i}
a_{t,\tau}^2
\rho(\theta_\tau-\theta_i)^2\,
\mathbb{E}\|\delta_\tau\|^2
},
}
$$

whereas the baseline behaves like

$$
\boxed{
\mathrm{SNR}_{\mathrm{base}}
\asymp
\frac{
a_{t,i}^2\,\mathbb{E}\|\delta_i\|^2
}
{
\sum_{\tau\neq i} a_{t,\tau}^2\,\mathbb{E}\|\delta_\tau\|^2
}.
}
$$

Whenever many collisions come from memories with similar content but incorrect
relative position, we get

$$
\mathrm{SNR}_{\mathrm{phase}} > \mathrm{SNR}_{\mathrm{base}}.
$$

### 7.5 SNR of RoPE Softmax Attention

It is useful to compare the phased DeltaNet result above with ordinary
**RoPE softmax attention**, since both use the same relative-position kernel

$$
\kappa_{\mathrm{rope}}((q_t,t),(k_\tau,\tau))
=
q_t^\top R(\theta_\tau-\theta_t)k_\tau.
$$

For causal softmax attention,

$$
o_t
=
\sum_{\tau=1}^t p_{t,\tau} v_\tau,
\qquad
p_{t,\tau}
=
\frac{\exp(\ell_{t,\tau})}{\sum_{j=1}^t \exp(\ell_{t,j})},
$$

with logits

$$
\ell_{t,\tau}
:=
\frac{1}{\sqrt{d_k}}
q_t^\top R(\theta_\tau-\theta_t)k_\tau.
$$

If the target memory is at index $i$, then

$$
o_t
=
\underbrace{p_{t,i}v_i}_{\text{signal}}
+
\underbrace{\sum_{\tau\neq i} p_{t,\tau} v_\tau}_{\text{noise}}.
$$

So, under the same retrieval-SNR template,

$$
\boxed{
\mathrm{SNR}_{\mathrm{softmax}}
=
\frac{
\mathbb{E}\bigl[p_{t,i}^2 \|v_i\|^2\bigr]
}{
\mathbb{E}\left\|
\sum_{\tau\neq i} p_{t,\tau} v_\tau
\right\|^2
}.
}
$$

If the non-target values are independent and isotropic with comparable norm, the
noise power becomes

$$
\mathbb{E}\|r_{\mathrm{softmax}}\|^2
\approx
\sum_{\tau\neq i}
\mathbb{E}[p_{t,\tau}^2]\,
\mathbb{E}\|v_\tau\|^2,
$$

so

$$
\boxed{
\mathrm{SNR}_{\mathrm{softmax}}
\asymp
\frac{\mathbb{E}[p_{t,i}^2]}
{\sum_{\tau\neq i}\mathbb{E}[p_{t,\tau}^2]}.
}
$$

This is the key difference from phased DeltaNet:

* in phased DeltaNet, the denominator is built from squared **kernel
  coefficients** such as
  $a_{t,\tau}^2\bigl[q_t^\top R(\theta_\tau-\theta_t)k_\tau\bigr]^2$;
* in softmax attention, the denominator is built from squared **normalized
  probabilities** $p_{t,\tau}^2$.

So RoPE affects softmax SNR through the logits, and then the softmax nonlinearity
can amplify small logit differences into large probability differences.

#### Random-key regime

If all non-target keys are independent isotropic random vectors, then replacing
$q_t^\top k_\tau$ by $q_t^\top R(\theta_\tau-\theta_t)k_\tau$ does not change the
distribution of individual logits. Therefore the distribution of the softmax
weights is also unchanged to first order, and

$$
\mathrm{SNR}_{\mathrm{rope\_softmax}}
\asymp
\mathrm{SNR}_{\mathrm{plain\_softmax}}.
$$

So, just as in phased DeltaNet, **RoPE does not improve SNR in the fully random
memory regime by itself**.

#### Repeated-key / structured-collision regime

Now assume repeated content keys and a target-aligned query. Then the target
logit behaves like

$$
\ell_{t,i}
\approx
\frac{1}{\sqrt{d_k}}\,k^\top k,
$$

whereas a distractor at position $\tau$ has

$$
\ell_{t,\tau}
\approx
\frac{1}{\sqrt{d_k}}\,k^\top R(\theta_\tau-\theta_i)k.
$$

If we write

$$
\rho(\Delta)
:=
\mathbb{E}_{k}\bigl[k^\top R(\Delta)k\bigr],
$$

then wrong-position duplicates have reduced mean logit

$$
\ell_{t,\tau}
\approx
\frac{1}{\sqrt{d_k}}\rho(\theta_\tau-\theta_i).
$$

After softmax normalization, this implies an approximate distractor-to-target
weight ratio

$$
\frac{p_{t,\tau}}{p_{t,i}}
\approx
\exp\!\left(
\frac{
\rho(\theta_\tau-\theta_i)-\rho(0)
}{\sqrt{d_k}}
\right).
$$

Thus RoPE can improve softmax-attention SNR by suppressing distractor
probabilities **exponentially in the logit gap**. This is stronger than the
linear-kernel suppression in phased DeltaNet, where the denominator is reduced
only through multiplicative factors like $\rho(\Delta)^2$.

So the qualitative comparison is:

1. phased DeltaNet:
   relative phase reduces interference directly in the linear read coefficients;
2. RoPE softmax attention:
   relative phase reduces interference in the logits, and softmax can sharpen
   that reduction further.

This means RoPE softmax attention can have a stronger SNR advantage in
collision-dominated retrieval, but it pays for that with quadratic sequence
interaction and no fixed-size recurrent memory.

### 7.6 Gaussian Toy Model for Phased DeltaNet

The same RoPE-SNR mechanism can be ported from softmax attention to phased
DeltaNet almost verbatim, as long as we remember that phased DeltaNet retrieves
**write vectors** $\delta_\tau$ through a decayed linear memory rather than raw
values through softmax weights.

For phased DeltaNet,

$$
o_t
=
\sum_{\tau=1}^t
a_{t,\tau}\,
s_{t,\tau}^{\mathrm{phase}}\,
\delta_\tau,
\qquad
s_{t,\tau}^{\mathrm{phase}}
:=
q_t^\top R(\theta_\tau-\theta_t)k_\tau,
$$

with

$$
a_{t,\tau}:=\prod_{u=\tau+1}^t \alpha_u.
$$

So the RoPE-sensitive object is still the same score
$s_{t,\tau}^{\mathrm{phase}}$, but each memory is also weighted by the DeltaNet
retention factor $a_{t,\tau}$.

#### Toy key model

The noncausal toy assumption

$$
k_\tau \propto R(-\Delta_i)q_t
$$

should **not** be interpreted literally as a write-time generation rule, because
the stored key at time $\tau$ cannot know the future query time $t$ or the
future relative offset $\Delta_i$.

The causal way to express the same idea is to use an absolute-phase model.
Assume there is a latent content template $u$ and that keys/queries are formed
locally as noisy copies of that template:

$$
k_\tau
=
\rho_\tau u + \eta_\tau,
\qquad
q_t
=
\rho_q u + \xi_t,
\qquad
\eta_\tau \sim \mathcal N(0,\sigma_k^2 I).
$$

The phased score is still

$$
s_{t,\tau}^{\mathrm{phase}}
=
q_t^\top R(\theta_\tau-\theta_t)k_\tau.
$$

Ignoring the query noise term for simplicity and conditioning on the target
content direction, we get the mean

$$
\mathbb E[s_{t,\tau}^{\mathrm{phase}}]
\approx
\rho_q \rho_\tau\,
u^\top R(\theta_\tau-\theta_t)u.
$$

Define the normalized RoPE self-correlation of the latent template

$$
C_u(\delta)
:=
\frac{u^\top R(\delta)u}{\|u\|^2}.
$$

Then

$$
\mathbb E[s_{t,\tau}^{\mathrm{phase}}]
\approx
\rho_q \rho_\tau \|u\|^2 C_u(\theta_\tau-\theta_t).
$$

Now suppose the target memory is stored at index $i$ and the query is aligned
to that target. Then the target score mean is

$$
\mathbb E[s_{t,i}^{\mathrm{phase}}]
\approx
\rho_q \rho_i \|u\|^2 C_u(\theta_i-\theta_t),
$$

while a distractor at another position $\tau$ has

$$
\mathbb E[s_{t,\tau}^{\mathrm{phase}}]
\approx
\rho_q \rho_\tau \|u\|^2 C_u(\theta_\tau-\theta_t).
$$

If we measure offsets relative to the target and define

$$
\delta_\tau := \theta_\tau-\theta_i,
$$

then the distractor-vs-target attenuation is controlled by

$$
C_u(\delta_\tau)
=
\frac{u^\top R(\theta_\tau-\theta_i)u}{\|u\|^2}.
$$

This recovers the same RoPE suppression mechanism, but now in a causal form:
each key only depends on local content, and the relative offset appears only
through the composition of the read-time and write-time absolute phases.

The random score perturbation is

$$
q_t^\top R(\theta_\tau-\theta_t)\eta_\tau,
$$

whose variance is

$$
\mathrm{Var}(q_t^\top R(\theta_\tau-\theta_t)\eta_\tau)
=
\sigma_k^2 \|q_t\|^2,
$$

because $R(\theta_\tau-\theta_t)$ is orthogonal.

If we further specialize to the idealized aligned-query regime

$$
q_t \approx \rho_q R(\theta_i-\theta_t)u,
$$

then

$$
s_{t,\tau}^{\mathrm{phase}}
=
q_t^\top R(\theta_\tau-\theta_t)k_\tau
\approx
\rho_q \rho_\tau\,
u^\top R(\theta_\tau-\theta_i)u
+
q_t^\top R(\theta_\tau-\theta_t)\eta_\tau.
$$

This is the causal replacement for the earlier noncausal ansatz.

For notational continuity below, we write

$$
C(\delta)
:=
\frac{u^\top R(\delta)u}{\|u\|^2}.
$$

Then the target has $C(0)=1$, while wrong-offset distractors typically satisfy

$$
|C(\delta)|<1.
$$

So RoPE suppresses the **structured distractor mean** exactly as in the
attention toy model, but now inside the phased DeltaNet read coefficient.

In the aligned-query specialization,

$$
s_{t,\tau}^{\mathrm{phase}}
 \approx
\rho_q \rho_\tau \|u\|^2 C(\delta_\tau)
+
q_t^\top R(\theta_\tau-\theta_t)\eta_\tau.
$$

#### Toy SNR for phased DeltaNet

If we neglect cross-covariances among the write vectors $\delta_\tau$, then

$$
\mathbb E\|r_{\mathrm{phase}}\|^2
\approx
\sum_{\tau\neq i}
a_{t,\tau}^2\,
\mathbb E\Bigl[(s_{t,\tau}^{\mathrm{phase}})^2\Bigr]\,
\mathbb E\|\delta_\tau\|^2.
$$

Using the Gaussian score model,

$$
\mathbb E\Bigl[(s_{t,\tau}^{\mathrm{phase}})^2\Bigr]
\approx
\rho_q^2 \rho_\tau^2 \|u\|^4 C(\delta_\tau)^2
+
\sigma_k^2 \|q_t\|^2.
$$

Therefore a useful toy approximation is

$$
\boxed{
\mathrm{SNR}_{\mathrm{phase\_DN}}
\approx
\frac{
a_{t,i}^2 \rho_q^2 \rho_i^2 \|u\|^4 \,\mathbb E\|\delta_i\|^2
}{
\sum_{\tau\neq i}
a_{t,\tau}^2
\left[
\rho_q^2 \rho_\tau^2 \|u\|^4 C(\delta_\tau)^2
+
\sigma_k^2 \|q_t\|^2
\right]
\mathbb E\|\delta_\tau\|^2
}.
}
$$

Without phase, the corresponding toy model is

$$
\boxed{
\mathrm{SNR}_{\mathrm{plain\_DN}}
\approx
\frac{
a_{t,i}^2 \rho_q^2 \rho_i^2 \|u\|^4 \,\mathbb E\|\delta_i\|^2
}{
\sum_{\tau\neq i}
a_{t,\tau}^2
\left[
\rho_q^2 \rho_\tau^2 \|u\|^4
+
\sigma_k^2 \|q_t\|^2
\right]
\mathbb E\|\delta_\tau\|^2
}.
}
$$

So in this toy model, the phased improvement comes from replacing the
structured distractor term

$$
\rho_q^2 \rho_\tau^2 \|u\|^4
$$

by

$$
\rho_q^2 \rho_\tau^2 \|u\|^4 C(\delta_\tau)^2.
$$

Since $C(\delta)^2 \le 1$, phase never increases the structured distractor
term in this approximation, and usually reduces it for wrong offsets.

#### Meaning of $C(\delta)$

RoPE acts in 2D frequency pairs. If we decompose the latent template into RoPE frequency
blocks, then

$$
u^\top R(\delta)u
=
\sum_m \|u^{(m)}\|^2 \cos(\omega_m \delta),
$$

so

$$
\boxed{
C(\delta)
=
\frac{
\sum_m \|u^{(m)}\|^2 \cos(\omega_m \delta)
}{
\sum_m \|u^{(m)}\|^2
}.
}
$$

Thus $C(\delta)$ is a weighted cosine kernel over relative offset. Wrong
offsets cause phase cancellation across frequencies, which is exactly why RoPE
reduces structured distractor power.

#### What this toy model does and does not capture

This toy model correctly captures the **readout-time** advantage of phased
DeltaNet:

* target-aligned memories keep full mean score;
* wrong-offset distractors are attenuated by $C_q(\Delta_\tau-\Delta_i)$;
* the DeltaNet decay factors $a_{t,\tau}$ further shape which memories matter.

But it does **not** yet capture:

1. write-time overwrite effects in $\delta_\tau$;
2. storage-time cancellation inside the matrix state $S_t$;
3. correlations between the score coefficients and the write vectors.

So it should be read as a local SNR model for phased DeltaNet addressing, not a
complete capacity theory of the full recurrent memory.

### 7.7 Interpretation

The comparison is therefore:

1. **Random-memory regime.**
   Phase gives little or no asymptotic SNR gain; it mainly preserves the
   baseline scaling.
2. **Collision-dominated regime.**
   Phase improves SNR by splitting one content address into many
   position-dependent sub-addresses through $R(\theta_\tau-\theta_t)$.
3. **Practical implication.**
   The main value of phased DeltaNet is not increasing the raw dimension from
   $d_k$ to something larger; it is reducing structured interference from
   repeated or aliased memories by making readout depend on relative position.

This is precisely the retrieval pattern we want: the model reads memory through

$$
q_t^\top R(\Delta_{t,\tau})k_\tau,
$$

so memories that look similar in content can still be separated by their phase.

---

## 8. Cancellation Risk Without Nonnegativity

If the memory writes are no longer constrained to be nonnegative, then
**destructive cancellation becomes possible**.

In the phased model,

$$
S_t
=
\sum_{\tau=1}^t
a_{t,\tau}\,
\bar k_\tau \delta_\tau^\top,
\qquad
\bar k_\tau = R(\theta_\tau)k_\tau.
$$

The squared memory magnitude contains cross-terms

$$
\|S_t\|_F^2
=
\sum_\tau a_{t,\tau}^2 \|\bar k_\tau \delta_\tau^\top\|_F^2
+
\sum_{\tau\neq j}
a_{t,\tau} a_{t,j}\,
\langle \bar k_\tau,\bar k_j\rangle\,
\langle \delta_\tau,\delta_j\rangle.
$$

When the cross-terms are negative, different memories cancel each other and the
stored signal energy shrinks. This is genuine memory loss, not just a readout
artifact.

So there is a real tradeoff:

1. signed writes increase expressivity and allow overwrite / error-correcting
   behavior;
2. signed writes also introduce the possibility of destructive superposition;
3. phase helps separate memories at readout, but it does **not** by itself
   prevent algebraic cancellation inside $S_t$.

In other words, phased addressing reduces **retrieval interference**, whereas
nonnegativity would reduce **storage-time cancellation**.

### 8.1 Why a Nonnegative Baseline Uses Only a Cone / Half-Space

If a baseline model enforces nonnegativity on the effective features used to
write memory, then its addresses or write vectors lie in a restricted region of
space. Geometrically, the model does not use the full sphere of directions in
$\mathbb{R}^{d_k}$ or $\mathbb{R}^{d_v}$; it uses only a cone-like subset
(informally, "half the space").

That has two opposite effects:

1. **Benefit:** fewer sign reversals, hence less destructive cancellation.
2. **Cost:** fewer usable directions, hence less angular resolution and weaker
   ability to separate different memories.

So the nonnegative baseline is more stable in one sense, but less expressive in
its address geometry.

### 8.2 Potential Benefit of the Phased Array

The phased construction partially compensates for the loss of the nonnegative
constraint by giving each content key a **position-dependent family of rotated
addresses**:

$$
\bar k_\tau = R(\theta_\tau)k_\tau.
$$

This creates several potential benefits.

First, it expands the set of effective addresses seen at readout. Even if the
underlying content key $k$ is reused many times, the actual memory term depends
on the rotated address

$$
q_t^\top R(\theta_\tau-\theta_t)k.
$$

So one content direction is no longer tied to a single address; it becomes a
structured orbit of addresses indexed by relative position.

Second, this helps recover some of the discrimination power that a nonnegative
model loses by occupying only a cone. A nonnegative model avoids cancellation,
but it also compresses many memories into similar directions. The phased model
uses signed directions, but phase spreads those directions across relative
position, which reduces structured aliasing.

Third, in repeated-pattern settings, the phased array behaves like a
position-sensitive codebook:

* baseline nonnegative addressing may map many similar memories into the same
  cone;
* phased addressing maps them into different rotated variants of that cone;
* the query can then select the correct variant through the relative phase
  factor $R(\theta_\tau-\theta_t)$.

So the qualitative comparison is:

1. **Nonnegative baseline:** lower cancellation risk, but reduced geometric
   coverage of the address space.
2. **Signed phased model:** higher cancellation risk, but richer address
   geometry and better separation of repeated memories across position.

The hope is that the phased array recovers enough extra addressing capacity to
outweigh the additional cancellation risk, especially in tasks where the main
failure mode is **collision between similar memories at different positions**
rather than pure random noise.

### 8.3 Net Effect

From a memory-capacity perspective, the phased model is not simply "better" or
"worse" than a nonnegative baseline. It changes the tradeoff:

$$
\text{nonnegativity}:
\text{less cancellation, less directional coverage},
$$

$$
\text{phase + signed writes}:
\text{more cancellation risk, more effective address diversity}.
$$

That is exactly why the phased design is attractive for long-context retrieval:
it uses relative phase to turn one content key space into a larger
position-sensitive family of addresses, which can reduce structured collisions
even though it no longer benefits from nonnegative accumulation.

---

## 9. How to Extend This to Increase Model Capacity

The main capacity bottleneck in baseline DeltaNet is that too many memories must
share the same finite key space $\mathbb{R}^{d_k}$. In the SNR view, retrieval
quality degrades once too many memories compete through the same inner products
$q_t^\top k_\tau$.

The phased construction already helps by replacing this with the richer address
family

$$
q_t^\top R(\theta_\tau-\theta_t)k_\tau.
$$

But if the goal is to **increase capacity further**, the right question is:

> how can we enlarge the number of effectively distinguishable memory addresses
> without breaking the DeltaNet update rule?

Below are the cleanest extensions.

### 9.1 Increase Address-Space Dimension

The most direct way is still to increase the key/query dimension $d_k$.

From the retrieval-SNR perspective,

$$
\mathrm{SNR}_{\mathrm{base}} \asymp \frac{d_k}{N_{\mathrm{eff}}},
$$

so larger $d_k$ gives more nearly-orthogonal addresses. In the phased model,
this also increases the number of RoPE pairs that can carry relative phase.

So the first extension is simply:

1. increase $d_k$;
2. dedicate a larger fraction of $d_k$ to phased channels;
3. keep the remaining channels unphased for pure content matching.

This produces a hybrid address space:

$$
k_t = [k_t^{\mathrm{phase}} ; k_t^{\mathrm{content}}],
\qquad
q_t = [q_t^{\mathrm{phase}} ; q_t^{\mathrm{content}}],
$$

with only the first block rotated by $R(\theta_t)$.

This is likely better than phasing all channels, because:

* phased channels help separate memories by relative position;
* unphased channels preserve pure semantic/content matching.

### 9.2 Use Multiple Independent Phase Banks

A stronger extension is to use not one RoPE system, but several independent
phase banks:

$$
\bar k_t =
\big[
R_1(\theta^{(1)}_t)k_t^{(1)};
\dots;
R_M(\theta^{(M)}_t)k_t^{(M)}
\big],
$$

and similarly for $\bar q_t$.

Then each memory is addressed simultaneously through several relative phase
codes. The resulting read score becomes

$$
\bar q_t^\top \bar k_\tau
=
\sum_{m=1}^M
(q_t^{(m)})^\top
R_m(\theta_\tau^{(m)}-\theta_t^{(m)})
k_\tau^{(m)}.
$$

This can increase capacity because two memories now collide only if they are
similar across **multiple** phase systems at once.

Intuitively:

* one phase bank gives one position-sensitive code;
* many phase banks give a product code.

That is much closer to an associative array with a larger effective address
space.

### 9.3 Make Phase Learnable or Data-Dependent

Fixed RoPE gives a predetermined positional code. Capacity can be improved if
the model learns how to place memories in phase space:

$$
\bar k_t = R(\theta_t(x_{\le t}))\,k_t,
\qquad
\bar q_t = R(\theta_t(x_{\le t}))\,q_t.
$$

Here $\theta_t$ can depend on token content, segment identity, memory type, or
context state.

This creates a larger effective codebook because different tokens need not share
the same phase schedule. Instead of using only absolute position, the model can
learn to route memories into different phase sectors.

A practical constrained version is:

1. keep a fixed RoPE backbone;
2. add a small learned phase offset $\Delta\theta_t$;
3. use
   $$
   \bar k_t = R(\theta^{\mathrm{rope}}_t + \Delta\theta_t)k_t.
   $$

That preserves relative-position structure while giving the model extra freedom
to de-conflict memories.

### 9.4 Use Complex / Oscillatory Multi-Timescale Addressing

Capacity improves when different memories decorrelate across lag. This suggests
using many frequencies, not just the standard RoPE schedule.

Let the phased key be decomposed into frequency groups:

$$
\bar k_t =
\big[
R(\omega_1 t)k_t^{(1)};
\dots;
R(\omega_M t)k_t^{(M)}
\big].
$$

Then the relative-address score is

$$
\bar q_t^\top \bar k_\tau
=
\sum_{m=1}^M
(q_t^{(m)})^\top R\bigl(\omega_m(\tau-t)\bigr)k_\tau^{(m)}.
$$

Using a broad frequency set helps because memories that alias at one lag/frequency
will often separate at another. This is analogous to increasing capacity by
using a richer kernel feature map.

From the kernel viewpoint, phased DeltaNet is no longer using only the linear
kernel $q^\top k$; it is using a structured feature map over

$$
(k,\tau)\mapsto R(\theta_\tau)k.
$$

Adding more frequencies enlarges that feature map and therefore enlarges the
effective associative-memory capacity.

### 9.5 Add Head-Wise Specialization

Another clean extension is to let different heads specialize to different memory
roles:

1. some heads mostly store content-stable memories;
2. some heads specialize in high-frequency phase discrimination;
3. some heads use slow phase for coarse long-range localization.

This matters because capacity is not just raw dimensionality; it is also how
well the model allocates that dimensionality. A heterogeneous head layout can be
more efficient than making every head identical.

In particular, one can assign per-head tuples

$$
(d_k^{(h)},\; d_{\mathrm{phase}}^{(h)},\; \omega^{(h)},\; \text{decay scale}^{(h)}),
$$

so that some heads become short-range precise associative arrays and others
become long-range coarse memories.

### 9.6 Increase Capacity on the Value Side Too

The address space determines **which** memory is retrieved, while the value side
determines **how much distinct information** can be attached to each address.

So another extension is to increase $d_v$ or to factorize values into multiple
submemories:

$$
S_t = \sum_{m=1}^M S_t^{(m)},
\qquad
S_t^{(m)} \in \mathbb{R}^{d_k^{(m)} \times d_v^{(m)}}.
$$

This does not by itself solve address collisions, but paired with phased
addressing it increases the total amount of retrievable information per
distinguished memory slot.

### 9.7 Use Sparse or Routed Writing

Capacity can also be increased by reducing how many memories interfere inside a
single state.

Instead of writing every token into every head/state, use a routing function
$\pi_t$ so that each token writes only to a subset of memory banks:

$$
S_{t+1}^{(m)} =
\begin{cases}
\alpha_t^{(m)} S_t^{(m)} + \bar k_t^{(m)}(\delta_t^{(m)})^\top, & m \in \pi_t, \\
\alpha_t^{(m)} S_t^{(m)}, & m \notin \pi_t.
\end{cases}
$$

This increases capacity because each bank sees a smaller effective
$N_{\mathrm{eff}}$.

From the SNR viewpoint, this is often as important as increasing dimension:

$$
\mathrm{SNR} \propto \frac{\text{address resolution}}{\text{memories per bank}}.
$$

So routing and sparsity improve capacity by reducing denominator pressure.

### 9.8 The Most Promising Combined Design

A practical high-capacity phased Gated DeltaNet would likely combine:

1. a larger $d_k$ than baseline;
2. a split between phased and nonphased key channels;
3. multiple independent phase banks or frequency groups;
4. head-wise specialization across timescales;
5. sparse/routed writing into memory banks.

One possible formulation is:

$$
S_t^{(m,b)}
=
\alpha_t^{(m,b)} S_{t-1}^{(m,b)}
+
\bar k_t^{(m,b)} (\delta_t^{(m,b)})^\top,
$$

with

$$
\bar k_t^{(m,b)} = R_b(\theta_t^{(b)})k_t^{(m,b)},
\qquad
\bar q_t^{(m,b)} = R_b(\theta_t^{(b)})q_t^{(m,b)},
$$

where:

* $m$ indexes memory heads,
* $b$ indexes phase banks / frequency groups,
* routing controls which banks each token writes to.

This turns the memory into a **banked phased associative array** rather than one
single global store.

### 9.9 Summary

To increase capacity, the phased model should not rely on phase alone. The best
path is to enlarge the number of distinguishable addresses along several axes:

1. more key dimension,
2. more independent phase codes,
3. more frequencies / timescales,
4. more memory banks with sparse routing,
5. balanced phased and nonphased channels.

Conceptually, the phased model increases capacity by transforming a single
address space

$$
q^\top k
$$

into a richer structured address family

$$
q^\top R(\Delta)k,
$$

and then scaling that family through multiple banks, frequencies, and routing.
That is the natural way to turn phased DeltaNet into a genuinely higher-capacity
memory architecture.
