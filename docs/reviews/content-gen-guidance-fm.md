# Review: gen.guidance and gen.flow-matching

Critic pass, 2026-10-02. Standard: `docs/review-rubric.md`, `docs/writing-brief.md`. Plan: `docs/content-plan.md` B.4, B.5 §170 and §180, plus `docs/reviews/plan-generative-review.md`.
Checks run: every number in both lessons recomputed in Python; both figure modules re-run (`_tradeoff_data`, `_sat_data`, `_paths2d`, `_field`); all 6 figures viewed in workbench, plus terminal or editorial. `validate.py` reports 0 errors. The only warning is the expected missing prereq `gen.flows`.

## Writer's flagged points (checked independently)

1. **$u_t(x)=(x+(1-t)\nabla\log p_t(x))/t$: correct.**
   - *Analytic check.* $x_t\mid x_1\sim\mathcal N(tx_1,(1-t)^2I)$ gives $\nabla\log p_t=-\mathbb E[x_0\mid x_t]/(1-t)$. Then $u=\mathbb E[x_1-x_0\mid x]$, and $x=t\mathbb E[x_1\mid x]+(1-t)\mathbb E[x_0\mid x]$ gives $u=(x-\mathbb E[x_0\mid x])/t$, which is the formula.
   - *Numerical check.* I compared it with the exact mixture field in `gen.flow-matching.py` at 9 $(t,x)$ points. All agree to 1e-10.
   - *Convention check.* It matches Lipman's convention ($x_0$ noise, $t=1$ data). It also matches Gao et al.'s PF-ODE ("Diffusion models" section, $dz=(f_tz-\tfrac12g_t^2\nabla\log p_t)dt$) with $\alpha=1-\tau$, $\sigma=\tau$: there $f=-1/(1-\tau)$ and $g^2=2\tau/(1-\tau)$, and $t=1-\tau$ gives the same formula.
   - **Citation is not honest as written.** Neither cited place states it. Lipman §4 never writes the linear-path score relation. App. D covers VP/VE only. Gao's "How do we choose what the network should output?" section has no score row. Fix: cite it as "derived here (Tweedie); equivalently Gao et al. 2024, 'Diffusion models' (PF-ODE with $f_t,g_t$), time reversed". Apply this to guidance explainer[8] and f11, and to FM explainer[8] and f5.
2. **Toy 4% / 20% / 30%: numbers confirmed, framing mostly fine.** Re-running `_tradeoff_data` at $s=3$:
   - the guided ODE keeps 0.041 of samples in the minor mode;
   - the exact tilt $p(x)p(A\mid x)^3$ keeps 0.205;
   - the data has 0.30, of class A.

   It is clearly labelled a toy with exact scores. Two problems remain:
   - (a) The figure plots coverage **normalised by 30%** (0.14 and 0.68 at $s=3$), and the y-axis has no label. A reader can't match 4%/20% to the plot (see G-F3).
   - (b) Kynkäänniemi Fig. 2 is a 1-D toy at $w=6$ with ideal denoisers, showing mode drop *relative to the conditional data*. It does not compare against the exact tilted density. So "show the same effect" overstates it: say "a related effect". It is consistent otherwise: high-noise guidance pushes trajectories off-distribution.
3. **w/s mapping: correct.**
   - Imagen eq. 2 / §2.2: "Setting w = 1 disables classifier-free guidance".
   - Kynkäänniemi eqs. 3–4: $wD(x|c)+(1-w)D(x)$.
   - Luo: $\gamma\nabla\log p(x|y)+(1-\gamma)\nabla\log p(x)$, p.21.
   - Dieleman's $\gamma$, and Ho eq. 6 $(1+w)$, are also right.
   - **Missing: Lin et al.'s $w$ is also $s$.** Lin eq. 13 is $x_{neg}+w(x_{pos}-x_{neg})$. Explainer[7] and f7 write "$w=7.5$" without flagging this, in a lesson that has just taught "$w$ = Ho's $s-1$" (see G-M2).
4. **FM = ε-loss weighted by $(1+\mathrm{SNR}^{-1/2})^2$: correct.** Kingma & Gao App. D.3 eq. 77 gives $(1+e^{-\lambda/2})^2$, and so does Gao's table. My own derivation: $\hat u-u=(1+\sigma/\alpha)(\hat\epsilon-\epsilon)$. The lesson asserts it without the 2-line derivation, though (see FM-M2).
5. **$dz/dt=-\tfrac{\pi}{2}v$: correct.** $v=dz_\phi/d\phi$ by Salimans & Ho App. D, and $d\phi/dt=-\pi/2$.
6. **CFG rescale, $\phi=0.7$ at $w=7.5$: correct.** Lin §3.4 says "w = 7.5, ϕ = 0.7 works great". Alg. 2 matches. $\phi\in[0.5,0.75]$ is from §5.3 / Fig. 6. "Overly plain" at full rescale and the near-zero-terminal-SNR sensitivity are both verbatim in substance.
7. **EDM2 FID 1.81 → 1.40: correct.** Kynkäänniemi Table 1 and §4: EDM2-XXL, interval $(0.19,1.61]$, $w=2.0$ vs 1.2.
8. **FM sign convention: correct.**
   - Lipman: $u=x_1-x_0$.
   - SD3 eq. for $z_t=(1-t)x_0+t\epsilon$: "velocity prediction target ε − x0" (§3.1).
   - Gao: $u=\epsilon-x$.

---

## gen.guidance

**Verdict: revise** (no blocking errors; several major clarity/consistency fixes)

### Findings

**Major**
- **G-M1 `explainer[8]` (flows paragraph): $x_0$ silently changes meaning.** The lesson's notation card defines $x_0$ as *data* ($x_t=\alpha_tx_0+\sigma_t\epsilon$). This card then writes $x_t=t\,x_1+(1-t)\,x_0$ with $x_0$ as *noise*, and never says so. The card also relies on a derivation "in gen.flow-matching", which comes *after* this lesson (order 180, not a prereq). Fix: cut the paragraph to two sentences. Velocity is an affine function of the score with condition-independent coefficients, so CFG on velocities is CFG on scores with the same $s$ (derived in gen.flow-matching). Drop f11, which duplicates FM f5/f11. If you keep it, state "$x_0\sim\mathcal N(0,I)$ is noise here (Lipman's labels, opposite to this lesson)" and include the one-line Tweedie step. This also helps the length (see Length below).
- **G-M2 `explainer[7]`, `f7`, `explainer[5]`: Lin's $w$ is used unflagged.** "They use $w=7.5$". f7 writes $x_{cfg}=x_{neg}+w(x_{pos}-x_{neg})$. In this lesson $w$ means Ho's $w=s-1$, so a careful reader would read this as $s=8.5$. Fix: write "$s=7.5$ (Lin et al. call it $w$, but their $w$ is this lesson's $s$)". Also add Lin to the convention-clash bullet list in explainer[3].
- **G-M3 `explainer[5]` figure / caption: the figure's units don't match the text.** The text says 4% vs 20% vs 30%. The plot shows "coverage" normalised so 1 = 30% (0.14 and 0.68 at $s=3$), and its y-axis is unlabelled. Fix: add a y-label ("fraction of data's minor-mode share") or plot absolute shares. Add to the caption: "at $s=3$: 0.14 (4% of samples) vs 0.68 (20%) under the exact tilt".
- **G-M4 `explainer[5]`: evaluation terms used before they are taught.** IS, FID and precision/recall appear (and q6 depends on them), but gen.evaluation is order 220. Fix: add one line each. IS rewards confident, class-distinct samples. FID is the Fréchet distance between Gaussian fits to Inception features of real and generated sets, so it penalises lost diversity. Precision is the share of samples on the data manifold, and recall the share of data covered by samples (Kynkäänniemi 2019).

**Minor**
- **G-m1 `explainer[4]`, third caveat bullet** ("skip the unconditional pass…"). This is faithful to Ho §5, but it is not a caveat, and it hides a catch. $\nabla\log\sum_cp_t(x|c)p(c)=\sum_cp_t(c|x)\nabla\log p_t(x|c)$ needs the posterior weights $p_t(c|x_t)$, which an ε/score network doesn't provide. Either add that catch or cut the bullet. Cutting it saves space.
- **G-m2 `explainer[5]`:** change "Kynkäänniemi et al. show the same effect" to "a related effect (their Fig. 2: 1-D, $w=6$, ideal denoisers; the mode drops relative to the conditional data)".
- **G-m3 `explainer[7]`, guidance interval:** $\sigma\in(0.19,1.61]$ is EDM's noise level ($x=y+\sigma n$), which equals this lesson's $\sigma_t/\alpha_t$, not $\sigma_t$. Say so.
- **G-m4 `explainer[8]`, Cost:** add the punchline that makes Ho's point interesting: at the compute-fair $T=128$, CFG *underperforms* ADM-G on FID (Ho §4.3).
- **G-m5 Figures `tradeoff-toy`, `oversaturation`:** the toys are VE ($x_0+\sigma\epsilon$, PF-ODE in σ), while the lesson's notation is VP. Add "VE parameterisation" to the captions. The oversaturation figure also shows the 0.35 mode vanishing at $s\ge3$. One clause in the caption ("and the minor mode at 0.35 is dropped") would tie it to explainer[5].
- **G-m6 Redundancy:** the $w\leftrightarrow s$ conversion appears in explainer[3], explainer[4] ("Equivalently, it is weight $w=s-1$…"), [9], [10], q1, q2 (its 1.4 distractor) and f8. Cut the sentence in explainer[4]: explainer[3] already covers it.
- **G-m7 Questions:** q3, q5 and q10 are near-recall. q5 is close to a giveaway for anyone who has read the CFG card. Turn q10 into a predict-question, for example: "With guidance on only at high σ, which failure do you expect?" (mode drop / template collapse).
- **G-m8 `explainer[0]`:** "Guidance is the knob that works." adds nothing. Cut it.

### Strengths
The implicit-classifier derivation, the over-saturation worked example ($\hat x_0$ = 0.1 / 0.5 / 1.7 / 3.1, all verified), and the "noisy classifier doesn't commute with tilting" point are excellent interview material. All numbers check out: sharpening means/sds/tails, the toy shares, q2/q8/q9, Ho's w-sweep optima (1.55 at w=0.1 for 64², 2.43 at w=0.3 for 128²), Imagen 1.35, SD3 5.0, and Dhariwal's ~50%.

### Length (11 cards)
This is slightly over. The plan budget was 9–10, and the brief says 6–10. Cutting explainer[8]'s FM derivation (G-M1) and the class-sum bullet (G-m1) frees about half a card. Fold the remaining Costs/variants content (cost, negative prompts, AR guidance, pointers) into a trimmed card. Explainer[9], the probes card, mostly restates cards 2–8, so trim it to the five sharpest follow-ups. That gets the lesson to 10 cards without losing a derivation.

---

## gen.flow-matching

**Verdict: revise** (one blocking figure/caption error; teaching gaps on OT and the weighting)

### Findings

**Blocking**
- **FM-B1 `explainer[7]` figure caption, plus card text, f10 and the probes card: "They bend, never cross" is false for the 2-D picture shown.** I re-ran `_paths2d`. **18 of 231 pairs of marginal trajectories intersect in space**, at *different* times (e.g. trajectory 0 at t=0.24 crosses trajectory 4's path at t=0.5). Non-crossing holds in $(x,t)$: two particles never occupy the same $x$ at the same $t$. Spatial projections can cross. As written, the caption plants a classic misconception and is contradicted by its own figure. Fixes:
  - Caption: "They bend, and never meet at the same place at the same time."
  - Card text after "solution through each $(x,t)$ is unique": add "(their 2-D projections can still cross at different times)".
  - f10: say the same in "ODE trajectories also cannot cross".

  The 1-D `probability-path` caption is fine as it stands, because it plots $(t,x)$.

**Major**
- **FM-M1 Optimal transport is never defined** ("OT path", "OT displacement", "minibatch OT", "dynamic OT problem"; explainer[5], [7], f7, q9). A reader new to OT can't follow. Add a 2–3 sentence refresher in explainer[5]:
  - Static OT: the coupling $\pi(x_0,x_1)$ with the given marginals that minimises $\mathbb E\|x_1-x_0\|^2$.
  - Dynamic OT: the velocity field of least kinetic energy $\int\mathbb E\|u_t\|^2dt$ carrying $p_0$ to $p_1$. Its particles move in straight lines at constant speed.
  - Displacement interpolation: $(1-t)x_0+tT(x_0)$.
  - Minibatch OT: solve the discrete assignment on each batch (Hungarian / Sinkhorn).
- **FM-M2 `explainer[8]`, the weighting is asserted.** "$\|\hat u-u\|^2=(1+\mathrm{SNR}^{-1/2})^2\|\hat\epsilon-\epsilon\|^2$" can be derived in two lines, so derive it. In diffusion time, $\hat x=(z-\tau\hat\epsilon)/(1-\tau)$ and $u=\epsilon-x$, so
  $$\hat u-u=\Big(1+\tfrac{\tau}{1-\tau}\Big)(\hat\epsilon-\epsilon)=\tfrac{\hat\epsilon-\epsilon}{\alpha_\tau},$$
  and $1/\alpha_\tau=1+\sigma_\tau/\alpha_\tau=1+\mathrm{SNR}^{-1/2}$ because $\alpha+\sigma=1$.

  Also define SNR $=\alpha^2/\sigma^2=((1-\tau)/\tau)^2$ here, and scope it: this is per-sample at fixed $t$. With $t\sim U[0,1]$ the implied weight in λ-space is $e^{-\lambda/2}$, which Kingma & Gao show equals v-prediction with a cosine schedule. That last fact links neatly to the v-prediction paragraph just above.
- **FM-M3 `explainer[8]`: the score formula skips its key step and is mis-cited.** "Hence $u_t(x)=\ldots$" jumps over the algebra. Add: "$u=\mathbb E[x_1|x]-\mathbb E[x_0|x]$ and $x=t\mathbb E[x_1|x]+(1-t)\mathbb E[x_0|x]$, so $u=(x-\mathbb E[x_0|x])/t$". Also show the Tweedie line: $\nabla\log p_t=\mathbb E[-(x-tx_1)/(1-t)^2\mid x]$. Fix the source (flagged point 1).
- **FM-M4 `explainer[8]` overlaps the next lesson.** The plan gives "the diffusion–flow equivalence" and the $(1+\mathrm{SNR}^{-1/2})^2$ weighting to gen.rectified-flow (plan review C13; B.5 §190 item 4). This card pre-empts it. That is acceptable, since the user's draft scope listed "relation to diffusion, v-prediction" under flow matching. But **tell the coordinator**, so that gen.rectified-flow builds on it rather than repeating it. Within this lesson, keep the sign convention, the conversions and v-prediction. The DDIM-equals-Euler and weighting details could move to rectified-flow if this card needs to slim down (it is 224 words, so that is not urgent).

**Minor**
- **FM-m1 `explainer[6]`:** "(last card)" should be "the diffusion card": explainer[8] is not the last card.
- **FM-m2 `explainer[0]`:** Hutchinson is named but not stated. Add "$\mathrm{tr}J=\mathbb E_\epsilon[\epsilon^\top J\epsilon]$, one vector–Jacobian product per sample".
- **FM-m3 `explainer[6]`, Lipman Table 1:** FM with the VP path also had NLL 3.10 (vs DDPM 3.12). Include it so the three rows are comparable.
- **FM-m4 `explainer[7]`** is 301 words with two figures, at the limit. The "Numbers" bullets repeat what the posterior-average caption already says. Keep one of the two.
- **FM-m5 q10 explanation:** "Its velocity is $x_0=x/(1+t)$" reads oddly. Write "$\tfrac{d}{dt}(1+t)x_0=x_0=x/(1+t)$", and explain the $u=x$ distractor as "confusing the current position $x$ with the start point $x_0$".
- **FM-m6 q6** is close to recall of Thm 3. That's acceptable given the derivation-step framing, but the stem could be "Which step of the derivation is wrong" to make it reasoning.

### Strengths
Theorems 1 and 2 are derived cleanly, with each step justified. "Linear in the flux" is a great one-line intuition. The straight-conditional/curved-marginal treatment (the posterior-average figure, $u_0=\mathbb E[x_1]-x$, the one-step-gives-the-mean result) is exactly what interviewers probe. All numbers verified:
- the q1/q3/q12 calculations;
- the posterior weights 0.982/0.018 → 2.86 and 0.881/0.119 → 2.55;
- the worked example;
- the OT-path field algebra;
- the VP/VE paths;
- Lipman Table 1 (6.35/142/2.99, 7.48/274/3.12, 8.06/183).

### Length (11 cards)
Mostly earned. The continuity-equation refresher is required, because this W1 lesson precedes gen.flows. Thms 1–2, the Gaussian/OT paths and the curvature card are each dense derivations. The diffusion card is the only one beyond the plan's 9-card map (FM-M4), and the probes card partly restates the derivation cards. Keep 11 if the diffusion card stays. If gen.rectified-flow will take the equivalence, cut to 10.

---

## Cross-lesson
- Both lessons' f11 cards (guidance f11 and FM f5/f11) test the same velocity–score identity. Keep it in FM and drop it from guidance (G-M1).
- Notation: guidance uses $x_0$ = data, while FM uses $x_0$ = noise. Each lesson states its own convention, but the guidance lesson breaks its convention once (G-M1).
