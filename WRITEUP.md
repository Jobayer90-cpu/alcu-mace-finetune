# Fine-tuning a foundation model for a disordered Al-Cu alloy

Assessing MACE-MP-0 on Al75Cu25, improving it with 19 DFT structures, and testing
whether standard accuracy metrics predict a physical observable.

## 1. Motivation

Foundation models such as MACE-MP-0 are trained largely on relaxed, ordered structures
from materials databases. A random Al-Cu solid solution at finite temperature is unlike
that training data in two ways at once: chemically disordered, and thermally displaced
from ideal sites. Two questions follow: how much accuracy is lost, and can a small
amount of targeted DFT recover it?

A third question emerged during the work and turned out to be more interesting: does
force RMSE -- the metric almost universally reported for machine-learned potentials --
predict whether a model gets a physical property right?

## 2. Method

System: Al75Cu25 (Al24Cu8), FCC, 32-atom supercell. NOTE: at 25 at% Cu this is a
hypothetical random solid solution roughly 10x beyond the equilibrium solubility limit
of Cu in fcc Al (2.53 at% at 548 C). The equilibrium state at this composition is
alpha-Al + theta-Al2Cu (tetragonal). The structure is therefore a model system for
methodological comparison, not a synthesisable alloy, and no experimental reference
exists for its thermal expansion. All conclusions below are model-vs-model or
model-vs-DFT and do not depend on an experimental anchor. Chemical order quantified by the
first-shell Warren-Cowley parameter a1 = 1 - P(Cu|Al)/c_Cu, verified against analytic
limits (L1_2 gives exactly -1/3).

DFT: Quantum ESPRESSO 6.7, PBE, PAW pseudopotentials from PSLibrary. 45 Ry / 360 Ry
cutoffs, 3x3x3 k-mesh, Marzari-Vanderbilt smearing (0.02 Ry), conv_thr 1e-6.
Variable-cell relaxation converged in 26 ionic steps to a_eq = 3.9217 A
(V = 482.53 A^3, P = -0.47 kbar), consistent with Vegard's law between Al and Cu.

Training data: 30 structures spanning what production MD would sample -- 9 volume-scaled
(+-4%), 8 rattled (0.05-0.15 A), 8 with differing chemical order (a1 from -0.333 to
+0.111), 5 snapshots from 500 K MD. Forces span 0.019-4.350 eV/A; energies span 5.83 eV.

Split: 19 train / 11 test, stratified across all four types, fixed BEFORE any model
was evaluated.

Fine-tuning: MACE-MP-0 medium, single-head, lr 1e-4, 120 epochs, isolated-atom energies
set from training data. Two loss weightings compared: force-weighted (E:F = 1:100) and
balanced (10:10).

## 3. Results

### 3.1 Baseline accuracy degrades with distance from database-like structures

  volume-scaled        58.5 meV/A
  chemically ordered  112.6
  rattled             133.4
  MD snapshots        151.5
  overall             116.7

A 2.6x gradient from near-ideal lattice positions to realistic thermal geometries --
the predicted signature of training on relaxed database structures, measured directly.

### 3.2 Fine-tuning improves forces uniformly

Held-out force RMSE falls 116.7 -> 34.7 meV/A (3.4x, 70% reduction), every category
improving 2.8-4.4x. Energy RMSE reaches 4-6 meV/atom.

### 3.3 ...but degrades the equation of state

  model            F RMSE    V_min err    B (GPa)
  MACE-MP-0         116.7      -0.28%       99.4
  force-weighted     34.7      -1.85%      121.8
  balanced           38.7      -1.21%      116.1
  DFT reference        --   (489.00 A^3)    96.4

The baseline reproduces the DFT equation of state almost exactly. Fine-tuning degrades
it up to 6.6x, and the error is ASYMMETRIC: good under compression, poor under
expansion, where the fine-tuned curves rise roughly twice as steeply as DFT.

Rebalancing the loss recovers about a third of the V_min error for 4 meV/A in forces,
but does not close the gap. The limitation is training-data COVERAGE, not the objective:
the training set spans +-4% in lattice parameter while the E-V curve spans +-12% in
volume, so the fine-tuned model extrapolates where the baseline retains database coverage.

### 3.4 Thermal expansion: neither metric predicts the answer

NPT MD (Melchionna barostat), 5 ps equilibration + 20 ps production at 300/400/500 K.
Uncertainties from block averaging over five segments.

  model            V(300)  V(400)  V(500)   alpha_L (1e-6/K)
  MACE-MP-0         496.3   503.8   506.1     32.8 +- 10.0
  force-weighted    481.6   484.0   486.7     17.7 +-  0.3
  balanced          485.3   488.0   490.7     18.4 +-  0.3

The three models span 17.7 to 32.8 -- a factor of 1.9 -- with the baseline carrying a
30% uncertainty against 2% for the fine-tuned models. Its volumes are visibly non-linear in T
(steps of 7.5 then 2.3 A^3) and its volume fluctuations are 60% larger, so 20 ps was
insufficient for the mean to converge.

The two fine-tuned models agree within ~2 sigma despite differing in V_min by 0.6
percentage points, indicating the prediction is robust to the loss weighting.

Neither force RMSE nor equation-of-state accuracy ranked the models correctly for
alpha_L. Sampling quality dominated.

### 3.5 Catastrophic forgetting

Single-head fine-tuning on one composition shifted the predicted pure-Al lattice constant
from 4.0605 to 3.9826 A (-78 mA), well beyond a 30 mA tolerance. The model traded
transferability for specialisation. This does not affect the results above -- all
production runs are at the training composition -- but the fine-tuned models must not
be applied to other Al-Cu compositions.

### 3.6 Ordering energetics: below resolution

DFT energies across eight configurations spanning a1 = -0.333 to +0.111 vary by under
30 meV/atom, with a fitted trend of +5.3 meV/atom per unit a1. Two configurations with
IDENTICAL a1 = +0.111 differ by 29.6 meV/atom -- scatter as large as the trend. At 32
atoms this is not resolvable; no conclusion about ordering preference is drawn.

## 4. Two failure modes worth documenting

Energy-reference mismatch destroys the model silently. QE reports absolute total energies
(about -1128 eV/atom here); MACE-MP-0 uses a formation-energy-like reference. Fine-tuning
with --E0s=foundation left a 1124 eV/atom constant for the network to fit. Over 120 epochs
the loss decreased monotonically while force RMSE rose from 112 to 33,439 meV/A -- a 300x
degradation with no warning in the training log. Setting E0 from the training data fixed
it completely.

Multi-head fine-tuning is the default and is expensive. Recent MACE versions replay the
foundation dataset alongside user data to prevent forgetting. This raised epoch time from
8 s to 90 min (~700x). Disabling it made training tractable but produced the forgetting
in section 3.5 -- the two are directly connected.

## 5. Limitations

- 32 atoms. Random configurations scatter +-0.058 in a1 at this size; 8 Cu atoms is
  coarse for chemical-order statistics.
- 20 ps trajectories. Marginal for resolving volume differences of a few A^3, as the
  baseline's non-linear V(T) demonstrates.
- 45/360 Ry cutoffs, below the SSSP recommendation of ~55/440 Ry for Cu -- a deliberate
  speed compromise.
- Classical nuclei. Al's Debye temperature (~428 K) is within the sampled range.
- 19 training structures, one composition. Transferability cost quantified in 3.5.
- No experimental reference. A disordered fcc Al75Cu25 phase does not exist at
  equilibrium, so alpha_L cannot be validated against measurement. Pure Al's
  23.1e-6/K is shown for scale only. The comparison that matters here is between
  models, which needs no experimental anchor.
- No active learning. Structures chosen by hand, not by model uncertainty.

## 6. Next steps

1. Extend training to the volumes MD actually samples (+-10%), addressing the asymmetric
   equation-of-state error.
2. Longer trajectories (100 ps+) and a larger cell, to test whether the baseline's
   alpha_L converges toward the fine-tuned value.
3. On-the-fly active learning (FLARE or DP-GEN) so structures are chosen by uncertainty.
4. Multi-head fine-tuning where 90 min/epoch is affordable, to retain transferability.
5. Quasi-harmonic phonons for a quantum-corrected alpha_L, avoiding the classical-nuclei
   and sampling limitations entirely.

## 7. Figures

1. fig1_parity.png      force parity, baseline vs fine-tuned, held-out set
2. fig2_ev.png          equation of state, DFT vs both models
3. fig3_breakdown.png   error by structure type and vs Warren-Cowley a1
4. fig4_expansion.png   V(T) and fitted alpha_L for all three models

## Appendix: reproducing this

  stage01.py   gates, order parameter, vc-relax        ~5 h
  stage02.py   generate 30 structures                  ~2 min
  stage03.py   label with QE (4 workers x 8 ranks)     ~8.7 h
  stage04.py   split + baseline evaluation             ~2 min
  stage05.sh   fine-tune                               ~16 min
  stage06.py   parity, E-V, breakdown                  ~3 min
  stage07.py   NPT, 3 models x 3 temperatures          ~4.5 h
  stage08.py   fit alpha_L, final figure               ~1 min

Set OMP_NUM_THREADS=1 for QE (MPI-parallel) and 16 for MACE (thread-parallel). Reversing
this caused a 40x slowdown from thread oversubscription.
