# Fine-tuning a foundation model for a disordered Al–Cu alloy

**Does force accuracy predict thermodynamic accuracy?**

A pilot study assessing MACE-MP-0 on Al₇₅Cu₂₅, improving it with 19 structures of
purpose-generated DFT, and testing whether the standard accuracy metrics for
machine-learned potentials predict which model gets a physical observable right.

| Model | Force RMSE (meV/Å) | V<sub>min</sub> error | α<sub>L</sub> (10⁻⁶ K⁻¹) |
|---|---|---|---|
| MACE-MP-0 (as downloaded) | 116.7 | −0.28 % | 32.8 ± 10.0 |
| Fine-tuned, force-weighted | 34.7 | −1.85 % | 17.7 ± 0.3 |
| Fine-tuned, balanced | 38.7 | −1.21 % | 18.4 ± 0.3 |

Fine-tuning improved forces 3.4× and degraded the equation of state 6.6×.
Neither metric ranked the models correctly for thermal expansion.

---

## 1. Background

Machine-learned interatomic potentials let molecular dynamics run at near-DFT
accuracy for a fraction of the cost. Until recently each system needed its own
potential, trained on its own reference data. Foundation models such as
MACE-MP-0 change that: trained on a large public database of DFT calculations,
they claim useful accuracy across most of the periodic table with no training
at all.

That claim deserves testing on systems unlike the training data. Materials
database entries are overwhelmingly relaxed, ordered, stoichiometric crystals.
A random solid solution at finite temperature is neither: its atoms sit on a
disordered chemical lattice and are thermally displaced from ideal sites.

This study asks three questions.

1. How much accuracy does MACE-MP-0 lose on a chemically disordered,
   thermally distorted alloy?
2. Can fine-tuning on a small set of purpose-generated DFT recover it?
3. **Does force RMSE — the metric almost universally reported for these
   models — predict whether a model gets a physical property right?**

The third question emerged during the work and turned out to be the most
interesting.

## 2. System

Al₇₅Cu₂₅ (Al₂₄Cu₈), FCC, 32-atom supercell — 2×2×2 of the conventional cell.

**This is not a synthesisable alloy.** The terminal solid solubility of Cu in
fcc Al is 2.53 at % at 548 °C and falls with temperature; 25 at % is roughly ten
times that limit. The equilibrium state at this composition is α-Al + θ-Al₂Cu
(tetragonal). The structure here is a model system for methodological
comparison, chosen before the phase diagram was checked.

The consequence is that no experimental thermal expansion coefficient exists
for it. Every conclusion below is model-versus-model or model-versus-DFT and
needs no experimental anchor; pure Al's 23.1 × 10⁻⁶ K⁻¹ appears for scale only.

Chemical order is quantified by the first-shell Warren–Cowley parameter

    α₁ = 1 − P(Cu|Al) / c_Cu

where α₁ = 0 is random, α₁ < 0 means unlike neighbours are preferred, and
α₁ > 0 means like atoms cluster. The implementation was verified against
analytic limits: perfect L1₂ Al₃Cu returns exactly −1/3.

## 3. Methods

### 3.1 DFT reference

Quantum ESPRESSO 6.7, PBE, PAW pseudopotentials from PSLibrary 1.0.0.
45 Ry wavefunction / 360 Ry charge-density cutoff, 3×3×3 Monkhorst-Pack mesh,
Marzari–Vanderbilt cold smearing at 0.02 Ry, `conv_thr` 10⁻⁶.

Variable-cell relaxation converged in 26 ionic steps (≈340 SCF iterations,
4 h 50 min on 16 MPI ranks) to a_eq = 3.9217 Å, V = 482.53 Å³, final pressure
−0.47 kbar. The lattice constant is consistent with Vegard's-law interpolation
between Al (4.05 Å) and Cu (3.61 Å).

### 3.2 Training structures

Thirty structures were generated to span the configuration space that
production MD would visit:

| Count | Type | Purpose |
|---|---|---|
| 9 | Volume-scaled, ±4 % in lattice parameter | volume response |
| 8 | Rattled, 0.05–0.15 Å displacements | thermal disorder |
| 8 | Differing chemical order, α₁ = −0.333 … +0.111 | chemical disorder |
| 5 | Snapshots from 500 K MD | realistic MD geometries |

Forces across the set span 0.019–4.350 eV/Å and energies span 5.83 eV, so the
set is not 30 near-identical configurations.

Labelling ran as four concurrent jobs of 8 MPI ranks each, 8 h 40 min total.
Cost varied more than 10× across structures — 480 s for the easiest
volume-scaled cell, 4360 s for the hardest MD snapshot — while SCF iteration
count stayed nearly constant at 13–14. The slow structures were slow per
iteration, not slow to converge.

The set was split 19 train / 11 test, stratified across all four types, **before
any model was evaluated**.

### 3.3 Fine-tuning

MACE-MP-0 (medium), single-head, learning rate 10⁻⁴, 120 epochs, isolated-atom
energies set from the training data. Two loss weightings were compared:
force-weighted (energy:force = 1:100) and balanced (10:10).

### 3.4 Molecular dynamics

NPT with the Melchionna barostat in ASE, 2 fs timestep, `ttime` 25 fs,
`pfactor` = (75 fs)² × 96.4 GPa, zero external stress. 5 ps equilibration
followed by 20 ps production at 300, 400 and 500 K, for all three models —
nine runs in total. Volume sampled every 10 steps; uncertainties from block
averaging over five segments.

LAMMPS was attempted but not used: exporting a MACE model to the TorchScript
format LAMMPS requires calls `torch.jit.script`, which fails inside `e3nn` on
Python 3.14. This is an environment limitation, not a methodological one, and
at 32 atoms ASE and LAMMPS implement the same integrator and barostat.

## 4. Results

### 4.1 Baseline accuracy degrades with distance from database-like structures

![force parity](figures/fig1_parity.png)

*Figure 1. Predicted versus DFT forces on the 11 held-out structures.*

| Structure type | Force RMSE (meV/Å) |
|---|---|
| Volume-scaled | 58.5 |
| Chemically ordered | 112.6 |
| Rattled | 133.4 |
| MD snapshots | 151.5 |
| **Overall** | **116.7** |

A 2.6× gradient from near-ideal lattice positions to realistic thermal
geometries. Volume-scaled structures keep atoms on perfect sites — which is
what a materials database looks like — and the model handles them best. MD
snapshots are furthest from the training distribution and the model is worst
there. This is the predicted signature of training on relaxed database
structures, measured directly.

### 4.2 Fine-tuning improves forces uniformly

Held-out force RMSE falls from 116.7 to 34.7 meV/Å — a 3.4× improvement, 70 %
error reduction — with every category improving by 2.8–4.4×. Energy RMSE
reaches 4–6 meV/atom.

![error breakdown](figures/fig3_breakdown.png)

*Figure 3. Left: force error by structure type, before and after fine-tuning.
Right: error against Warren–Cowley α₁, showing no dependence on chemical order
for either model.*

### 4.3 …but degrades the equation of state

![equation of state](figures/fig2_ev.png)

*Figure 2. Energy–volume curves, each shifted to its own minimum.*

| Model | V<sub>min</sub> error | B (GPa) |
|---|---|---|
| MACE-MP-0 | −0.28 % | 99.4 |
| Force-weighted | −1.85 % | 121.8 |
| Balanced | −1.21 % | 116.1 |
| DFT reference | (489.00 Å³) | 96.4 |

The baseline reproduces the DFT equation of state almost exactly. Fine-tuning
degrades it by up to 6.6× in V<sub>min</sub> and overstiffens the bulk modulus
by 26 %.

The error is **asymmetric**, which the single bulk-modulus figure hides.
Figure 2 shows the fine-tuned curve tracking DFT closely under compression and
diverging badly under expansion — by 543 Å³ it sits at 2960 meV against DFT's
1640 meV, nearly double.

Rebalancing the loss recovers about a third of the V<sub>min</sub> error at a
cost of 4 meV/Å in forces, but does not close the gap. That the gap survives
equal weighting indicates the limitation is **training-data coverage**, not the
objective function: the training set spans ±4 % in lattice parameter while the
E–V curve spans ±12 % in volume, so the fine-tuned model extrapolates in a
region where the baseline retains broad database coverage.

### 4.4 Thermal expansion: neither metric predicts the answer

![thermal expansion](figures/fig4_expansion.png)

*Figure 4. Left: mean volume against temperature with linear fits. Right:
fitted α_L for the three models.*

| Model | ⟨V⟩ 300 K | 400 K | 500 K | α_L (10⁻⁶ K⁻¹) |
|---|---|---|---|---|
| MACE-MP-0 | 496.3 | 503.8 | 506.1 | 32.8 ± 10.0 |
| Force-weighted | 481.6 | 484.0 | 486.7 | 17.7 ± 0.3 |
| Balanced | 485.3 | 488.0 | 490.7 | 18.4 ± 0.3 |

The model with the best equation of state produces the least reliable α_L. Its
volumes are visibly non-linear in temperature — steps of 7.5 then 2.3 Å³ — and
its volume fluctuations are 60 % larger than the fine-tuned models', so 20 ps
was insufficient for the mean to converge. The resulting 30 % uncertainty means
the baseline's value is unresolved rather than simply wrong.

The two fine-tuned models agree within about 2σ despite differing in
V<sub>min</sub> by 0.6 percentage points, so the 17–18 range is robust to the
loss weighting rather than an artifact of one arbitrary choice.

**Neither force RMSE nor equation-of-state accuracy ranked the models correctly
for α_L.** Sampling quality dominated both.

### 4.5 Catastrophic forgetting

Single-head fine-tuning on a single composition shifted the predicted pure-Al
lattice constant from 4.0605 to 3.9826 Å — a −78 mÅ move, well beyond a 30 mÅ
tolerance. The model traded transferability for specialisation, pulling pure Al
toward the alloy's contracted lattice.

This does not affect the results above, since every production run is at the
training composition. It does mean the fine-tuned models must not be applied to
other Al–Cu compositions without retraining.

### 4.6 Ordering energetics: below resolution

DFT energies across eight configurations spanning α₁ = −0.333 to +0.111 vary by
under 30 meV/atom, with a fitted trend of +5.3 meV/atom per unit α₁. Two
configurations with *identical* α₁ = +0.111 differ by 29.6 meV/atom — scatter as
large as the trend itself.

At 32 atoms this question is not resolvable and no conclusion about ordering
preference is drawn. The lowest-energy configuration was the clustered one,
which is at least consistent with a composition that wants to precipitate θ
rather than remain a solid solution.

## 5. Two failure modes

Both were encountered, diagnosed and fixed during the work. Both are silent —
neither produces an error message — and both seem worth documenting.

### 5.1 Energy-reference mismatch destroys the model while the loss falls

Quantum ESPRESSO reports absolute total energies, about −1128 eV/atom for this
system. MACE-MP-0 uses a formation-energy-like reference. Fine-tuning with
`--E0s=foundation` therefore left a constant of ≈1124 eV/atom for the network
to fit, and that constant dominated the loss.

| Epoch | Energy RMSE (meV/atom) | Force RMSE (meV/Å) |
|---|---|---|
| 0 | 1,123,888 | 112 |
| 20 | 1,119,296 | 995 |
| 50 | 992,388 | 14,606 |
| 80 | 584,420 | 33,697 |
| 119 | 275,804 | 21,674 |

The energy error fell by 75 %. The force error rose by 190×. The total loss
decreased monotonically throughout, so nothing in the training log flagged the
problem; it surfaced only on evaluating the saved model against the held-out
set, where it scored 33,439 meV/Å against the baseline's 117.

Setting E₀ from the training data fixed it completely. Note that epoch 0 forces
(112 meV/Å) were slightly *better* than the untrained baseline — the model was
fine before training degraded it.

### 5.2 Multi-head fine-tuning is the default and is expensive

Recent MACE versions default to replaying the foundation dataset alongside user
data, specifically to prevent the forgetting documented in §4.5. On this
hardware that raised epoch time from 8 s to 90 min — a factor of ~700, which
would have made 120 epochs a five-day run.

Disabling it made training tractable and directly produced the forgetting in
§4.5. The two are not independent choices.

## 6. Discussion

The headline finding is negative and, I think, useful: **the metric the field
reports did not predict the property anyone cares about.**

Force RMSE ranked the fine-tuned models far above the baseline. Equation-of-state
accuracy ranked them far below it. Thermal expansion agreed with neither,
because what limited it was trajectory convergence — a property of the
*dynamics* the model produces, not of its pointwise accuracy on static
structures.

That mechanism is worth stating precisely. In NPT, forces move atoms within the
cell and stress moves the cell itself. Volume is set by the stress the model
predicts — the derivative of energy with respect to strain — not by forces, the
derivative with respect to position. These are different derivatives of the same
network and there is no reason improving one improves the other. Weighting the
loss 100:1 toward forces sharpened one and degraded the other, exactly as
instructed.

A secondary finding is that MACE-MP-0 was better than expected. Reproducing a
DFT equation of state to 0.28 % on V<sub>min</sub> and 3 GPa in bulk modulus,
for a structure it had never seen, is a real result. The "you must train your
own potential" orthodoxy deserves more scrutiny than it usually gets — though
Al and Cu are both well represented in the Materials Project, so this is close
to the model's home territory and the result should not be generalised.

## 7. Limitations

- **32 atoms.** Random configurations scatter by ±0.058 in α₁ at this size, and
  8 Cu atoms is coarse for chemical-order statistics.
- **20 ps trajectories.** Marginal for resolving volume differences of a few
  Å³, as the baseline's non-linear V(T) demonstrates. This is the largest single
  weakness.
- **45/360 Ry cutoffs**, below the SSSP recommendation of ≈55/440 Ry for Cu.
  No convergence study was performed; the relaxed lattice constant's agreement
  with Vegard's law is suggestive but not a substitute.
- **Classical nuclei.** No quantum correction; Al's Debye temperature (≈428 K)
  lies within the sampled range.
- **19 training structures, one composition.** Small-data fine-tuning, with the
  transferability cost quantified in §4.5.
- **No experimental reference.** A disordered fcc Al₇₅Cu₂₅ phase does not exist
  at equilibrium, so α_L cannot be validated against measurement.
- **N = 1.** One system, one composition. Whether the force-versus-EOS trade is
  general or specific to this case is untested.
- **No active learning.** Structures were chosen by hand rather than by model
  uncertainty.

## 8. Conclusions

1. MACE-MP-0 loses accuracy progressively with distance from relaxed,
   database-like structures — 58 meV/Å on volume-scaled cells rising to
   152 meV/Å on MD snapshots.
2. Fine-tuning on 19 structures improves forces 3.4× uniformly across
   structure types.
3. The same fine-tuning degrades the equation of state up to 6.6×,
   asymmetrically, in a way that rebalancing the loss only partly recovers —
   implicating training-data coverage rather than the objective.
4. Neither force RMSE nor equation-of-state accuracy predicted which model
   produced the most reliable thermal expansion coefficient.
5. Two silent failure modes in standard fine-tuning workflows are documented
   with before-and-after numbers.

## 9. Next steps

1. **Extend training to the volumes MD samples** (±10 %), which should test the
   coverage explanation for the asymmetric equation-of-state error directly.
2. **Longer trajectories** (100 ps+) and a larger cell, to establish whether the
   baseline's α_L converges toward the fine-tuned value or genuinely differs.
   This is the single most informative follow-up.
3. **Repeat on 3–4 systems inside their phase fields**, so experimental
   references exist and the N = 1 limitation is removed.
4. **Sweep loss weightings systematically** to map the force/EOS trade rather
   than sampling two points.
5. **On-the-fly active learning** (FLARE or DP-GEN) so structures are chosen by
   model uncertainty.
6. **Quasi-harmonic phonons** for a quantum-corrected α_L, avoiding the
   classical-nuclei and sampling limitations entirely.

## 10. Repository contents

    README.md              this report
    figures/               the four figures
    scripts/               stage01-08, fine-tuning scripts, input generators
    data/                  labelled structures, train/test split
    results/               JSON results from each stage
    dft/                   all QE inputs and outputs, provenance table
    pseudos/               PSLibrary PAW pseudopotentials

`data/data_labelled.xyz` is the expensive asset — 30 structures with DFT
energies, forces and stresses, 8 h 40 min of compute. Everything downstream
rebuilds from it, including training a different model architecture entirely.

### Reproducing

Stage timings on a 32-core workstation:

    stage01.py   gates, order parameter, vc-relax        ~5 h
    stage02.py   generate 30 structures                  ~2 min
    stage03.py   label with QE (4 workers x 8 ranks)     ~8.7 h
    stage04.py   split + baseline evaluation             ~2 min
    stage05.sh   fine-tune                               ~16 min
    stage06.py   parity, E-V, breakdown                  ~3 min
    stage07.py   NPT, 3 models x 3 temperatures          ~4.5 h
    stage08.py   fit alpha_L, final figure               ~1 min

Set `OMP_NUM_THREADS=1` for Quantum ESPRESSO, which is MPI-parallel, and 16 for
MACE, which is thread-parallel. Leaving it unset for QE let each of 64 ranks
spawn a full thread pool — roughly 4000 threads on 32 physical cores — and
produced a 40× slowdown that took an hour to notice.

## References

- Batatia et al., *MACE: Higher Order Equivariant Message Passing Neural
  Networks for Fast and Accurate Force Fields*, NeurIPS (2022)
- Batatia et al., *A foundation model for atomistic materials chemistry*,
  arXiv:2401.00096 (2023) — MACE-MP-0
- Giannozzi et al., *Advanced capabilities for materials modelling with Quantum
  ESPRESSO*, J. Phys. Condens. Matter **29**, 465901 (2017)
- Larsen et al., *The Atomic Simulation Environment*, J. Phys. Condens. Matter
  **29**, 273002 (2017)
- Prandini et al., *Precision and efficiency in solid-state pseudopotential
  calculations*, npj Comput. Mater. **4**, 72 (2018) — SSSP
- Cowley, *An Approximate Theory of Order in Alloys*, Phys. Rev. **77**, 669
  (1950)
