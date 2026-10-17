# DFT reference calculations

All Quantum ESPRESSO inputs and outputs for this study. Regenerate the inputs
with `python make_qe_inputs.py`; collect the outputs with
`python collect_qe_outputs.py`.

## Settings

| | |
|---|---|
| Code | Quantum ESPRESSO 6.7 |
| Functional | PBE |
| Pseudopotentials | PSLibrary 1.0.0 PAW (`Al.pbe-n-kjpaw`, `Cu.pbe-dn-kjpaw`) |
| Wavefunction cutoff | 45 Ry |
| Charge-density cutoff | 360 Ry |
| k-mesh | 3x3x3 Monkhorst-Pack, unshifted |
| Smearing | Marzari-Vanderbilt, 0.02 Ry |
| `conv_thr` | 1e-6 Ry |
| `mixing_beta` | 0.4 |

The cutoffs are below the SSSP recommendation of ~55/440 Ry for Cu. This was a
deliberate speed compromise and no convergence study was performed; see the
limitations section of the main write-up.

## Contents

    relax/alcu.vc-relax.in/.out    Stage 1: variable-cell relaxation
    scf/s000..s029.scf.in/.out     Stage 3: 30 single-point labels
    summary.csv                    one row per calculation
    pseudos.sha256                 checksums for ../pseudos/*.UPF
    get_pseudos.sh                 re-download the pseudopotentials

Each input carries a comment header naming its structure type, Warren-Cowley
alpha_1, cell volume, and the exact mpirun line used.

## Stage 1: variable-cell relaxation

32-atom Al75Cu25, `calculation = 'vc-relax'`, `forc_conv_thr = 1e-3` Ry/bohr,
`press_conv_thr = 0.5` kbar.

Converged in 26 ionic steps (~340 SCF iterations, 4 h 50 min on 16 MPI ranks) to:

- a_eq = 3.9217 A
- V = 482.53 A^3
- final total force 0.0024 Ry/bohr
- final pressure -0.47 kbar

## Stage 3: single-point labels

30 structures spanning what production MD samples: 9 volume-scaled (+-4% in
lattice parameter), 8 rattled (0.05-0.15 A displacements), 8 with differing
chemical order (alpha_1 from -0.333 to +0.111), 5 snapshots from 500 K MD.

Energy, forces and stress from each run are the training labels, stored in
`../data_labelled.xyz`.

Run with 4 concurrent jobs x 8 MPI ranks, 8 h 40 min total. Cost varied by more
than 10x across structures: the easiest volume-scaled cell took 480 s, the
thermally distorted MD snapshots up to 4360 s.

## Reproducing

    bash get_pseudos.sh
    cd relax && mpirun -np 16 pw.x -in alcu.vc-relax.in > alcu.vc-relax.out
    cd ../scf
    for f in s*.scf.in; do
        mpirun -np 8 pw.x -in $f > ${f%.in}.out
    done

Set `OMP_NUM_THREADS=1`. QE is MPI-parallel, and leaving the variable unset lets
each rank spawn a full thread pool; on a 32-core machine this caused a 40x
slowdown from oversubscription.

## summary.csv columns

`struct_type`, `warren_cowley`, `volume_A3` and `fmax_eV_A` come from
`../data_labelled.xyz`; the rest are scraped from the QE outputs.
`total_force_eV_A` is QE's total force norm over all atoms, not the maximum
per-atom component -- use `fmax_eV_A` for that.
