#!/usr/bin/env python3
"""
Regenerate every Quantum ESPRESSO input used in this study.

Writes:
  dft/relax/alcu.vc-relax.in      variable-cell relaxation (Stage 1)
  dft/scf/s000.scf.in ... s029    the 30 single-point labels (Stage 3)

Reads data_labelled.xyz and data_ref_relaxed.xyz from the working directory.
"""
import io, os
from pathlib import Path
from ase.io import read
from ase.io.espresso import write_espresso_in

# ----- settings, exactly as used in the study -------------------------------
ECUTWFC, ECUTRHO = 45, 360          # Ry  (SSSP recommends ~55/440 for Cu)
KPTS             = (3, 3, 3)        # Monkhorst-Pack, unshifted
DEGAUSS          = 0.02             # Ry, Marzari-Vanderbilt cold smearing
CONV_THR         = 1e-6             # Ry, electronic
MIXING_BETA      = 0.4
FORC_CONV_THR    = 1e-3             # Ry/bohr, ionic   (vc-relax only)
PRESS_CONV_THR   = 0.5              # kbar, cell       (vc-relax only)

PSEUDOS = {"Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF",
           "Cu": "Cu.pbe-dn-kjpaw_psl.1.0.0.UPF"}

# QE 6.7 does not know &FCP or &RISM; ASE emits them empty. Strip them.
DROP = ("&IONS\n/\n", "&CELL\n/\n", "&FCP\n/\n", "&RISM\n/\n")


def namelists(calc, prefix, pseudo_dir):
    d = {
        "control": {"calculation": calc, "prefix": prefix, "outdir": "./tmp",
                    "pseudo_dir": pseudo_dir, "tprnfor": True, "tstress": True,
                    "disk_io": "none"},
        "system": {"ecutwfc": ECUTWFC, "ecutrho": ECUTRHO,
                   "occupations": "smearing", "smearing": "mv",
                   "degauss": DEGAUSS},
        "electrons": {"conv_thr": CONV_THR, "mixing_beta": MIXING_BETA},
    }
    if calc == "vc-relax":
        d["control"]["forc_conv_thr"] = FORC_CONV_THR
        d["control"]["etot_conv_thr"] = 1e-4
        d["ions"] = {"ion_dynamics": "bfgs"}
        d["cell"] = {"cell_dynamics": "bfgs", "press_conv_thr": PRESS_CONV_THR}
    return d


def write_one(atoms, path, calc, prefix, pseudo_dir, header):
    buf = io.StringIO()
    write_espresso_in(buf, atoms, input_data=namelists(calc, prefix, pseudo_dir),
                      pseudopotentials=PSEUDOS, kpts=KPTS)
    text = buf.getvalue()
    for d in DROP:
        if d in text and not (calc == "vc-relax" and d in ("&IONS\n/\n", "&CELL\n/\n")):
            text = text.replace(d, "")
    text = "\n".join(f"! {l}" for l in header.strip().split("\n")) + "\n\n" + text
    Path(path).write_text(text)
    return len(text.split("\n"))


def main():
    out = Path("dft")
    (out / "relax").mkdir(parents=True, exist_ok=True)
    (out / "scf").mkdir(parents=True, exist_ok=True)

    src = next((c for c in ("data_ref_relaxed.xyz", "data_ref_unrelaxed.xyz")
                if os.path.exists(c)), None)
    if src:
        n = write_one(read(src), out / "relax" / "alcu.vc-relax.in", "vc-relax",
                      "alcu", "../../pseudos",
                      f"Stage 1: variable-cell relaxation of Al75Cu25 (32 atoms).\n"
                      f"Geometry from {src}. Converged in 26 ionic steps to\n"
                      f"a_eq = 3.9217 A, V = 482.53 A^3, P = -0.47 kbar.\n"
                      f"Run: mpirun -np 16 pw.x -in alcu.vc-relax.in > alcu.vc-relax.out")
        print(f"  relax/alcu.vc-relax.in    ({n} lines, from {src})")
    else:
        print("  ! no reference structure found - skipped vc-relax input")

    src = next((c for c in ("data_labelled.xyz", "data_unlabelled.xyz")
                if os.path.exists(c)), None)
    if not src:
        print("  ! no structure set found - skipped scf inputs"); return

    frames = read(src, index=":")
    kinds = []
    for i, at in enumerate(frames):
        kind = at.info.get("kind", "?")
        alpha = at.info.get("alpha", float("nan"))
        kinds.append(kind)
        write_one(at, out / "scf" / f"s{i:03d}.scf.in", "scf", f"s{i:03d}",
                  "../../pseudos",
                  f"Stage 3, structure {i:03d}  |  type: {kind}  |  "
                  f"Warren-Cowley alpha_1 = {alpha:+.4f}  |  V = {at.get_volume():.2f} A^3\n"
                  f"Single-point SCF; energy, forces and stress are the training label.\n"
                  f"Run: mpirun -np 8 pw.x -in s{i:03d}.scf.in > s{i:03d}.scf.out")
    print(f"  scf/s000..s{len(frames)-1:03d}.scf.in   ({len(frames)} files, from {src})")
    for k in sorted(set(kinds)):
        print(f"      {k:8s} {kinds.count(k)}")


if __name__ == "__main__":
    main()
