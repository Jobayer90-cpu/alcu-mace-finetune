#!/usr/bin/env python3
"""
Collect the raw Quantum ESPRESSO outputs into dft/ alongside the inputs,
with matching names, and extract a provenance table.

Writes:
  dft/relax/alcu.vc-relax.out
  dft/scf/s000.scf.out ... s029.scf.out
  dft/summary.csv          one row per calculation
"""
import csv, os, re, shutil
from pathlib import Path

RE = {
    "energy":  re.compile(r"^!\s+total energy\s+=\s+(-?[\d.]+)\s+Ry", re.M),
    "force":   re.compile(r"Total force =\s+([\d.]+)", re.M),
    "press":   re.compile(r"P=\s*(-?[\d.]+)", re.M),
    "niter":   re.compile(r"convergence has been achieved in\s+(\d+) iterations", re.M),
    "wall":    re.compile(r"PWSCF\s+:.*?([\dhms. ]+)WALL", re.M),
    "nk":      re.compile(r"number of k points=\s*(\d+)", re.M),
    "ecutwfc": re.compile(r"kinetic-energy cutoff\s+=\s+([\d.]+)", re.M),
    "ecutrho": re.compile(r"charge density cutoff\s+=\s+([\d.]+)", re.M),
    "nat":     re.compile(r"number of atoms/cell\s+=\s+(\d+)", re.M),
    "done":    re.compile(r"JOB DONE", re.M),
}
RY, RY_BOHR = 13.605693122994, 25.71104309541616


def scrape(text):
    g = lambda k: (RE[k].findall(text) or [None])[-1]
    d = {"energy_Ry":   float(g("energy")) if g("energy") else None,
         "force_Ry_au": float(g("force")) if g("force") else None,
         "press_kbar":  float(g("press")) if g("press") else None,
         "scf_iters":   int(g("niter")) if g("niter") else None,
         "walltime":    (g("wall") or "").strip() or None,
         "nkpts":       int(g("nk")) if g("nk") else None,
         "ecutwfc_Ry":  float(g("ecutwfc")) if g("ecutwfc") else None,
         "ecutrho_Ry":  float(g("ecutrho")) if g("ecutrho") else None,
         "natoms":      int(g("nat")) if g("nat") else None,
         "job_done":    bool(RE["done"].search(text))}
    d["n_scf_cycles"] = len(RE["energy"].findall(text))
    if d["energy_Ry"] is not None:
        d["energy_eV"] = d["energy_Ry"] * RY
    if d["force_Ry_au"] is not None:
        d["total_force_eV_A"] = d["force_Ry_au"] * RY_BOHR
    return d


def main():
    out = Path("dft")
    (out / "relax").mkdir(parents=True, exist_ok=True)
    (out / "scf").mkdir(parents=True, exist_ok=True)
    rows = []

    meta = {}
    if os.path.exists("data_labelled.xyz"):
        try:
            import numpy as np
            from ase.io import read
            for i, a in enumerate(read("data_labelled.xyz", index=":")):
                meta[f"s{i:03d}"] = {
                    "struct_type": a.info.get("kind"),
                    "warren_cowley": round(float(a.info.get("alpha", float("nan"))), 4),
                    "volume_A3": round(a.get_volume(), 3),
                    "fmax_eV_A": round(float(np.abs(a.get_forces()).max()), 4)}
            print(f"  (metadata for {len(meta)} structures from data_labelled.xyz)")
        except Exception as exc:
            print(f"  ! could not read data_labelled.xyz: {exc}")

    src = Path("qe_relax/espresso.pwo")
    if src.exists():
        shutil.copy(src, out / "relax" / "alcu.vc-relax.out")
        r = scrape(src.read_text()); r["id"] = "vc-relax"; r["kind"] = "relax"
        rows.append(r)
        print(f"  relax/alcu.vc-relax.out   {src.stat().st_size//1024} KB, "
              f"{r['n_scf_cycles']} ionic steps, done={r['job_done']}")
    else:
        print("  ! qe_relax/espresso.pwo not found")

    n_ok = n_bad = 0
    dirs = sorted(Path("qe_lab").glob("s[0-9][0-9][0-9]")) if Path("qe_lab").is_dir() else []
    for d in dirs:
        f = d / "espresso.pwo"
        if not f.exists():
            print(f"  ! {d.name}: no espresso.pwo"); n_bad += 1; continue
        shutil.copy(f, out / "scf" / f"{d.name}.scf.out")
        r = scrape(f.read_text()); r["id"] = d.name; r["kind"] = "scf"
        r.update(meta.get(d.name, {}))
        rows.append(r); n_ok += 1
        if not r["job_done"] or r["energy_Ry"] is None:
            print(f"  ! {d.name}: incomplete (done={r['job_done']}, E={r['energy_Ry']})")
    print(f"  scf/*.scf.out             {n_ok} files copied"
          + (f", {n_bad} missing" if n_bad else ""))

    if not rows:
        print("  ! nothing collected"); return

    cols = ["id", "kind", "struct_type", "warren_cowley", "volume_A3",
            "natoms", "energy_Ry", "energy_eV", "fmax_eV_A", "total_force_eV_A",
            "press_kbar", "scf_iters", "n_scf_cycles", "nkpts", "ecutwfc_Ry",
            "ecutrho_Ry", "walltime", "job_done"]
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  summary.csv               {len(rows)} rows")

    e = [r["energy_eV"] for r in rows if r["kind"] == "scf" and r.get("energy_eV")]
    f = [r.get("fmax_eV_A") for r in rows if r["kind"] == "scf" and r.get("fmax_eV_A")]
    if e:
        print(f"\n  energy spread  {max(e)-min(e):.3f} eV over {len(e)} structures")
    if f:
        print(f"  |F|max range   {min(f):.3f} .. {max(f):.3f} eV/A")
    it = [r["scf_iters"] for r in rows if r["kind"] == "scf" and r["scf_iters"]]
    if it:
        print(f"  SCF iterations {min(it)} .. {max(it)}  (mean {sum(it)/len(it):.1f})")


if __name__ == "__main__":
    main()
