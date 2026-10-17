import os, sys, json, time
import numpy as np
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator
from concurrent.futures import ProcessPoolExecutor
from stage01 import PSEUDOS, PSEUDO_DIR, ECUTWFC, ECUTRHO, KPTS, DEGAUSS

NWORKERS   = int(sys.argv[1]) if len(sys.argv) > 1 else 4
CORES_EACH = int(sys.argv[2]) if len(sys.argv) > 2 else 16

def label_one(n):
    from ase.calculators.espresso import Espresso, EspressoProfile
    at = read("data_unlabelled.xyz", index=n)
    prof = EspressoProfile(command=f"mpirun -np {CORES_EACH} pw.x",
                           pseudo_dir=PSEUDO_DIR)
    at.calc = Espresso(profile=prof, directory=f"qe_lab/s{n:03d}",
        pseudopotentials=PSEUDOS, kpts=KPTS,
        input_data={"control":{"calculation":"scf","tprnfor":True,
                               "tstress":True,"disk_io":"none"},
                    "system":{"ecutwfc":ECUTWFC,"ecutrho":ECUTRHO,
                              "occupations":"smearing","smearing":"mv",
                              "degauss":DEGAUSS},
                    "electrons":{"conv_thr":1e-6,"mixing_beta":0.4}})
    t0 = time.time()
    e, f, s = at.get_potential_energy(), at.get_forces(), at.get_stress()
    dt = time.time() - t0
    out = read("data_unlabelled.xyz", index=n)
    out.calc = SinglePointCalculator(out, energy=float(e), forces=f, stress=s)
    write(f"qe_lab/labelled_{n:03d}.xyz", out)
    return n, float(e), float(np.abs(f).max()), dt

if __name__ == "__main__":
    os.makedirs("qe_lab", exist_ok=True)
    meta = json.load(open("data_meta.json"))
    todo = [n for n in range(len(meta))
            if not os.path.exists(f"qe_lab/labelled_{n:03d}.xyz")]
    print(f"{len(todo)} to label | {NWORKERS} workers x {CORES_EACH} cores")

    if todo:                                   # probe: time structure 0 alone
        t0 = time.time()
        n, e, fm, dt = label_one(todo[0])
        print(f"  [{n:02d}] PROBE  E={e:.4f}  |F|max={fm:.3f}  {dt:.0f} s")
        rest = todo[1:]
        est = dt * len(rest) / NWORKERS / 60
        print(f"  -> estimate for remaining {len(rest)}: {est:.0f} min\n")

        t0 = time.time()
        with ProcessPoolExecutor(max_workers=NWORKERS) as ex:
            for n, e, fm, dt in ex.map(label_one, rest):
                print(f"  [{n:02d}] E={e:.4f}  |F|max={fm:.3f}  {dt:.0f} s"
                      f"   [{(time.time()-t0)/60:.0f} min elapsed]")

    done = sorted(f for f in os.listdir("qe_lab") if f.startswith("labelled_"))
    lab = [read(f"qe_lab/{f}") for f in done]
    for a, m in zip(lab, [meta[int(f[9:12])] for f in done]):
        a.info.update({"kind":m["kind"], "alpha":m["alpha"]})
    write("data_labelled.xyz", lab)
    print(f"\n{len(lab)}/{len(meta)} labelled -> data_labelled.xyz")
