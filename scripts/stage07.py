import json, time, sys
import numpy as np
from ase.io import read
from ase.units import fs, GPa
from mace.calculators import mace_mp, MACECalculator
try:
    from ase.md.melchionna import MelchionnaNPT as NPT
except ImportError:
    from ase.md.npt import NPT
try:
    from ase.md.velocitydistribution import thermalize_momenta as initv
except ImportError:
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution as initv
from ase.md.velocitydistribution import Stationary

TEMPS    = [300, 400, 500]
EQUIL_PS = 5
PROD_PS  = 20
DT       = 2.0
B_GUESS  = 96.4

MODELS = {
    "baseline": lambda: mace_mp(model="medium", default_dtype="float64", device="cpu"),
    "ft2":      lambda: MACECalculator(model_paths="AlCu_ft2.model", device="cpu",
                                       default_dtype="float64"),
    "ft3":      lambda: MACECalculator(model_paths="AlCu_ft3.model", device="cpu",
                                       default_dtype="float64"),
}

def run(calc, T, tag):
    at = read("data_ref_relaxed.xyz"); at.calc = calc
    C = at.cell.array
    if abs(C[1,0])+abs(C[2,0])+abs(C[2,1]) > 1e-8:
        at.set_cell(at.cell.standard_form()[0], scale_atoms=True)
    initv(at, temperature_K=T); Stationary(at)
    dyn = NPT(at, timestep=DT*fs, temperature_K=T, externalstress=0.0,
              ttime=25*fs, pfactor=(75*fs)**2 * B_GUESS*GPa,
              mask=np.eye(3, dtype=bool))
    t0 = time.time()
    dyn.run(int(EQUIL_PS*1000/DT))
    V, Tk = [], []
    dyn.attach(lambda: (V.append(at.get_volume()),
                        Tk.append(at.get_temperature())), interval=10)
    dyn.run(int(PROD_PS*1000/DT))
    V, Tk = np.array(V), np.array(Tk)
    blocks = np.array_split(V, 5)
    sem = np.std([b.mean() for b in blocks], ddof=1)/np.sqrt(5)
    print(f"  {tag:9s} {T} K   <V>={V.mean():7.2f} +/- {sem:.2f}   "
          f"sd={V.std():5.2f}   <T>={Tk.mean():5.1f} K   ({time.time()-t0:.0f} s)")
    np.save(f"md_{tag}_{T}.npy", V)
    return float(V.mean()), float(sem)

if __name__ == "__main__":
    which = sys.argv[1:] or list(MODELS)
    out = {}
    for tag in which:
        print(f"\n=== {tag} ===")
        calc = MODELS[tag]()
        out[tag] = {T: run(calc, T, tag) for T in TEMPS}
    json.dump({k: {str(t): list(v) for t, v in d.items()} for k, d in out.items()},
              open("res_md.json", "w"), indent=1)
    print("\nwrote res_md.json")
