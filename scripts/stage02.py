import json
import numpy as np
from ase.io import read, write
from stage01 import base_cell, make_random, warren_cowley, N_CU, A_AL

base = read("data_ref_relaxed.xyz")
A_EQ = (base.get_volume()/8)**(1/3)
print(f"A_EQ = {A_EQ:.4f} A")

seeds = np.arange(300)
alphas = np.array([warren_cowley(make_random(s)) for s in seeds])

def make_l12():
    at = base_cell()
    frac = (at.get_scaled_positions()*2.0) % 1.0
    corner = np.all(np.isclose(frac,0,atol=1e-6)|np.isclose(frac,1,atol=1e-6), axis=1)
    s = np.array(at.get_chemical_symbols(), dtype=object); s[corner] = "Cu"
    at.set_chemical_symbols(list(s)); return at

def make_clustered():
    at = base_cell(); p = at.positions
    o = np.lexsort((p[:,2], p[:,1], p[:,0]))
    s = np.array(at.get_chemical_symbols(), dtype=object); s[o[:N_CU]] = "Cu"
    at.set_chemical_symbols(list(s)); return at

structs, meta = [], []
def add(at, kind):
    at = at.copy(); at.calc = None
    structs.append(at)
    meta.append({"kind":kind, "alpha":float(warren_cowley(at, a=A_EQ)),
                 "volume":float(at.get_volume())})

for s in np.linspace(0.96, 1.04, 9):
    at = base.copy(); at.set_cell(at.cell*s, scale_atoms=True); add(at, "volume")

for k, sd in enumerate(np.linspace(0.05, 0.15, 8)):
    at = base.copy(); at.rattle(stdev=sd, seed=100+k); add(at, "rattle")

scale = A_EQ / A_AL
oset = [make_l12(), make_clustered()]
for t in [-0.20,-0.10,0.0,0.05,0.10,-0.15]:
    oset.append(make_random(int(seeds[np.argmin(np.abs(alphas-t))])))
for at in oset:
    at = at.copy(); at.set_cell(at.cell*scale, scale_atoms=True)
    at.rattle(stdev=0.05, seed=7); add(at, "order")

from mace.calculators import mace_mp
from ase.md.langevin import Langevin
from ase.units import fs
try:
    from ase.md.velocitydistribution import thermalize_momenta as initv
except ImportError:
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution as initv

mace0 = mace_mp(model="medium", default_dtype="float64", device="cpu")
at = base.copy(); at.calc = mace0
initv(at, temperature_K=500)
dyn = Langevin(at, timestep=2*fs, temperature_K=500, friction=0.01/fs)
for _ in range(5):
    dyn.run(200); add(at, "md")

print(f"\n{len(structs)} structures")
for k in ["volume","rattle","order","md"]:
    print(f"  {k:7s} {sum(m['kind']==k for m in meta)}")
print(f"alpha range: {min(m['alpha'] for m in meta):+.3f} .. {max(m['alpha'] for m in meta):+.3f}")
write("data_unlabelled.xyz", structs)
json.dump(meta, open("data_meta.json","w"), indent=1)
print("wrote data_unlabelled.xyz, data_meta.json")
