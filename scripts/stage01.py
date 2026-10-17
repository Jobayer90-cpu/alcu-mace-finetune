import os, time
import numpy as np
from ase.build import bulk
from ase.io import read, write
from ase.neighborlist import neighbor_list
from ase.calculators.espresso import Espresso, EspressoProfile

A_AL, N_CU = 4.05, 8
NCORES = 16
PSEUDO_DIR = os.path.abspath("pseudos")
ECUTWFC, ECUTRHO, KPTS, DEGAUSS = 45, 360, (3,3,3), 0.02

profile = EspressoProfile(command=f"mpirun -np {NCORES} pw.x", pseudo_dir=PSEUDO_DIR)
PSEUDOS = {"Al":"Al.pbe-n-kjpaw_psl.1.0.0.UPF", "Cu":"Cu.pbe-dn-kjpaw_psl.1.0.0.UPF"}

def qe(directory, calc="scf", kpts=KPTS, extra=None):
    data = {"control":{"calculation":calc,"tprnfor":True,"tstress":True,"disk_io":"none"},
            "system":{"ecutwfc":ECUTWFC,"ecutrho":ECUTRHO,"occupations":"smearing",
                      "smearing":"mv","degauss":DEGAUSS},
            "electrons":{"conv_thr":1e-6,"mixing_beta":0.4}}
    if extra: data.update(extra)
    return Espresso(profile=profile, directory=directory,
                    pseudopotentials=PSEUDOS, kpts=kpts, input_data=data)

def base_cell(): return bulk("Al","fcc",a=A_AL,cubic=True)*(2,2,2)

def make_random(seed, n_cu=N_CU):
    at = base_cell()
    idx = np.random.default_rng(seed).choice(len(at), size=n_cu, replace=False)
    s = np.array(at.get_chemical_symbols(), dtype=object); s[idx] = "Cu"
    at.set_chemical_symbols(list(s)); return at

def warren_cowley(atoms, a=A_AL, host="Al", solute="Cu"):
    i, j = neighbor_list("ij", atoms, 0.854*a)
    s = np.array(atoms.get_chemical_symbols())
    c = np.count_nonzero(s==solute)/len(s); fh = s[i]==host
    return 1.0 - (np.count_nonzero(fh & (s[j]==solute))/np.count_nonzero(fh))/c

if __name__ == "__main__":
    print("pseudo_dir:", PSEUDO_DIR, "exists:", os.path.isdir(PSEUDO_DIR))

    print("\n--- GATE 0: single Al atom ---")
    t = bulk("Al","fcc",a=A_AL,cubic=True); t.calc = qe("qe_gate0", kpts=(4,4,4))
    t0 = time.time(); e = t.get_potential_energy()
    print(f"E = {e:.4f} eV   ({time.time()-t0:.0f} s)   GATE 0 PASSED")

    print("\n--- GATE 1a: order parameter ---")
    print(f"random(0) alpha = {warren_cowley(make_random(0)):+.3f}")
    seeds = np.arange(300)
    al = np.array([warren_cowley(make_random(s)) for s in seeds])
    SEED = int(seeds[np.argmin(np.abs(al))])
    print(f"mean {al.mean():+.3f} +/- {al.std():.3f}   best seed {SEED}")

    ref = make_random(SEED)
    write("data_ref_unrelaxed.xyz", ref)
    print(f"reference: {ref.get_chemical_formula()}  alpha={warren_cowley(ref):+.4f}")

    print("\n--- GATE 1b: vc-relax (this takes a while) ---")
    r = ref.copy()
    r.calc = qe("qe_relax", calc="vc-relax",
                extra={"cell":{"press_conv_thr":0.5}})
    t0 = time.time(); r.get_potential_energy()
    rel = read("qe_relax/espresso.pwo", index=-1)
    A_EQ = (rel.get_volume()/8)**(1/3)
    print(f"a_eq = {A_EQ:.4f} A   V = {rel.get_volume():.2f} A^3   ({time.time()-t0:.0f} s)")
    write("data_ref_relaxed.xyz", rel)
