import json
import numpy as np
from ase.io import read, write

lab = read("data_labelled.xyz", index=":")
print(f"{len(lab)} labelled structures")

fmax = [np.abs(a.get_forces()).max() for a in lab]
ener = [a.get_potential_energy() for a in lab]
print(f"|F|max range : {min(fmax):.3f} .. {max(fmax):.3f} eV/A")
print(f"energy range : {max(ener)-min(ener):.3f} eV spread")
kinds = np.array([a.info["kind"] for a in lab])
for k in np.unique(kinds):
    print(f"  {k:7s} {np.sum(kinds==k)}")

rng = np.random.default_rng(42)
test_idx = []
for k in np.unique(kinds):
    idx = np.where(kinds == k)[0]
    n = max(1, round(len(idx) * 10 / len(lab)))
    test_idx += list(rng.choice(idx, size=n, replace=False))
test_idx = sorted(test_idx)
train_idx = [i for i in range(len(lab)) if i not in test_idx]

write("data_train.xyz", [lab[i] for i in train_idx])
write("data_test.xyz",  [lab[i] for i in test_idx])
print(f"\ntrain {len(train_idx)} | test {len(test_idx)}")
print("test composition:", {k:int(np.sum(kinds[test_idx]==k)) for k in np.unique(kinds)})

from mace.calculators import mace_mp
mace0 = mace_mp(model="medium", default_dtype="float64", device="cpu")

def evaluate(calc, frames, label):
    Fp, Fr, Ep, Er, rows = [], [], [], [], []
    for a in frames:
        fr, er = a.get_forces(), a.get_potential_energy()
        b = a.copy(); b.calc = calc
        fp = b.get_forces()
        Fp.append(fp); Fr.append(fr)
        Ep.append(b.get_potential_energy()); Er.append(er)
        rows.append({"kind":a.info["kind"], "alpha":a.info["alpha"],
                     "f_rmse":float(np.sqrt(np.mean((fp-fr)**2)))})
    Fp, Fr = np.concatenate(Fp).ravel(), np.concatenate(Fr).ravel()
    Ep, Er = np.array(Ep), np.array(Er)
    dE = (Ep-Ep.mean()) - (Er-Er.mean())
    out = {"label":label,
           "f_rmse":float(np.sqrt(np.mean((Fp-Fr)**2))),
           "f_mae":float(np.mean(np.abs(Fp-Fr))),
           "e_rmse_per_atom":float(np.sqrt(np.mean(dE**2))/len(frames[0])),
           "rows":rows}
    print(f"\n{label}")
    print(f"  force RMSE  {out['f_rmse']*1000:7.1f} meV/A")
    print(f"  force MAE   {out['f_mae']*1000:7.1f} meV/A")
    print(f"  energy RMSE {out['e_rmse_per_atom']*1000:7.2f} meV/atom")
    np.save(f"pred_{label}.npy", {"F_pred":Fp, "F_ref":Fr})
    return out

test = read("data_test.xyz", index=":")
base = evaluate(mace0, test, "baseline")
json.dump(base, open("res_baseline.json","w"), indent=1)

print("\nby structure type:")
for k in sorted({r["kind"] for r in base["rows"]}):
    v = [r["f_rmse"] for r in base["rows"] if r["kind"]==k]
    print(f"  {k:7s} {1000*np.mean(v):7.1f} meV/A   (n={len(v)})")
