import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io import read
from mace.calculators import mace_mp, MACECalculator

lab  = read("data_labelled.xyz", index=":")
test = read("data_test.xyz", index=":")
test_ids = {(a.info["kind"], round(a.get_volume(), 4)) for a in test}

base = mace_mp(model="medium", default_dtype="float64", device="cpu")
ft   = MACECalculator(model_paths="AlCu_ft2.model", device="cpu", default_dtype="float64")

def forces_on(calc, frames):
    P, R, per = [], [], []
    for a in frames:
        r = a.get_forces(); b = a.copy(); b.calc = calc
        p = b.get_forces(); P.append(p); R.append(r)
        per.append(float(np.sqrt(np.mean((p - r) ** 2))))
    return np.concatenate(P).ravel(), np.concatenate(R).ravel(), per

print("=== FIGURE 1: force parity ===")
Pb, Rb, perb = forces_on(base, test)
Pf, Rf, perf = forces_on(ft,   test)
rb = np.sqrt(np.mean((Pb - Rb) ** 2)); rf = np.sqrt(np.mean((Pf - Rf) ** 2))
print(f"  baseline {rb*1000:.1f}  finetuned {rf*1000:.1f} meV/A")

fig, ax = plt.subplots(figsize=(5.2, 5))
lim = 1.05 * np.abs(Rb).max()
ax.plot([-lim, lim], [-lim, lim], "k-", lw=.8, zorder=0)
ax.scatter(Rb, Pb, s=10, alpha=.4, c="tab:orange", marker="o",
           label=f"MACE-MP-0  ({rb*1000:.0f} meV/$\\AA$)")
ax.scatter(Rf, Pf, s=10, alpha=.5, c="tab:blue", marker="^",
           label=f"fine-tuned  ({rf*1000:.0f} meV/$\\AA$)")
ax.set_xlabel("DFT force (eV/$\\AA$)"); ax.set_ylabel("model force (eV/$\\AA$)")
ax.set_title("Held-out test set (11 structures)"); ax.legend(fontsize=8)
ax.set_aspect("equal"); fig.tight_layout()
fig.savefig("fig1_parity.png", dpi=180); plt.close()

print("\n=== FIGURE 2: equation of state ===")
vol = sorted([a for a in lab if a.info["kind"] == "volume"], key=lambda a: a.get_volume())
V = np.array([a.get_volume() for a in vol])
E_dft = np.array([a.get_potential_energy() for a in vol])
def curve(calc):
    out = []
    for a in vol:
        b = a.copy(); b.calc = calc; out.append(b.get_potential_energy())
    return np.array(out)
E_b, E_f = curve(base), curve(ft)

def para(V, E):
    c = np.polyfit(V, E, 2)
    v0 = -c[1] / (2 * c[0])
    B = v0 * 2 * c[0] * 160.2177
    return v0, B, c

fig, ax = plt.subplots(figsize=(5.8, 4.3))
rows = []
for E, lb, st, col in [(E_dft, "DFT (QE)", "o-", "k"),
                       (E_b, "MACE-MP-0", "o--", "tab:orange"),
                       (E_f, "fine-tuned", "^-", "tab:blue")]:
    v0, B, _ = para(V, E)
    rows.append((lb, v0, B))
    ax.plot(V, (E - E.min()) * 1000, st, color=col, ms=5, label=lb)
ax.set_xlabel("volume ($\\AA^3$)"); ax.set_ylabel("$E-E_{min}$ (meV)")
ax.set_title("Equation of state, Al$_{75}$Cu$_{25}$"); ax.legend()
fig.tight_layout(); fig.savefig("fig2_ev.png", dpi=180); plt.close()

print(f"  {'':12s} {'V_min':>9s} {'B':>8s}")
for lb, v0, B in rows:
    print(f"  {lb:12s} {v0:9.2f} {B:8.1f} GPa")
print(f"  V_min error: baseline {100*(rows[1][1]-rows[0][1])/rows[0][1]:+.2f}%"
      f"   fine-tuned {100*(rows[2][1]-rows[0][1])/rows[0][1]:+.2f}%")

print("\n=== FIGURE 3: breakdown ===")
kinds = [a.info["kind"] for a in test]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
ks = sorted(set(kinds)); x = np.arange(len(ks)); w = .36
for off, per, c, lb in [(-w/2, perb, "tab:orange", "MACE-MP-0"),
                        (w/2,  perf, "tab:blue",   "fine-tuned")]:
    v = [1000*np.mean([p for p, k in zip(per, kinds) if k == kk]) for kk in ks]
    a1.bar(x + off, v, w, color=c, label=lb)
a1.set_xticks(x); a1.set_xticklabels(ks); a1.set_ylabel("force RMSE (meV/$\\AA$)")
a1.set_title("Error by structure type"); a1.legend(fontsize=8)

order = [a for a in lab if a.info["kind"] == "order"]
al = np.array([a.info["alpha"] for a in order])
_, _, pob = forces_on(base, order); _, _, pof = forces_on(ft, order)
intest = np.array([(a.info["kind"], round(a.get_volume(),4)) in test_ids for a in order])
for per, c, mk, lb in [(pob,"tab:orange","o","MACE-MP-0"), (pof,"tab:blue","^","fine-tuned")]:
    per = np.array(per)*1000
    a2.scatter(al[~intest], per[~intest], c=c, marker=mk, s=45, alpha=.45,
               edgecolors="none", label=f"{lb} (train)")
    a2.scatter(al[intest], per[intest], c=c, marker=mk, s=80,
               edgecolors="k", linewidths=1.2, label=f"{lb} (held out)")
a2.axvline(0, color="k", lw=.6, ls=":")
a2.set_xlabel("Warren-Cowley $\\alpha_1$   ($<0$ ordered, $>0$ clustered)")
a2.set_ylabel("force RMSE (meV/$\\AA$)"); a2.set_title("Error vs chemical order")
a2.legend(fontsize=7)
fig.tight_layout(); fig.savefig("fig3_breakdown.png", dpi=180); plt.close()
for k in ks:
    vb = 1000*np.mean([p for p,kk in zip(perb,kinds) if kk==k])
    vf = 1000*np.mean([p for p,kk in zip(perf,kinds) if kk==k])
    print(f"  {k:8s} {vb:7.1f} -> {vf:6.1f} meV/A")

print("\n=== DFT ordering energetics ===")
n = len(order[0])
eo = np.array([a.get_potential_energy()/n for a in order])
print(f"  {'alpha':>7s} {'E/atom (eV)':>14s} {'rel (meV)':>10s}")
for i in np.argsort(al):
    print(f"  {al[i]:+7.3f} {eo[i]:14.5f} {1000*(eo[i]-eo.min()):10.1f}")
sl = np.polyfit(al, eo*1000, 1)[0]
print(f"\n  dE/d(alpha) = {sl:+.1f} meV/atom per unit alpha")

json.dump({"f_rmse_base": float(rb), "f_rmse_ft": float(rf),
           "improvement_pct": float(100*(1-rf/rb)),
           "V_min": {r[0]: float(r[1]) for r in rows},
           "B_GPa": {r[0]: float(r[2]) for r in rows},
           "by_kind": {k: [float(1000*np.mean([p for p,kk in zip(perb,kinds) if kk==k])),
                           float(1000*np.mean([p for p,kk in zip(perf,kinds) if kk==k]))]
                       for k in ks},
           "order_slope_meV": float(sl)},
          open("res_stage6.json","w"), indent=1)
print("\nwrote fig1_parity.png fig2_ev.png fig3_breakdown.png res_stage6.json")
