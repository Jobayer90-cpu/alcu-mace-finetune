import json, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

LBL = {"baseline":"MACE-MP-0", "ft2":"force-weighted", "ft3":"balanced"}
COL = {"baseline":"tab:orange", "ft2":"tab:blue", "ft3":"tab:green"}
MK  = {"baseline":"o", "ft2":"^", "ft3":"s"}

d = defaultdict(dict)
for f in sorted(glob.glob("md_*.npy")):
    tag, T = re.match(r"md_(.+)_(\d+)\.npy", f).groups()
    V = np.load(f); b = np.array_split(V, 5)
    d[tag][int(T)] = (V.mean(), np.std([x.mean() for x in b], ddof=1)/np.sqrt(5), V.std())

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
res = {}
for tag in ["baseline", "ft2", "ft3"]:
    if tag not in d or len(d[tag]) < 3: continue
    Ts = np.array(sorted(d[tag]), float)
    V  = np.array([d[tag][t][0] for t in sorted(d[tag])])
    E  = np.array([d[tag][t][1] for t in sorted(d[tag])])
    p, cov = np.polyfit(Ts, V, 1, cov=True)
    V300 = np.polyval(p, 300)
    aL, sd = p[0]/(3*V300), np.sqrt(cov[0,0])/(3*V300)
    res[tag] = {"alpha_L_1e6": aL*1e6, "sd": sd*1e6,
                "V": {int(t): float(d[tag][t][0]) for t in sorted(d[tag])}}
    a1.errorbar(Ts, V, yerr=E, fmt=MK[tag], color=COL[tag], ms=7, capsize=3,
                label=f"{LBL[tag]}: {aL*1e6:.1f}$\\pm${sd*1e6:.1f}")
    tt = np.linspace(280, 520, 30)
    a1.plot(tt, np.polyval(p, tt), "-", color=COL[tag], lw=1.1, alpha=.7)
a1.set_xlabel("temperature (K)"); a1.set_ylabel("$\\langle V\\rangle$ ($\\AA^3$)")
a1.set_title("Thermal expansion, Al$_{75}$Cu$_{25}$")
a1.legend(fontsize=8, title="$\\alpha_L$ ($10^{-6}$/K)", title_fontsize=8)

tags = list(res)
x = np.arange(len(tags))
a2.bar(x, [res[t]["alpha_L_1e6"] for t in tags],
       yerr=[res[t]["sd"] for t in tags], capsize=5,
       color=[COL[t] for t in tags])
a2.axhline(23.1, color="k", ls="--", lw=1.2, label="pure Al, expt 23.1")
a2.set_xticks(x); a2.set_xticklabels([LBL[t] for t in tags], fontsize=8)
a2.set_ylabel("$\\alpha_L$ ($10^{-6}$ K$^{-1}$)")
a2.set_title("Linear expansion coefficient"); a2.legend(fontsize=8)
fig.tight_layout(); fig.savefig("fig4_expansion.png", dpi=180); plt.close()

print(f"{'model':16s} {'alpha_L':>10s} {'+/-':>6s}   V(300) V(400) V(500)")
for t in tags:
    r = res[t]; v = r["V"]
    print(f"{LBL[t]:16s} {r['alpha_L_1e6']:10.1f} {r['sd']:6.1f}   "
          + "  ".join(f"{v[k]:6.1f}" for k in sorted(v)))
print(f"\n{'pure Al experiment':16s} {23.1:10.1f}")

try:
    s6 = json.load(open("res_stage6.json")); s6b = json.load(open("res_stage6_ft3.json"))
    dft = s6['V_min']['DFT (QE)']
    print(f"\n{'model':16s} {'F RMSE':>9s} {'V_min err':>10s} {'alpha_L':>9s}")
    print(f"{'MACE-MP-0':16s} {s6['f_rmse_base']*1000:9.1f} "
          f"{100*(s6['V_min']['MACE-MP-0']-dft)/dft:9.2f}% "
          f"{res.get('baseline',{}).get('alpha_L_1e6',float('nan')):9.1f}")
    print(f"{'force-weighted':16s} {s6['f_rmse_ft']*1000:9.1f} "
          f"{100*(s6['V_min']['fine-tuned']-dft)/dft:9.2f}% "
          f"{res.get('ft2',{}).get('alpha_L_1e6',float('nan')):9.1f}")
    print(f"{'balanced':16s} {s6b['f_rmse_ft']*1000:9.1f} "
          f"{100*(s6b['V_min']['fine-tuned']-dft)/dft:9.2f}% "
          f"{res.get('ft3',{}).get('alpha_L_1e6',float('nan')):9.1f}")
except Exception as e:
    print("(summary table skipped:", e, ")")

json.dump(res, open("res_alpha.json","w"), indent=1)
print("\nwrote fig4_expansion.png res_alpha.json")
