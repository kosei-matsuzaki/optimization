import sys
sys.path.insert(0, '/home/user/optimization')
sys.path.insert(0, '/home/user/optimization/analysis/mmo2024/e115')
import importlib.util, numpy as np
from analyze import rule_indices, aggregate           # e115
spec = importlib.util.spec_from_file_location(
    "e142mod", "/home/user/optimization/analysis/mmo2024/e142/analyze.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
runs = m.load_runs()
arm_fn = lambda r: rule_indices(m.ARM, r["f"], r["K"], r["x"], m.ARM_R)
agg = {mm: aggregate([r for r in runs if r["method"] == mm], arm_fn)
       for mm in ("r3pso", "r3pso-p400", "Restart-Lander")}
probs = sorted(agg["r3pso-p400"])
print("部分結果（判定に使わない。n=3、事前登録の逃げ道で腕を落としたため）")
print(f"{'問題':<16}{'r3pso(30)':>12}{'r3pso(400)':>12}{'400-30':>10}{'null':>10}")
for p in probs:
    x, y = agg["r3pso"][p]["score"], agg["r3pso-p400"][p]["score"]
    print(f"{p:<16}{x:>12.4f}{y:>12.4f}{y-x:>+10.4f}{agg['Restart-Lander'][p]['score']:>10.4f}")
print(f"{'3 問平均':<14}"
      f"{np.mean([agg['r3pso'][p]['score'] for p in probs]):>12.4f}"
      f"{np.mean([agg['r3pso-p400'][p]['score'] for p in probs]):>12.4f}"
      f"{np.mean([agg['r3pso-p400'][p]['score'] - agg['r3pso'][p]['score'] for p in probs]):>+10.4f}"
      f"{np.mean([agg['Restart-Lander'][p]['score'] for p in probs]):>10.4f}")
