# per-seed baseline 들에서 best-test 를 골라 canonical baseline_{ds}_pool.npz 로 저장.
#   cry  → cache/baseline_{ds}_pool.npz  (※ 기존 파일 보존을 위해 덮어쓰지 않음. 이미
#          존재하면 스킵하고 경고. cond 재실행이 CRY 기준 baseline 을 망가뜨리지 않게.)
#   cond → cache/cond/baseline_{ds}_pool.npz
import os, sys, glob, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, circuits

ap = argparse.ArgumentParser()
ap.add_argument("--method", default="cond", choices=circuits.METHODS)
a = ap.parse_args()

seed_dir = circuits.cache_subdir("baseline_seeds", a.method)
out_dir = config.CACHE_DIR if a.method == "cry" else os.path.join(config.CACHE_DIR, "cond")
os.makedirs(out_dir, exist_ok=True)

for name in config.DATASETS:
    files = sorted(glob.glob(os.path.join(seed_dir, f"{name}_pool_s*.npz")))
    if not files:
        print(f"[{name}] per-seed 파일 없음 (스킵)"); continue
    recs = [np.load(f, allow_pickle=True) for f in files]
    best = max(recs, key=lambda z: float(z["test_acc"]))
    outp = os.path.join(out_dir, f"baseline_{name}_pool.npz")
    if a.method == "cry" and os.path.exists(outp):
        print(f"[{name}] cry baseline 이미 존재 → 보존(스킵): {outp}"); continue
    np.savez(outp, params=best["params"], n_q=int(best["n_q"]),
             pooling=bool(best["pooling"]), reupload=bool(best["reupload"]),
             n_blocks=int(best["n_blocks"]), dataset=name, method=a.method,
             split_seed=int(best["split_seed"]), best_seed=int(best["seed"]),
             train_idx=best["train_idx"], test_idx=best["test_idx"],
             train_acc=float(best["train_acc"]), test_acc=float(best["test_acc"]),
             n_train=int(best["n_train"]), n_test=int(best["n_test"]),
             n_steps=int(best["n_steps"]), lr=float(best["lr"]))
    allte = {int(z["seed"]): round(float(z["test_acc"]), 4) for z in recs}
    print(f"[{name}] best seed={int(best['seed'])} test={float(best['test_acc']):.4f} "
          f"(all={allte}) → {outp}")
