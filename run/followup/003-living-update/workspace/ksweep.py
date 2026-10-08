"""K-sensitivity: reuse cached mean_head_activations + indirect_effect (NO recompute),
rebuild the FV with several K and re-run the ZS layer sweep. Tests whether the
antonym/country-capital FV failure is a too-small-head-set artifact."""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from qwen35_fv import (load_qwen35, shape_guard, compute_function_vector, fv_eval,
                        load_dataset, tenshot_filter)
from utils.model_utils import set_seed

ROOT = "results_fu/qwen35-9b_seed42"
DATA = os.path.join(os.path.dirname(__file__), "codebase", "dataset_files")
torch.set_grad_enabled(False)
model, tok, dec, mc = load_qwen35(); shape_guard(dec, mc)
out = {}
for task in ["antonym", "country-capital", "present-past"]:
    mean_act = torch.load(f"{ROOT}/{task}/mean_head_activations.pt")
    ie = torch.load(f"{ROOT}/{task}/indirect_effect.pt")
    set_seed(42)
    ds = load_dataset(task, root_data_dir=DATA, test_size=0.3, seed=42)
    set_seed(42)
    ft = tenshot_filter(ds, "test", model, tok, dec, mc)
    if len(ft) > 300: ft = ft[:300]
    out[task] = {}
    for K in [10, 20, 30, 40]:
        fv, top = compute_function_vector(mean_act, ie, dec, mc, K=K)
        cache = {}
        best = (-1, -1.0)
        curve = {}
        for L in range(mc["n_layers"]):
            set_seed(42)
            iv, base, _ = fv_eval(ds, fv, L, 0, model, tok, dec, mc, ft, clean_cache=cache)
            curve[L] = iv
            if iv > best[1]: best = (L, iv)
        out[task][K] = {"peak_layer": best[0], "peak_fv": round(best[1], 4),
                        "peak_frac": round(best[0]/mc["n_layers"], 3)}
        print(f"{task} K={K}: peak L{best[0]} ({best[0]/mc['n_layers']:.2f}) fv={best[1]:.3f}")
    json.dump(out, open(f"{ROOT}/ksweep.json", "w"), indent=2)
print("DONE", json.dumps(out, indent=2))
