"""Head localization (C5) + induction-head prefix-matching (C13) for a followup model.
Aggregates this session's per-task indirect_effect.pt (mean over trials, then mean
over tasks) -> top-K AIE (layer,head) heads; computes concentration (fraction in the
first 2/3 of layers, mean/median layer as fraction of depth); and prefix_matching_score
for the top-K heads. Mirrors replication_plan step 5, generalized to any model.
"""
import os, sys, json, glob, argparse
import torch, numpy as np
from utils.model_utils import load_gpt_model_and_tokenizer
from utils.extract_utils import prefix_matching_score

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', required=True)
    ap.add_argument('--nickname', required=True)
    ap.add_argument('--n_top_heads', type=int, required=True)
    ap.add_argument('--save_root', default='results_fu')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    root = f"{args.save_root}/{args.nickname}_seed{args.seed}"
    ie_files = sorted(glob.glob(f"{root}/*/*_indirect_effect.pt"))
    assert ie_files, f"no indirect_effect.pt under {root}"
    ies = [torch.load(f) for f in ie_files]              # each (T, n_layers, n_heads)
    aie = torch.stack([ie.float().mean(0) for ie in ies]).mean(0)  # (n_layers, n_heads)
    n_layers, n_heads = aie.shape
    K = args.n_top_heads
    vals, inds = torch.topk(aie.view(-1), K, largest=True)
    top = [(int(np.unravel_index(int(i), aie.shape)[0]), int(np.unravel_index(int(i), aie.shape)[1]), float(v))
           for v, i in zip(vals, inds)]
    layer_idx = [t[0] for t in top]
    frac_first_two_thirds = float(np.mean([L < (2/3)*n_layers for L in layer_idx]))
    mean_layer_frac = float(np.mean(layer_idx) / n_layers)
    median_layer_frac = float(np.median(layer_idx) / n_layers)

    torch.set_grad_enabled(False)
    model, tokenizer, cfg = load_gpt_model_and_tokenizer(args.model_name)
    try:
        pms = prefix_matching_score(model, cfg)   # (n_layers, n_heads)
    except TypeError:
        # SDPA models (e.g. Qwen3 on transformers>=4.5x) return attentions=None;
        # reload with eager attention just for the prefix-match forward.
        import torch as _t
        from transformers import AutoModelForCausalLM
        print("[hl] attentions were None; reloading with attn_implementation='eager'")
        del model; _t.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=_t.bfloat16,
                                                     attn_implementation='eager').to('cuda')
        pms = prefix_matching_score(model, cfg)
    top_pms = {f"{L}-{H}": round(float(pms[L, H]), 4) for L, H, _ in top}
    mean_pms_all = round(float(pms.float().mean()), 4)
    mean_pms_topK = round(float(np.mean(list(top_pms.values()))), 4)

    out = {
        "nickname": args.nickname, "model": args.model_name, "n_layers": n_layers, "n_heads": n_heads,
        "n_top_heads": K, "tasks_aggregated": [os.path.basename(os.path.dirname(f)) for f in ie_files],
        "topK_AIE": [[L, H, round(v, 5)] for L, H, v in top],
        "topK_layer_indices": layer_idx,
        "concentration_frac_first_two_thirds": round(frac_first_two_thirds, 3),
        "mean_layer_frac_depth": round(mean_layer_frac, 3),
        "median_layer_frac_depth": round(median_layer_frac, 3),
        "topK_prefix_match": top_pms,
        "mean_prefix_match_topK": mean_pms_topK,
        "mean_prefix_match_all_heads": mean_pms_all,
    }
    os.makedirs(root, exist_ok=True)
    json.dump(out, open(f"{root}/head_localization.json", 'w'), indent=2)
    print(json.dumps(out, indent=2))
