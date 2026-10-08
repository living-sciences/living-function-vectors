"""C16 (#heads plateau) for a followup model, faithful to THIS session's CIE.
For k = 0..N, build the FV from the top-k AIE heads (compute_function_vector) and
evaluate zero-shot at the task's peak layer. Reuses saved mean_activations,
indirect_effect, the 10-shot filter set, and peak layer from fu2_summary.json.
Filter set capped for budget (disclosed)."""
import os, json, argparse
import torch, numpy as np
from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from utils.prompt_utils import load_dataset
from utils.extract_utils import compute_function_vector
from utils.eval_utils import n_shot_eval

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', required=True)
    ap.add_argument('--nickname', required=True)
    ap.add_argument('--tasks', required=True)
    ap.add_argument('--max_heads', type=int, default=30)
    ap.add_argument('--cap_filter', type=int, default=150)
    ap.add_argument('--save_root', default='results_fu')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    root = f"{args.save_root}/{args.nickname}_seed{args.seed}"

    torch.set_grad_enabled(False)
    model, tokenizer, cfg = load_gpt_model_and_tokenizer(args.model_name)
    os.makedirs(f"{root}/numheads", exist_ok=True)

    for task in args.tasks.split(','):
        spr = f"{root}/{task}"
        summ = json.load(open(f"{spr}/fu2_summary.json"))
        peak = summ['peak_layer']
        mean_act = torch.load(f"{spr}/{task}_mean_head_activations.pt")
        ie = torch.load(f"{spr}/{task}_indirect_effect.pt")
        fs = json.load(open(f"{spr}/fs_results_layer_sweep.json"))
        filt = np.where(np.array(fs['clean_rank_list']) == 0)[0]
        if len(filt) > args.cap_filter:
            filt = filt[:args.cap_filter]
        set_seed(args.seed)
        dataset = load_dataset(task, root_data_dir='../dataset_files', test_size=0.3, seed=args.seed)
        res = {}
        for k in range(args.max_heads + 1):
            fv, _ = compute_function_vector(mean_act, ie, model, cfg, n_top_heads=k)
            set_seed(args.seed)
            r = n_shot_eval(dataset=dataset, fv_vector=fv, edit_layer=peak, n_shots=0, model=model,
                            model_config=cfg, tokenizer=tokenizer, filter_set=filt)
            res[k] = r['intervention_topk'][0][1]
        json.dump({"task": task, "peak_layer": peak, "cap_filter": int(len(filt)),
                   "acc_by_k": res}, open(f"{root}/numheads/{task}_perf_v_heads.json", 'w'), indent=2)
        # locate plateau: smallest k reaching 95% of max
        mx = max(res.values())
        plat = next((k for k in range(args.max_heads + 1) if res[k] >= 0.95 * mx), args.max_heads)
        print(f"[{args.nickname}/{task}] peakL{peak} max={mx:.3f} @k*; 95%-plateau at k={plat}; "
              f"k1={res[1]:.2f} k5={res[5]:.2f} k10={res[10]:.2f} k20={res[20]:.2f}")
