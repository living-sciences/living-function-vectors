"""Follow-up 001 FAST driver.

Same measurement as run_step2/evaluate_function_vector (baukit path, CIE ->
top-K AIE heads -> function vector), but the layer sweep is trimmed to fit the
8 GPU-h / 90-min-per-model budget:
  * ZS FV curve: FULL sweep over all layers (needed for the peak layer, C2).
  * SL (few-shot-shuffled) FV: evaluated ONLY at the peak layer (C1 SL column),
    not the full 32-layer sweep (which is the 10-shot-prompt cost sink).
  * baseline (no-FV) accuracy is read for free from clean_topk in the ZS sweep.

Load model ONCE, shape-guard, loop tasks. Resumable via cached artifacts.
Disclosed deviations: bf16 (native precision) for all new models; SL@peak only.
"""
import os, json, time, argparse
import torch, numpy as np

from utils.prompt_utils import *
from utils.intervention_utils import *
from utils.model_utils import *
from utils.eval_utils import *
from utils.extract_utils import *
from compute_indirect_effect import compute_indirect_effect
from run_fu import shape_guard


def run_one_fast(dataset_name, seed, model, tokenizer, model_config, save_root, root_data_dir,
                 n_top_heads, n_mean=50, n_cie=15, n_shots=10, test_split=0.3):
    prefixes = {"input": "Q:", "output": "A:", "instructions": ""}
    separators = {"input": "\n", "output": "\n\n", "instructions": ""}
    spr = f"{save_root}/{dataset_name}"
    os.makedirs(spr, exist_ok=True)
    done = f'{spr}/fu2_summary.json'
    if os.path.exists(done):
        print(f"[skip] {dataset_name} already done (fu2_summary.json)")
        return json.load(open(done))

    t0 = time.time()
    set_seed(seed)
    dataset = load_dataset(dataset_name, root_data_dir=root_data_dir, test_size=test_split, seed=seed)

    # 1. 10-shot filter (valid w/ seed+42, test w/ seed) -- reuse if present
    fs_file = f'{spr}/fs_results_layer_sweep.json'
    set_seed(seed + 42)
    fs_val = n_shot_eval_no_intervention(dataset=dataset, n_shots=n_shots, model=model, model_config=model_config,
                                         tokenizer=tokenizer, compute_ppl=True, test_split='valid',
                                         prefixes=prefixes, separators=separators)
    filter_val = np.where(np.array(fs_val['clean_rank_list']) == 0)[0]
    set_seed(seed)
    fs_res = n_shot_eval_no_intervention(dataset=dataset, n_shots=n_shots, model=model, model_config=model_config,
                                         tokenizer=tokenizer, compute_ppl=True, prefixes=prefixes, separators=separators)
    filter_test = np.where(np.array(fs_res['clean_rank_list']) == 0)[0]
    json.dump(fs_res, open(fs_file, 'w'), indent=2)
    print(f"  filter_test={len(filter_test)}/{len(dataset['test'])} filter_val={len(filter_val)}/{len(dataset['valid'])}")

    # 2. mean head activations (reuse if cached)
    ma_path = f'{spr}/{dataset_name}_mean_head_activations.pt'
    if os.path.exists(ma_path):
        mean_activations = torch.load(ma_path); print("  [reuse] mean_activations")
    else:
        set_seed(seed)
        mean_activations = get_mean_head_activations(dataset, model=model, model_config=model_config, tokenizer=tokenizer,
                                                     n_icl_examples=n_shots, N_TRIALS=n_mean, prefixes=prefixes,
                                                     separators=separators, filter_set=filter_val)
        torch.save(mean_activations, ma_path)

    # 3. CIE (reuse if cached)
    ie_path = f'{spr}/{dataset_name}_indirect_effect.pt'
    if os.path.exists(ie_path):
        indirect_effect = torch.load(ie_path); print("  [reuse] indirect_effect")
    else:
        tcie = time.time()
        set_seed(seed)
        indirect_effect = compute_indirect_effect(dataset, mean_activations, model=model, model_config=model_config,
                                                  tokenizer=tokenizer, n_shots=n_shots, n_trials=n_cie,
                                                  last_token_only=True, prefixes=prefixes, separators=separators,
                                                  filter_set=filter_val)
        torch.save(indirect_effect, ie_path)
        print(f"  CIE done in {(time.time()-tcie)/60:.1f} min")

    # 4. function vector from top-K AIE heads
    fv, top_heads = compute_function_vector(mean_activations, indirect_effect, model, model_config=model_config,
                                            n_top_heads=n_top_heads)

    # 5. ZS FULL layer sweep
    tsw = time.time()
    zs = {}
    for L in range(model_config['n_layers']):
        set_seed(seed)
        zs[L] = n_shot_eval(dataset=dataset, fv_vector=fv, edit_layer=L, n_shots=0, model=model,
                            model_config=model_config, tokenizer=tokenizer, filter_set=filter_test,
                            prefixes=prefixes, separators=separators)
    json.dump(zs, open(f'{spr}/zs_results_layer_sweep.json', 'w'), indent=2)
    # peak layer = argmax ZS FV top1
    peak = max(range(model_config['n_layers']), key=lambda L: zs[L]['intervention_topk'][0][1])
    print(f"  ZS sweep done in {(time.time()-tsw)/60:.1f} min; peak L{peak} FV={zs[peak]['intervention_topk'][0][1]:.3f}")

    # 6. SL (few-shot shuffled) FV at PEAK layer only
    set_seed(seed)
    sl = n_shot_eval(dataset=dataset, fv_vector=fv, edit_layer=peak, n_shots=n_shots, model=model,
                     model_config=model_config, tokenizer=tokenizer, filter_set=filter_test, shuffle_labels=True,
                     prefixes=prefixes, separators=separators)
    json.dump({str(peak): sl}, open(f'{spr}/sl_at_peak.json', 'w'), indent=2)

    # depth-1/3 canonical layer
    canon = model_config['n_layers'] // 3
    top_heads_ser = [[int(x[0]), int(x[1]), float(x[2])] for x in top_heads]
    summary = {
        "task": dataset_name, "seed": seed, "n_layers": model_config['n_layers'],
        "n_heads": model_config['n_heads'], "n_top_heads": n_top_heads,
        "peak_layer": int(peak), "peak_frac_depth": round(peak / model_config['n_layers'], 3),
        "zs_fv_peak": zs[peak]['intervention_topk'][0][1],
        "zs_baseline_peak": zs[peak]['clean_topk'][0][1],
        "canonical_layer": int(canon), "zs_fv_canonical": zs[canon]['intervention_topk'][0][1],
        "zs_baseline_canonical": zs[canon]['clean_topk'][0][1],
        "sl_fv_at_peak": sl['intervention_topk'][0][1],
        "sl_baseline_at_peak": sl['clean_topk'][0][1],
        "top_heads": top_heads_ser,
        "filter_test": int(len(filter_test)), "filter_val": int(len(filter_val)),
        "minutes": round((time.time() - t0) / 60, 1),
        "dtype": str(next(model.parameters()).dtype),
    }
    json.dump(summary, open(done, 'w'), indent=2)
    print(f"[done] {dataset_name} in {summary['minutes']} min  peak L{peak}({summary['peak_frac_depth']}) "
          f"ZS FV={summary['zs_fv_peak']:.3f} base={summary['zs_baseline_peak']:.3f} SL={summary['sl_fv_at_peak']:.3f}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', required=True)
    ap.add_argument('--nickname', required=True)
    ap.add_argument('--n_top_heads', type=int, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tasks', default='antonym,country-capital,present-past')
    ap.add_argument('--save_root', default='results_fu')
    ap.add_argument('--root_data_dir', default='../dataset_files')
    ap.add_argument('--n_mean', type=int, default=50)
    ap.add_argument('--n_cie', type=int, default=15)
    args = ap.parse_args()

    tasks = args.tasks.split(',')
    torch.set_grad_enabled(False)
    print("Loading model", args.model_name)
    t0 = time.time()
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name)
    print(f"model loaded in {time.time()-t0:.0f}s dtype={next(model.parameters()).dtype}")
    print("MODEL_CONFIG:", {k: v for k, v in model_config.items() if 'hook' not in k})
    shape_guard(model, model_config)

    save_root = f"{args.save_root}/{args.nickname}_seed{args.seed}"
    tstudy = time.time()
    for task in tasks:
        try:
            run_one_fast(task, args.seed, model, tokenizer, model_config, save_root, args.root_data_dir,
                         n_top_heads=args.n_top_heads, n_mean=args.n_mean, n_cie=args.n_cie)
        except Exception as e:
            import traceback
            print(f"[ERROR] {task}: {e}"); traceback.print_exc()
    print(f"[MODEL DONE] {args.nickname} total {(time.time()-tstudy)/60:.1f} min")
