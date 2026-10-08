"""Driver for Step 2: main GPT-J causal-triggering FV evaluation.
Loads the model ONCE and loops over seeds x tasks, mirroring evaluate_function_vector.py
exactly (baukit path). Saves per-(seed,task) artifacts and is resumable.
"""
import os, json, time, argparse
import torch, numpy as np

from utils.prompt_utils import *
from utils.intervention_utils import *
from utils.model_utils import *
from utils.eval_utils import *
from utils.extract_utils import *
from compute_indirect_effect import compute_indirect_effect


def run_one(dataset_name, seed, model, tokenizer, model_config, save_path_root_base,
            root_data_dir, n_top_heads=10, n_shots=10, test_split=0.3,
            n_mean_activations_trials=100, n_indirect_effect_trials=25,
            prefixes=None, separators=None):
    if prefixes is None:
        prefixes = {"input": "Q:", "output": "A:", "instructions": ""}
    if separators is None:
        separators = {"input": "\n", "output": "\n\n", "instructions": ""}

    save_path_root = f"{save_path_root_base}/{dataset_name}"
    os.makedirs(save_path_root, exist_ok=True)

    done_marker = f'{save_path_root}/zs_results_layer_sweep.json'
    if os.path.exists(done_marker):
        print(f"[skip] {dataset_name} seed{seed} already done")
        return

    t0 = time.time()
    set_seed(seed)
    dataset = load_dataset(dataset_name, root_data_dir=root_data_dir, test_size=test_split, seed=seed)

    # 1. 10-shot filter (valid split with seed+42, test split with seed)
    fs_results_file_name = f'{save_path_root}/fs_results_layer_sweep.json'
    set_seed(seed + 42)
    fs_results_validation = n_shot_eval_no_intervention(dataset=dataset, n_shots=n_shots, model=model, model_config=model_config,
                                                        tokenizer=tokenizer, compute_ppl=True, test_split='valid',
                                                        prefixes=prefixes, separators=separators)
    filter_set_validation = np.where(np.array(fs_results_validation['clean_rank_list']) == 0)[0]
    set_seed(seed)
    fs_results = n_shot_eval_no_intervention(dataset=dataset, n_shots=n_shots, model=model, model_config=model_config,
                                             tokenizer=tokenizer, compute_ppl=True, prefixes=prefixes, separators=separators)
    filter_set = np.where(np.array(fs_results['clean_rank_list']) == 0)[0]
    with open(fs_results_file_name, 'w') as f:
        json.dump(fs_results, f, indent=2)
    print(f"  filter_set(test)={len(filter_set)}/{len(dataset['test'])}  filter_set(valid)={len(filter_set_validation)}/{len(dataset['valid'])}")

    # 2. Mean head activations
    set_seed(seed)
    mean_activations = get_mean_head_activations(dataset, model=model, model_config=model_config, tokenizer=tokenizer,
                                                 n_icl_examples=n_shots, N_TRIALS=n_mean_activations_trials,
                                                 prefixes=prefixes, separators=separators, filter_set=filter_set_validation)
    mean_activations_path = f'{save_path_root}/{dataset_name}_mean_head_activations.pt'
    torch.save(mean_activations, mean_activations_path)

    # 3. Indirect effect (CIE) -- baukit patching loop
    set_seed(seed)
    indirect_effect = compute_indirect_effect(dataset, mean_activations, model=model, model_config=model_config,
                                              tokenizer=tokenizer, n_shots=n_shots, n_trials=n_indirect_effect_trials,
                                              last_token_only=True, prefixes=prefixes, separators=separators,
                                              filter_set=filter_set_validation)
    indirect_effect_path = f'{save_path_root}/{dataset_name}_indirect_effect.pt'
    torch.save(indirect_effect, indirect_effect_path)

    # 4. Function vector
    fv, top_heads = compute_function_vector(mean_activations, indirect_effect, model, model_config=model_config, n_top_heads=n_top_heads)

    # 5. Layer sweep: zero-shot + few-shot-shuffled
    zs_results = {}
    fs_shuffled_results = {}
    for edit_layer in range(0, model_config['n_layers']):
        set_seed(seed)
        zs_results[edit_layer] = n_shot_eval(dataset=dataset, fv_vector=fv, edit_layer=edit_layer, n_shots=0,
                                             model=model, model_config=model_config, tokenizer=tokenizer,
                                             filter_set=filter_set, prefixes=prefixes, separators=separators)
        set_seed(seed)
        fs_shuffled_results[edit_layer] = n_shot_eval(dataset=dataset, fv_vector=fv, edit_layer=edit_layer, n_shots=n_shots,
                                                      model=model, model_config=model_config, tokenizer=tokenizer,
                                                      filter_set=filter_set, shuffle_labels=True,
                                                      prefixes=prefixes, separators=separators)
    with open(f'{save_path_root}/zs_results_layer_sweep.json', 'w') as f:
        json.dump(zs_results, f, indent=2)
    with open(f'{save_path_root}/fs_shuffled_results_layer_sweep.json', 'w') as f:
        json.dump(fs_shuffled_results, f, indent=2)

    # 6. Model baseline (0..n_shots)
    baseline_results = compute_dataset_baseline(dataset, model, model_config, tokenizer, n_shots=n_shots, seed=seed,
                                                prefixes=prefixes, separators=separators)
    with open(f'{save_path_root}/model_baseline.json', 'w') as f:
        json.dump(baseline_results, f, indent=2)

    # args record
    with open(f'{save_path_root}/fv_eval_args.txt', 'w') as f:
        json.dump({"dataset_name": dataset_name, "seed": seed, "n_top_heads": n_top_heads,
                   "top_heads": [[int(x[0]), int(x[1]), float(x[2])] for x in top_heads],
                   "n_mean_activations_trials": n_mean_activations_trials,
                   "n_indirect_effect_trials": n_indirect_effect_trials, "path": "baukit"}, f, indent=2)

    # quick console summary: layer-9 zs top1
    l9 = zs_results[9]['intervention_topk'][0][1]
    l9c = zs_results[9]['clean_topk'][0][1]
    print(f"[done] {dataset_name} seed{seed} in {time.time()-t0:.0f}s  L9 zs FV top1={l9:.3f} (clean {l9c:.3f})  top5heads={top_heads[:5]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', default='EleutherAI/gpt-j-6b')
    ap.add_argument('--seeds', default='42,1,2,3,4')
    ap.add_argument('--tasks', default='antonym,capitalize,country-capital,english-french,present-past,singular-plural')
    ap.add_argument('--save_root', default='results/gptj_seed')  # seed appended
    ap.add_argument('--root_data_dir', default='../dataset_files')
    ap.add_argument('--n_mean', type=int, default=100)
    ap.add_argument('--n_cie', type=int, default=25)
    ap.add_argument('--half', action='store_true')
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    tasks = args.tasks.split(',')

    torch.set_grad_enabled(False)
    print("Loading model", args.model_name)
    t0 = time.time()
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name)
    if args.half:
        model = model.half()
        print("model converted to fp16")
    print(f"model loaded in {time.time()-t0:.0f}s")

    for seed in seeds:
        for task in tasks:
            save_root = f"{args.save_root}{seed}"
            try:
                run_one(task, seed, model, tokenizer, model_config, save_root, args.root_data_dir,
                        n_mean_activations_trials=args.n_mean, n_indirect_effect_trials=args.n_cie)
            except Exception as e:
                import traceback
                print(f"[ERROR] {task} seed{seed}: {e}")
                traceback.print_exc()
