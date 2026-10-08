"""Step 3: layer-average (h_bar) baseline. Mirrors compute_avg_hidden_state.py but
loads the model once, runs all 6 tasks, casts mean_layer_activations to model dtype
(needed for fp16 in-place FV add), and uses a fixed reproducible seed set.
"""
import os, json, time, argparse
import torch, numpy as np
from utils.prompt_utils import *
from utils.model_utils import *
from utils.eval_utils import *
from utils.extract_utils import get_mean_layer_activations

TASKS = ['antonym', 'capitalize', 'country-capital', 'english-french', 'present-past', 'singular-plural']

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--save_root', default='results/gptj_avg_hs')
    ap.add_argument('--root_data_dir', default='../dataset_files')
    ap.add_argument('--n_seeds', type=int, default=3)
    ap.add_argument('--n_trials', type=int, default=100)
    ap.add_argument('--half', action='store_true')
    ap.add_argument('--seed_list', default=None, help='comma-separated explicit seeds; overrides n_seeds draw')
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    model, tokenizer, cfg = load_gpt_model_and_tokenizer('EleutherAI/gpt-j-6b')
    if args.half:
        model = model.half()
        print("fp16")

    if args.seed_list is not None:
        seeds = np.array([int(s) for s in args.seed_list.split(',')])
    else:
        np.random.seed(0)  # make the seed draw reproducible (repo leaves it unseeded)
        seeds = np.random.choice(100000, size=args.n_seeds)
    print("seeds:", seeds.tolist())

    for dataset_name in TASKS:
        save_path_root = f"{args.save_root}/{dataset_name}"
        os.makedirs(save_path_root, exist_ok=True)
        for seed in seeds:
            marker = f'{save_path_root}/mean_layer_intervention_zs_results_sweep_{seed}.json'
            if os.path.exists(marker):
                print(f"[skip] {dataset_name} seed{seed}")
                continue
            t0 = time.time()
            set_seed(seed)
            dataset = load_dataset(dataset_name, root_data_dir=args.root_data_dir, test_size=0.3, seed=seed)
            mean_activations = get_mean_layer_activations(dataset, model=model, model_config=cfg, tokenizer=tokenizer,
                                                          n_icl_examples=10, N_TRIALS=args.n_trials)
            mean_activations = mean_activations.to(model.dtype)  # fp16 in-place add needs matching dtype
            torch.save(mean_activations, f'{save_path_root}/{dataset_name}_mean_layer_activations.pt')

            fs_results = n_shot_eval_no_intervention(dataset, 10, model, cfg, tokenizer)
            filter_set = np.where(np.array(fs_results['clean_rank_list']) == 0)[0]

            zs_res, fss_res = {}, {}
            for i in range(cfg['n_layers']):
                set_seed(seed)
                zs_res[i] = n_shot_eval(dataset, mean_activations[i].unsqueeze(0), i, 0, model, cfg, tokenizer, filter_set=filter_set)
                set_seed(seed)
                fss_res[i] = n_shot_eval(dataset, mean_activations[i].unsqueeze(0), i, 10, model, cfg, tokenizer,
                                         filter_set=filter_set, shuffle_labels=True)
            with open(f'{save_path_root}/mean_layer_intervention_zs_results_sweep_{seed}.json', 'w') as f:
                json.dump(zs_res, f, indent=2)
            with open(f'{save_path_root}/mean_layer_intervention_fss_results_sweep_{seed}.json', 'w') as f:
                json.dump(fss_res, f, indent=2)
            l12 = zs_res[12]['intervention_topk'][0][1] if 12 in zs_res else None
            print(f"[done] {dataset_name} seed{seed} in {time.time()-t0:.0f}s  L12 zs top1={l12}")
