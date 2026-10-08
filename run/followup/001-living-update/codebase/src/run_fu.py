"""Follow-up 001 driver: load model ONCE, shape-guard, then run the step-2
CIE+layer-sweep (run_one from run_step2) for each core task with the per-model K.

Adds two things run_step2 lacks: a CLI --n_top_heads, and the MANDATORY shape
guard (instruction section 3.4) that gates the whole study.
"""
import os, sys, time, argparse
import torch

from utils.model_utils import load_gpt_model_and_tokenizer
from run_step2 import run_one


def shape_guard(model, model_config):
    """Assert the o_proj-input head decomposition is valid (instruction 3.4).

    resid_dim must split evenly into n_heads, and the o_proj INPUT width must
    equal n_heads * (resid_dim // n_heads). GQA shrinks K/V heads only; the
    o_proj input stays n_heads*head_dim, so this holds for the main ladder.
    Gemma-3 / gpt-oss would fail here -> skip.
    """
    n_heads = model_config['n_heads']
    resid_dim = model_config['resid_dim']
    assert resid_dim % n_heads == 0, f"resid_dim {resid_dim} not divisible by n_heads {n_heads}"
    head_dim = resid_dim // n_heads
    # find an o_proj module from the first attn hook name
    hook = model_config['attn_hook_names'][0]
    mod = model
    for part in hook.split('.'):
        mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
    in_features = mod.in_features
    assert in_features == n_heads * head_dim, (
        f"o_proj in_features {in_features} != n_heads*head_dim {n_heads*head_dim}")
    print(f"[shape-guard PASS] n_heads={n_heads} resid_dim={resid_dim} head_dim={head_dim} o_proj.in={in_features}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', required=True)
    ap.add_argument('--nickname', required=True)
    ap.add_argument('--n_top_heads', type=int, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tasks', default='antonym,capitalize,country-capital,english-french,present-past,singular-plural')
    ap.add_argument('--save_root', default='results_fu')  # <save_root>/<nick>_seed<seed>
    ap.add_argument('--root_data_dir', default='../dataset_files')
    ap.add_argument('--n_mean', type=int, default=50)
    ap.add_argument('--n_cie', type=int, default=15)
    ap.add_argument('--wall_cap_min', type=float, default=90.0)
    args = ap.parse_args()

    tasks = args.tasks.split(',')
    torch.set_grad_enabled(False)

    print("Loading model", args.model_name)
    t0 = time.time()
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name)
    print(f"model loaded in {time.time()-t0:.0f}s  dtype={next(model.parameters()).dtype}")
    print("MODEL_CONFIG:", {k: v for k, v in model_config.items() if k not in ('attn_hook_names', 'layer_hook_names')})

    shape_guard(model, model_config)

    save_root = f"{args.save_root}/{args.nickname}_seed{args.seed}"
    tstudy = time.time()
    for task in tasks:
        elapsed_min = (time.time() - tstudy) / 60.0
        if elapsed_min > args.wall_cap_min:
            print(f"[WALL-CAP HIT] {elapsed_min:.1f} min > {args.wall_cap_min} min after task loop; stopping before {task}")
            break
        try:
            run_one(task, args.seed, model, tokenizer, model_config, save_root, args.root_data_dir,
                    n_top_heads=args.n_top_heads, n_mean_activations_trials=args.n_mean,
                    n_indirect_effect_trials=args.n_cie)
        except Exception as e:
            import traceback
            print(f"[ERROR] {task} seed{args.seed}: {e}")
            traceback.print_exc()
    print(f"[STUDY-MODEL DONE] {args.nickname} total {(time.time()-tstudy)/60.0:.1f} min")
