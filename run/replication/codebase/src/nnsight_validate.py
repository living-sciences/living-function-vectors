"""Directive compliance: implement FV activation-extraction and the FV-add
intervention with nnsight, and validate they reproduce the baukit path on antonym.

- Extraction: read model.transformer.h[L].attn.out_proj.input (per directive).
- Intervention: add FV to model.transformer.h[edit_layer].output[0][:, -1] inside a trace.

We compare (a) per-head mean activations extracted by nnsight vs baukit on identical
prompts, and (b) zero-shot FV top-1 accuracy at layer 9 on antonym (nnsight vs baukit).
"""
import os, sys, json, time
import torch, numpy as np

from utils.prompt_utils import *
from utils.model_utils import *
from utils.eval_utils import *
from utils.extract_utils import get_mean_head_activations, compute_function_vector
from compute_indirect_effect import compute_indirect_effect

from nnsight import LanguageModel

DATASET = 'antonym'
SEED = 42
EDIT_LAYER = 9
PREF = {"input": "Q:", "output": "A:", "instructions": ""}
SEP = {"input": "\n", "output": "\n\n", "instructions": ""}


def split_by_head(x, cfg):
    ns = x.size()[:-1] + (cfg['n_heads'], cfg['resid_dim'] // cfg['n_heads'])
    return x.view(*ns)


def nnsight_gather_head_acts(nnmodel, prompt_string, dummy_labels, tokenizer, cfg, token_labels):
    """Replicates gather_attn_activations using nnsight; returns filtered (heads,tok,dim)."""
    idx_map, idx_avg = compute_duplicated_labels(token_labels, dummy_labels)
    saved = []
    with nnmodel.trace(prompt_string):
        for l in range(cfg['n_layers']):
            saved.append(nnmodel.transformer.h[l].attn.out_proj.input.save())
    # each saved[l]: (1, seq, resid) -> split heads -> (heads, seq, dim)
    stacks = []
    for l in range(cfg['n_layers']):
        a = saved[l]
        if isinstance(a, tuple):
            a = a[0]
        a = a.detach().float().cpu()          # (1, seq, resid)
        a = split_by_head(a, cfg)             # (1, seq, heads, dim)
        stacks.append(a.permute(0, 2, 1, 3))  # (1, heads, seq, dim)
    stack_initial = torch.vstack(stacks)      # (layers, heads, seq, dim)
    stack_filtered = stack_initial[:, :, list(idx_map.keys())]
    for (i, j) in idx_avg.values():
        stack_filtered[:, :, idx_map[i]] = stack_initial[:, :, i:j+1].mean(axis=2)
    return stack_filtered


def nnsight_mean_head_activations(dataset, nnmodel, cfg, tokenizer, n_icl=10, N_TRIALS=100, filter_set=None):
    dummy_labels = get_dummy_token_labels(n_icl, tokenizer=tokenizer, prefixes=PREF, separators=SEP, model_config=cfg)
    storage = torch.zeros(N_TRIALS, cfg['n_layers'], cfg['n_heads'], len(dummy_labels), cfg['resid_dim']//cfg['n_heads'])
    if filter_set is None:
        filter_set = np.arange(len(dataset['valid']))
    prepend_bos = False if cfg['prepend_bos'] else True
    for n in range(N_TRIALS):
        word_pairs = dataset['train'][np.random.choice(len(dataset['train']), n_icl, replace=False)]
        wpt = dataset['valid'][np.random.choice(filter_set, 1, replace=False)]
        prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=wpt, prepend_bos_token=prepend_bos,
                                                shuffle_labels=False, prefixes=PREF, separators=SEP)
        query = prompt_data['query_target']['input']
        token_labels, prompt_string = get_token_meta_labels(prompt_data, tokenizer, query, prepend_bos=cfg['prepend_bos'])
        storage[n] = nnsight_gather_head_acts(nnmodel, prompt_string, dummy_labels, tokenizer, cfg, token_labels)
    return storage.mean(dim=0)


def nnsight_fv_zshot_eval(dataset, fv, nnmodel, cfg, tokenizer, filter_set, edit_layer):
    """Zero-shot eval with FV added via nnsight; returns top-1 acc (intervention & clean)."""
    prepend_bos = False if cfg['prepend_bos'] else True
    clean_ranks, interv_ranks = [], []
    fv = fv.reshape(-1).to(nnmodel.device)
    for j in range(len(dataset['test'])):
        if j not in filter_set:
            continue
        word_pairs = {'input': [], 'output': []}
        wpt = dataset['test'][j]
        prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=wpt, prepend_bos_token=prepend_bos,
                                                shuffle_labels=False, prefixes=PREF, separators=SEP)
        target = prompt_data['query_target']['output']
        target = target[0] if isinstance(target, list) else target
        sentence = create_prompt(prompt_data)
        target_id = get_answer_id(sentence, target, tokenizer)
        with nnmodel.trace(sentence):
            clean_logits = nnmodel.lm_head.output[:, -1, :].save()
        with nnmodel.trace(sentence):
            nnmodel.transformer.h[edit_layer].output[0][:, -1] += fv
            interv_logits = nnmodel.lm_head.output[:, -1, :].save()
        clean_ranks.append(compute_individual_token_rank(clean_logits.detach().float().cpu(), target_id))
        interv_ranks.append(compute_individual_token_rank(interv_logits.detach().float().cpu(), target_id))
    c1 = compute_top_k_accuracy(clean_ranks, 1)
    i1 = compute_top_k_accuracy(interv_ranks, 1)
    return float(i1), float(c1)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    out = {}
    print("Loading HF (baukit) model...")
    model, tokenizer, cfg = load_gpt_model_and_tokenizer('EleutherAI/gpt-j-6b')
    model = model.half()

    set_seed(SEED)
    dataset = load_dataset(DATASET, root_data_dir='../dataset_files', test_size=0.3, seed=SEED)
    set_seed(SEED + 42)
    fsv = n_shot_eval_no_intervention(dataset=dataset, n_shots=10, model=model, model_config=cfg, tokenizer=tokenizer,
                                      compute_ppl=True, test_split='valid', prefixes=PREF, separators=SEP)
    filter_set_validation = np.where(np.array(fsv['clean_rank_list']) == 0)[0]
    set_seed(SEED)
    fst = n_shot_eval_no_intervention(dataset=dataset, n_shots=10, model=model, model_config=cfg, tokenizer=tokenizer,
                                      compute_ppl=True, prefixes=PREF, separators=SEP)
    filter_set = np.where(np.array(fst['clean_rank_list']) == 0)[0]

    # ---- baukit mean activations + FV (reuse cached step-2 artifacts if present) ----
    cache = 'results/gptj_seed42/antonym'
    set_seed(SEED)
    ma_bk = get_mean_head_activations(dataset, model=model, model_config=cfg, tokenizer=tokenizer, n_icl_examples=10,
                                      N_TRIALS=100, prefixes=PREF, separators=SEP, filter_set=filter_set_validation)
    if os.path.exists(f'{cache}/antonym_indirect_effect.pt'):
        ie = torch.load(f'{cache}/antonym_indirect_effect.pt')
    else:
        set_seed(SEED)
        ie = compute_indirect_effect(dataset, ma_bk, model=model, model_config=cfg, tokenizer=tokenizer, n_shots=10,
                                     n_trials=25, last_token_only=True, prefixes=PREF, separators=SEP,
                                     filter_set=filter_set_validation)
    fv_bk, top_heads = compute_function_vector(ma_bk, ie, model, model_config=cfg, n_top_heads=10)

    # ---- nnsight model ----
    print("Loading nnsight model...")
    nnmodel = LanguageModel('EleutherAI/gpt-j-6b', device_map='cuda', torch_dtype=torch.float16, dispatch=True)

    # (a) extraction match: same seed => same prompts
    set_seed(SEED)
    ma_nn = nnsight_mean_head_activations(dataset, nnmodel, cfg, tokenizer, n_icl=10, N_TRIALS=100,
                                          filter_set=filter_set_validation)
    diff = (ma_nn - ma_bk).abs()
    rel = diff.mean().item() / (ma_bk.abs().mean().item() + 1e-9)
    out['extraction_mean_abs_diff'] = diff.mean().item()
    out['extraction_max_abs_diff'] = diff.max().item()
    out['extraction_rel_diff'] = rel
    out['baukit_mean_abs'] = ma_bk.abs().mean().item()
    print(f"extraction mean|diff|={diff.mean().item():.3e} max={diff.max().item():.3e} rel={rel:.3e}")

    # build nnsight FV from nnsight-extracted activations + shared IE, and also from baukit acts
    fv_nn, _ = compute_function_vector(ma_nn, ie, model, model_config=cfg, n_top_heads=10)

    # (b) zero-shot FV top-1 accuracy at layer 9: nnsight intervention vs baukit intervention
    set_seed(SEED)
    zs_bk = n_shot_eval(dataset=dataset, fv_vector=fv_bk, edit_layer=EDIT_LAYER, n_shots=0, model=model,
                        model_config=cfg, tokenizer=tokenizer, filter_set=filter_set, prefixes=PREF, separators=SEP)
    bk_i1 = zs_bk['intervention_topk'][0][1]
    bk_c1 = zs_bk['clean_topk'][0][1]

    set_seed(SEED)
    nn_i1, nn_c1 = nnsight_fv_zshot_eval(dataset, fv_nn, nnmodel, cfg, tokenizer, filter_set, EDIT_LAYER)

    out['top_heads_baukit'] = [[int(a), int(b), float(c)] for a, b, c in top_heads]
    out['antonym_L9_zs_top1_baukit'] = float(bk_i1)
    out['antonym_L9_clean_top1_baukit'] = float(bk_c1)
    out['antonym_L9_zs_top1_nnsight'] = float(nn_i1)
    out['antonym_L9_clean_top1_nnsight'] = float(nn_c1)
    out['filter_set_size'] = int(len(filter_set))
    print(json.dumps(out, indent=2))
    with open('results/nnsight_validation.json', 'w') as f:
        json.dump(out, f, indent=2)
