"""Follow-up 003 — Qwen3.5-9B function-vector pipeline via TRANSFORMERS FORWARD HOOKS.

The 2026 Qwen3.5-9B is a MULTIMODAL, HYBRID linear/full-attention model
(Qwen3_5ForConditionalGeneration): 32 decoder layers, of which only 8 are
`full_attention` (indices 3,7,11,15,19,23,27,31, class Qwen3_5Attention with a
standard self_attn.o_proj); the other 24 are `linear_attention`
(Qwen3_5GatedDeltaNet, a Mamba-like mixer with no softmax attention heads).
nnsight cannot wrap it; the pinned baukit pipeline assumes a uniform
`model.layers.{L}.self_attn.o_proj` on every layer. So per the 003 instruction
we REIMPLEMENT the FV head-decomposition with plain transformers forward hooks
on the attention out-projection input inside model.language_model.layers[L],
reshaped to n_heads x head_dim.

Genuine attention heads exist ONLY in the 8 full-attention layers -> the
per-head decomposition (mean-head-activations, CIE, FV construction) runs over
those 8 layers x 16 heads = 128 candidate heads (head_dim 256, 16*256=4096 =
o_proj.in_features -> shape-guard valid). The FV (a residual-stream vector) is
then added at each of the 32 layers to find the peak edit layer, exactly as in
the paper. All prompt construction, dataset splits, filtering, and top-1 rank
accuracy reuse the replication codebase's utils verbatim for comparability with
followup-001. bf16 (native precision), seed 42.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "codebase", "src"))
from utils.prompt_utils import (load_dataset, word_pairs_to_prompt_data, create_prompt,
                                get_token_meta_labels, get_dummy_token_labels,
                                compute_duplicated_labels, update_idx_map)
from utils.eval_utils import get_answer_id, compute_individual_token_rank, compute_top_k_accuracy
from utils.model_utils import set_seed

PATH_BASE = "/net/projects2/chai-lab/shared_models/hub/models--Qwen--Qwen3.5-9B-Base/snapshots/2d021f1887f1fe402bf2c53ed69d7f0fc4709ec9"
PREFIXES = {"input": "Q:", "output": "A:", "instructions": ""}
SEPARATORS = {"input": "\n", "output": "\n\n", "instructions": ""}


def load_qwen35(path=PATH_BASE):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map="auto").eval()
    dec = model.model.language_model
    tcfg = model.config.text_config
    layer_types = list(tcfg.layer_types)
    full_layers = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    mc = {
        "n_heads": tcfg.num_attention_heads,          # 16
        "n_layers": tcfg.num_hidden_layers,           # 32 (for the FV edit-layer sweep)
        "resid_dim": tcfg.hidden_size,                # 4096 (residual/hidden width)
        "head_dim": tcfg.head_dim,                    # 256
        "attn_in_dim": tcfg.num_attention_heads * tcfg.head_dim,  # 4096 = o_proj input width
        "name_or_path": "Qwen/Qwen3.5-9B-Base",
        "prepend_bos": False,                          # Qwen tokenizer adds no bos; eval prepends <|endoftext|>
        "full_attn_layers": full_layers,              # [3,7,11,15,19,23,27,31]
        "n_attn_layers": len(full_layers),            # 8
        "head_dims": [tcfg.head_dim] * len(full_layers),   # uniform 256
        "layer_types": layer_types,
    }
    return model, tok, dec, mc


def load_gemma4(path):
    """gemma-4-12B: Gemma4UnifiedForConditionalGeneration, 48 layers ALL attention
    (full_attention + sliding_attention, both standard o_proj). head_dim 256,
    n_heads 16 -> o_proj input 4096; hidden 3840 (decoupled). BASE model (paper
    convention). Gemma tokenizer prepends BOS -> prepend_bos=True."""
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map="auto").eval()
    dec = model.model.language_model
    tcfg = model.config.text_config
    layer_types = list(tcfg.layer_types)
    n_heads = tcfg.num_attention_heads  # 16, uniform across layers
    # gemma-4 is HETEROGENEOUS in head_dim only: 40 sliding_attention layers
    # (o_proj in 4096 = 16*256) and 8 full/global attention layers (o_proj in
    # 8192 = 16*512). n_heads is uniform (16), so we decompose ALL 48 layers with
    # a PER-LAYER head_dim (hd_L = o_proj.in // n_heads).
    attn_layers = list(range(tcfg.num_hidden_layers))
    head_dims = [dec.layers[L].self_attn.o_proj.in_features // n_heads for L in attn_layers]
    mc = {
        "n_heads": n_heads, "n_layers": tcfg.num_hidden_layers,
        "resid_dim": tcfg.hidden_size, "head_dim": max(head_dims),  # nominal (padded storage)
        "attn_in_dim": None,  # per-layer (n_heads*head_dims[i]); see head_dims
        "name_or_path": "google/gemma-4-12B", "prepend_bos": True,
        "full_attn_layers": attn_layers, "n_attn_layers": len(attn_layers),
        "head_dims": head_dims,
        "layer_types": layer_types,
    }
    return model, tok, dec, mc


def shape_guard(dec, mc):
    """Assert the o_proj-input per-head decomposition is valid on every attention layer.
    Real invariant (per layer): o_proj.in_features == n_heads*head_dim_L (head_dim may be
    DECOUPLED from resid_dim//n_heads, as in Gemma, and may VARY across layers).
    o_proj.out_features == resid_dim on every layer."""
    nh, rd = mc["n_heads"], mc["resid_dim"]
    for i, L in enumerate(mc["full_attn_layers"]):
        hd = mc["head_dims"][i]
        o = dec.layers[L].self_attn.o_proj
        assert o.in_features == nh * hd, f"layer {L}: o_proj.in {o.in_features} != n_heads*head_dim {nh*hd}"
        assert o.out_features == rd, f"layer {L}: o_proj.out {o.out_features} != resid_dim {rd}"
    uniq = sorted(set(mc["head_dims"]))
    print(f"[shape-guard PASS] n_heads={nh} head_dim(s)={uniq} o_proj.out={rd} on {mc['n_attn_layers']} attn layers")


# ---------- prompt helpers (mirror run_fu2 / eval_utils) ----------
def build_prompt(word_pairs, query_pair, shuffle_labels, prepend_bos):
    return word_pairs_to_prompt_data(word_pairs, query_target_pair=query_pair,
                                     prepend_bos_token=prepend_bos, shuffle_labels=shuffle_labels,
                                     prefixes=PREFIXES, separators=SEPARATORS)


@torch.no_grad()
def clean_last_logits(model, tok, sentence):
    inputs = tok(sentence, return_tensors="pt").to(model.device)
    return model(**inputs).logits[:, -1, :]


# ---------- mean head activations (full-attn layers only) ----------
@torch.no_grad()
def get_mean_head_activations(dataset, model, tok, dec, mc, n_icl=10, N=50, filter_set=None):
    nh = mc["n_heads"]
    hds = mc["head_dims"]
    max_hd = max(hds)
    full = mc["full_attn_layers"]
    dummy = get_dummy_token_labels(n_icl, tokenizer=tok, model_config=mc, prefixes=PREFIXES, separators=SEPARATORS)
    store = torch.zeros(N, len(full), nh, len(dummy), max_hd)  # padded to max head_dim
    if filter_set is None:
        filter_set = np.arange(len(dataset["valid"]))
    prepend_bos = not mc["prepend_bos"]

    cap = {}
    def mk(i):
        def hook(mod, inp, out):
            cap[i] = inp[0].detach()  # (1,T, nh*hd_i) input to o_proj
        return hook
    handles = [dec.layers[L].self_attn.o_proj.register_forward_hook(mk(i)) for i, L in enumerate(full)]
    try:
        for n in range(N):
            wp = dataset["train"][np.random.choice(len(dataset["train"]), n_icl, replace=False)]
            wpt = dataset["valid"][np.random.choice(filter_set, 1, replace=False)]
            pd = build_prompt(wp, wpt, False, prepend_bos)
            q = pd["query_target"]["input"]
            token_labels, prompt_string = get_token_meta_labels(pd, tok, q, prepend_bos=mc["prepend_bos"])
            idx_map, idx_avg = compute_duplicated_labels(token_labels, dummy)
            cap.clear()
            model(**tok([prompt_string], return_tensors="pt").to(model.device))
            for i in range(len(full)):
                hd = hds[i]
                x = cap[i].view(1, -1, nh, hd)[0].permute(1, 0, 2)  # (nh,T,hd)
                x_f = x[:, list(idx_map.keys())]                     # (nh, n_dummy, hd)
                for (a, b) in idx_avg.values():
                    x_f[:, idx_map[a]] = x[:, a:b + 1].mean(axis=1)
                store[n, i, :, :, :hd] = x_f.cpu().float()
    finally:
        for h in handles:
            h.remove()
    return store.mean(0)  # (n_attn_layers, nh, n_dummy, max_hd) [padded]


# ---------- CIE over full-attn heads ----------
@torch.no_grad()
def compute_indirect_effect(dataset, mean_act, model, tok, dec, mc, n_shots=10, n_trials=15, filter_set=None):
    nh = mc["n_heads"]
    hds = mc["head_dims"]
    full = mc["full_attn_layers"]
    dummy = get_dummy_token_labels(n_shots, tokenizer=tok, model_config=mc, prefixes=PREFIXES, separators=SEPARATORS)
    if filter_set is None:
        filter_set = np.arange(len(dataset["valid"]))
    prepend_bos = not mc["prepend_bos"]
    ie = torch.zeros(n_trials, len(full), nh)

    # per-head input patch via forward_pre_hook on o_proj (per-layer head_dim)
    state = {"head": None, "vec": None, "hd": None}
    def prehook(mod, args):
        if state["head"] is None:
            return None
        x = args[0].clone()
        xv = x.view(*x.shape[:-1], nh, state["hd"])   # view shares storage with x
        xv[0, -1, state["head"], :] = state["vec"].to(x.dtype).to(x.device)
        return (x,) + tuple(args[1:])

    for t in range(n_trials):
        wp = dataset["train"][np.random.choice(len(dataset["train"]), n_shots, replace=False)]
        wpt = dataset["valid"][np.random.choice(filter_set, 1, replace=False)]
        pd = build_prompt(wp, wpt, True, prepend_bos)  # shuffled-label (corrupted) ICL prompt
        q = pd["query_target"]["input"]
        token_labels, prompt_string = get_token_meta_labels(pd, tok, q, prepend_bos=mc["prepend_bos"])
        tgt = pd["query_target"]["output"]
        tok_id = get_answer_id(prompt_string, tgt, tok)
        tok_id = tok_id[0] if isinstance(tok_id, list) else tok_id
        inputs = tok([prompt_string], return_tensors="pt").to(model.device)
        clean_probs = torch.softmax(model(**inputs).logits[:, -1, :][0], dim=-1)
        for i, L in enumerate(full):
            hd = hds[i]
            state["hd"] = hd
            handle = dec.layers[L].self_attn.o_proj.register_forward_pre_hook(prehook)
            try:
                for h in range(nh):
                    state["head"] = h
                    state["vec"] = mean_act[i, h, -1, :hd]   # trim padding
                    p = torch.softmax(model(**inputs).logits[:, -1, :][0], dim=-1)
                    ie[t, i, h] = (p[tok_id] - clean_probs[tok_id]).item()
            finally:
                state["head"] = None
                handle.remove()
    return ie  # (n_trials, n_attn_layers, nh)


@torch.no_grad()
def compute_function_vector(mean_act, ie, dec, mc, K=10):
    full = mc["full_attn_layers"]
    nh, rd = mc["n_heads"], mc["resid_dim"]
    hds = mc["head_dims"]
    mean_ie = ie.mean(0)  # (n_attn_layers, nh)
    vals, inds = torch.topk(mean_ie.view(-1), k=K, largest=True)
    top = []
    dev = dec.layers[full[0]].self_attn.o_proj.weight.device
    dtype = dec.layers[full[0]].self_attn.o_proj.weight.dtype
    fv = torch.zeros(1, 1, rd, device=dev, dtype=dtype)   # o_proj OUTPUT = residual width
    for v, idx in zip(vals, inds):
        i = int(idx // nh); h = int(idx % nh); L = full[i]; hd = hds[i]
        top.append([int(L), int(h), round(float(v), 5)])  # REAL layer index
        aid = nh * hd                                      # o_proj INPUT width for this layer
        x = torch.zeros(aid, dtype=dtype, device=dev)
        x[h * hd:(h + 1) * hd] = mean_act[i, h, -1, :hd].to(dtype).to(dev)
        fv = fv + dec.layers[L].self_attn.o_proj(x.view(1, 1, aid))
    return fv.view(1, rd), top


# ---------- FV zero/few-shot eval with a residual-stream hook ----------
@torch.no_grad()
def fv_eval(dataset, fv, edit_layer, n_shots, model, tok, dec, mc, filter_set,
            shuffle_labels=False, clean_cache=None):
    """Returns (intervention_top1, baseline_top1, clean_rank_list). If clean_cache
    (list aligned to dataset['test']) is provided, reuse cached clean ranks."""
    prepend_bos = not mc["prepend_bos"]
    fvv = fv.view(-1).to(dtype=next(model.parameters()).dtype)

    state = {"on": False}
    def hook(mod, inp, out):
        if not state["on"]:
            return out
        if isinstance(out, tuple):
            out[0][:, -1, :] = out[0][:, -1, :] + fvv.to(out[0].device)
            return out
        out[:, -1, :] = out[:, -1, :] + fvv.to(out.device)
        return out
    handle = dec.layers[edit_layer].register_forward_hook(hook)

    interv_ranks, clean_ranks = [], []
    try:
        for j in range(len(dataset["test"])):
            if j not in filter_set:
                continue
            if n_shots == 0:
                wp = {"input": [], "output": []}
            else:
                wp = dataset["train"][np.random.choice(len(dataset["train"]), n_shots, replace=False)]
            wpt = dataset["test"][j]
            pd = build_prompt(wp, wpt, shuffle_labels, prepend_bos)
            tgt = pd["query_target"]["output"]
            tgt = tgt[0] if isinstance(tgt, list) else tgt
            sentence = create_prompt(pd)
            tok_id = get_answer_id(sentence, tgt, tok)
            inputs = tok([sentence], return_tensors="pt").to(model.device)
            if clean_cache is not None and clean_cache.get(j) is not None:
                clean_ranks.append(clean_cache[j])
            else:
                state["on"] = False
                clean_logits = model(**inputs).logits[:, -1, :]
                cr = compute_individual_token_rank(clean_logits, tok_id)
                clean_ranks.append(cr)
                if clean_cache is not None:
                    clean_cache[j] = cr
            state["on"] = True
            iv_logits = model(**inputs).logits[:, -1, :]
            state["on"] = False
            interv_ranks.append(compute_individual_token_rank(iv_logits, tok_id))
    finally:
        handle.remove()
    return (float(compute_top_k_accuracy(interv_ranks, 1)),
            float(compute_top_k_accuracy(clean_ranks, 1)),
            clean_ranks)


@torch.no_grad()
def tenshot_filter(dataset, split, model, tok, dec, mc, n_shots=10, seed=42):
    """Indices in `split` the model gets top-1 right with n-shot ICL (no intervention)."""
    prepend_bos = not mc["prepend_bos"]
    ranks = []
    for j in range(len(dataset[split])):
        wp = dataset["train"][np.random.choice(len(dataset["train"]), n_shots, replace=False)]
        wpt = dataset[split][j]
        pd = build_prompt(wp, wpt, False, prepend_bos)
        tgt = pd["query_target"]["output"]; tgt = tgt[0] if isinstance(tgt, list) else tgt
        sentence = create_prompt(pd)
        tok_id = get_answer_id(sentence, tgt, tok)
        logits = clean_last_logits(model, tok, [sentence])
        ranks.append(compute_individual_token_rank(logits, tok_id))
    return np.where(np.array(ranks) == 0)[0]


def run_task(task, model, tok, dec, mc, save_root, root_data, K, n_mean, n_cie, seed=42, eval_cap=None):
    spr = f"{save_root}/{task}"
    os.makedirs(spr, exist_ok=True)
    done = f"{spr}/fu2_summary.json"
    if os.path.exists(done):
        print(f"[skip] {task} done"); return json.load(open(done))
    t0 = time.time()
    set_seed(seed)
    dataset = load_dataset(task, root_data_dir=root_data, test_size=0.3, seed=seed)

    set_seed(seed + 42)
    filter_val = tenshot_filter(dataset, "valid", model, tok, dec, mc)
    set_seed(seed)
    filter_test = tenshot_filter(dataset, "test", model, tok, dec, mc)
    if eval_cap is not None and len(filter_test) > eval_cap:
        filter_test = filter_test[:eval_cap]
    print(f"  filter_test={len(filter_test)} filter_val={len(filter_val)}")

    ma_path = f"{spr}/mean_head_activations.pt"
    if os.path.exists(ma_path):
        mean_act = torch.load(ma_path)
    else:
        set_seed(seed)
        mean_act = get_mean_head_activations(dataset, model, tok, dec, mc, N=n_mean, filter_set=filter_val)
        torch.save(mean_act, ma_path)
    ie_path = f"{spr}/indirect_effect.pt"
    if os.path.exists(ie_path):
        ie = torch.load(ie_path)
    else:
        tc = time.time(); set_seed(seed)
        ie = compute_indirect_effect(dataset, mean_act, model, tok, dec, mc, n_trials=n_cie, filter_set=filter_val)
        torch.save(ie, ie_path); print(f"  CIE done {(time.time()-tc)/60:.1f} min")

    fv, top_heads = compute_function_vector(mean_act, ie, dec, mc, K=K)

    n_layers = mc["n_layers"]
    zs = {}
    clean_cache = {}
    tsw = time.time()
    for L in range(n_layers):
        set_seed(seed)
        iv, base, _ = fv_eval(dataset, fv, L, 0, model, tok, dec, mc, filter_test, clean_cache=clean_cache)
        zs[L] = {"fv": iv, "base": base}
    json.dump(zs, open(f"{spr}/zs_layer_sweep.json", "w"), indent=2)
    peak = max(range(n_layers), key=lambda L: zs[L]["fv"])
    print(f"  ZS sweep {(time.time()-tsw)/60:.1f} min; peak L{peak} FV={zs[peak]['fv']:.3f} base={zs[peak]['base']:.3f}")

    set_seed(seed)
    sl_iv, sl_base, _ = fv_eval(dataset, fv, peak, 10, model, tok, dec, mc, filter_test, shuffle_labels=True)

    canon = n_layers // 3
    summary = {
        "task": task, "seed": seed, "n_layers": n_layers, "n_heads": mc["n_heads"],
        "n_attn_layers": mc["n_attn_layers"], "full_attn_layers": mc["full_attn_layers"],
        "n_top_heads": K, "peak_layer": int(peak), "peak_frac_depth": round(peak / n_layers, 3),
        "zs_fv_peak": zs[peak]["fv"], "zs_baseline_peak": zs[peak]["base"],
        "canonical_layer": int(canon), "zs_fv_canonical": zs[canon]["fv"], "zs_baseline_canonical": zs[canon]["base"],
        "sl_fv_at_peak": sl_iv, "sl_baseline_at_peak": sl_base,
        "top_heads": top_heads,
        "filter_test": int(len(filter_test)), "filter_val": int(len(filter_val)),
        "minutes": round((time.time() - t0) / 60, 1), "dtype": "torch.bfloat16",
    }
    json.dump(summary, open(done, "w"), indent=2)
    print(f"[done] {task} {summary['minutes']}min peak L{peak}({summary['peak_frac_depth']}) "
          f"ZS FV={summary['zs_fv_peak']:.3f} base={summary['zs_baseline_peak']:.3f} SL={summary['sl_fv_at_peak']:.3f}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="antonym,country-capital,present-past")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--n_mean", type=int, default=50)
    ap.add_argument("--n_cie", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_cap", type=int, default=None)
    ap.add_argument("--save_root", default=None)
    ap.add_argument("--model", default="qwen35", choices=["qwen35", "gemma4"])
    ap.add_argument("--path", default=None)
    ap.add_argument("--nick", default=None)
    ap.add_argument("--root_data", default=os.path.join(os.path.dirname(__file__), "codebase", "dataset_files"))
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    print(f"loading {args.model} ...")
    t0 = time.time()
    if args.model == "qwen35":
        model, tok, dec, mc = load_qwen35(args.path or PATH_BASE)
        nick = args.nick or "qwen35-9b"
    else:
        assert args.path, "--path required for gemma4"
        model, tok, dec, mc = load_gemma4(args.path)
        nick = args.nick or "gemma4-12b"
    print(f"loaded in {time.time()-t0:.0f}s; n_layers={mc['n_layers']} attn_layers={mc['n_attn_layers']}")
    shape_guard(dec, mc)
    save_root = args.save_root or os.path.join(os.path.dirname(__file__), "results_fu", f"{nick}_seed{args.seed}")
    os.makedirs(save_root, exist_ok=True)

    tstudy = time.time()
    for task in args.tasks.split(","):
        try:
            run_task(task, model, tok, dec, mc, save_root, args.root_data, args.K, args.n_mean, args.n_cie,
                     seed=args.seed, eval_cap=args.eval_cap)
        except Exception as e:
            import traceback; print(f"[ERROR] {task}: {e}"); traceback.print_exc()
    print(f"[MODEL DONE] {nick} total {(time.time()-tstudy)/60:.1f} min")
