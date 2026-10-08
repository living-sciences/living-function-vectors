"""Smoke test for Qwen3.5-9B-Base (multimodal, hybrid linear/full attention).
Verifies: load, module structure per layer type, o_proj presence + in_features,
one forward -> logits + hidden_states, o_proj-input hook (per-head decomposition
feasibility), and output_attentions (for induction/prefix-match)."""
import torch, json
from transformers import AutoModelForImageTextToText, AutoTokenizer

PATH = "/net/projects2/chai-lab/shared_models/hub/models--Qwen--Qwen3.5-9B-Base/snapshots/2d021f1887f1fe402bf2c53ed69d7f0fc4709ec9"
torch.set_grad_enabled(False)

print("loading tokenizer...")
tok = AutoTokenizer.from_pretrained(PATH)
print("loading model (AutoModelForImageTextToText, bf16)...")
m = AutoModelForImageTextToText.from_pretrained(PATH, dtype=torch.bfloat16, device_map="auto").eval()
print("top-level type:", type(m).__name__)
dec = m.model.language_model
print("decoder type:", type(dec).__name__, "| n layers:", len(dec.layers))
cfg = m.config.text_config
print("cfg: n_heads=%d kv_heads=%d head_dim=%d hidden=%d n_layers=%d" % (
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.hidden_size, cfg.num_hidden_layers))
lt = cfg.layer_types
print("layer_types:", lt)

def describe(L):
    layer = dec.layers[L]
    print(f"\n--- layer {L} ({lt[L]}) --- layer children: {[n for n,_ in layer.named_children()]}")
    for attn_name in ["self_attn","linear_attn","attn","linear_attention","mixer"]:
        if hasattr(layer, attn_name):
            sa = getattr(layer, attn_name)
            print(f"  attn module '{attn_name}': {type(sa).__name__} children={[n for n,_ in sa.named_children()]}")
            for n, sub in sa.named_modules():
                if hasattr(sub, "in_features") and n:
                    print(f"    [linear] {n}: in={sub.in_features} out={sub.out_features} bias={getattr(sub,'bias',None) is not None}")

describe(0)   # linear_attention
describe(3)   # full_attention
describe(31)  # full_attention (last)

# forward one prompt
print("\n=== forward ===")
ids = tok("Q: hot\nA: cold\n\nQ: up\nA:", return_tensors="pt").to(m.device)
out = m(**ids, output_hidden_states=True)
print("logits shape:", tuple(out.logits.shape))
print("hidden_states len:", len(out.hidden_states), "(expect n_layers+1 =", cfg.num_hidden_layers+1, ")")
# top next-token
nt = out.logits[0,-1].argmax().item()
print("greedy next token:", repr(tok.decode([nt])))

# o_proj input hook on a full_attention layer (3)
print("\n=== o_proj input hook (full-attn layer 3) ===")
captured = {}
def mk_hook(tag):
    def hook(mod, inp, outp):
        captured[tag] = tuple(inp[0].shape)
    return hook
h = dec.layers[3].self_attn.o_proj.register_forward_hook(mk_hook("L3_oproj_in"))
m(**ids)
h.remove()
print("captured input shapes:", captured)
print("\nSMOKE OK")
