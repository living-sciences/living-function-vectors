import torch, glob
from transformers import AutoModelForImageTextToText, AutoTokenizer
PATH = glob.glob("/net/projects2/chai-lab-models/haokunliu/alignment-batch/hf-cache/hub/models--google--gemma-4-12B/snapshots/*/")[0]
torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained(PATH)
print("bos:", tok.bos_token, "add_bos:", getattr(tok,"add_bos_token","n/a"), "| eos:", tok.eos_token)
ids = tok("Q: up\nA:", return_tensors="pt")
print("first tok:", repr(tok.decode(ids.input_ids[0,:1])), ids.input_ids[0,0].item(), "| len", ids.input_ids.shape[1])
m = AutoModelForImageTextToText.from_pretrained(PATH, dtype=torch.bfloat16, device_map="auto").eval()
print("top:", type(m).__name__, "children:", [n for n,_ in m.named_children()])
import torch.nn as nn
layers_path=None
for name, mod in m.named_modules():
    if name.endswith("layers") and isinstance(mod, nn.ModuleList) and len(mod)>=40:
        layers_path=name; layers=mod; break
print("layers at:", layers_path, "len", len(layers))
# check o_proj on a few layers
for L in [0,1,47]:
    sa = layers[L].self_attn
    o = sa.o_proj
    print(f"  L{L}: self_attn={type(sa).__name__} o_proj in={o.in_features} out={o.out_features} bias={o.bias is not None}")
dec = m.model.language_model
tc = m.config.text_config
print("cfg n_heads=%d head_dim=%d hidden=%d n_layers=%d"%(tc.num_attention_heads,tc.head_dim,tc.hidden_size,tc.num_hidden_layers))
# forward
d = tok("Q: hot\nA: cold\n\nQ: up\nA:", return_tensors="pt").to(m.device)
out = m(**d, output_hidden_states=True)
print("logits", tuple(out.logits.shape), "hs len", len(out.hidden_states), "greedy:", repr(tok.decode([out.logits[0,-1].argmax().item()])))
# o_proj input hook L1 + decoder layer output type
cap={}
def h1(mod,inp,o): cap['oin']=tuple(inp[0].shape)
def h2(mod,inp,o): cap['ltype']='tuple' if isinstance(o,tuple) else 'tensor'; cap['oshape']=tuple((o[0] if isinstance(o,tuple) else o).shape)
hh=[dec.layers[1].self_attn.o_proj.register_forward_hook(h1), dec.layers[1].register_forward_hook(h2)]
m(**d)
for x in hh: x.remove()
print("o_proj input shape:", cap.get('oin'), "| layer out:", cap.get('ltype'), cap.get('oshape'))
print("SMOKE OK")
