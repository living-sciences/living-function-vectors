import torch, time
from transformers import AutoModelForImageTextToText, AutoTokenizer
PATH = "/net/projects2/chai-lab/shared_models/hub/models--Qwen--Qwen3.5-9B-Base/snapshots/2d021f1887f1fe402bf2c53ed69d7f0fc4709ec9"
torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained(PATH)
m = AutoModelForImageTextToText.from_pretrained(PATH, dtype=torch.bfloat16, device_map="auto").eval()
dec = m.model.language_model

# tokenizer behavior
print("=== tokenizer ===")
print("add_bos_token attr:", getattr(tok, "add_bos_token", "n/a"), "| bos:", tok.bos_token, "| eos:", tok.eos_token)
ids_plain = tok("Q: up\nA:", return_tensors="pt").input_ids
print("plain ids[:6]:", ids_plain[0,:6].tolist(), "-> decoded first:", repr(tok.decode(ids_plain[0,:1])))
ids_eot = tok("<|endoftext|>Q: up\nA:", return_tensors="pt").input_ids
print("<|endoftext|> prefixed len vs plain:", ids_eot.shape[1], ids_plain.shape[1], "| first tok:", repr(tok.decode(ids_eot[0,:1])), ids_eot[0,0].item())

# residual add hook: inspect decoder layer output type
print("\n=== decoder layer output type ===")
cap = {}
def outhook(mod, inp, out):
    cap['type'] = type(out).__name__
    cap['is_tuple'] = isinstance(out, tuple)
    if isinstance(out, tuple):
        cap['elem0shape'] = tuple(out[0].shape); cap['len'] = len(out)
    else:
        cap['shape'] = tuple(out.shape)
h = dec.layers[5].register_forward_hook(outhook)
ids = tok("Q: hot\nA: cold\n\nQ: up\nA:", return_tensors="pt").to(m.device)
m(**ids)
h.remove()
print("layer5 output:", cap)

# timing: single forward
for _ in range(2): m(**ids)  # warmup
torch.cuda.synchronize(); t=time.time()
N=20
for _ in range(N): m(**ids)
torch.cuda.synchronize()
print(f"\n=== timing === {(time.time()-t)/N*1000:.1f} ms/forward (seq len {ids.input_ids.shape[1]})")
