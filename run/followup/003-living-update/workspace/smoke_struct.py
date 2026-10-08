import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
PATH = "/net/projects2/chai-lab/shared_models/hub/models--Qwen--Qwen3.5-9B-Base/snapshots/2d021f1887f1fe402bf2c53ed69d7f0fc4709ec9"
torch.set_grad_enabled(False)
m = AutoModelForImageTextToText.from_pretrained(PATH, dtype=torch.bfloat16, device_map="auto").eval()
print("top:", type(m).__name__)
print("top children:", [n for n,_ in m.named_children()])
# walk 2 levels
for n, c in m.named_children():
    print(f"  {n}: {type(c).__name__} -> children {[x for x,_ in c.named_children()][:12]}")
# find modules named 'layers' that are ModuleList of decoder layers
import torch.nn as nn
for name, mod in m.named_modules():
    if name.endswith("layers") and isinstance(mod, nn.ModuleList) and len(mod) >= 8:
        print("FOUND layers at:", name, "len", len(mod), "elem0:", type(mod[0]).__name__)
# get_output_embeddings / input
print("get_output_embeddings:", type(m.get_output_embeddings()).__name__ if m.get_output_embeddings() is not None else None)
print("get_input_embeddings:", type(m.get_input_embeddings()).__name__ if m.get_input_embeddings() is not None else None)
