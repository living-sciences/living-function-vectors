import os, time
from huggingface_hub import snapshot_download
t0 = time.time()
p = snapshot_download('EleutherAI/gpt-neox-20b',
                      allow_patterns=['*.json', '*.txt', '*.safetensors', 'tokenizer*'],
                      max_workers=8)
print("neox downloaded to", p, "in", round(time.time()-t0), "s")
