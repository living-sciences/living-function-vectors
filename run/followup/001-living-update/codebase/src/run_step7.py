"""Step 7: logit-lens decode of the function vector D(v_t) for the 6 core tasks (GPT-J).
Follows notebooks/fv_demo.ipynb decode_to_vocab usage."""
import torch
from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from utils.prompt_utils import load_dataset
from utils.extract_utils import get_mean_head_activations, compute_universal_function_vector
from utils.eval_utils import decode_to_vocab

torch.set_grad_enabled(False)
model, tok, cfg = load_gpt_model_and_tokenizer('EleutherAI/gpt-j-6b')
dec = torch.nn.Sequential(model.transformer.ln_f, model.lm_head)

tasks = ['antonym','capitalize','country-capital','english-french','present-past','singular-plural']
for d in tasks:
    set_seed(0)
    ds = load_dataset(d, seed=0)
    ma = get_mean_head_activations(ds, model, cfg, tok)
    fv, _ = compute_universal_function_vector(ma, model, cfg)
    top = decode_to_vocab(dec(fv), tok, k=10)  # decode_to_vocab softmaxes internally; pass logits
    print(f"{d}: {top}")
