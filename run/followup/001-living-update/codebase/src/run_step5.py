"""Step 5: head-localization (AIE map) + induction-head prefix-matching comparison."""
import glob, json
import torch, numpy as np
from utils.model_utils import load_gpt_model_and_tokenizer
from utils.extract_utils import prefix_matching_score

ie_files = sorted(glob.glob('results/gptj_seed42/*/*_indirect_effect.pt'))
print("IE files:", ie_files)
ies = [torch.load(f) for f in ie_files]
aie = torch.stack([ie.mean(0) for ie in ies]).mean(0)  # mean over trials, then over 6 tasks -> (28,16)
vals, inds = torch.topk(aie.view(-1), 10)
top10 = [tuple(int(x) for x in np.unravel_index(int(i), aie.shape)) + (round(float(v), 4),)
         for i, v in zip(inds, vals)]
print("Top-10 highest-AIE (layer,head,score):")
for t in top10:
    print("  ", t)
layers = [t[0] for t in top10]
print(f"layer indices of top-10: {layers}  (min {min(layers)}, max {max(layers)}, mean {np.mean(layers):.1f}); "
      f"GPT-J has 28 layers")

torch.set_grad_enabled(False)
model, tok, cfg = load_gpt_model_and_tokenizer('EleutherAI/gpt-j-6b')
model = model.half()
torch.manual_seed(0)
pms = prefix_matching_score(model, cfg)  # returns (n_heads, n_layers) due to a .T in the repo
print("pms shape", tuple(pms.shape))

flagged = [(8, 1), (12, 10), (24, 6)]  # (layer, head)
flagged_scores = {f"{l}-{h}": round(float(pms[h, l]), 4) for l, h in flagged}
top10_pms = {f"{t[0]}-{t[1]}": round(float(pms[t[1], t[0]]), 4) for t in top10}
mean_pms = float(pms.mean())
print("prefix-matching scores for flagged induction heads:", flagged_scores)
print("prefix-matching scores for the top-10 AIE heads:", top10_pms)
print(f"mean prefix-matching over all heads: {mean_pms:.4f}")

out = {"top10_AIE": top10, "top10_layer_indices": layers,
       "flagged_induction_head_pms": flagged_scores, "top10_head_pms": top10_pms,
       "mean_pms_all_heads": mean_pms}
with open('results/head_localization.txt', 'w') as f:
    f.write(json.dumps(out, indent=2))
print("saved results/head_localization.txt")
