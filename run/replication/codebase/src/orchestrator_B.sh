#!/bin/bash
# Remaining GPT-J steps across GPU1 and GPU3, parallel with step2 seeds1,2 (GPU0,2).
set -x
RL=/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/function-vectors/run/replication
export TQDM_DISABLE=1 HF_HOME=/net/projects2/chai-lab/shared_models HF_HUB_OFFLINE=1

# ---------- Stream A: GPU1 = step3 (avg hidden state) then step10 (numheads) ----------
(
  export CUDA_VISIBLE_DEVICES=1
  FV_FP16=1 python run_step3.py --save_root results/gptj_avg_hs --n_seeds 3 --half > "$RL/step3.log" 2>&1
  for D in antonym capitalize country-capital english-french present-past singular-plural; do
    FV_FP16=1 python test_numheads.py --dataset_name="$D" --model_name='EleutherAI/gpt-j-6b' --model_nickname=gptj \
       --n_heads=40 --edit_layer=9 --save_path_root=results --mean_act_root=gptj_seed42 >> "$RL/step10.log" 2>&1
  done
) &
SA=$!

# ---------- Stream B: GPU3 = step9 (portability) then step8 (nattext) then step6 (vocab fp32) ----------
(
  export CUDA_VISIBLE_DEVICES=3
  for D in antonym capitalize country-capital english-french present-past singular-plural; do
    FV_FP16=1 python portability_eval.py --dataset_name="$D" --n_eval_templates=20 --edit_layer=9 \
       --model_name='EleutherAI/gpt-j-6b' --save_path_root='results/gptj' >> "$RL/step9.log" 2>&1
  done
  FV_FP16=1 python natural_text_eval.py --model_name='EleutherAI/gpt-j-6b' --edit_layer=9 --n_seeds 3 \
       --save_path_root='results/nattext' > "$RL/step8.log" 2>&1
  # step6 vocab reconstruction needs fp32 (gradient optimization); FV_FP16 unset
  python vocab_reconstruction.py --model_name='EleutherAI/gpt-j-6b' --n_seeds=3 --save_path_root='results/vocab_recon' > "$RL/step6.log" 2>&1
) &
SB=$!

wait $SA; echo "STREAM A (step3,step10) done exit $?"
wait $SB; echo "STREAM B (step9,step8,step6) done exit $?"
echo "ORCHESTRATOR B COMPLETE"
