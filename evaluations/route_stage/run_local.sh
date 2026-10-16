#!/usr/bin/env bash
# v24 routing stage on a machine without SLURM (see RUNBOOK.md). With the pred-distr env active:
#   bash evaluations/route_stage/run_local.sh smoke | diag | mc | train | report | all
# Env: THREADS (torch threads per process, 8), JOBS (trainings in parallel, 1), DEVICE (cpu | cuda, cpu),
#      EPOCHS (40), SEEDS ("5 6 7"), MC (Poisson samples per schedule, 24), GPUS ("0 1": trainings spread round-robin)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
export PYTHONPATH="$HERE/../../src${PYTHONPATH:+:$PYTHONPATH}"
THREADS=${THREADS:-8}; JOBS=${JOBS:-1}; EP_TASKS=${EP_TASKS:-4}; DEVICE=${DEVICE:-cpu}; EPOCHS=${EPOCHS:-40}; SEEDS=${SEEDS:-"5 6 7"}; MC=${MC:-24}
LAB=../compile_check/results/esp_labels.pt
SPLITS="qaoa variational control loso:dense,complete"
LOGS=$HERE/local_logs
mkdir -p "$LOGS"
name() { local s=${1//:/_}; echo "${s//,/_}"; }        # loso:dense,complete -> loso_dense_complete

smoke() {   # ~10 min: every code path on a few circuits
    python3 -u diag_scale.py --split qaoa --ckpt results/v24_qaoa/model.pt --out results/diag_scale_smoke.json \
        --n 10 --n-train 5 --K 4 --threads "$THREADS" 2>&1 | tee "$LOGS/smoke_diag.log"
    python3 -u stage1.py eval --split qaoa --ckpt results/v24_qaoa/model.pt --out results/smoke --max-test 20 \
        --mc 4 --esp-labels "$LAB" --device "$DEVICE" --threads "$THREADS" 2>&1 | tee "$LOGS/smoke_eval.log"
}

diag() {    # why stage 1 under-estimates routed durations (seed-5 checkpoints, no training)
    for s in qaoa variational control; do
        python3 -u diag_scale.py --split $s --ckpt results/v24_$s/model.pt --out results/diag_scale_$s.json \
            --n 800 --n-train 300 --K 24 --threads "$THREADS" > "$LOGS/diag_$s.log" 2>&1
        echo "diag $s done"
    done
}

mc() {      # seed-5 checkpoints re-evaluated: sampled SWAPs + dataset target
    for s in $SPLITS; do
        python3 -u stage1.py eval --split $s --ckpt results/v24_$(name $s)/model.pt --out results/v24_$(name $s) \
            --mc "$MC" --esp-labels "$LAB" --tag _mc --device "$DEVICE" --threads "$THREADS" > "$LOGS/mc_$(name $s).log" 2>&1
        echo "mc $s done"
    done
}

train() {   # v24b = v24 with EPOCHS epochs, 3 seeds, evaluated with MC and on the dataset target
    : > "$LOGS/train_cmds.txt"
    local gpus=(${GPUS:-}) i=0 pre=""
    for seed in $SEEDS; do for s in $SPLITS; do
        out=results/v24b_$(name $s)_s$seed
        [ -f "$out/eval_$(name $s | sed 's/_dense_complete/_dense-complete/').json" ] && continue    # done already
        [ ${#gpus[@]} -gt 0 ] && pre="CUDA_VISIBLE_DEVICES=${gpus[i++ % ${#gpus[@]}]} "
        echo "${pre}python3 -u stage1.py train --split $s --epochs $EPOCHS --seed $seed --out $out --mc $MC --esp-labels $LAB --device $DEVICE --threads $THREADS > $LOGS/train_$(name $s)_s$seed.log 2>&1 && echo 'train $s s$seed done'" >> "$LOGS/train_cmds.txt"
    done; done
    xargs -P "$JOBS" -I{} bash -c '{}' < "$LOGS/train_cmds.txt"
}

train25() { # v25 = v24b + loss on the routed duration through the differentiable schedule (--dur-lambda), same seeds / splits
    : > "$LOGS/train25_cmds.txt"
    local gpus=(${GPUS:-}) i=0 pre=""
    for seed in $SEEDS; do for s in $SPLITS; do
        out=results/v25_$(name $s)_s$seed
        [ -f "$out/eval_$(name $s | sed 's/_dense_complete/_dense-complete/').json" ] && continue    # done already
        [ ${#gpus[@]} -gt 0 ] && pre="CUDA_VISIBLE_DEVICES=${gpus[i++ % ${#gpus[@]}]} "
        echo "${pre}python3 -u stage1.py train --split $s --epochs $EPOCHS --seed $seed --out $out --dur-lambda ${DUR_LAMBDA:-1} --dur-every ${DUR_EVERY:-1} --c2-target ${C2_TARGET:-sabre} --mc $MC --esp-labels $LAB --device $DEVICE --threads $THREADS > $LOGS/train25_$(name $s)_s$seed.log 2>&1 && echo 'train25 $s s$seed done'" >> "$LOGS/train25_cmds.txt"
    done; done
    xargs -P "$JOBS" -I{} bash -c '{}' < "$LOGS/train25_cmds.txt"
}

train26() { # v26 = v25 + contention: our greedy router's wait per gate as input and as the base of the wait (--contention)
    : > "$LOGS/train26_cmds.txt"
    local gpus=(${GPUS:-}) i=0 pre=""
    for seed in $SEEDS; do for s in $SPLITS; do
        out=results/v26_$(name $s)_s$seed
        [ -f "$out/eval_$(name $s | sed 's/_dense_complete/_dense-complete/').json" ] && continue    # done already
        [ ${#gpus[@]} -gt 0 ] && pre="CUDA_VISIBLE_DEVICES=${gpus[i++ % ${#gpus[@]}]} "
        echo "${pre}python3 -u stage1.py train --split $s --epochs $EPOCHS --seed $seed --out $out --dur-lambda ${DUR_LAMBDA:-1} --dur-every ${DUR_EVERY:-1} --c2-target ${C2_TARGET:-sabre} --contention --mc $MC --esp-labels $LAB --device $DEVICE --threads $THREADS > $LOGS/train26_$(name $s)_s$seed.log 2>&1 && echo 'train26 $s s$seed done'" >> "$LOGS/train26_cmds.txt"
    done; done
    xargs -P "$JOBS" -I{} bash -c '{}' < "$LOGS/train26_cmds.txt"
}

contention() { # once, before train26 (it is cached in results/contention.pt; trainings in parallel would all compute it)
    python3 -u -c "import stage1 as S; from pathlib import Path; p = Path('results/stage1.pt'); S.load_all(p, True); \
print('contention cached:', (p.parent / 'contention.pt').is_file())" 2>&1 | tee "$LOGS/contention.log"
}

smoke26() { # ~5 min after `contention`: 2 epochs of v26 on qaoa (prints the router-vs-Sabre wait correlation)
    python3 -u stage1.py train --split qaoa --epochs 2 --out results/smoke26 --dur-lambda 1 --contention --max-test 20 --mc 4 \
        --esp-labels "$LAB" --device "$DEVICE" --threads "$THREADS" 2>&1 | tee "$LOGS/smoke26.log"
}

synth2() {  # CPU, once: the low-parallelism synthetic class (layered / seqdense), Sabre labels, stage-1 items
    local n=${N_SYNTH2:-6000} j=${JOBS:-8} pids=()
    for i in $(seq 0 $((j - 1))); do python3 -u ../synthetic_routing/synth_layered.py gen --task-id $i --n-tasks $j --n-circuits $n > "$LOGS/synth2_gen_$i.log" 2>&1 & pids+=($!); done
    for p in "${pids[@]}"; do wait "$p"; done; pids=()
    python3 -u ../synthetic_routing/synth_layered.py gen --merge | tee "$LOGS/synth2_merge.log"
    for i in $(seq 0 $((j - 1))); do python3 -u ../synthetic_routing/synth_layered.py sabre --task-id $i --n-tasks $j > "$LOGS/synth2_sabre_$i.log" 2>&1 & pids+=($!); done
    for p in "${pids[@]}"; do wait "$p"; done
    python3 -u ../synthetic_routing/synth_layered.py prep | tee "$LOGS/synth2_prep.log"
}

train27() { # v27 = v25 + the synth2 circuits on the training side; + split loso:layered,seqdense (the new class held out)
    : > "$LOGS/train27_cmds.txt"
    local gpus=(${GPUS:-}) i=0 pre=""
    for seed in $SEEDS; do for s in $SPLITS loso:layered,seqdense; do
        out=results/v27_$(name $s)_s$seed
        [ -f "$out/eval_$(name $s | sed 's/_dense_complete/_dense-complete/; s/_layered_seqdense/_layered-seqdense/').json" ] && continue
        [ ${#gpus[@]} -gt 0 ] && pre="CUDA_VISIBLE_DEVICES=${gpus[i++ % ${#gpus[@]}]} "
        echo "${pre}python3 -u stage1.py train --split $s --epochs $EPOCHS --seed $seed --out $out --dur-lambda ${DUR_LAMBDA:-1} --dur-every ${DUR_EVERY:-1} --c2-target ${C2_TARGET:-sabre} --synth2 --mc $MC --esp-labels $LAB --device $DEVICE --threads $THREADS > $LOGS/train27_$(name $s)_s$seed.log 2>&1 && echo 'train27 $s s$seed done'" >> "$LOGS/train27_cmds.txt"
    done; done
    xargs -P "$JOBS" -I{} bash -c '{}' < "$LOGS/train27_cmds.txt"
}

light() {   # CPU, once: the light Sabre (1 trial) on every circuit of stage1.pt (+ synth2) -> results/sabre_light.pt
    local j=${JOBS:-8} pids=()
    for i in $(seq 0 $((j - 1))); do python3 -u sabre_light.py run --task-id $i --n-tasks $j > "$LOGS/light_$i.log" 2>&1 & pids+=($!); done
    for p in "${pids[@]}"; do wait "$p"; done
    python3 -u sabre_light.py merge | tee "$LOGS/light_merge.log"
    python3 -u sabre_light.py check --n 3000 | tee "$LOGS/light_check.log"
}

train28() { # v28 = v27 with the light Sabre as routing input instead of the greedy router (--prior sabre-light)
    : > "$LOGS/train28_cmds.txt"
    local gpus=(${GPUS:-}) i=0 pre=""
    for seed in $SEEDS; do for s in $SPLITS loso:layered,seqdense; do
        out=results/v28_$(name $s)_s$seed
        [ -f "$out/eval_$(name $s | sed 's/_dense_complete/_dense-complete/; s/_layered_seqdense/_layered-seqdense/').json" ] && continue
        [ ${#gpus[@]} -gt 0 ] && pre="CUDA_VISIBLE_DEVICES=${gpus[i++ % ${#gpus[@]}]} "
        echo "${pre}python3 -u stage1.py train --split $s --epochs $EPOCHS --seed $seed --out $out --dur-lambda ${DUR_LAMBDA:-1} --dur-every ${DUR_EVERY:-1} --c2-target ${C2_TARGET:-sabre} --synth2 --prior sabre-light --mc $MC --esp-labels $LAB --device $DEVICE --threads $THREADS > $LOGS/train28_$(name $s)_s$seed.log 2>&1 && echo 'train28 $s s$seed done'" >> "$LOGS/train28_cmds.txt"
    done; done
    xargs -P "$JOBS" -I{} bash -c '{}' < "$LOGS/train28_cmds.txt"
}

evalpath() { # v28 end to end on the test splits (CPU): switch -> exact | light Sabre + v28; duration, ESP, F, times
    : > "$LOGS/evalpath_cmds.txt"
    local F9=../generalization_v9/results
    for seed in $SEEDS; do for s in qaoa variational control; do
        ck=results/v28_${s}_s$seed/model.pt
        [ -f "$ck" ] || { echo "missing $ck (run train28 first)"; continue; }
        case $s in qaoa) sd=leave_one_family_out/qaoa ;; variational) sd=leave_one_group_out/variational ;;
                   control) sd=random_control/random_seed$seed ;; esac
        fr=v8kdc10_s$seed; [ $seed = 5 ] && fr=v8kdc10
        fp=$F9/$fr/$sd/predictions.json         # used only when $F9/$fr/$sd/model.pth is missing (then F is not timed)
        for i in $(seq 0 $((EP_TASKS - 1))); do
            echo "RAYON_NUM_THREADS=1 python3 -u evalpath.py run --split $s --ckpt $ck --f-model $F9/$fr/$sd --f-pred $fp --baselines --mc $MC --time-full-every ${TIME_FULL_EVERY:-1} --max-circuits ${EP_MAX:-0} --esp-labels $LAB --task-id $i --n-tasks $EP_TASKS > $LOGS/evalpath_${s}_s${seed}_$i.log 2>&1" >> "$LOGS/evalpath_cmds.txt"
        done
    done; done
    xargs -P "$JOBS" -I{} bash -c '{}' < "$LOGS/evalpath_cmds.txt"
    python3 -u evalpath.py report | tee "$LOGS/evalpath_report.log"
}

test29() {  # ~30 s, no data: unified.py on small real circuits with synthetic labels (run after any code change)
    python3 -u test_unified.py 2>&1 | tee "$LOGS/test29.log"
}

smoke29() { # ~10 min: 2 epochs of v29 on qaoa, 300 training / 20 test circuits, 20 timed (checks speed and the losses)
    python3 -u unified.py train --split qaoa --epochs 2 --out results/smoke29 --synth2 --max-train 300 --max-test 20 --mc 4 \
        --time-n 20 --esp-labels "$LAB" --device "$DEVICE" --threads "$THREADS" 2>&1 | tee "$LOGS/smoke29.log"
}

train29() { # v29 = ONE model: F backbone v8kdfi10 (frozen) + v27 route stage on its embeddings + ESP on its placement
    : > "$LOGS/train29_cmds.txt"
    local gpus=(${GPUS:-}) i=0 pre=""
    for seed in $SEEDS; do for s in $SPLITS loso:layered,seqdense; do
        out=results/v29_$(name $s)_s$seed
        [ -f "$out/eval_$(name $s | sed 's/_dense_complete/_dense-complete/; s/_layered_seqdense/_layered-seqdense/').json" ] && continue
        [ ${#gpus[@]} -gt 0 ] && pre="CUDA_VISIBLE_DEVICES=${gpus[i++ % ${#gpus[@]}]} "
        echo "${pre}python3 -u unified.py train --split $s --epochs $EPOCHS --seed $seed --out $out --dur-lambda ${DUR_LAMBDA:-1} --dur-every ${DUR_EVERY:-1} --c2-target ${C2_TARGET:-sabre} --esp-lambda ${ESP_LAMBDA:-1} --esp-grad ${ESP_GRAD:-head} --synth2 --mc $MC --time-n ${TIME_N:-300} --esp-labels $LAB --device $DEVICE --threads $THREADS > $LOGS/train29_$(name $s)_s$seed.log 2>&1 && echo 'train29 $s s$seed done'" >> "$LOGS/train29_cmds.txt"
    done; done
    xargs -P "$JOBS" -I{} bash -c '{}' < "$LOGS/train29_cmds.txt"
}

smoke25() { # ~5 min: 2 epochs of v25 on qaoa, 20 test circuits (checks speed and that the duration loss falls)
    python3 -u stage1.py train --split qaoa --epochs 2 --out results/smoke25 --dur-lambda 1 --max-test 20 --mc 4 \
        --esp-labels "$LAB" --device "$DEVICE" --threads "$THREADS" 2>&1 | tee "$LOGS/smoke25.log"
}

report() { python3 -u report_v24.py | tee "$LOGS/report.log"; }

case ${1:-all} in
    smoke) smoke ;; diag) diag ;; mc) mc ;; train) train ;; report) report ;; smoke25) smoke25 ;; train25) train25 ;; contention) contention ;; smoke26) smoke26 ;; train26) train26 ;; synth2) synth2 ;; train27) train27 ;; light) light ;; train28) train28 ;; evalpath) evalpath ;; test29) test29 ;; smoke29) smoke29 ;; train29) train29 ;;
    all) smoke; diag; mc; report; train; report ;;
    *) echo "usage: $0 smoke|diag|mc|train|report|smoke25|train25|contention|smoke26|train26|synth2|train27|light|train28|evalpath|test29|smoke29|train29|all"; exit 1 ;;
esac
