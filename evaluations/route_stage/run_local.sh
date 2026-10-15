#!/usr/bin/env bash
# v24 routing stage on a machine without SLURM (see RUNBOOK.md). With the pred-distr env active:
#   bash evaluations/route_stage/run_local.sh smoke | diag | mc | train | report | all
# Env: THREADS (torch threads per process, 8), JOBS (trainings in parallel, 1), DEVICE (cpu | cuda, cpu),
#      EPOCHS (40), SEEDS ("5 6 7"), MC (Poisson samples per schedule, 24), GPUS ("0 1": trainings spread round-robin)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
export PYTHONPATH="$HERE/../../src${PYTHONPATH:+:$PYTHONPATH}"
THREADS=${THREADS:-8}; JOBS=${JOBS:-1}; DEVICE=${DEVICE:-cpu}; EPOCHS=${EPOCHS:-40}; SEEDS=${SEEDS:-"5 6 7"}; MC=${MC:-24}
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

smoke25() { # ~5 min: 2 epochs of v25 on qaoa, 20 test circuits (checks speed and that the duration loss falls)
    python3 -u stage1.py train --split qaoa --epochs 2 --out results/smoke25 --dur-lambda 1 --max-test 20 --mc 4 \
        --esp-labels "$LAB" --device "$DEVICE" --threads "$THREADS" 2>&1 | tee "$LOGS/smoke25.log"
}

report() { python3 -u report_v24.py | tee "$LOGS/report.log"; }

case ${1:-all} in
    smoke) smoke ;; diag) diag ;; mc) mc ;; train) train ;; report) report ;; smoke25) smoke25 ;; train25) train25 ;; contention) contention ;; smoke26) smoke26 ;; train26) train26 ;;
    all) smoke; diag; mc; report; train; report ;;
    *) echo "usage: $0 smoke|diag|mc|train|report|smoke25|train25|contention|smoke26|train26|all"; exit 1 ;;
esac
