#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
METHODS="${METHODS:-thermal_lstm_spawnhist_latency_v11,thermal_lstm_spawnhist_latency_v11_no_pred,thermal_lstm_spawnhist_latency_v11_no_carry,thermal_lstm_spawnhist_latency_v11_no_latency_reward,thermal_lstm_spawnhist_latency_v11_no_attention,thermal_lstm_spawnhist_latency_v11_no_residual}"
SEEDS="${SEEDS:-0,1,2}"
TRAIN_JOBS="${TRAIN_JOBS:-2}"
DEVICE="${DEVICE:-cpu}"

EVAL_STAGES="${EVAL_STAGES:-multi_low,multi_realistic,multi_hard,multi_extreme}"
EVAL_SEEDS="${EVAL_SEEDS:-100,101,102}"
EVAL_EPISODES="${EVAL_EPISODES:-3}"
EVAL_STEPS="${EVAL_STEPS:-3200}"
DECISION_DT_SECONDS="${DECISION_DT_SECONDS:-0.05}"
BASELINE_POLICIES="${BASELINE_POLICIES:-horizon2,dynamic_weighted,planner_ensemble}"

SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
RUN_SIM_TIME_EVAL="${RUN_SIM_TIME_EVAL:-0}"
RUN_CHECKPOINT_SWEEP="${RUN_CHECKPOINT_SWEEP:-0}"
FAIL_ON_MISSING="${FAIL_ON_MISSING:-1}"

SUITE_NAME="${SUITE_NAME:-v11_paper_suite_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-runs/${SUITE_NAME}/train}"
EVAL_DIR="${EVAL_DIR:-runs/${SUITE_NAME}/eval}"
COMMANDS_FILE="${COMMANDS_FILE:-${RUN_DIR}/train_commands.txt}"

CHECKPOINT_SWEEP_METHODS="${CHECKPOINT_SWEEP_METHODS:-thermal_lstm_spawnhist_latency_v11}"
CHECKPOINT_SWEEP_TAGS="${CHECKPOINT_SWEEP_TAGS:-}"
CHECKPOINT_SWEEP_STRIDE="${CHECKPOINT_SWEEP_STRIDE:-100}"
CHECKPOINT_SWEEP_STAGES="${CHECKPOINT_SWEEP_STAGES:-multi_hard,multi_extreme}"

normalize_csv() {
  local value="$1"
  value="${value//[[:space:]]/}"
  printf "%s" "$value"
}

csv_to_array() {
  local value
  value="$(normalize_csv "$1")"
  IFS=',' read -r -a CSV_ARRAY <<< "$value"
}

print_config() {
  echo "=== V11 full paper suite ==="
  echo "methods:              $(normalize_csv "$METHODS")"
  echo "seeds:                $(normalize_csv "$SEEDS")"
  echo "train jobs:           $TRAIN_JOBS"
  echo "run dir:              $RUN_DIR"
  echo "eval dir:             $EVAL_DIR"
  echo "eval stages:          $(normalize_csv "$EVAL_STAGES")"
  echo "eval seeds:           $(normalize_csv "$EVAL_SEEDS")"
  echo "eval episodes:        $EVAL_EPISODES"
  echo "eval steps:           $EVAL_STEPS"
  echo "decision dt seconds:  $DECISION_DT_SECONDS"
  echo "device:               $DEVICE"
  echo "run sim-time eval:    $RUN_SIM_TIME_EVAL"
  echo "run checkpoint sweep: $RUN_CHECKPOINT_SWEEP"
}

train_suite() {
  if [[ "$SKIP_TRAIN" == "1" ]]; then
    echo "SKIP_TRAIN=1, training phase skipped."
    return
  fi

  mkdir -p "$RUN_DIR"
  "$PYTHON" -m coverage_schemes.scheme_d_paper_base.run_training_suite \
    --methods "$(normalize_csv "$METHODS")" \
    --seeds "$(normalize_csv "$SEEDS")" \
    --run-dir "$RUN_DIR" \
    --output "$COMMANDS_FILE" \
    --execute \
    --jobs "$TRAIN_JOBS"
}

checkpoint_path() {
  local method="$1"
  local seed="$2"
  printf "%s/%s_seed%s/checkpoints/scheme_d_paper_base_latest_full.pt" "$RUN_DIR" "$method" "$seed"
}

find_baseline_model() {
  local methods_array seeds_array method seed checkpoint
  csv_to_array "$METHODS"
  methods_array=("${CSV_ARRAY[@]}")
  csv_to_array "$SEEDS"
  seeds_array=("${CSV_ARRAY[@]}")

  for method in "${methods_array[@]}"; do
    for seed in "${seeds_array[@]}"; do
      checkpoint="$(checkpoint_path "$method" "$seed")"
      if [[ -f "$checkpoint" ]]; then
        printf "%s" "$checkpoint"
        return 0
      fi
    done
  done
  return 1
}

run_matrix() {
  local output_dir="$1"
  local policies="$2"
  local model="$3"
  local timebase="$4"
  local command

  command=(
    "$PYTHON" -m coverage_schemes.scheme_d_paper_base.run_matrix
    --policies "$policies"
    --stages "$(normalize_csv "$EVAL_STAGES")"
    --seeds "$(normalize_csv "$EVAL_SEEDS")"
    --episodes "$EVAL_EPISODES"
    --steps "$EVAL_STEPS"
    --demo-mode
    --model "$model"
    --device "$DEVICE"
    --output-dir "$output_dir"
  )
  if [[ "$timebase" == "real_time" ]]; then
    command+=(--decision-dt-seconds "$DECISION_DT_SECONDS")
  fi

  echo
  echo ">>> ${command[*]}"
  "${command[@]}"
}

eval_latest_checkpoints_for_timebase() {
  local timebase="$1"
  local root="$EVAL_DIR/latest_${timebase}"
  local baseline_model methods_array seeds_array method seed checkpoint out_dir

  baseline_model="$(find_baseline_model)" || {
    echo "ERROR: no latest full checkpoint found under $RUN_DIR" >&2
    return 1
  }

  mkdir -p "$root"
  run_matrix "$root/baselines" "$BASELINE_POLICIES" "$baseline_model" "$timebase"

  csv_to_array "$METHODS"
  methods_array=("${CSV_ARRAY[@]}")
  csv_to_array "$SEEDS"
  seeds_array=("${CSV_ARRAY[@]}")

  for method in "${methods_array[@]}"; do
    for seed in "${seeds_array[@]}"; do
      checkpoint="$(checkpoint_path "$method" "$seed")"
      if [[ ! -f "$checkpoint" ]]; then
        if [[ "$FAIL_ON_MISSING" == "1" ]]; then
          echo "ERROR: missing checkpoint: $checkpoint" >&2
          return 1
        fi
        echo "Skipping missing checkpoint: $checkpoint"
        continue
      fi
      out_dir="$root/ppo_latest/${method}_seed${seed}"
      run_matrix "$out_dir" "ppo" "$checkpoint" "$timebase"
    done
  done

  "$PYTHON" -m coverage_schemes.scheme_d_paper_base.collect_eval_summaries \
    --input-dir "$root" \
    --combined-output "$root/combined_summary.csv" \
    --paper-output "$root/paper_summary.csv"
}

run_checkpoint_sweeps() {
  if [[ "$RUN_CHECKPOINT_SWEEP" != "1" ]]; then
    echo "RUN_CHECKPOINT_SWEEP=0, checkpoint sweep skipped."
    return
  fi

  local methods_array seeds_array method seed run_path command output_dir
  csv_to_array "$CHECKPOINT_SWEEP_METHODS"
  methods_array=("${CSV_ARRAY[@]}")
  csv_to_array "$SEEDS"
  seeds_array=("${CSV_ARRAY[@]}")

  for method in "${methods_array[@]}"; do
    for seed in "${seeds_array[@]}"; do
      run_path="$RUN_DIR/${method}_seed${seed}"
      if [[ ! -d "$run_path/checkpoints" ]]; then
        if [[ "$FAIL_ON_MISSING" == "1" ]]; then
          echo "ERROR: missing checkpoint directory: $run_path/checkpoints" >&2
          return 1
        fi
        echo "Skipping missing checkpoint directory: $run_path/checkpoints"
        continue
      fi

      output_dir="$EVAL_DIR/checkpoint_sweeps/${method}_seed${seed}"
      command=(
        "$PYTHON" -m coverage_schemes.scheme_d_paper_base.checkpoint_sweep
        --run-dir "$run_path"
        --policies ppo
        --stages "$(normalize_csv "$CHECKPOINT_SWEEP_STAGES")"
        --seeds "$(normalize_csv "$EVAL_SEEDS")"
        --episodes "$EVAL_EPISODES"
        --steps "$EVAL_STEPS"
        --decision-dt-seconds "$DECISION_DT_SECONDS"
        --demo-mode
        --rank-objective latency
        --device "$DEVICE"
        --stride "$CHECKPOINT_SWEEP_STRIDE"
        --output-dir "$output_dir"
      )
      if [[ -n "$CHECKPOINT_SWEEP_TAGS" ]]; then
        command+=(--checkpoint-tags "$(normalize_csv "$CHECKPOINT_SWEEP_TAGS")")
      fi

      echo
      echo ">>> ${command[*]}"
      "${command[@]}"
    done
  done
}

eval_suite() {
  if [[ "$SKIP_EVAL" == "1" ]]; then
    echo "SKIP_EVAL=1, evaluation phase skipped."
    return
  fi

  eval_latest_checkpoints_for_timebase real_time
  if [[ "$RUN_SIM_TIME_EVAL" == "1" ]]; then
    eval_latest_checkpoints_for_timebase sim_time
  fi
  run_checkpoint_sweeps
}

print_config
train_suite
eval_suite

echo
echo "Done."
echo "Training commands: $COMMANDS_FILE"
echo "Real-time paper summary: $EVAL_DIR/latest_real_time/paper_summary.csv"
if [[ "$RUN_SIM_TIME_EVAL" == "1" ]]; then
  echo "Sim-time paper summary:  $EVAL_DIR/latest_sim_time/paper_summary.csv"
fi
