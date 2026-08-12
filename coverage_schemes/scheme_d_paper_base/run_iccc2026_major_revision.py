"""Run the two experiments required by the ICCC 2026 major revision.

The protocol keeps the submitted V12 LSTM-PPO architecture fixed.  It adds:

1. a disclosed tuning protocol for direct LSTM-PPO, using validation seeds
   50--52 and never using held-out test seeds 100--102 for selection; and
2. a planner-mismatch variant that replaces the Horizon-2 teacher/base with
   Horizon-3 while holding the remaining V12 configuration fixed.

All artifacts are written below the requested run root.  Completed seed runs
and evaluations are skipped, so the command can be restarted after an
interruption at seed boundaries.
"""

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path


DIRECT_METHOD = "thermal_lstm_spawnhist_latency_v12_fast_vanilla_lstm_ppo"
H3_METHOD = "thermal_lstm_spawnhist_latency_v12_fast_horizon3_residual"
STAGES = "multi_low,multi_realistic,multi_hard,multi_extreme"
PILOT_STAGES = "multi_hard,multi_extreme"

DIRECT_CANDIDATES = {
    "default": {
        "ppo_lr": "1e-5",
        "ppo_epochs": "1",
        "ppo_clip": "0.05",
        "ppo_value_clip": "0.05",
        "ppo_entropy_start": "0.0008",
        "ppo_entropy_end": "0.00018",
        "ppo_max_grad_norm": "0.35",
    },
    "medium": {
        "ppo_lr": "3e-5",
        "ppo_epochs": "2",
        "ppo_clip": "0.10",
        "ppo_value_clip": "0.10",
        "ppo_entropy_start": "0.0015",
        "ppo_entropy_end": "0.00030",
        "ppo_max_grad_norm": "0.50",
    },
    "standard": {
        "ppo_lr": "1e-4",
        "ppo_epochs": "4",
        "ppo_clip": "0.20",
        "ppo_value_clip": "0.20",
        "ppo_entropy_start": "0.0030",
        "ppo_entropy_end": "0.00050",
        "ppo_max_grad_norm": "0.50",
    },
}


def parse_int_csv(value):
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def shell_join(command):
    return " ".join(shlex.quote(str(item)) for item in command)


def candidate_flags(candidate):
    config = DIRECT_CANDIDATES[candidate]
    flags = []
    for key, value in config.items():
        flags.extend([f"--{key.replace('_', '-')}", value])
    return flags


class RevisionRunner:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.run_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "experiment_manifest.json"
        self.manifest = {
            "protocol": "ICCC2026_IC069_major_revision",
            "python": args.python,
            "device": args.device,
            "commands": [],
            "direct_candidates": DIRECT_CANDIDATES,
            "selection_rule": (
                "Maximize the equally weighted mean coverage over Hard and Extreme "
                "on validation seeds 50--52; break exact ties using lower mean p90 latency."
            ),
            "held_out_test_seeds": parse_int_csv(args.test_seeds),
        }

    def record(self, label, command, skipped=False):
        self.manifest["commands"].append(
            {"label": label, "command": shell_join(command), "skipped": bool(skipped)}
        )
        write_json(self.manifest_path, self.manifest)

    def run(self, label, command, complete_path=None):
        if complete_path is not None and Path(complete_path).is_file() and self.args.skip_existing:
            print(f"[SKIP] {label}: {complete_path}", flush=True)
            self.record(label, command, skipped=True)
            return
        print(f"\n>>> [{label}] {shell_join(command)}", flush=True)
        self.record(label, command)
        if self.args.dry_run:
            return
        subprocess.run(command, check=True)

    def training_command(self, method, seeds, run_dir, extra_flags=None, quick=False):
        command = [
            self.args.python,
            "-m",
            "coverage_schemes.scheme_d_paper_base.run_training_suite",
            "--methods",
            method,
            "--seeds",
            ",".join(str(seed) for seed in seeds),
            "--run-dir",
            str(run_dir),
            "--output",
            str(run_dir / "commands.txt"),
            "--device",
            self.args.device,
            "--jobs",
            str(min(self.args.train_jobs, len(seeds))),
            "--execute",
        ]
        if quick:
            command.extend(
                [
                    "--stage-episodes",
                    "multi_low:2,multi_realistic:2,multi_hard:3,multi_extreme:1",
                    "--episode-steps",
                    "300",
                    "--bc-episodes",
                    "8",
                    "--bc-epochs",
                    "2",
                    "--bc-stage-episodes",
                    "multi_low:2,multi_realistic:2,multi_hard:2,multi_extreme:2",
                    "--save-interval",
                    "20",
                ]
            )
        if extra_flags:
            command.extend(extra_flags)
        return command

    def matrix_command(self, model, output_dir, seeds, stages, episodes, steps):
        return [
            self.args.python,
            "-m",
            "coverage_schemes.scheme_d_paper_base.run_matrix",
            "--policies",
            "ppo",
            "--stages",
            stages,
            "--seeds",
            ",".join(str(seed) for seed in seeds),
            "--episodes",
            str(episodes),
            "--steps",
            str(steps),
            "--model",
            str(model),
            "--device",
            self.args.device,
            "--decision-dt-seconds",
            "0.05",
            "--demo-mode",
            "--output-dir",
            str(output_dir),
        ]

    @staticmethod
    def checkpoint(run_dir, method, seed):
        return (
            Path(run_dir)
            / f"{method}_seed{seed}"
            / "checkpoints"
            / "scheme_d_paper_base_latest_full.pt"
        )

    def train_missing(self, label, method, seeds, run_dir, extra_flags=None, quick=False):
        missing = [seed for seed in seeds if not self.checkpoint(run_dir, method, seed).is_file()]
        if not missing and self.args.skip_existing:
            command = self.training_command(method, seeds, run_dir, extra_flags, quick)
            self.record(label, command, skipped=True)
            print(f"[SKIP] {label}: all checkpoints exist", flush=True)
            return
        command = self.training_command(method, missing or seeds, run_dir, extra_flags, quick)
        complete = self.checkpoint(run_dir, method, (missing or seeds)[-1])
        self.run(label, command, complete_path=complete)

    def run_pilot(self):
        pilot_seed = [self.args.pilot_train_seed]
        validation_seeds = parse_int_csv(self.args.validation_seeds)
        quick = self.args.quick
        for candidate in DIRECT_CANDIDATES:
            train_dir = self.root / "pilot" / candidate / "train"
            flags = candidate_flags(candidate)
            if not quick:
                flags.extend(
                    [
                        "--stage-episodes",
                        self.args.pilot_stage_episodes,
                        "--episode-steps",
                        "800",
                    ]
                )
            self.train_missing(
                f"pilot_train_{candidate}",
                DIRECT_METHOD,
                pilot_seed,
                train_dir,
                extra_flags=flags,
                quick=quick,
            )
            model = self.checkpoint(train_dir, DIRECT_METHOD, pilot_seed[0])
            eval_dir = self.root / "pilot" / candidate / "validation"
            command = self.matrix_command(
                model,
                eval_dir,
                validation_seeds[:1] if quick else validation_seeds,
                PILOT_STAGES,
                1 if quick else self.args.validation_episodes,
                600 if quick else self.args.eval_steps,
            )
            self.run(
                f"pilot_eval_{candidate}",
                command,
                complete_path=eval_dir / "summary.csv",
            )
        if not self.args.dry_run:
            self.select_direct_candidate()

    def select_direct_candidate(self):
        rows = []
        for candidate in DIRECT_CANDIDATES:
            summary_path = self.root / "pilot" / candidate / "validation" / "summary.csv"
            if not summary_path.is_file():
                raise FileNotFoundError(f"Missing validation summary: {summary_path}")
            summary = [row for row in read_csv(summary_path) if row["stage"] in PILOT_STAGES.split(",")]
            coverage = sum(float(row["coverage_rate_mean"]) for row in summary) / len(summary)
            p90 = sum(float(row["cover_latency_p90_seconds_mean"]) for row in summary) / len(summary)
            rows.append({"candidate": candidate, "coverage_score": coverage, "p90_tiebreak": p90})
        rows.sort(key=lambda row: (-row["coverage_score"], row["p90_tiebreak"], row["candidate"]))
        selection = {
            "selected": rows[0]["candidate"],
            "ranking": rows,
            "selection_rule": self.manifest["selection_rule"],
            "validation_seeds": parse_int_csv(self.args.validation_seeds),
            "test_seeds_not_used": parse_int_csv(self.args.test_seeds),
        }
        write_json(self.root / "pilot_selection.json", selection)
        self.manifest["pilot_selection"] = selection
        write_json(self.manifest_path, self.manifest)
        print(f"Selected direct LSTM-PPO candidate: {selection['selected']}", flush=True)
        return selection["selected"]

    def selected_candidate(self):
        path = self.root / "pilot_selection.json"
        if not path.is_file():
            if self.args.dry_run:
                return "medium"
            return self.select_direct_candidate()
        return json.loads(path.read_text(encoding="utf-8"))["selected"]

    def eval_learned_method(self, label, method, train_dir, eval_dir, train_seeds):
        test_seeds = parse_int_csv(self.args.test_seeds)
        for train_seed in train_seeds:
            model = self.checkpoint(train_dir, method, train_seed)
            output_dir = eval_dir / f"{method}_seed{train_seed}"
            command = self.matrix_command(
                model,
                output_dir,
                test_seeds[:1] if self.args.quick else test_seeds,
                STAGES if not self.args.quick else PILOT_STAGES,
                1 if self.args.quick else self.args.test_episodes,
                600 if self.args.quick else self.args.eval_steps,
            )
            self.run(
                f"{label}_seed{train_seed}",
                command,
                complete_path=output_dir / "summary.csv",
            )

    def run_final(self):
        train_seeds = parse_int_csv(self.args.train_seeds)
        if self.args.quick:
            train_seeds = train_seeds[:1]

        selected = self.selected_candidate()
        direct_train = self.root / "final" / "direct_lstm_ppo" / selected / "train"
        self.train_missing(
            f"final_train_direct_{selected}",
            DIRECT_METHOD,
            train_seeds,
            direct_train,
            extra_flags=candidate_flags(selected),
            quick=self.args.quick,
        )
        self.eval_learned_method(
            f"final_eval_direct_{selected}",
            DIRECT_METHOD,
            direct_train,
            self.root / "final" / "direct_lstm_ppo" / selected / "eval",
            train_seeds,
        )

        h3_train = self.root / "final" / "horizon3_residual" / "train"
        self.train_missing(
            "final_train_horizon3_residual",
            H3_METHOD,
            train_seeds,
            h3_train,
            quick=self.args.quick,
        )
        self.eval_learned_method(
            "final_eval_horizon3_residual",
            H3_METHOD,
            h3_train,
            self.root / "final" / "horizon3_residual" / "eval",
            train_seeds,
        )

    def summarize(self):
        summaries = []
        for path in sorted((self.root / "final").glob("**/summary.csv")):
            for row in read_csv(path):
                row = dict(row)
                row["source"] = str(path.relative_to(self.root))
                summaries.append(row)
        output = self.root / "final_summary_rows.csv"
        if summaries:
            with output.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
                writer.writeheader()
                writer.writerows(summaries)
        print(f"Collected {len(summaries)} final summary rows in {output}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "final", "all", "summarize"), default="all")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--run-root",
        default="runs/v12_small_paper_suite/iccc2026_major_revision/experiments",
    )
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--train-jobs", type=int, default=2)
    parser.add_argument("--train-seeds", default="0,1,2")
    parser.add_argument("--pilot-train-seed", type=int, default=0)
    parser.add_argument("--validation-seeds", default="50,51,52")
    parser.add_argument("--test-seeds", default="100,101,102")
    parser.add_argument(
        "--pilot-stage-episodes",
        default="multi_low:20,multi_realistic:40,multi_hard:180,multi_extreme:48",
    )
    parser.add_argument("--validation-episodes", type=int, default=3)
    parser.add_argument("--test-episodes", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=3200)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main():
    args = build_parser().parse_args()
    runner = RevisionRunner(args)
    if args.phase in ("pilot", "all"):
        runner.run_pilot()
    if args.phase in ("final", "all"):
        runner.run_final()
    if args.phase in ("summarize", "all") and not args.dry_run:
        runner.summarize()


if __name__ == "__main__":
    main()
