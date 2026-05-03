import argparse
import csv
import hashlib
import json
import os
from contextlib import contextmanager
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

from datasets.trellis_drug_classes import decode_treatment_code
from distributional_regression_notebook_utils import ConditionalSampler, fit_conditional_sampler
from evaluate_trellis_experimental import compute_latents, compute_metric, is_conditioned_encoder
from utils.experiment_utils import find_experiment_dir, is_mfm_model, load_config, load_experiment


SPLITS = ("replicas-1", "replicas-2", "pdo21", "pdo27", "pdo75")
METRICS = ("mmd_energy", "mmd_rbf", "swd")


def _to_tensor(array, device):
    return torch.tensor(array, dtype=torch.float32, device=device)


def build_sample_info(samples):
    sample_info = []
    for i, sample in enumerate(samples):
        culture, x0, x1, _, _, treat_cond, patient = sample
        sample_info.append(
            {
                "index": i,
                "culture": culture,
                "patient": patient,
                "treatment": decode_treatment_code(treat_cond),
                "n_source": int(x0.shape[0]),
                "n_target": int(x1.shape[0]),
            }
        )
    return sample_info


def compute_all_metrics(pred: torch.Tensor, target: torch.Tensor, swd_subsample_rounds: int) -> Dict[str, float]:
    return {
        metric: compute_metric(
            pred,
            target,
            metric=metric,
            swd_subsample_rounds=swd_subsample_rounds,
        )
        for metric in METRICS
    }


def summarize(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "sem": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()),
        "sem": float(arr.std() / np.sqrt(arr.size)),
        "std": float(arr.std()),
        "n": int(arr.size),
    }


def initialize_result_store(k_values: List[int]):
    return {
        "best_of_k": {k: {metric: [] for metric in METRICS} for k in k_values},
        "mixture_of_k": {k: {metric: [] for metric in METRICS} for k in k_values},
        "oracle": {metric: [] for metric in METRICS},
    }


def result_stem(split_name: str, args) -> str:
    if args.num_sample_shards > 1:
        return f"{split_name}_shard{args.sample_shard:02d}-of-{args.num_sample_shards:02d}"
    return split_name


def get_sample_indices(num_samples: int, args) -> List[int]:
    if args.num_sample_shards < 1:
        raise ValueError("--num_sample_shards must be >= 1")
    if args.sample_shard < 0 or args.sample_shard >= args.num_sample_shards:
        raise ValueError(
            f"--sample_shard must be in [0, {args.num_sample_shards - 1}], got {args.sample_shard}"
        )
    return [i for i in range(num_samples) if i % args.num_sample_shards == args.sample_shard]


def cache_dir_for_args(args) -> str:
    return args.cache_dir or os.path.join(args.results_dir, "_cache")


def stable_hash(payload: Dict) -> str:
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.md5(encoded).hexdigest()[:12]


@contextmanager
def cache_lock(path: str):
    lock_path = f"{path}.lock"
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "w") as lock_file:
        try:
            import fcntl

            fcntl.flock(lock_file, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
            except Exception:
                pass


def latent_cache_path(args, experiment_dir: str, use_cell_cond: bool) -> str:
    key = stable_hash(
        {
            "kind": "trellis_distreg_latents",
            "experiment_dir": os.path.abspath(experiment_dir),
            "checkpoint_epoch": args.checkpoint_epoch,
            "split_name": args.split_name,
            "use_cell_cond": use_cell_cond,
        }
    )
    return os.path.join(cache_dir_for_args(args), f"latents_{args.split_name}_{key}.npz")


def sampler_cache_path(args, experiment_dir: str, train_source_latents: np.ndarray) -> str:
    key = stable_hash(
        {
            "kind": "trellis_distreg_sampler",
            "experiment_dir": os.path.abspath(experiment_dir),
            "checkpoint_epoch": args.checkpoint_epoch,
            "split_name": args.split_name,
            "seed": args.seed,
            "k": args.k_max,
            "hidden": args.sampler_hidden,
            "lr": args.sampler_lr,
            "epochs": args.sampler_epochs,
            "batch_size": args.sampler_batch_size,
            "z_dim": int(train_source_latents.shape[1]),
        }
    )
    return os.path.join(cache_dir_for_args(args), f"sampler_{args.split_name}_{key}.pt")


def load_or_compute_latents(args, encoder, train_samples, test_samples, device, use_cell_cond):
    path = latent_cache_path(args, args.experiment_dir, use_cell_cond)
    if not args.no_cache and os.path.exists(path):
        print(f"Loading cached latents from {path}")
        data = np.load(path)
        return (
            data["train_source_latents"],
            data["train_target_latents"],
            data["train_treat_conds"],
            data["test_source_latents"],
            data["test_target_latents"],
            data["test_treat_conds"],
        )

    if args.no_cache:
        train_source_latents, train_target_latents, train_treat_conds = compute_latents(
            encoder, train_samples, device, split_name="train", use_cell_cond=use_cell_cond
        )
        test_source_latents, test_target_latents, test_treat_conds = compute_latents(
            encoder, test_samples, device, split_name="test", use_cell_cond=use_cell_cond
        )
        return (
            train_source_latents,
            train_target_latents,
            train_treat_conds,
            test_source_latents,
            test_target_latents,
            test_treat_conds,
        )

    with cache_lock(path):
        if os.path.exists(path):
            print(f"Loading cached latents from {path}")
            data = np.load(path)
            return (
                data["train_source_latents"],
                data["train_target_latents"],
                data["train_treat_conds"],
                data["test_source_latents"],
                data["test_target_latents"],
                data["test_treat_conds"],
            )

        train_source_latents, train_target_latents, train_treat_conds = compute_latents(
            encoder, train_samples, device, split_name="train", use_cell_cond=use_cell_cond
        )
        test_source_latents, test_target_latents, test_treat_conds = compute_latents(
            encoder, test_samples, device, split_name="test", use_cell_cond=use_cell_cond
        )

        tmp_path = f"{path}.tmp.{os.getpid()}.npz"
        np.savez_compressed(
            tmp_path,
            train_source_latents=train_source_latents,
            train_target_latents=train_target_latents,
            train_treat_conds=train_treat_conds,
            test_source_latents=test_source_latents,
            test_target_latents=test_target_latents,
            test_treat_conds=test_treat_conds,
        )
        os.replace(tmp_path, path)
        print(f"Cached latents to {path}")

        return (
            train_source_latents,
            train_target_latents,
            train_treat_conds,
            test_source_latents,
            test_target_latents,
            test_treat_conds,
        )


def load_or_fit_sampler(args, X_train, Y_train, z_dim, device):
    path = sampler_cache_path(args, args.experiment_dir, X_train[:, :z_dim])
    x_dim = X_train.shape[1]
    y_dim = Y_train.shape[1]

    if not args.no_cache and os.path.exists(path):
        print(f"Loading cached ConditionalSampler from {path}")
        payload = torch.load(path, map_location=device)
        sampler = ConditionalSampler(x_dim, z_dim, y_dim, hidden=args.sampler_hidden).to(device)
        sampler.load_state_dict(payload["state_dict"])
        sampler.eval()
        return sampler

    if args.no_cache:
        return fit_conditional_sampler(
            X_train,
            Y_train,
            z_dim=z_dim,
            hidden=args.sampler_hidden,
            lr=args.sampler_lr,
            epochs=args.sampler_epochs,
            batch_size=args.sampler_batch_size,
            k=args.k_max,
            device=device,
        )

    with cache_lock(path):
        if os.path.exists(path):
            print(f"Loading cached ConditionalSampler from {path}")
            payload = torch.load(path, map_location=device)
            sampler = ConditionalSampler(x_dim, z_dim, y_dim, hidden=args.sampler_hidden).to(device)
            sampler.load_state_dict(payload["state_dict"])
            sampler.eval()
            return sampler

        sampler = fit_conditional_sampler(
            X_train,
            Y_train,
            z_dim=z_dim,
            hidden=args.sampler_hidden,
            lr=args.sampler_lr,
            epochs=args.sampler_epochs,
            batch_size=args.sampler_batch_size,
            k=args.k_max,
            device=device,
        )

        tmp_path = f"{path}.tmp.{os.getpid()}"
        torch.save(
            {
                "state_dict": sampler.state_dict(),
                "metadata": {
                    "x_dim": x_dim,
                    "y_dim": y_dim,
                    "z_dim": z_dim,
                    "hidden": args.sampler_hidden,
                    "k": args.k_max,
                },
            },
            tmp_path,
        )
        os.replace(tmp_path, path)
        print(f"Cached ConditionalSampler to {path}")

        return sampler


def add_row_to_store(store, row):
    mode = row["mode"]
    metric = row["metric"]
    value = float(row["value"])
    if mode == "oracle":
        store["oracle"][metric].append(value)
    else:
        store[mode][int(row["k"])][metric].append(value)


def load_eval_checkpoint(checkpoint_path: str, store, args):
    if not os.path.exists(checkpoint_path):
        return set(), []
    with open(checkpoint_path, "r") as f:
        checkpoint = json.load(f)
    checkpoint_args = checkpoint.get("args")
    if checkpoint_args is None:
        print(f"Ignoring checkpoint without argument metadata: {checkpoint_path}")
        return set(), []
    if not shard_args_are_compatible(args, checkpoint_args):
        print(f"Ignoring checkpoint with different arguments: {checkpoint_path}")
        return set(), []
    detailed_rows = checkpoint.get("detailed_rows", [])
    for row in detailed_rows:
        add_row_to_store(store, row)
    completed = set(int(i) for i in checkpoint.get("completed_indices", []))
    print(f"Loaded checkpoint from {checkpoint_path}: {len(completed)} samples complete")
    return completed, detailed_rows


def save_eval_checkpoint(checkpoint_path: str, completed_indices, detailed_rows, args):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    tmp_path = f"{checkpoint_path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(
            {
                "args": vars(args),
                "completed_indices": sorted(int(i) for i in completed_indices),
                "detailed_rows": detailed_rows,
            },
            f,
            indent=2,
        )
    os.replace(tmp_path, checkpoint_path)


def evaluate_distributional_regression(
    generator,
    sampler,
    test_samples,
    source_latents,
    target_latents,
    treat_conds,
    sample_info,
    device,
    k_max: int,
    k_values: List[int],
    m_draws: int,
    swd_subsample_rounds: int,
    rng: np.random.Generator,
    sample_indices: List[int],
    checkpoint_path: str,
    args,
):
    store = initialize_result_store(k_values)
    completed_indices, detailed_rows = load_eval_checkpoint(checkpoint_path, store, args)

    for sample_idx in sample_indices:
        if sample_idx in completed_indices:
            continue

        sample = test_samples[sample_idx]
        culture, x0, x1, _, _, _, patient = sample
        info = sample_info[sample_idx]
        sample_rows = []

        x0_tensor = _to_tensor(x0, device)
        x1_tensor = _to_tensor(x1, device)
        source_latent_tensor = _to_tensor(source_latents[sample_idx : sample_idx + 1], device)
        target_latent_tensor = _to_tensor(target_latents[sample_idx : sample_idx + 1], device)

        sampler_input_np = np.concatenate(
            [source_latents[sample_idx : sample_idx + 1], treat_conds[sample_idx : sample_idx + 1]],
            axis=1,
        )
        sampler_input = _to_tensor(sampler_input_np, device)

        with torch.no_grad():
            latent_samples = sampler.sample(sampler_input, k=k_max).squeeze(1)

            candidate_metrics = {metric: np.zeros(k_max, dtype=float) for metric in METRICS}
            x0_batched = x0_tensor.repeat(k_max, 1)
            source_latent_batched = source_latent_tensor.expand(k_max, -1)
            candidate_preds = generator.sample(
                x0_batched,
                source_latent_batched,
                latent_samples,
            )
            for candidate_idx in range(k_max):
                pred = candidate_preds[candidate_idx]
                metrics = compute_all_metrics(
                    pred, x1_tensor, swd_subsample_rounds=swd_subsample_rounds
                )
                for metric, value in metrics.items():
                    candidate_metrics[metric][candidate_idx] = value

            oracle_pred = generator.sample(
                x0_tensor,
                source_latent_tensor,
                target_latent_tensor,
            ).squeeze(0)
            oracle_metrics = compute_all_metrics(
                oracle_pred, x1_tensor, swd_subsample_rounds=swd_subsample_rounds
            )

        for metric, value in oracle_metrics.items():
            store["oracle"][metric].append(value)
            sample_rows.append(
                {**info, "mode": "oracle", "k": "", "metric": metric, "value": value}
            )

        n_pred = candidate_preds.shape[1]
        for k in k_values:
            m_eff = 1 if k == k_max else m_draws
            best_sums = {metric: 0.0 for metric in METRICS}
            mixture_sums = {metric: 0.0 for metric in METRICS}

            for _ in range(m_eff):
                chosen = rng.choice(k_max, size=k, replace=False)
                for metric in METRICS:
                    best_sums[metric] += float(candidate_metrics[metric][chosen].min())

                chosen_t = torch.as_tensor(chosen, device=device, dtype=torch.long)
                pooled = candidate_preds.index_select(0, chosen_t).reshape(k * n_pred, -1)
                resample_idx = torch.randperm(pooled.shape[0], device=device)[:n_pred]
                mixture = pooled[resample_idx]
                mixture_metrics = compute_all_metrics(
                    mixture, x1_tensor, swd_subsample_rounds=swd_subsample_rounds
                )
                for metric, value in mixture_metrics.items():
                    mixture_sums[metric] += value

            for metric in METRICS:
                best_value = best_sums[metric] / m_eff
                mixture_value = mixture_sums[metric] / m_eff
                store["best_of_k"][k][metric].append(best_value)
                store["mixture_of_k"][k][metric].append(mixture_value)
                sample_rows.append(
                    {
                        **info,
                        "mode": "best_of_k",
                        "k": k,
                        "metric": metric,
                        "value": best_value,
                    }
                )
                sample_rows.append(
                    {
                        **info,
                        "mode": "mixture_of_k",
                        "k": k,
                        "metric": metric,
                        "value": mixture_value,
                    }
                )

        detailed_rows.extend(sample_rows)
        completed_indices.add(sample_idx)
        save_eval_checkpoint(checkpoint_path, completed_indices, detailed_rows, args)

        print(
            f"Sample {sample_idx + 1}/{len(test_samples)} "
            f"({culture}, patient={patient}, treatment={info['treatment']}): "
            f"oracle mmd_energy={oracle_metrics['mmd_energy']:.6f}"
        )

        del candidate_preds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return store, detailed_rows


def build_summary(split_name, store, k_values):
    summary = {
        "split": split_name,
        "best_of_k": defaultdict(dict),
        "mixture_of_k": defaultdict(dict),
        "oracle": {},
    }

    for mode in ("best_of_k", "mixture_of_k"):
        for k in k_values:
            for metric in METRICS:
                summary[mode][str(k)][metric] = summarize(store[mode][k][metric])

    for metric in METRICS:
        summary["oracle"][metric] = summarize(store["oracle"][metric])

    summary["best_of_k"] = dict(summary["best_of_k"])
    summary["mixture_of_k"] = dict(summary["mixture_of_k"])
    return summary


def fmt_stat(stat: Dict[str, float]) -> str:
    return f"{stat['mean']:.4f} +/- {stat['sem']:.4f}"


def print_tables(summary, k_values):
    split = summary["split"]
    for metric in METRICS:
        print()
        print(f"{split} {metric}")
        header = ["Mode"] + [f"k={k}" for k in k_values] + ["Oracle"]
        print("| " + " | ".join(header) + " |")
        print("| " + " | ".join(["---"] * len(header)) + " |")
        for mode_key, label in (("best_of_k", "Best-of-k"), ("mixture_of_k", "Mixture-of-k")):
            row = [label]
            for k in k_values:
                row.append(fmt_stat(summary[mode_key][str(k)][metric]))
            row.append(fmt_stat(summary["oracle"][metric]))
            print("| " + " | ".join(row) + " |")


def write_outputs(results_dir, split_name, summary, detailed_rows, args, stem_override: Optional[str] = None):
    os.makedirs(results_dir, exist_ok=True)
    stem = stem_override or result_stem(split_name, args)
    json_path = os.path.join(results_dir, f"trellis_distributional_regression_{stem}.json")
    summary_csv_path = os.path.join(
        results_dir, f"trellis_distributional_regression_{stem}_summary.csv"
    )
    detailed_csv_path = os.path.join(
        results_dir, f"trellis_distributional_regression_{stem}_per_sample.csv"
    )

    payload = {
        "args": vars(args),
        "summary": summary,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    with open(summary_csv_path, "w", newline="") as f:
        fieldnames = ["split", "mode", "k", "metric", "mean", "sem", "std", "n"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for mode in ("best_of_k", "mixture_of_k"):
            for k, metrics in summary[mode].items():
                for metric, stat in metrics.items():
                    writer.writerow(
                        {
                            "split": split_name,
                            "mode": mode,
                            "k": k,
                            "metric": metric,
                            **stat,
                        }
                    )
        for metric, stat in summary["oracle"].items():
            writer.writerow(
                {
                    "split": split_name,
                    "mode": "oracle",
                    "k": "",
                    "metric": metric,
                    **stat,
                }
            )

    with open(detailed_csv_path, "w", newline="") as f:
        fieldnames = [
            "index",
            "culture",
            "patient",
            "treatment",
            "n_source",
            "n_target",
            "mode",
            "k",
            "metric",
            "value",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detailed_rows)

    print(f"Wrote {json_path}")
    print(f"Wrote {summary_csv_path}")
    print(f"Wrote {detailed_csv_path}")


def read_detailed_rows(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["index"] = int(row["index"])
            row["n_source"] = int(row["n_source"])
            row["n_target"] = int(row["n_target"])
            row["value"] = float(row["value"])
            if row["k"] != "":
                row["k"] = int(row["k"])
            rows.append(row)
    return rows


def shard_args_are_compatible(current_args, shard_args: Dict) -> bool:
    keys = [
        "split_name",
        "outputs_dir",
        "experiment_name",
        "experiment_dir",
        "checkpoint_epoch",
        "seed",
        "k_max",
        "k_values",
        "m",
        "sampler_epochs",
        "sampler_hidden",
        "sampler_lr",
        "sampler_batch_size",
        "swd_subsample_rounds",
        "num_sample_shards",
    ]
    current = vars(current_args)
    return all(current.get(key) == shard_args.get(key) for key in keys)


def try_write_combined_shard_outputs(results_dir, split_name, args, k_values):
    if args.num_sample_shards == 1:
        return

    combined_rows = []
    for shard in range(args.num_sample_shards):
        stem = f"{split_name}_shard{shard:02d}-of-{args.num_sample_shards:02d}"
        json_path = os.path.join(results_dir, f"trellis_distributional_regression_{stem}.json")
        detailed_csv_path = os.path.join(
            results_dir, f"trellis_distributional_regression_{stem}_per_sample.csv"
        )
        if not os.path.exists(json_path) or not os.path.exists(detailed_csv_path):
            print("Not all shard outputs are present yet; skipping combined summary for now.")
            return
        with open(json_path, "r") as f:
            payload = json.load(f)
        if not shard_args_are_compatible(args, payload.get("args", {})):
            print(f"Shard output {json_path} was produced with different arguments; skipping combine.")
            return
        combined_rows.extend(read_detailed_rows(detailed_csv_path))

    store = initialize_result_store(k_values)
    for row in combined_rows:
        add_row_to_store(store, row)
    summary = build_summary(split_name, store, k_values)
    print("All shard outputs found; writing combined split-level outputs.")
    write_outputs(results_dir, split_name, summary, combined_rows, args, stem_override=split_name)


def parse_k_values(raw_values: Optional[List[int]], k_max: int) -> List[int]:
    k_values = raw_values or [1, 2, 4, 8, 16, 32, 64]
    k_values = sorted(set(k_values))
    if any(k < 1 for k in k_values):
        raise ValueError(f"k_values must all be >= 1, got {k_values}")
    if max(k_values) > k_max:
        raise ValueError(f"max k_values={max(k_values)} exceeds k_max={k_max}")
    return k_values


def main():
    parser = argparse.ArgumentParser(
        description="Trellis latent distributional-regression evaluation with ConditionalSampler."
    )
    parser.add_argument("--split_name", choices=SPLITS, required=True)
    parser.add_argument("--outputs_dir", default="outputs_energy_and_swd_generators_never_delete_01_29_2026")
    parser.add_argument("--experiment_name", default="trellis_a2a_energy")
    parser.add_argument("--experiment_dir", default=None)
    parser.add_argument("--checkpoint_epoch", type=int, default=None)
    parser.add_argument("--results_dir", default="trellis_distributional_regression_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--k_max", type=int, default=64)
    parser.add_argument("--k_values", nargs="+", type=int, default=None)
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--sampler_epochs", type=int, default=500)
    parser.add_argument("--sampler_hidden", type=int, default=128)
    parser.add_argument("--sampler_lr", type=float, default=1e-3)
    parser.add_argument("--sampler_batch_size", type=int, default=256)
    parser.add_argument("--swd_subsample_rounds", type=int, default=100)
    parser.add_argument("--num_sample_shards", type=int, default=1)
    parser.add_argument("--sample_shard", type=int, default=0)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    k_values = parse_k_values(args.k_values, args.k_max)

    if args.experiment_dir is None:
        args.experiment_dir = find_experiment_dir(
            outputs_dir=args.outputs_dir,
            match_criteria={
                "experiment.name": args.experiment_name,
                "experiment.split_name": args.split_name,
            },
        )

    print(f"Using experiment directory: {args.experiment_dir}")
    print(f"Device: {device}")
    print(f"k_max={args.k_max}, k_values={k_values}, m={args.m}")
    print(
        f"Sample shard {args.sample_shard}/{args.num_sample_shards} "
        f"(cache={'off' if args.no_cache else cache_dir_for_args(args)})"
    )

    cfg = load_config(args.experiment_dir)
    if is_mfm_model(cfg):
        raise ValueError("This evaluator is intended for a2a models, not MFM/source-only models.")

    encoder, generator, dataset, cfg = load_experiment(
        args.experiment_dir,
        device,
        cfg=cfg,
        checkpoint_epoch=args.checkpoint_epoch,
    )

    use_cell_cond = is_conditioned_encoder(cfg)
    train_samples = dataset.samples_train
    test_samples = dataset.samples_test

    (
        train_source_latents,
        train_target_latents,
        train_treat_conds,
        test_source_latents,
        test_target_latents,
        test_treat_conds,
    ) = load_or_compute_latents(
        args, encoder, train_samples, test_samples, device, use_cell_cond
    )
    test_info = build_sample_info(test_samples)
    sample_indices = get_sample_indices(len(test_samples), args)
    print(f"Evaluating {len(sample_indices)}/{len(test_samples)} test samples in this job")

    X_train = np.concatenate([train_source_latents, train_treat_conds], axis=1)
    Y_train = train_target_latents
    print(f"ConditionalSampler input shape: {X_train.shape} -> {Y_train.shape}")

    sampler = load_or_fit_sampler(
        args,
        X_train,
        Y_train,
        z_dim=train_source_latents.shape[1],
        device=device,
    )

    checkpoint_path = os.path.join(
        args.results_dir,
        f"trellis_distributional_regression_{result_stem(args.split_name, args)}_checkpoint.json",
    )

    store, detailed_rows = evaluate_distributional_regression(
        generator=generator,
        sampler=sampler,
        test_samples=test_samples,
        source_latents=test_source_latents,
        target_latents=test_target_latents,
        treat_conds=test_treat_conds,
        sample_info=test_info,
        device=device,
        k_max=args.k_max,
        k_values=k_values,
        m_draws=args.m,
        swd_subsample_rounds=args.swd_subsample_rounds,
        rng=rng,
        sample_indices=sample_indices,
        checkpoint_path=checkpoint_path,
        args=args,
    )

    summary = build_summary(args.split_name, store, k_values)
    print_tables(summary, k_values)
    write_outputs(args.results_dir, args.split_name, summary, detailed_rows, args)
    try_write_combined_shard_outputs(args.results_dir, args.split_name, args, k_values)


if __name__ == "__main__":
    main()
