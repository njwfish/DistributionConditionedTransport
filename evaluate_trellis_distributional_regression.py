import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

from datasets.trellis_drug_classes import decode_treatment_code
from distributional_regression_notebook_utils import fit_conditional_sampler
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
):
    store = initialize_result_store(k_values)
    detailed_rows = []

    for sample_idx, sample in enumerate(test_samples):
        culture, x0, x1, _, _, _, patient = sample
        info = sample_info[sample_idx]

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

            candidate_preds = []
            candidate_metrics = {metric: np.zeros(k_max, dtype=float) for metric in METRICS}
            for candidate_idx in range(k_max):
                pred = generator.sample(
                    x0_tensor,
                    source_latent_tensor,
                    latent_samples[candidate_idx : candidate_idx + 1],
                ).squeeze(0)
                candidate_preds.append(pred)
                metrics = compute_all_metrics(
                    pred, x1_tensor, swd_subsample_rounds=swd_subsample_rounds
                )
                for metric, value in metrics.items():
                    candidate_metrics[metric][candidate_idx] = value

            candidate_preds = torch.stack(candidate_preds, dim=0)

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
            detailed_rows.append(
                {
                    **info,
                    "mode": "oracle",
                    "k": "",
                    "metric": metric,
                    "value": value,
                }
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
                detailed_rows.append(
                    {
                        **info,
                        "mode": "best_of_k",
                        "k": k,
                        "metric": metric,
                        "value": best_value,
                    }
                )
                detailed_rows.append(
                    {
                        **info,
                        "mode": "mixture_of_k",
                        "k": k,
                        "metric": metric,
                        "value": mixture_value,
                    }
                )

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


def write_outputs(results_dir, split_name, summary, detailed_rows, args):
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, f"trellis_distributional_regression_{split_name}.json")
    summary_csv_path = os.path.join(
        results_dir, f"trellis_distributional_regression_{split_name}_summary.csv"
    )
    detailed_csv_path = os.path.join(
        results_dir, f"trellis_distributional_regression_{split_name}_per_sample.csv"
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
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    k_values = parse_k_values(args.k_values, args.k_max)
    sampler_loss_samples = args.sampler_loss_samples or args.k_max

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

    train_source_latents, train_target_latents, train_treat_conds = compute_latents(
        encoder, train_samples, device, split_name="train", use_cell_cond=use_cell_cond
    )
    test_source_latents, test_target_latents, test_treat_conds = compute_latents(
        encoder, test_samples, device, split_name="test", use_cell_cond=use_cell_cond
    )
    test_info = build_sample_info(test_samples)

    X_train = np.concatenate([train_source_latents, train_treat_conds], axis=1)
    Y_train = train_target_latents
    print(f"ConditionalSampler input shape: {X_train.shape} -> {Y_train.shape}")

    sampler = fit_conditional_sampler(
        X_train,
        Y_train,
        z_dim=train_source_latents.shape[1],
        hidden=args.sampler_hidden,
        lr=args.sampler_lr,
        epochs=args.sampler_epochs,
        batch_size=args.sampler_batch_size,
        k=sampler_loss_samples,
        device=device,
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
    )

    summary = build_summary(args.split_name, store, k_values)
    print_tables(summary, k_values)
    write_outputs(args.results_dir, args.split_name, summary, detailed_rows, args)


if __name__ == "__main__":
    main()
