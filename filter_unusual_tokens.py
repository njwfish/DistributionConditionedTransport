#!/usr/bin/env python3
import argparse
import os
from typing import Dict, List, Optional, Sequence

import torch
from transformers import EsmTokenizer, AutoTokenizer, PreTrainedTokenizerBase

from utils.hf_local import resolve_local_or_repo


def load_dataset(path: str) -> List[dict]:
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, list):
        raise TypeError(f"Expected list at root of {path}, got {type(data)}")
    for i, el in enumerate(data):
        if not isinstance(el, dict) or "samples" not in el:
            raise ValueError(f"Element {i} missing 'samples'")
    return data


def standard_aa_ids(tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
    aa_tokens = [
        "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
        "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
    ]
    aa_ids = [int(tokenizer.convert_tokens_to_ids(a)) for a in aa_tokens]
    return torch.tensor(sorted(set(aa_ids)), dtype=torch.long)


def x_id(tokenizer: PreTrainedTokenizerBase) -> Optional[int]:
    vocab = tokenizer.get_vocab()
    xid = vocab.get("X", None)
    if xid is None:
        val = tokenizer.convert_tokens_to_ids("X")
        xid = int(val) if val is not None else None
    return xid


def tensor_isin(values: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """Boolean mask of values membership in allowed (torch >=1.10 supports torch.isin)."""
    try:
        return torch.isin(values, allowed)
    except AttributeError:
        if allowed.numel() == 0:
            return torch.zeros_like(values, dtype=torch.bool)
        out = values == allowed[0]
        for k in range(1, allowed.numel()):
            out |= values == allowed[k]
        return out


def progen_bos_token_id(tokenizer: PreTrainedTokenizerBase) -> Optional[int]:
    """Return the id of the literal '<|bos|>' token if present; otherwise fall back to tokenizer.bos_token_id/cls_token_id."""
    try:
        vocab = tokenizer.get_vocab()
        if "<|bos|>" in vocab:
            return int(vocab["<|bos|>"])
    except Exception:
        pass
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is not None:
        return int(bos_id)
    cls_id = getattr(tokenizer, "cls_token_id", None)
    return int(cls_id) if cls_id is not None else None


def filter_indices_for_esm(
    esm_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    x_fraction_threshold: float,
) -> torch.Tensor:
    """
    Returns a boolean mask of shape [N] where True means keep the sequence.

    ESM exclusion criteria:
      - Must have CLS at index 0 and EOS at index -1
      - All inner tokens (1..L-2) must be one of 20 AAs or 'X'
      - If more than x_fraction_threshold of inner tokens are 'X', exclude
    """
    assert esm_ids.ndim == 2, f"Expected 2D esm_input_ids, got {esm_ids.shape}"
    N, L = esm_ids.shape
    if L < 2:
        # Cannot satisfy BOS/EOS layout; drop all
        return torch.zeros(N, dtype=torch.bool)

    bos_id = tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id
    if bos_id is None or eos_id is None:
        raise ValueError("ESM tokenizer must provide cls_token_id and eos_token_id")

    # Rule 1: CLS at 0 and EOS at -1
    has_bos = esm_ids[:, 0] == bos_id
    has_eos = esm_ids[:, -1] == eos_id
    ok_bos_eos = has_bos & has_eos

    # Build allowed set for inner positions (20 AA + X)
    allowed = standard_aa_ids(tokenizer).tolist()
    xid = x_id(tokenizer)
    if xid is not None:
        allowed.append(int(xid))
    allowed_ids = torch.tensor(sorted(set(allowed)), dtype=torch.long)

    if L >= 3:
        inner = esm_ids[:, 1:-1]
        inner_is_allowed = tensor_isin(inner, allowed_ids)
        # Rule 2: any disallowed token inside -> drop
        inner_all_allowed = inner_is_allowed.all(dim=1)

        # Rule 3: X fraction threshold on inner tokens
        if xid is None:
            x_fraction_ok = torch.ones(N, dtype=torch.bool)
        else:
            inner_len = inner.size(1)
            # Avoid division by zero (shouldn't happen when L>=3)
            inner_len = max(inner_len, 1)
            x_counts = (inner == int(xid)).sum(dim=1)
            x_fraction = x_counts.to(torch.float32) / float(inner_len)
            x_fraction_ok = x_fraction <= float(x_fraction_threshold)
    else:
        # No inner positions when L==2
        inner_all_allowed = torch.ones(N, dtype=torch.bool)
        x_fraction_ok = torch.ones(N, dtype=torch.bool)

    keep = ok_bos_eos & inner_all_allowed & x_fraction_ok
    return keep


def filter_indices_for_progen(
    progen_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
) -> torch.Tensor:
    """
    Returns a boolean mask of shape [N] where True means keep the sequence.

    ProGen exclusion criteria:
      - Must have BOS at index 0
      - All tokens from index 1..end must be one of 20 AAs or 'X'
      - In particular, sequences should NOT end with EOS; last token must be AA or 'X'
    """
    assert progen_ids.ndim == 2, f"Expected 2D progen_input_ids, got {progen_ids.shape}"
    N, L = progen_ids.shape
    if L < 1:
        return torch.zeros(N, dtype=torch.bool)

    bos_id = progen_bos_token_id(tokenizer)
    if bos_id is None:
        # Many GPT-like tokenizers expose bos_token_id
        raise ValueError("ProGen tokenizer must provide '<|bos|>' or bos_token_id (or cls_token_id)")

    has_bos = progen_ids[:, 0] == bos_id

    # Allowed tokens for positions 1..end (20 AA + X)
    allowed = standard_aa_ids(tokenizer).tolist()
    xid = x_id(tokenizer)
    if xid is not None:
        allowed.append(int(xid))
    allowed_ids = torch.tensor(sorted(set(allowed)), dtype=torch.long)

    if L >= 2:
        tail = progen_ids[:, 1:]
        tail_is_allowed = tensor_isin(tail, allowed_ids)
        tail_all_allowed = tail_is_allowed.all(dim=1)
    else:
        # Sequence with only BOS -> not acceptable (must have at least one AA)
        tail_all_allowed = torch.zeros(N, dtype=torch.bool)

    keep = has_bos & tail_all_allowed
    return keep


def apply_keep_mask_to_samples(samples: Dict[str, object], keep_mask: torch.Tensor) -> None:
    """In-place filters all per-sequence arrays in samples to the rows where keep_mask is True.

    We apply the mask to:
      - torch.Tensor objects with leading dimension N
      - Python lists/tuples of length N
    Other entries are left untouched.
    """
    # Determine N from any tensor/list present
    N: Optional[int] = None
    for v in samples.values():
        if isinstance(v, torch.Tensor) and v.ndim >= 1:
            N = int(v.shape[0])
            break
        if isinstance(v, (list, tuple)):
            N = len(v)
            break
    if N is None:
        return

    idx = keep_mask.nonzero(as_tuple=False).view(-1).tolist()

    for k, v in list(samples.items()):
        if isinstance(v, torch.Tensor) and v.ndim >= 1 and int(v.shape[0]) == N:
            samples[k] = v.index_select(0, keep_mask.nonzero(as_tuple=False).view(-1))
        elif isinstance(v, list) and len(v) == N:
            samples[k] = [v[i] for i in idx]
        elif isinstance(v, tuple) and len(v) == N:
            samples[k] = tuple(v[i] for i in idx)
        # else leave unchanged


def _convert_ids_to_tokens(tokenizer: PreTrainedTokenizerBase, ids: Sequence[int]) -> List[str]:
    return [tokenizer.convert_ids_to_tokens(int(t)) for t in ids]


def esm_row_reasons(
    row_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    allowed_ids: torch.Tensor,
    xid: Optional[int],
    x_fraction_threshold: float,
) -> List[str]:
    reasons: List[str] = []
    bos_id = tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id
    L = int(row_ids.numel())
    if L < 2:
        reasons.append("ESM: too short to contain CLS and EOS")
        return reasons
    if bos_id is None or eos_id is None:
        reasons.append("ESM: tokenizer missing CLS/EOS ids")
        return reasons
    if int(row_ids[0].item()) != int(bos_id):
        reasons.append(f"ESM: missing CLS at index 0 (found id={int(row_ids[0].item())}, expected={int(bos_id)})")
    if int(row_ids[-1].item()) != int(eos_id):
        reasons.append(f"ESM: missing EOS at index -1 (found id={int(row_ids[-1].item())}, expected={int(eos_id)})")
    if L >= 3:
        inner = row_ids[1:-1]
        allowed_mask = tensor_isin(inner, allowed_ids)
        if not bool(allowed_mask.all().item()):
            bad_ids = inner[~allowed_mask]
            bad_ids_list = [int(x) for x in bad_ids[:10].tolist()]
            bad_toks = _convert_ids_to_tokens(tokenizer, bad_ids_list)
            reasons.append(
                f"ESM: contains non-AA/non-X tokens in inner positions (count={int((~allowed_mask).sum().item())}, e.g., ids={bad_ids_list}, toks={bad_toks})"
            )
        if xid is not None and inner.numel() > 0:
            x_count = int((inner == int(xid)).sum().item())
            inner_len = int(inner.numel())
            x_frac = x_count / max(inner_len, 1)
            if x_frac > float(x_fraction_threshold):
                reasons.append(
                    f"ESM: 'X' fraction {x_frac:.3f} exceeds threshold {float(x_fraction_threshold):.3f} (X_count={x_count}/{inner_len})"
                )
    return reasons


def progen_row_reasons(
    row_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    allowed_ids: torch.Tensor,
    xid: Optional[int],
) -> List[str]:
    reasons: List[str] = []
    bos_id = progen_bos_token_id(tokenizer)
    eos_id = tokenizer.eos_token_id
    L = int(row_ids.numel())
    if bos_id is None:
        reasons.append("ProGen: tokenizer missing BOS id")
        return reasons
    if L == 0:
        reasons.append("ProGen: empty sequence")
        return reasons
    if int(row_ids[0].item()) != int(bos_id):
        reasons.append(f"ProGen: missing BOS <|bos|> at index 0 (found id={int(row_ids[0].item())}, expected={int(bos_id)})")
    if L >= 2:
        tail = row_ids[1:]
        allowed_mask = tensor_isin(tail, allowed_ids)
        if not bool(allowed_mask.all().item()):
            # Be explicit if last token is EOS
            if eos_id is not None and int(row_ids[-1].item()) == int(eos_id):
                reasons.append(f"ProGen: ends with EOS at last index (id={int(eos_id)})")
            # Generic disallowed tokens in the tail
            bad_ids = tail[~allowed_mask]
            bad_ids_list = [int(x) for x in bad_ids[:10].tolist()]
            bad_toks = _convert_ids_to_tokens(tokenizer, bad_ids_list)
            reasons.append(
                f"ProGen: contains non-AA/non-X tokens in tail (count={int((~allowed_mask).sum().item())}, e.g., ids={bad_ids_list}, toks={bad_toks})"
            )
    else:
        reasons.append("ProGen: has BOS only and no amino-acid tokens")
    return reasons


def main():
    parser = argparse.ArgumentParser(description="Filter sequences with unusual tokens for ESM2 and ProGen2")
    parser.add_argument("--data_in", default="data/spikeprot0430/virus_tokenized_data_for_tde_downsampled100.pt", help="Path to aggregated dataset .pt to filter")
    parser.add_argument("--data_out", default="data/spikeprot0430/virus_tokenized_data_for_tde_downsampled100_filtered_test.pt", help="Path to save filtered dataset .pt (default: <data_in>.filtered.pt)")
    parser.add_argument("--esm_name", default="facebook/esm2_t6_8M_UR50D", help="ESM model id or local path")
    parser.add_argument("--progen_name", default="hugohrban/progen2-base", help="ProGen2 model id or local path")
    parser.add_argument("--x_fraction_threshold", type=float, default=0.10, help="Maximum allowed fraction of 'X' tokens in ESM inner positions")
    args = parser.parse_args()

    data_in = os.path.abspath(args.data_in)
    if not os.path.isfile(data_in):
        raise FileNotFoundError(f"Dataset not found: {data_in}")

    data_out = os.path.abspath(args.data_out) if args.data_out else f"{data_in}.filtered.pt"

    data = load_dataset(data_in)

    esm_model_id = resolve_local_or_repo(args.esm_name)
    esm_tokenizer = EsmTokenizer.from_pretrained(esm_model_id)

    # ProGen may be missing; handle gracefully
    progen_present = any("progen_input_ids" in el.get("samples", {}) for el in data)
    if progen_present:
        progen_model_id = resolve_local_or_repo(args.progen_name)
        progen_tokenizer = AutoTokenizer.from_pretrained(progen_model_id, trust_remote_code=True)
    else:
        progen_tokenizer = None

    total_before: int = 0
    total_after: int = 0
    excluded_so_far: int = 0

    for el in data:
        samples = el["samples"]

        esm_ids: Optional[torch.Tensor] = samples.get("esm_input_ids")
        progen_ids: Optional[torch.Tensor] = samples.get("progen_input_ids")

        if esm_ids is None and progen_ids is None:
            # Nothing to filter in this element
            continue

        # Validate shapes and determine N
        if esm_ids is not None:
            if esm_ids.ndim != 2:
                raise ValueError(f"esm_input_ids must be 2D, got {esm_ids.shape}")
            N, _ = esm_ids.shape
        else:
            if progen_ids is None or progen_ids.ndim != 2:
                raise ValueError("progen_input_ids must be 2D when esm_input_ids absent")
            N, _ = progen_ids.shape

        if progen_ids is not None and progen_ids.shape[0] != N:
            raise ValueError("esm_input_ids and progen_input_ids must have the same number of sequences")

        total_before += N

        keep_mask = torch.ones(N, dtype=torch.bool)

        # Compute keep masks separately to enable diagnostics
        if esm_ids is not None:
            esm_keep = filter_indices_for_esm(esm_ids, esm_tokenizer, args.x_fraction_threshold)
            keep_mask &= esm_keep
        else:
            esm_keep = None

        if progen_ids is not None and progen_tokenizer is not None:
            progen_keep = filter_indices_for_progen(progen_ids, progen_tokenizer)
            keep_mask &= progen_keep
        else:
            progen_keep = None

        # Diagnostics: print every 1000th excluded sequence with reasons and decoded sequences
        excluded_indices = (~keep_mask).nonzero(as_tuple=False).view(-1).tolist()
        if excluded_indices:
            # Precompute allowed sets for detailed reporting
            if esm_ids is not None:
                esm_allowed = standard_aa_ids(esm_tokenizer).tolist()
                esm_xid = x_id(esm_tokenizer)
                if esm_xid is not None:
                    esm_allowed.append(int(esm_xid))
                esm_allowed_ids = torch.tensor(sorted(set(esm_allowed)), dtype=torch.long)
            else:
                esm_allowed_ids = None
                esm_xid = None

            if progen_ids is not None and progen_tokenizer is not None:
                progen_allowed = standard_aa_ids(progen_tokenizer).tolist()
                progen_xid = x_id(progen_tokenizer)
                if progen_xid is not None:
                    progen_allowed.append(int(progen_xid))
                progen_allowed_ids = torch.tensor(sorted(set(progen_allowed)), dtype=torch.long)
            else:
                progen_allowed_ids = None
                progen_xid = None

            for seq_idx in excluded_indices:
                excluded_so_far += 1
                if excluded_so_far % 1000 == 0:
                    print(f"\n=== Excluded sequence #{excluded_so_far} (element={id(el)}, seq_idx={int(seq_idx)}) ===")
                    # Reasons
                    if esm_ids is not None and esm_allowed_ids is not None:
                        esm_reasons = esm_row_reasons(
                            esm_ids[int(seq_idx)], esm_tokenizer, esm_allowed_ids, esm_xid, args.x_fraction_threshold
                        )
                        for r in esm_reasons:
                            print(f"  - {r}")
                        # Decoded ESM sequence
                        try:
                            decoded_esm = esm_tokenizer.decode(esm_ids[int(seq_idx)].tolist(), skip_special_tokens=False)
                        except Exception as e:
                            decoded_esm = f"<decode_error: {e}>"
                        print(f"  ESM decoded: {decoded_esm}")
                    else:
                        print("  (No ESM ids available for diagnostics)")

                    if progen_ids is not None and progen_tokenizer is not None and progen_allowed_ids is not None:
                        progen_reas = progen_row_reasons(
                            progen_ids[int(seq_idx)], progen_tokenizer, progen_allowed_ids, progen_xid
                        )
                        for r in progen_reas:
                            print(f"  - {r}")
                        # Decoded ProGen sequence
                        try:
                            decoded_progen = progen_tokenizer.decode(progen_ids[int(seq_idx)].tolist(), skip_special_tokens=False)
                        except Exception as e:
                            decoded_progen = f"<decode_error: {e}>"
                        print(f"  ProGen decoded: {decoded_progen}")
                    else:
                        print("  (No ProGen ids available for diagnostics)")

        # Apply mask across all per-sequence arrays inside samples
        apply_keep_mask_to_samples(samples, keep_mask)

        # Track new N for this element
        new_esm_ids = samples.get("esm_input_ids")
        if isinstance(new_esm_ids, torch.Tensor) and new_esm_ids.ndim == 2:
            total_after += int(new_esm_ids.shape[0])
        else:
            # Fall back to any other per-seq tensor/list
            any_tensor = None
            for v in samples.values():
                if isinstance(v, torch.Tensor) and v.ndim >= 1:
                    any_tensor = v
                    break
            if any_tensor is not None:
                total_after += int(any_tensor.shape[0])

    # Save filtered dataset
    os.makedirs(os.path.dirname(data_out), exist_ok=True)
    torch.save(data, data_out)

    # Print per-element shapes for esm_input_ids
    for idx, el in enumerate(data):
        ids = el.get("samples", {}).get("esm_input_ids")
        shape_str = str(tuple(ids.shape)) if isinstance(ids, torch.Tensor) else "<missing>"
        print(f"Element {idx} esm_input_ids shape: {shape_str}")

    print(f"Total sequences before: {total_before}")
    print(f"Total sequences after:  {total_after}")
    print(f"Saved filtered dataset to: {data_out}")


if __name__ == "__main__":
    main()


