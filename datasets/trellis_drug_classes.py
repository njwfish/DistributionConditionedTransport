"""
Trellis treatment codes (one-hot order matches datasets.trellis.trellis_dataset)
and pharmacological class labels for per-condition evaluation reporting.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# Must match trellis_dataset(..., treatment=[...]) default order.
TRELLIS_TREATMENT_CODES: Tuple[str, ...] = (
    "O",
    "S",
    "VS",
    "L",
    "V",
    "F",
    "C",
    "SF",
    "CS",
    "CF",
    "CSF",
)

DRUG_CLASS_METADATA: Dict[str, Dict[str, object]] = {
    "O": {
        "drugs": ["Oxaliplatin"],
        "class": "Cytotoxic chemotherapy (platinum; ribosome biogenesis stress / DNA damage)",
    },
    "S": {
        "drugs": ["SN-38 (irinotecan metabolite)"],
        "class": "Cytotoxic chemotherapy (topoisomerase I inhibitor; replication stress)",
    },
    "VS": {
        "drugs": ["SN-38", "Berzosertib (VX-970)"],
        "class": "Chemotherapy + DNA damage response inhibition (ATR inhibitor combination)",
    },
    "L": {
        "drugs": ["LGK974"],
        "class": "WNT pathway inhibition (PORCN inhibitor; developmental signaling)",
    },
    "V": {
        "drugs": ["Berzosertib (VX-970)"],
        "class": "DNA damage response inhibition (ATR inhibitor)",
    },
    "F": {
        "drugs": ["5-Fluorouracil (5-FU)"],
        "class": "Cytotoxic chemotherapy (antimetabolite; thymidylate synthase inhibition)",
    },
    "C": {
        "drugs": ["Cetuximab"],
        "class": "Targeted therapy (EGFR inhibition)",
    },
    "SF": {
        "drugs": ["SN-38", "5-FU"],
        "class": "Combination chemotherapy (dual replication stress / DNA damage)",
    },
    "CS": {
        "drugs": ["Cetuximab", "SN-38"],
        "class": "Targeted + chemotherapy (EGFR inhibition + replication stress)",
    },
    "CF": {
        "drugs": ["Cetuximab", "5-FU"],
        "class": "Targeted + chemotherapy (EGFR inhibition + antimetabolite)",
    },
    "CSF": {
        "drugs": ["Cetuximab", "SN-38", "5-FU"],
        "class": "Triple combination therapy (EGFR inhibition + dual chemotherapy)",
    },
}


def decode_treatment_code(treat_cond) -> str:
    """Map one-hot treatment row(s) to Trellis condition code (e.g. 'O', 'CSF')."""
    import numpy as np

    arr = np.asarray(treat_cond)
    idx = int(np.argmax(arr[0]))
    return TRELLIS_TREATMENT_CODES[idx]


def class_description(code: str) -> str:
    meta = DRUG_CLASS_METADATA.get(code)
    if not meta:
        return ""
    return str(meta["class"])


def drugs_for_code(code: str) -> List[str]:
    meta = DRUG_CLASS_METADATA.get(code)
    if not meta:
        return []
    return list(meta["drugs"])  # type: ignore[arg-type]
