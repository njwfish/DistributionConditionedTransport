from transformers import EsmModel
from utils.hf_local import resolve_local_or_repo
from encoder.encoders import DistributionEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F

class ESMFeatureExtractor(nn.Module):
    def __init__(self, esm_model_name="facebook/esm2_t6_8M_UR50D", output_dim=320, freeze=False):
        super().__init__()
        local_or_repo = resolve_local_or_repo(esm_model_name)
        self.esm = EsmModel.from_pretrained(local_or_repo)
        if freeze:
            for p in self.esm.parameters(): p.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        # Ensure ESM runs in FP32 to avoid dtype mismatch with fp16 PLM when needed
        prev_dtype = None
        if hasattr(self.esm, 'dtype'):
            prev_dtype = next(self.esm.parameters()).dtype
        x = self.esm(input_ids, attention_mask=attention_mask).last_hidden_state

        return x

class ProteinSetEncoder(nn.Module):
    def __init__(self, esm_model_name="facebook/esm2_t6_8M_UR50D",
                 esm_dim=320, freeze=False):
        super().__init__()
        self.esm_extractor = ESMFeatureExtractor(esm_model_name, esm_dim, freeze)

    def forward(self, samples):
        b, s = samples['esm_input_ids'].shape[:2]
        ids = samples['esm_input_ids'].view(b * s, -1)
        mask = samples['esm_attention_mask'].view(b * s, -1)
        feats = self.esm_extractor(ids, mask).view(b, s, -1)
        return feats