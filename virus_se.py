import os
os.chdir("/orcd/archive/abugoot/001/Projects/paolo/main_tde/")

import hydra
import argparse
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import numpy as np
import torch
import logging
from utils.seed import seed_everything  # Import seeding utility
from torch.utils.data import DataLoader

from encoder.esm_baseline2 import ProteinSetEncoder
from Bio import pairwise2
from transformers import EsmTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
def load_all(ckpt_dir):
    # Resolve and validate checkpoint directory
    experiment_dir = os.path.abspath(os.path.expanduser(str(ckpt_dir)))
    cfg_path = os.path.join(experiment_dir, 'config.yaml')
    ckpt_path = os.path.join(experiment_dir, 'best_model.pt')
    # Load trained config and use it as the active config
    cfg = OmegaConf.load(cfg_path)

    encoder = hydra.utils.instantiate(cfg.encoder).to(device)
    generator = hydra.utils.instantiate(cfg.generator).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    generator.load_state_dict(checkpoint['generator_state_dict'])
    encoder.eval()
    generator.eval()
    
    dataset = hydra.utils.instantiate(cfg.dataset)

    return cfg, ckpt_path, encoder, generator, dataset


#ckpt_dir = "outputs/virus_time_and_location_5b834ff383274001ef4622150e1d9f12"
ckpt_dir = "outputs/virus_time_only_54488d0be1a9b1ccea388879c81eb082"

cfg, ckpt_path, encoder, generator, dataset = load_all(ckpt_dir)
idx = 0
print(dataset[idx]["source_samples"]["esm_input_ids"].shape)


ESM_baseline_model = ProteinSetEncoder().to(device).eval()

# Initialize ESM tokenizer for decoding
esm_tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')

loader = DataLoader(dataset, batch_size=1, shuffle=True)


#def edit_distance(seq1, seq2):
#    """Compute the edit (Hamming) distance between two sequences of equal length."""
#    if len(seq1) != len(seq2):
#        raise ValueError(f"Sequences must be of equal length for Hamming distance but got {len(seq1)} and {len(seq2)}")
#    return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))

def edit_distance(seq1, seq2):
    score = pairwise2.align.globalxx(seq1, seq2, score_only=True)
    length = max(len(seq1), len(seq2))
    identity = score / length
    return 1 - identity

for j, batch in enumerate(loader):
    if j > 20:
        break
    if j < 10:
        continue
    # For dictionary samples (like PubMed dataset), move tensors to device
    source_samples = {}
    target_samples = {}
    for key, value in batch['source_samples'].items():
        if isinstance(value, torch.Tensor):
            source_samples[key] = value.to(device)
        else:
            source_samples[key] = value
            
    for key, value in batch['target_samples'].items():
        if isinstance(value, torch.Tensor):
            target_samples[key] = value.to(device)
        else:
            target_samples[key] = value
    
    #print(source_samples.keys())
            
    #x_source = x_samples["source_samples"]
    #x_target = x_samples["target_samples"]

    latent_source = encoder(source_samples)
    latent_target = encoder(target_samples)
    #print(batch['source_idx'].shape)
    #print(batch['target_idx'].shape)
    #print(batch['d'].shape)
    #print(source_samples['esm_input_ids'].shape)
    #print(target_samples['esm_attention_mask'].shape)
    #print(latent_source.shape)
    #print(latent_target.shape)
    
    with open("debug_log.log", "a") as f:
        f.write(f"latent_source: {latent_source.shape}\n")
        f.write(f"latent_target: {latent_target.shape}\n")

    ## Decode ESM input IDs back to protein sequences
    decoded_seqs_source = []
    for batch_idx in range(source_samples['esm_input_ids'][0].shape[0]):
        with open("debug_log.log", "a") as f:
            f.write(f"batch_idx: {batch_idx}\n")

        #print("UNDerstanding", source_samples['esm_input_ids'].shape, source_samples['esm_input_ids'][0][batch_idx].shape)
        ids = source_samples['esm_input_ids'][0][batch_idx].cpu().tolist()
        decoded_seq = esm_tokenizer.decode(ids, skip_special_tokens=True).replace(" ", "")
        decoded_seqs_source.append(decoded_seq)

    decoded_seqs_target = []
    for batch_idx in range(target_samples['esm_input_ids'][0].shape[0]):
        with open("debug_log.log", "a") as f:
            f.write(f"batch_idx target: {batch_idx}\n")

        #print("UNDerstanding", source_samples['esm_input_ids'].shape, source_samples['esm_input_ids'][0][batch_idx].shape)
        ids = target_samples['esm_input_ids'][0][batch_idx].cpu().tolist()
        decoded_seq = esm_tokenizer.decode(ids, skip_special_tokens=True).replace(" ", "")
        decoded_seqs_target.append(decoded_seq)
    
    scs_seqs = decoded_seqs_source
    tgt_seqs = decoded_seqs_target



    ## Calculate edit distance between decoded sequences and original raw texts
    #print("EDIT DISTANCES BETWEEN DECODED ESM IDs AND ORIGINAL SEQUENCES:")
    #for i, (decoded_seq, original_seq) in enumerate(zip(decoded_seqs, scs_seqs)):
    #    print(decoded_seq[:10], original_seq[0][:10])
    #    original_seq = original_seq[0][:-2]  # Extract from list if needed
    #    decoded_seq = decoded_seq
    #    edit_dist = edit_distance(decoded_seq, original_seq)
    #    print(f"  Sample {i}: {edit_dist:.4f} with {len(decoded_seq)} and {len(original_seq)}")
    #    print([idx for idx in range(len(decoded_seq)) if decoded_seq[idx] != original_seq[idx]])

    #print("TRUE SOURCE-TARGET EDIT DISTANCES:")
    #print("time-loc: ", source_samples['time-loc'], target_samples['time-loc'])

    #print(scs_seqs)
    #print(tgt_seqs)
    print([edit_distance(scs_seq[0], tgt_seq[0]) for scs_seq, tgt_seq in zip(scs_seqs, tgt_seqs)])
    #print("TEST SEQ:", len(scs_seqs[0]), len(tgt_seqs[0]), scs_seqs[0], tgt_seqs[0])
    latent_source_baseline = ESM_baseline_model(source_samples)
    latent_target_baseline = ESM_baseline_model(target_samples)
    
    #print("!",latent_source_baseline.shape)
    #print(latent_target_baseline.shape)

    
    _, texts = generator.sample(source_samples, latent_source, latent_target, num_samples=16, return_texts = True)
    #print(len(texts),len(texts[0]), len(texts[0][0]))
    print("PREDICTED SOURCE-TARGET EDIT DISTANCES:")
    #print([edit_distance(scs_seq[0][1:-1], tgt_seq) for scs_seq, tgt_seq in zip(scs_seqs, texts[0])])
    for j, text in enumerate(texts[0]):
        with open("debug_log.log", "a") as f:
            f.write(f"text is being compared \n")

            f.write(f"{np.amin([edit_distance(scs_seq[:-2], text) for scs_seq in scs_seqs])} \n")
    #print("--------------------------------")
