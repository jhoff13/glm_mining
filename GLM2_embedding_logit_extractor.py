import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import softmax
import time

TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device Loaded: {DEVICE}')

def get_args():
    parser = argparse.ArgumentParser(description="")

    # Mandatory arguments
    parser.add_argument('-i','--csv_path', type=str, help='Path to the input csv.')
    parser.add_argument('-o','--out_dir', type=str, help='Output directory where npz files saved.')
    # Optional args:
    parser.add_argument('-m','--model',default="tattabio/gLM2_650M", type=str, help='gLM2 Model to Load [Default: tattabio/gLM2_650M].')
    
    return parser.parse_args()
    
def find_motif(tokens, motif):
    """
    Find the motif in the tokens.

    Parameters:
        tokens: numpy array, list, or PyTorch tensor
            The input sequence to search for the motif.
        motif: list or numpy array
            The motif to search for.

    Returns:
        List of start indices where the motif matches in the tokens.
    """
    #convert ad detach tokens
    tokens = tokens.detach().cpu().numpy()
    motif = motif.detach().cpu().numpy()
    # Find matches using a sliding window
    matches = []
    motif_len = motif.shape[0] 
    for i in range(len(tokens) - motif_len + 1):
        if np.array_equal(tokens[i:i + motif_len], motif):
            matches.append(i)  # Store the start index of each match
    assert matches, "No matches found for the motif in the tokens."
    if len(matches)>1: print(f'{len(matches)} matches - reporting first hit')
    return matches[0], matches[0]+motif_len
    
def get_logits(X):
    """Adapted from Sergey's version"""
    ln = X.shape[1]
    f = lambda x: MODEL(x).logits[0, :, 4:24].detach().cpu().numpy()
    logits = []
    #print(f'get_logits: {X.shape}')
    with tqdm(total=ln, bar_format=TQDM_BAR_FORMAT) as pbar:
        for i in range(0,X.shape[1]):
          x_ = torch.clone(X)
          x_[0,i] = MASK_TOKEN_ID
          logits.append(f(x_)[i])
          pbar.update(1)
    logits = np.array(logits)
    return logits
    
def get_GLM_embeddings(X):
   # should this not include the starting token? X[1:]
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        embedding = MODEL(X.unsqueeze(0),output_hidden_states=True).hidden_states[-1]
    return embedding.detach().cpu().numpy()

def glm_workflow(DF, Outpath):
    #encode seq
    seq = []
    strand_dict = {'+':'<+>','-':'<->'}
    for i, row in DF.iterrows():
        print(f'Komputing {row.Name}')
        strand = strand_dict[row.strand]
        ref_seq = strand + row.cds_seq
        seq = row.five_prime_igs.lower() + ref_seq + row.three_prime_igs.lower()
        
        #tokenize - 
        full_x = torch.tensor(TOKENIZER(seq)["input_ids"]).to(DEVICE)
        ref_x = torch.tensor(TOKENIZER(ref_seq)["input_ids"]).to(DEVICE)
        ref_start, ref_end = find_motif(full_x, ref_x)
        
        if (full_x[ref_start:ref_end] == ref_x).all() == False:
            continue
        
        #calculations:
        emb = get_GLM_embeddings(full_x).squeeze()[ref_start:ref_end].mean(0) 
        log = get_logits(full_x.unsqueeze(0))#[ref_start:ref_end]
    
        #save to outfile
        outfile = os.path.join(Outpath,f'{row.Name}.npz')
        print(f'> Saved to {outfile}')
        np.savez_compressed(outfile, a=emb, b=log)

def main():
    t0 = time.time()
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    global MODEL, TOKENIZER, MASK_TOKEN_ID 
    MODEL_NAME = args.model
    MODEL = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True).eval().to(DEVICE)
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    MASK_TOKEN_ID = TOKENIZER.mask_token_id
    print(f'Model Loaded: {os.path.basename(MODEL_NAME)}')
    
    df = pd.read_csv(args.csv_path)
    df['strand'] = df.hit_ids.str.split('|').str[-2]
    df['Name'] = df.hit_ids.str.split('|').str[3]

    glm_workflow(df, args.out_dir)
    
    print(f'Kompute Komplete - Time Elapsed: {(time.time() - t0)/60:.3f} min')

if __name__ == "__main__":
    main()