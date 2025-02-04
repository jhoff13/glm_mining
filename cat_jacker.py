import os
from typing import List
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import time
import argparse
import re

#define globally - tisk tisk
MODEL_NAME = "tattabio/gLM2_650M"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
NUC_TOKENS = tuple(range(29, 33)) # 4 nucleotides a,t,c,g
AA_TOKENS = tuple(range(4,24)) # 20 amino acids
NUM_TOKENS = len(AA_TOKENS) + len(NUC_TOKENS)

model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True).eval().to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
MASK_TOKEN_ID = tokenizer.mask_token_id
TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]'

def get_args():
    parser = argparse.ArgumentParser(description="Compute and save the Cat Jacs of CDS with specified 5' or 3' Intergenic Sequence from a Jett formatted csv format.")
    parser.add_argument('-i','--input', type=str, help='Path to input csv with "csd_seq" and "five_prime_igs"/"three_prime_igs".')
    parser.add_argument('-p','--prime', type=str, help="Define to check the 5' or 3' Intergenic Sequence ['5' or '3'].")
    parser.add_argument('-o','--out_dir', type=str, help='Output directory where results will be saved.')
    parser.add_argument('-f','--fast', type=bool, default=False, help='Run in fast mode with just masking. [Default: False]')
    args = parser.parse_args()
    return args

def clean_string(input_string,verbose=True):
    # Define the pattern of intrusive characters you want to remove
    pattern = r"[\/\|\(\)\']"  # Add any other characters you want to remove inside the brackets
    # Replace all matches of the pattern with an empty string
    cleaned_string = re.sub(pattern, "", input_string)
    if verbose:
        print(f'>> Renamed {input_string} -> {cleaned_string}')
    return cleaned_string

def contact_to_dataframe(con):
  sequence_length = con.shape[0]
  idx = [str(i) for i in np.arange(1, sequence_length + 1)]
  df = pd.DataFrame(con, index=idx, columns=idx)
  df = df.stack().reset_index()
  df.columns = ['i', 'j', 'value']
  return df

def jac_to_contact(jac, symm=True, center=True, diag="remove", apc=True):

  X = jac.copy()
  Lx,Ax,Ly,Ay = X.shape

  if center:
    for i in range(4):
      if X.shape[i] > 1:
        X -= X.mean(i,keepdims=True)

  contacts = np.sqrt(np.square(X).sum((1,3)))

  if symm and (Ax != 20 or Ay != 20):
    contacts = (contacts + contacts.T)/2

  if diag == "remove":
    np.fill_diagonal(contacts,0)

  if diag == "normalize":
    contacts_diag = np.diag(contacts)
    contacts = contacts / np.sqrt(contacts_diag[:,None] * contacts_diag[None,:])

  if apc:
    ap = contacts.sum(0,keepdims=True) * contacts.sum(1, keepdims=True) / contacts.sum()
    contacts = contacts - ap

  if diag == "remove":
    np.fill_diagonal(contacts,0)

  return contacts


def get_categorical_jacobian(sequence: str, fast: bool = False):
  all_tokens = NUC_TOKENS + AA_TOKENS
  num_tokens = len(all_tokens)

  input_ids = torch.tensor(tokenizer.encode(sequence), dtype=torch.int)
  tokens = tokenizer.convert_ids_to_tokens(input_ids)
  seqlen = input_ids.shape[0]
  # [seqlen, 1, seqlen, 1].
  is_nuc_pos = torch.isin(input_ids, torch.tensor(NUC_TOKENS)).view(-1, 1, 1, 1).repeat(1, 1, seqlen, 1)
  # [1, num_tokens, 1, num_tokens].
  is_nuc_token = torch.isin(torch.tensor(all_tokens), torch.tensor(NUC_TOKENS)).view(1, -1, 1, 1).repeat(1, 1, 1, num_tokens)
  # [seqlen, 1, seqlen, 1].
  is_aa_pos = torch.isin(input_ids, torch.tensor(AA_TOKENS)).view(-1, 1, 1, 1).repeat(1, 1, seqlen, 1)
  # [1, num_tokens, 1, num_tokens].
  is_aa_token = torch.isin(torch.tensor(all_tokens), torch.tensor(AA_TOKENS)).view(1, -1, 1, 1).repeat(1, 1, 1, num_tokens)

  input_ids = input_ids.unsqueeze(0).to(DEVICE)
  with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
    f = lambda x:model(x)[0][..., all_tokens].cpu().float()

    x = torch.clone(input_ids).to(DEVICE)
    ln = x.shape[1]

    fx = f(x)[0]
    if fast:
      fx_h = torch.zeros((ln, 1 , ln, num_tokens), dtype=torch.float32)
    else:
      fx_h = torch.zeros((ln,num_tokens,ln,num_tokens),dtype=torch.float32)
      x = torch.tile(x,[num_tokens,1])
    with tqdm(total=ln, bar_format=TQDM_BAR_FORMAT) as pbar:
      for n in range(ln): # for each position
        x_h = torch.clone(x)
        if fast:
          x_h[:, n] = MASK_TOKEN_ID
        else:
          x_h[:, n] = torch.tensor(all_tokens)
        fx_h[n] = f(x_h)
        pbar.update(1)
    jac = fx_h-fx
    valid_nuc = is_nuc_pos & is_nuc_token
    valid_aa = is_aa_pos & is_aa_token
    # Zero out other modality
    jac = torch.where(valid_nuc | valid_aa, jac, 0.0)
    contact = jac_to_contact(jac.numpy())
  return jac, contact, tokens


def create_heatmap(contact_df: pd.DataFrame, tokens: List[str], title='CONSERVATION'):
    seqlen = len(tokens)
    
    # Ensure that the 'i' and 'j' columns are integers.
    contact_df['i'] = contact_df['i'].astype(int)
    contact_df['j'] = contact_df['j'].astype(int)
    
    # Optionally, add token labels to the DataFrame (for non-interactive annotation, these won't appear on the plot)
    contact_df['i_token'] = contact_df['i'].astype(str) + ': ' + contact_df['i'].map(lambda x: tokens[x-1])
    contact_df['j_token'] = contact_df['j'].astype(str) + ': ' + contact_df['j'].map(lambda x: tokens[x-1])
    
    # Pivot the data so that rows represent y-axis positions and columns represent x-axis positions.
    # This assumes that each (i, j) pair appears once.
    heatmap_data = contact_df.pivot(index='j', columns='i', values='value')
    
    # IMPORTANT: To have the origin in the top left, we want the first token (j=1) at the top.
    # Seaborn heatmap, by default, displays the first row at the top.
    # So, do NOT reverse the order of the index.
    heatmap_data = heatmap_data.sort_index(ascending=True)
    
    # Create the figure and axis.
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Use a Blues colormap similar to matplotlib's "Blues".
    cmap = sns.color_palette("Blues", as_cmap=True)
    
    # Plot the heatmap.
    # Note: xticklabels and yticklabels are set to False initially so that we can add custom ticks.
    cax = sns.heatmap(heatmap_data, ax=ax, cmap=cmap, cbar=True, square=True, cbar_kws={'shrink': 0.75}
,
                xticklabels=False, yticklabels=False)
    
    # Set up sparse tick labeling.
    xtick_positions = np.arange(0, seqlen, 20)
    xtick_labels = np.arange(0, seqlen, 20) 
    
    ytick_positions = np.arange(0, seqlen, 20)
    ytick_labels = np.arange(0, seqlen, 20)  

    # For heatmap cells, the ticks should be centered.
    ax.set_xticks(xtick_positions + 0.5)
    ax.set_xticklabels(xtick_labels, rotation=90) 
    
    ax.set_yticks(ytick_positions + 0.5)
    ax.set_yticklabels(ytick_labels, rotation=0)
    
    # Optionally, add labels and a title.
    ax.set_xlabel("Position (bp)")
    ax.set_ylabel("Position (bp)")
    ax.set_title(title)
    
    plt.tight_layout()
    return fig, ax

def create_figure(contact_df: pd.DataFrame, Tokens: List[str], Title, outpath, seq0_len, seq1_len, labels):
    # create heatmap
    fig, ax = create_heatmap(contact_df, Tokens, Title)
    # create boundary labels
    labels = labels.split(' + ')
    
    plt.vlines(seq0_len,0,seq0_len,colors='r',linestyles='dashed',label=labels[0]) 
    plt.hlines(seq0_len,0,seq0_len,colors='r',linestyles='dashed')
    
    plt.vlines(seq0_len,seq0_len,seq0_len+seq1_len,colors='orange',linestyles='dashed',label=labels[1]) 
    plt.hlines(seq0_len,seq0_len,seq0_len+seq1_len,colors='orange',linestyles='dashed')
    
    plt.legend()
    # save
    plt.savefig(outpath, format="png", dpi=500, transparent=True,bbox_inches="tight"); 

def welcome_txt():
    print(' ______  ________  _________   _________ ________  ______       ______  __  __    ')
    print('/_____/\/_______/\/________/\ /________//_______/\/_____/\     /_____/\/_/\/_/\   ')
    print('\:::__\/\::: _  \ \__.::.__\/ \__.::.__\\::: _  \ \:::__\/     \:::_ \ \ \ \ \ \  ')
    print(' \:\ \  _\::(_)  \ \ \::\ \     /_\::\ \ \::(_)  \ \:\ \  __  __\:(_) \ \:\_\ \ \ ')
    print('  \:\ \/_/\:: __  \ \ \::\ \    \:.\::\ \ \:: __  \ \:\ \/_/\/__/\: ___\/\::::_\/ ')
    print('   \:\_\ \ \:.\ \  \ \ \::\ \    \: \  \ \ \:.\ \  \ \:\_\ \ \::\ \ \ \    \::\ \ ')
    print('    \_____\/\__\/\__\/  \__\/     \_____\/  \__\/\__\/\_____\/\:_\/\_\/     \__\/ ')
    print('                                                      SummitHill Biotech, V1.00.01')

def main():
    t0 = time.time()
    welcome_txt()
    args = get_args()
    # load and parse rows in csv file
    DF = pd.read_csv(args.input)
    prime_dict = {'5':'five_prime_igs','3':'three_prime_igs'}
    name_dict = {'5':"5'IGS + icsB",'3':"tnpB + 3'IGS"}
    strand_dict = {'+':'<+>','-':'<->'}
    for i, row in DF.iterrows():
        print('-'*50,f'\n> {i}/{DF.shape[0]} [{(time.time()-t0):.2f} sec] {row.Name}')
        # define seq - 
        cds, five_prime_IGS, three_prime_IGS = row.cds_seq, row.five_prime_igs ,row.three_prime_igs 
        if args.prime == '5':
            seq0 = five_prime_IGS
            seq1 = f'{strand_dict[row.strand]}{cds}'
        elif args.prime =='3':
            seq0 = f'{strand_dict[row.strand]}{cds}'
            seq1 = three_prime_IGS
        else: 
            print(f'ERROR: Unrecognized parameter for --prime [{args.prime}]')
            return None
        sequence = f"{seq0}{seq1}"
        # run cat jac
        J, contact, tokens = get_categorical_jacobian(sequence, fast=args.fast)
        df = contact_to_dataframe(contact)
        # create and save figure:
        out_file = clean_string(row.Name)
        outpath = os.path.join(args.out_dir,f'{out_file}.png')
        figure_dict = {'Title':f"{row.Name}\n{name_dict[args.prime]} Conservation", 'outpath':outpath, 'seq0_len':len(seq0), 'seq1_len':len(seq1), 'labels':name_dict[args.prime]}
        create_figure(df, tokens, **figure_dict)
        print(f' >> Saved to: {outpath}')
    print(f'Kompute Komplete! Elapsed time: {((time.time()-t0)/60):.2f} min')

if __name__ == '__main__':
    main()