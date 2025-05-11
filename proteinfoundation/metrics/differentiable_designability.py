import argparse
import json, time, os, sys, glob
import shutil
import warnings
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import os.path
import subprocess

from ProteinMPNN.protein_mpnn_utils import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq, tied_featurize, parse_PDB, parse_fasta
from ProteinMPNN.protein_mpnn_utils import StructureDataset, StructureDatasetPDB, ProteinMPNN

from proteinfoundation.utils.align_utils.align_utils import kabsch_align_ind
from transformers import AutoTokenizer, EsmForProteinFolding

from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

def get_args():
    
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--suppress_print", type=int, default=0, help="0 for False, 1 for True")

  
    argparser.add_argument("--ca_only", action="store_true", default=True, help="Parse CA-only structures and use CA-only models (default: false)")   
    argparser.add_argument("--path_to_model_weights", type=str, default="", help="Path to model weights folder;") 
    argparser.add_argument("--model_name", type=str, default="v_48_020", help="ProteinMPNN model name: v_48_002, v_48_010, v_48_020, v_48_030; v_48_010=version with 48 edges 0.10A noise")
    argparser.add_argument("--use_soluble_model", action="store_true", default=False, help="Flag to load ProteinMPNN weights trained on soluble proteins only.")


    argparser.add_argument("--seed", type=int, default=0, help="If set to 0 then a random seed will be picked;")
 
    argparser.add_argument("--save_score", type=int, default=0, help="0 for False, 1 for True; save score=-log_prob to npy files")
    argparser.add_argument("--save_probs", type=int, default=0, help="0 for False, 1 for True; save MPNN predicted probabilites per position")

    argparser.add_argument("--score_only", type=int, default=0, help="0 for False, 1 for True; score input backbone-sequence pairs")
    argparser.add_argument("--path_to_fasta", type=str, default="", help="score provided input sequence in a fasta format; e.g. GGGGGG/PPPPS/WWW for chains A, B, C sorted alphabetically and separated by /")


    argparser.add_argument("--conditional_probs_only", type=int, default=0, help="0 for False, 1 for True; output conditional probabilities p(s_i given the rest of the sequence and backbone)")    
    argparser.add_argument("--conditional_probs_only_backbone", type=int, default=0, help="0 for False, 1 for True; if true output conditional probabilities p(s_i given backbone)") 
    argparser.add_argument("--unconditional_probs_only", type=int, default=0, help="0 for False, 1 for True; output unconditional probabilities p(s_i given backbone) in one forward pass")   
 
    argparser.add_argument("--backbone_noise", type=float, default=0.00, help="Standard deviation of Gaussian noise to add to backbone atoms")
    argparser.add_argument("--num_seq_per_target", type=int, default=8, help="Number of sequences to generate per target")
    argparser.add_argument("--batch_size", type=int, default=128, help="Batch size; can set higher for titan, quadro GPUs, reduce this if running out of GPU memory")
    argparser.add_argument("--max_length", type=int, default=200000, help="Max sequence length")
    argparser.add_argument("--sampling_temp", type=str, default="0.1", help="A string of temperatures, 0.2 0.25 0.5. Sampling temperature for amino acids. Suggested values 0.1, 0.15, 0.2, 0.25, 0.3. Higher values will lead to more diversity.")
    
    argparser.add_argument("--out_folder", type=str, help="Path to a folder to output sequences, e.g. /home/out/")
    argparser.add_argument("--pdb_path", type=str, default='', help="Path to a single PDB to be designed")
    argparser.add_argument("--pdb_path_chains", type=str, default='A', help="Define which chains need to be designed for a single PDB ")
    argparser.add_argument("--jsonl_path", type=str, help="Path to a folder with parsed pdb into jsonl")
    argparser.add_argument("--chain_id_jsonl",type=str, default='', help="Path to a dictionary specifying which chains need to be designed and which ones are fixed, if not specied all chains will be designed.")
    argparser.add_argument("--fixed_positions_jsonl", type=str, default='', help="Path to a dictionary with fixed positions")
    argparser.add_argument("--omit_AAs", type=list, default='X', help="Specify which amino acids should be omitted in the generated sequence, e.g. 'AC' would omit alanine and cystine.")
    argparser.add_argument("--bias_AA_jsonl", type=str, default='', help="Path to a dictionary which specifies AA composion bias if neededi, e.g. {A: -1.1, F: 0.7} would make A less likely and F more likely.")
   
    argparser.add_argument("--bias_by_res_jsonl", default='', help="Path to dictionary with per position bias.") 
    argparser.add_argument("--omit_AA_jsonl", type=str, default='', help="Path to a dictionary which specifies which amino acids need to be omited from design at specific chain indices")
    argparser.add_argument("--pssm_jsonl", type=str, default='', help="Path to a dictionary with pssm")
    argparser.add_argument("--pssm_multi", type=float, default=0.0, help="A value between [0.0, 1.0], 0.0 means do not use pssm, 1.0 ignore MPNN predictions")
    argparser.add_argument("--pssm_threshold", type=float, default=0.0, help="A value between -inf + inf to restric per position AAs")
    argparser.add_argument("--pssm_log_odds_flag", type=int, default=0, help="0 for False, 1 for True")
    argparser.add_argument("--pssm_bias_flag", type=int, default=0, help="0 for False, 1 for True")
    
    argparser.add_argument("--tied_positions_jsonl", type=str, default='', help="Path to a dictionary with tied positions")
    
    return argparser.parse_args(args = [])

class Designability:

    def __init__(self, device):

        args = get_args()

        print_all = args.suppress_print == 0

        if args.path_to_model_weights:
            model_folder_path = args.path_to_model_weights
            if model_folder_path[-1] != '/':
                model_folder_path = model_folder_path + '/'
        else: 
            file_path = os.path.realpath(__file__)
            k = file_path.rfind("/")
            file_path = file_path[:k-1]
            k = file_path.rfind("/")
            file_path = file_path[:k-1]
            k = file_path.rfind("/")
            if args.ca_only:
                print("Using CA-ProteinMPNN!")
                model_folder_path = file_path[:k] + '/ProteinMPNN/ca_model_weights/'
                if args.use_soluble_model:
                    print("WARNING: CA-SolubleMPNN is not available yet")
                    sys.exit()
            else:
                if args.use_soluble_model:
                    print("Using ProteinMPNN trained on soluble proteins only!")
                    model_folder_path = file_path[:k] + '/soluble_model_weights/'
                else:
                    model_folder_path = file_path[:k] + '/vanilla_model_weights/'

        checkpoint_path = model_folder_path + f'{args.model_name}.pt'

        hidden_dim = 128
        num_layers = 3

        checkpoint = torch.load(checkpoint_path, map_location=device)
        noise_level_print = checkpoint['noise_level']
        model = ProteinMPNN(ca_only=args.ca_only, num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers, augment_eps=args.backbone_noise, k_neighbors=checkpoint['num_edges'])
        model.to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        for param in model.parameters():
            param.requires_grad = False

        self.model = model

        if print_all:
            print(40*'-')
            print('Number of edges:', checkpoint['num_edges'])
            print(f'Training noise level: {noise_level_print}A')

        self.args = args

        local_dir = "/home/gridsan/mmorsy/proteins_project/models--facebook--esmfold_v1/snapshots/75a3841ee059df2bf4d56688166c8fb459ddd97a"
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_dir, local_files_only=True
        )
        self.esm_model = EsmForProteinFolding.from_pretrained(
            local_dir, local_files_only=True
        ).to(device)

    def proteinMPNN(self, proteins, return_grad = False):

        args = self.args

        if args.seed:
            seed=args.seed
        else:
            seed=int(np.random.randint(0, high=999, size=1, dtype=int)[0])

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        args.batch_size = min(args.batch_size, proteins.shape[0])
        
        NUM_BATCHES = proteins.shape[0]//args.batch_size
        temperatures = [float(item) for item in args.sampling_temp.split()]
        omit_AAs_list = args.omit_AAs
        alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
        alphabet_dict = dict(zip(alphabet, range(21)))    
        print_all = args.suppress_print == 0 
        omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)
        
        chain_id_dict = None
        fixed_positions_dict = None
        pssm_dict = None
        omit_AA_dict = None
        bias_AA_dict = None
        tied_positions_dict = None
        bias_by_res_dict = None
    
        bias_AAs_np = np.zeros(len(alphabet))

        all_seqs = []
        grads = []
        
        for ix in range(NUM_BATCHES):

            start = ix*args.batch_size
            end = start + args.batch_size

            X = proteins[start:end]
            B,N = X.shape[:2]
            S = torch.zeros((B,N), device=X.device)
            mask = torch.ones((B,N), device=X.device)
            chain_M = torch.ones((B,N), device=X.device)
            chain_encoding_all = torch.zeros((B,N), device=X.device)
            residue_idx = torch.arange(N, device=X.device).unsqueeze(0).expand(B, N)
            chain_M_pos = torch.ones((B,N), device=X.device)
            omit_AA_mask = torch.zeros((B,N,21), device=X.device)
            pssm_coef = torch.zeros((B,N), device=X.device)
            pssm_bias = torch.zeros((B,N,21), device=X.device)
            pssm_log_odds_mask = torch.ones((B,N,21), device=X.device)
            bias_by_res_all = torch.zeros((B,N,21), device=X.device)

            for temp in temperatures:
                randn_2 = torch.randn(chain_M.shape, device=X.device)
                sample_dict = self.model.sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=args.pssm_multi, pssm_log_odds_flag=bool(args.pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(args.pssm_bias_flag), bias_by_res=bias_by_res_all)
                for i in range(B):
                    all_seqs.append(_S_to_seq(sample_dict['S'][i], chain_M[i]))
                
                if return_grad:
                    log_probs = sample_dict['log_probs']
                    # grad = torch.autograd.grad(log_probs, X, grad_outputs=torch.ones_like(log_probs, device='cuda'))[0]
                    # print(torch.autograd.grad(log_probs, X, grad_outputs=torch.ones_like(log_probs, device='cuda'))[0][0])
                    # grads.append(grad)

        return all_seqs, grads

    def scRMSD(self, proteins, return_grad = False):

        proteins.requires_grad_(return_grad)
        
        ns = self.args.num_seq_per_target
        proteins_copied = proteins.repeat_interleave(ns, dim=0)
        with torch.set_grad_enabled(return_grad):
            seqs, log_prob_grads = self.proteinMPNN(proteins_copied, return_grad)
        batch_size = min(128, len(seqs))
        rmsd_list = []
        for i in range(0, len(seqs), batch_size):
            batch_seqs = seqs[i:i+batch_size]
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch_seqs,
                    return_tensors="pt",
                    add_special_tokens=False,
                    padding=True,
                )
                inputs = {k: inputs[k].cuda() for k in inputs}

                outputs = self.esm_model(**inputs)
                atom37_outputs = atom14_to_atom37(outputs["positions"][-1], outputs)
                pred_positions = atom37_outputs[:, :, 1, :]

            pred_positions.requires_grad_(return_grad)

            for j in range(len(pred_positions)):
                coors_1, coors_2 = kabsch_align_ind(pred_positions[j], proteins[(i+j)//ns], ret_both=True)
                sq_err = (coors_1 - coors_2) ** 2
                rmsd_list.append(sq_err.sum(dim=-1).mean().sqrt())

        rmsd_list = torch.stack(rmsd_list)
        rmsd_list = rmsd_list.view(-1, ns)
        scores = rmsd_list.min(dim=-1).values

        if return_grad:

            log_prob_grads = torch.cat(log_prob_grads, dim=0)
            log_prob_grads = log_prob_grads.view(-1, ns, *log_prob_grads.shape[1:]).sum(dim=1)
            rmsd_grad = torch.autograd.grad(scores, proteins, grad_outputs=torch.ones_like(scores))[0]

            return scores, scores[:, None, None] * log_prob_grads + rmsd_grad

        return scores
    
if __name__ == "__main__":
    proteins = torch.tensor([[[  5.3729,   8.8073,   2.3873],
         [  5.8942,   5.1946,   1.2521],
         [  3.5162,   4.0116,   4.1117],
         [  1.3949,   1.8357,   1.8339],
         [  3.5507,   0.1369,  -0.6823],
         [  6.6764,  -0.2397,   1.4322],
         [  5.2484,  -0.6529,   4.9443],
         [  2.0252,  -2.5815,   4.3260],
         [  2.2371,  -4.1959,   0.8842],
         [  5.8321,  -5.4820,   0.9087],
         [  5.7841,  -6.9982,   4.4547],
         [  2.3568,  -8.4808,   3.7001],
         [  3.4322, -10.0425,   0.4264],
         [  6.5987, -11.4922,   2.0883],
         [  4.5901, -12.9874,   4.9367],
         [  1.2327, -14.0800,   3.5104],
         [  2.0603, -14.5919,  -0.2223],
         [  5.6925, -15.9387,  -0.2915],
         [  4.8969, -17.7483,  -3.5211],
         [  3.9663, -14.4902,  -5.2696],
         [  6.9650, -12.5437,  -3.7242],
         [  9.4231, -15.1628,  -5.0334],
         [  7.7469, -15.5815,  -8.4210],
         [  7.8666, -11.6965,  -8.9373],
         [ 11.5162, -11.6978,  -7.9125],
         [ 12.5028, -14.5148, -10.3484],
         [ 10.2372, -13.7188, -13.1924],
         [  9.5747,  -9.9010, -12.9410],
         [  6.4526, -10.6089, -15.0488],
         [  3.8590, -10.1898, -12.2684],
         [  2.5240,  -6.5793, -11.6674],
         [ -0.0483,  -5.1349,  -9.1292],
         [ -0.9959,  -1.6181,  -8.2670],
         [ -0.6700,  -0.1409,  -4.7447],
         [ -2.4371,   3.2062,  -4.0095],
         [ -3.3580,   4.5631,  -0.5366],
         [ -6.1489,   7.0080,  -1.0140],
         [ -4.6122,  10.2619,   0.1361],
         [ -8.3003,  11.1778,   0.6474],
         [ -8.3310,  14.5230,   2.4702],
         [-12.0795,  15.2496,   2.8215],
         [-11.5478,  18.1635,   5.1130],
         [-15.0348,  19.6513,   5.2699],
         [-14.6812,  23.3535,   6.8158],
         [-16.8729,  22.5121,   9.8739],
         [-14.9543,  19.7736,  11.7473],
         [-11.7545,  21.9983,  12.0863],
         [-13.5754,  24.6518,  14.2080],
         [-14.6126,  21.9859,  16.9774],
         [-11.0121,  21.0250,  17.8184]]], device='cuda')
    
    designability = Designability()
    alpha = 1e-5
    scores = designability.scRMSD(proteins, return_grad=False)
    print(scores)