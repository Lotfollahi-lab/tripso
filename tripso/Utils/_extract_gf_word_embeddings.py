from pathlib import Path

import torch
from transformers import BertConfig, BertForMaskedLM

# Path to the Geneformer model
GF12L95M_PATH = '/nfs/team361/mm58/Geneformer/gf-12L-95M-i4096'

geneformer = BertForMaskedLM.from_pretrained(GF12L95M_PATH)

gene_emb_weight = geneformer.bert.embeddings.word_embeddings.weight.data

# Save to the same directory as this script
save_dir = Path(__file__).parent
torch.save(gene_emb_weight, save_dir / 'gf-12L-95M-i4096_word_embeddings_may2025.pt')
