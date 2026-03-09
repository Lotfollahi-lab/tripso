from geneformer import ENSEMBL_DICTIONARY_FILE, TOKEN_DICTIONARY_FILE, GENE_MEDIAN_FILE
import pandas as pd
import pickle
from pathlib import Path

# Save to the same directory as this script
save_dir = Path(__file__).parent

ens_dict = pd.read_pickle(ENSEMBL_DICTIONARY_FILE)

# Use pickle to save dictionary (these are plain dicts, not pandas objects)
with open(save_dir / "geneformer_ensembl_dictionary_may2025.pkl", "wb") as f:
    pickle.dump(ens_dict, f)

token_dict = pd.read_pickle(TOKEN_DICTIONARY_FILE)
with open(save_dir / "geneformer_token_dictionary_may2025.pkl", "wb") as f:
    pickle.dump(token_dict, f)
    
gene_median_file = pd.read_pickle(GENE_MEDIAN_FILE)
with open(save_dir / "geneformer_gene_median_file_may2025.pkl", "wb") as f:
    pickle.dump(gene_median_file, f)