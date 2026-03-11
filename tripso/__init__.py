"""
Modules for tripso method
"""

from pathlib import Path

from . import (
    Datamodules,
    Metrics,
    Models,
    Modules,
    Trainers,
    Utils,
)
from .Evaluate.downstream import gpEval
from .Preprocessing.preprocess import pp_and_tokenize
from .Train.training import run_training as train

# Dictionary files from Geneformer
ENSEMBL_DICTIONARY_FILE = (
    Path(__file__).parent / 'Utils/geneformer_ensembl_dictionary_may2025.pkl'
)
TOKEN_DICTIONARY_FILE = (
    Path(__file__).parent / 'Utils/geneformer_token_dictionary_may2025.pkl'
)

# Geneformer model paths
GF12L95M = '/nfs/team361/mm58/Geneformer/gf-12L-95M-i4096'

__all__ = [
    'Datamodules',
    'ENSEMBL_DICTIONARY_FILE',
    'GF12L95M',
    'Metrics',
    'Models',
    'Modules',
    'pp_and_tokenize',
    'TOKEN_DICTIONARY_FILE',
    'Utils',
    'Trainers',
    'train',
    'gpEval',
]
