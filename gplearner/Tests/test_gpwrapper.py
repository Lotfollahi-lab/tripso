import unittest

import numpy as np
import pandas as pd
import torch
from geneformer.tokenizer import TOKEN_DICTIONARY_FILE

from gplearner.Models.gp_model import GENE_NAME_FILE, gpWrapper
from gplearner.Utils.utils import convert_gene_names_to_tokens


class TestGpWrapper(unittest.TestCase):
    def setUp(self):
        # Set up the inputs
        self.gp_inputs = ['A', 'C']

        self.database = pd.DataFrame(
            {
                'A': ['TMPRSS2', 'CXCL8', 'BMP4', 'BCL2A1', 'HEY2'],
                'C': ['CXCL8', 'VEGFA', 'MMP10', 'OAS1', np.nan],
            }
        )

        self.gpdb_tokens = {}
        for gp in self.gp_inputs:
            self.gpdb_tokens[gp] = list(
                convert_gene_names_to_tokens(self.database[gp].values, gp_name=gp)
            )

        # Initialize the gpWrapper model
        self.gp_wrapper = gpWrapper(
            self.gp_inputs,
            self.database,
            do_ensembl_conversion=True,
            gene_token_path=TOKEN_DICTIONARY_FILE,
            gene_name_path=GENE_NAME_FILE,
            gp_latent_size=10,
            n_blocks=2,
            num_heads=2,
            mgm_mask_ratio=0.8,
            use_flash=False,
            model_type='Base',
            learn_new_gp=False,
            use_pos_emb=True,
        )

        # Mock inputs for the model
        self.gf_emb = torch.randn(2, 5, 10)
        self.input_ids = torch.tensor(
            [[15244, 7913, 12504, 1821, 254], [12504, 5616, 11834, 7067, 4093]]
        )

        self.input_dataset = {'input_ids': self.input_ids}

    def test_output_shape(self):
        output = self.gp_wrapper(self.gf_emb, self.input_dataset, masking=True)

        # Check the output shape
        self.assertEqual(output['z'].shape, (2, len(self.gp_inputs), 10))
        self.assertEqual(
            output['logits_lm_list'][0].shape,
            (2, len(self.gpdb_tokens['A']) + 1, len(self.gpdb_tokens['A'])),
        )
        self.assertEqual(
            output['logits_lm_list'][1].shape,
            (2, len(self.gpdb_tokens['C']) + 1, len(self.gpdb_tokens['C'])),
        )

    def test_gene_labels(self):
        output = self.gp_wrapper(self.gf_emb, self.input_dataset, masking=True)

        # Check the gene_labels_list
        self.assertTrue((output['gene_labels_list'][0] == -100).sum() != 0)
        self.assertTrue((output['gene_labels_list'][0] != -100).any())


if __name__ == '__main__':
    unittest.main()
