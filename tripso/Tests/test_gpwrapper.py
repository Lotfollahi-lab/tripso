import unittest

import numpy as np
import pandas as pd
import torch

from tripso.Models.gp_model import gpWrapper
from tripso.Utils.utils import convert_gene_names_to_tokens

from .. import ENSEMBL_DICTIONARY_FILE, TOKEN_DICTIONARY_FILE


class TestGpWrapper(unittest.TestCase):
    def setUp(self):
        # Set up the inputs
        self.gp_inputs = ['A', 'C']

        self.database = pd.DataFrame(
            {
                'A': ['TMPRSS2', 'CXCL8', 'BMP4', 'BCL2A1', 'HEY2'],
                # 14988, 12365, 5573, 7842, 7004
                'C': ['CXCL8', 'VEGFA', 'MMP10', 'OAS1', np.nan],
                # 12365, 4064, 11709, 1811
            }
        )

        self.gpdb_tokens = {}

        name_dict = pd.read_pickle(ENSEMBL_DICTIONARY_FILE)
        token_dict = pd.read_pickle(TOKEN_DICTIONARY_FILE)

        for gp in self.gp_inputs:
            self.gpdb_tokens[gp] = list(
                convert_gene_names_to_tokens(
                    self.database[gp].values,
                    gp_name=gp,
                    name_dictionary=name_dict,
                    token_dictionary=token_dict,
                )
            )

        # Initialize the gpWrapper model
        # this code matches dictionaries for Geneformer 4096
        self.gp_wrapper = gpWrapper(
            self.gp_inputs,
            self.database,
            do_ensembl_conversion=True,
            gene_token_path=TOKEN_DICTIONARY_FILE,
            gene_name_path=ENSEMBL_DICTIONARY_FILE,
            gp_latent_size=10,
            n_blocks=2,
            num_heads=2,
            mgm_mask_ratio=0.8,
            use_flash=False,
            model_type='Base',
            learn_new_gp=False,
            use_pos_emb='sin_cos',
            fm_model_input_size=4096,  # goes with dictionary files
            use_l2_norm=False,
        )

        # Mock inputs for the model
        self.gf_emb = {'gene_emb': torch.randn(2, 7, 10)}
        self.input_ids = torch.tensor(
            [
                [
                    14988,
                    7913,
                    5573,
                    1811,
                    12365,
                    41,
                    7004,
                ],  # 3 GP A tokens, 2 GP C tokens
                [14988, 4064, 12365, 7067, 7842, 39, 7003],
            ]  # 2 GP A tokens, 2 GP C tokens
        )

        self.input_dataset = {'input_ids': self.input_ids}

    def test_output_shape(self):
        output = self.gp_wrapper(self.gf_emb, self.input_dataset, masking=True)

        # Check the output shape
        # cls has shape (batch, num_gp, gp_latent_size)
        self.assertEqual(output['z'].shape, (2, len(self.gp_inputs), 10))

        # for each logits has shape
        # (batch, max num_tokens in batch + 1 (cls), num_tokens (num classes))
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
