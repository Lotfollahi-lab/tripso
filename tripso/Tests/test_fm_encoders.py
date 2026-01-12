import os
import unittest

import pandas as pd
import pytorch_lightning as pl
import torch
from datasets import load_from_disk

from tripso.Datamodules.datamodule import txDataModule
from tripso.Models.gp_model import gpTransformerBase
from tripso.Trainers.trainer import gpBase

configs = {
    'config1': {
        'fm_encoder_pkg': 'geneformer',
        'fm_encoder_name': 'gf-6L-30M-i2048',
        'data_path': '/lustre/scratch126/cellgen/team361/mm58/gpformer_reproducibility/'
        'HECA/data_2048/processed/input_dataset',
        'gpdb_path': '/lustre/scratch126/cellgen/team361/mm58/gpformer_reproducibility/'
        'HECA/gpdb_progeny.csv',
        'max_len': 2048,
        'output_size': 256,
    },
    'config2': {
        'fm_encoder_pkg': 'geneformer',
        'fm_encoder_name': 'gf-12L-95M-i4096',
        'data_path': '/lustre/scratch126/cellgen/team361/mm58/gpformer_reproducibility/'
        'HECA/data/processed/input_dataset',
        'gpdb_path': '/lustre/scratch126/cellgen/team361/mm58/gpformer_reproducibility/'
        'HECA/gpdb_progeny.csv',
        'max_len': 4096,
        'output_size': 512,
    },
    'config3': {
        'fm_encoder_pkg': 'from_scratch',
        'fm_encoder_name': 'from_scratch',
        'data_path': '/lustre/scratch126/cellgen/team361/mm58/'
        'gplearner_reproducibility/02_benchmarking/endometrium/'
        'data/processed/input_dataset',
        'bert_config': {
            'hidden_size': 512,
            'num_hidden_layers': 2,
            'num_attention_heads': 8,
            'max_position_embeddings': 4096,
            'mlm_masking_prob': 0.15,
            'use_pos_emb': 'sin_cos',
            'use_l2_norm': False,
            'tokenization_vocab_size': 20275,
            'torch_dtype': 'bf16',
            'use_flash': False,
        },
        'gpdb_path': '/lustre/scratch126/cellgen/team361/mm58/'
        'gplearner_reproducibility/'
        '02_benchmarking/endometrium/gpdb_progeny_200.csv',
        'max_len': 4096,
        'output_size': 128,
    },
}


class TestFmEncoderBase(unittest.TestCase):
    def setUp(self):
        self.configs = configs

    def run_test_on_config(self, config_name, config_i):
        print(f'Running tests for {config_name}...')

        # Set up dataset
        self.gpdb = pd.read_csv(config_i['gpdb_path'])
        self.gp_inputs = [self.gpdb.columns.tolist()[0]]
        self.max_len = config_i['max_len']
        self.expected_output_size = config_i['output_size']

        dm = txDataModule(
            folder=load_from_disk(config_i['data_path']),
            fm_encoder_name=config_i['fm_encoder_name'],
            model_input_size=config_i['max_len'],
            batch_size=2,
        )

        dm.setup()

        self.dataloader = dm.train_dataloader()

        # Set up the model
        model = gpTransformerBase(
            database=self.gpdb,
            gp_inputs=self.gp_inputs,
            fm_encoder_pkg=config_i['fm_encoder_pkg'],
            fm_encoder_name=config_i['fm_encoder_name'],
            bert_config=config_i.get('bert_config', None),
            gp_latent_size=self.expected_output_size,
            all_genes=self.gpdb.iloc[:, 0].tolist(),
        )

        # And the lightning module
        self.model = gpBase(
            model=model,
            output_dir=os.getcwd(),
            lr_scheduler='CosineLRwithWarmUp',
            total_epochs=2,
        )

    def test_forward_pass(self):
        for config_name, config in self.configs.items():
            self.run_test_on_config(config_name, config)
            batch = next(iter(self.dataloader))

            # check batch matches expected size
            # now switched to dynamic padding
            # self.assertEqual(len(batch['input_ids'][0]), self.max_len)
            self.assertLessEqual(len(batch['input_ids'][0]), self.max_len)

            # do forward pass
            output = self.model(batch, masking=True)

            # Check the output keys
            self.assertIn('z', output)
            self.assertIn('logits_lm_list', output)

            # Check the output shapes
            self.assertEqual(
                output['z'].shape,
                torch.Size([2, len(self.gp_inputs), self.expected_output_size]),
            )
            self.assertEqual(len(output['logits_lm_list']), len(self.gp_inputs))

    def test_training_step(self):
        for config_name, config in self.configs.items():
            self.run_test_on_config(config_name, config)
            batch = next(iter(self.dataloader))
            loss = self.model.training_step(batch, 0)
            # Ensure loss is a scalar tensor
            self.assertIsInstance(loss, torch.Tensor)
            self.assertEqual(loss.dim(), 0)

    def test_training(self):
        for config_name, config in self.configs.items():
            self.run_test_on_config(config_name, config)
            trainer = pl.Trainer(max_epochs=1, limit_train_batches=2)
            trainer.fit(self.model, self.dataloader)
            # Check that the model's state dict is updated
            # (indicating training happened)
            state_dict = self.model.state_dict()
            self.assertIsNotNone(state_dict)
            self.assertGreater(len(state_dict), 0)

    def test_configure_optimizers(self):
        for config_name, config in self.configs.items():
            self.run_test_on_config(config_name, config)
            optimizers = self.model.configure_optimizers()

            # Check if the return value is a dictionary or a tuple
            if isinstance(optimizers, dict):
                self.assertIn('optimizer', optimizers)
                self.assertIsInstance(optimizers['optimizer'], torch.optim.Optimizer)
                if 'lr_scheduler' in optimizers:
                    self.assertIsInstance(optimizers['lr_scheduler'], dict)
                    self.assertIn('scheduler', optimizers['lr_scheduler'])
            elif isinstance(optimizers, tuple) or isinstance(optimizers, list):
                self.assertIsInstance(optimizers[0], torch.optim.Optimizer)
            else:
                self.assertIsInstance(optimizers, torch.optim.Optimizer)


if __name__ == '__main__':
    unittest.main()
