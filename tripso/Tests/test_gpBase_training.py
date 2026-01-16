import os
import unittest

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from datasets import Dataset

from tripso.Datamodules.datamodule import txDataModule
from tripso.Models.gp_model import gpTransformerBase
from tripso.Trainers.trainer import gpBase


class TestGpBase(unittest.TestCase):
    def setUp(self):
        self.model = gpBase()
        # Create a small dummy dataset
        self.gp_inputs = ['A', 'C']

        self.gpdb = pd.DataFrame(
            {
                'A': ['TMPRSS2', 'CXCL8', 'BMP4', 'BCL2A1', 'HEY2'],
                'C': ['CXCL8', 'VEGFA', 'MMP10', 'OAS1', np.nan],
            }
        )

        dummy_dataset = Dataset.from_dict(
            {
                'input_ids': [
                    [14988, 7913, 5573, 1811, 12365],
                    [14988, 7913, 5573, 1811, 12365],
                    [14988, 7913, 5573, 1811, 12365],
                    [14988, 7913, 5573, 1811, 12365],
                ],
                'length': [5, 5, 5, 5],
            }
        )

        dm = txDataModule(folder=dummy_dataset, batch_size=2)

        dm.setup()

        self.dataloader = dm.train_dataloader()

        # Set up the model
        model = gpTransformerBase(
            database=self.gpdb,
        )

        # And the lightning module
        self.model = gpBase(
            model=model,
            output_dir=os.getcwd(),
            lr_scheduler='CosineLRwithWarmUp',
            total_epochs=2,
        )

    def test_forward_pass(self):
        batch = next(iter(self.dataloader))
        # x, _ = batch
        output = self.model(batch, masking=True)
        # Check the output keys
        self.assertIn('z', output)
        self.assertIn('logits_lm_list', output)

        # Check the output shapes
        # shape.[0] is the batch size
        self.assertEqual(output['z'].shape, torch.Size([2, len(self.gp_inputs), 256]))
        self.assertEqual(len(output['logits_lm_list']), len(self.gp_inputs))
        self.assertEqual(
            output['logits_lm_list'][0].shape,
            torch.Size([2, 6, len(self.gpdb['A'].dropna())]),
        )
        self.assertEqual(
            output['logits_lm_list'][1].shape,
            torch.Size([2, 5, len(self.gpdb['C'].dropna())]),
        )

    def test_training_step(self):
        batch = next(iter(self.dataloader))
        loss = self.model.training_step(batch, 0)
        # Ensure loss is a scalar tensor
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)

    def test_training(self):
        trainer = pl.Trainer(max_epochs=1, limit_train_batches=2)
        trainer.fit(self.model, self.dataloader)
        # Check that the model's state dict is updated (indicating training happened)
        state_dict = self.model.state_dict()
        self.assertIsNotNone(state_dict)
        self.assertGreater(len(state_dict), 0)

    def test_configure_optimizers(self):
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
