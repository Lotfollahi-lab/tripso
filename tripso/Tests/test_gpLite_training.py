import os
import unittest

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from datasets import Dataset

from tripso.Datamodules.datamodule import txDataModule
from tripso.Models.gp_model import gpTransformerBase, gpTransformerLite
from tripso.Trainers.trainer import gpLite


class TestGpLite(unittest.TestCase):
    def setUp(self):
        # Create a small dummy dataset
        self.gp_inputs = ['A', 'C']

        self.gpdb = pd.DataFrame(
            {
                'A': ['TMPRSS2', 'CXCL8', 'BMP4', 'BCL2A1', 'HEY2'],
                'C': ['CXCL8', 'VEGFA', 'MMP10', 'OAS1', np.nan],
            }
        )

        # Build the model first so the dummy input_ids can be derived from the
        # GP tokens the current tokenizer actually resolves. Hard-coding token
        # ids couples the test to one Geneformer vocab version; a GP with zero
        # matches yields an empty sequence and crashes build_gp_input_matrix.
        model = gpTransformerLite(
            database=self.gpdb,
        )
        self.gp_token_counts = [
            len(getattr(model.multi_gp_encoder, f'gp{i}_tokens'))
            for i in range(len(self.gp_inputs))
        ]
        # Union of every GP's tokens -> each cell contains every GP gene, so
        # each GP's cropped sequence length equals its full token count.
        all_tokens = sorted(
            {
                int(t)
                for i in range(len(self.gp_inputs))
                for t in getattr(model.multi_gp_encoder, f'gp{i}_tokens')
            }
        )

        dummy_dataset = Dataset.from_dict(
            {
                'input_ids': [all_tokens] * 4,
                'length': [len(all_tokens)] * 4,
            }
        )

        dm = txDataModule(
            folder=dummy_dataset,
            batch_size=2,
            model_input_size=max(10, len(all_tokens)),
        )

        dm.setup()

        self.dataloader = dm.train_dataloader()

        # And the lightning module
        self.model = gpLite(
            model=model,
            output_dir=os.getcwd(),
            lr_scheduler='CosineLRwithWarmUp',
            total_epochs=2,
        )

    def test_forward_pass(self):
        batch = next(iter(self.dataloader))
        output = self.model(batch, masking=True, epoch=0)
        # Check the output keys
        self.assertIn('z', output)
        self.assertIn('logits_lm_list', output)

        # Output shapes match gpTransformerBase exactly (only the GP body is
        # shared; heads stay per-GP sized). Sequence length = GP token count + 1
        # (CLS); vocab dim = GP token count.
        n0, n1 = self.gp_token_counts
        self.assertEqual(output['z'].shape, torch.Size([2, len(self.gp_inputs), 256]))
        self.assertEqual(len(output['logits_lm_list']), len(self.gp_inputs))
        self.assertEqual(
            output['logits_lm_list'][0].shape,
            torch.Size([2, n0 + 1, n0]),
        )
        self.assertEqual(
            output['logits_lm_list'][1].shape,
            torch.Size([2, n1 + 1, n1]),
        )

    def test_shared_body(self):
        # A single shared transformer body (not a per-GP ModuleList) ...
        wrapper = self.model.model.multi_gp_encoder
        self.assertTrue(hasattr(wrapper.encoder, 'blocks'))
        # ... plus one MLM head per gene program.
        self.assertEqual(len(wrapper.decoders), len(self.gp_inputs))

        # The GP-transformer parameter count is materially lower than the Base
        # model, which builds one full transformer per GP.
        base = gpTransformerBase(database=self.gpdb)

        def gp_param_count(m):
            return sum(p.numel() for p in m.multi_gp_encoder.parameters())

        self.assertLess(gp_param_count(self.model.model), gp_param_count(base))

    def test_training_step(self):
        batch = next(iter(self.dataloader))
        loss = self.model.training_step(batch, 0)
        # Ensure loss is a scalar tensor
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)

    def test_training(self):
        trainer = pl.Trainer(max_epochs=1, limit_train_batches=2)
        trainer.fit(self.model, self.dataloader)
        # Check that the model's state dict is updated
        state_dict = self.model.state_dict()
        self.assertIsNotNone(state_dict)
        self.assertGreater(len(state_dict), 0)

    def test_configure_optimizers(self):
        optimizers = self.model.configure_optimizers()

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
