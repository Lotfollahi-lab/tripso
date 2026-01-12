import unittest

import torch

from tripso.Utils.utils import build_gp_input_matrix


class TestBuildGPInputMatrix(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

        # gf embeddings (b, s, e)
        self.gf = torch.randn(2, 5, 10)

        # input ids (b, s)
        self.input_ids = torch.tensor([[1, 4, 3, 8, 24], [32, 7, 6, 9, 1]])

        self.gp_tokens = torch.tensor([1, 4, 5, 8, 9])

        # Expected values
        self.expected_labels = torch.tensor([[1, 4, 8], [9, 1, 0]])
        self.expected_num_genes_per_cell = torch.tensor([3, 2])
        self.expected_attn_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
        self.expected_result_matrix_zeros = torch.zeros(10)

    def test_input_matrix_shapes(self):
        (
            result_matrix,
            masked_labels_output,
            num_genes_per_cell,
            attn_mask,
        ) = build_gp_input_matrix(self.gf, self.input_ids, self.gp_tokens)

        # Check the shapes
        self.assertEqual(result_matrix.shape, (2, 3, 10))
        self.assertEqual(masked_labels_output.shape, (2, 3))
        self.assertEqual(num_genes_per_cell.shape, (2,))
        self.assertEqual(attn_mask.shape, (2, 4))

    def test_input_matrix_values(self):
        (
            result_matrix,
            masked_labels_output,
            num_genes_per_cell,
            attn_mask,
        ) = build_gp_input_matrix(self.gf, self.input_ids, self.gp_tokens)

        # Check values
        self.assertTrue(torch.allclose(masked_labels_output, self.expected_labels))
        self.assertTrue(
            torch.equal(num_genes_per_cell, self.expected_num_genes_per_cell)
        )
        self.assertTrue(torch.allclose(attn_mask, self.expected_attn_mask))

        # Check the values of the result matrix
        self.assertTrue(torch.equal(result_matrix[0, 0], self.gf[0, 0]))
        self.assertTrue(torch.equal(result_matrix[0, 1], self.gf[0, 1]))
        self.assertTrue(torch.equal(result_matrix[0, 2], self.gf[0, 3]))

        self.assertTrue(torch.equal(result_matrix[1, 0], self.gf[1, 3]))
        self.assertTrue(torch.equal(result_matrix[1, 1], self.gf[1, 4]))
        self.assertTrue(
            torch.allclose(result_matrix[1, 2], self.expected_result_matrix_zeros)
        )


if __name__ == '__main__':
    unittest.main()
