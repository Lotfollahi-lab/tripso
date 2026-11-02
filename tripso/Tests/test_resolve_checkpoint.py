import os
import tempfile
import unittest

from tripso.Utils.utils import resolve_checkpoint


class TestResolveCheckpoint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.model_dir = self.tmp.name
        self.ckpt_dir = os.path.join(self.model_dir, 'checkpoints')
        os.makedirs(self.ckpt_dir)

        self.dated = '2026-01-01_gp_transformer_pert_Base_abc.ckpt'
        self._touch(self.dated)
        self._touch('last.ckpt')

    def tearDown(self):
        self.tmp.cleanup()

    def _touch(self, name):
        path = os.path.join(self.ckpt_dir, name)
        open(path, 'w').close()
        return path

    def test_defaults_to_last(self):
        # the dated best-loss file must not be picked over last.ckpt
        self.assertEqual(
            resolve_checkpoint(self.model_dir),
            os.path.join(self.ckpt_dir, 'last.ckpt'),
        )

    def test_versioned_last_raises(self):
        self._touch('last-v1.ckpt')
        with self.assertRaises(RuntimeError):
            resolve_checkpoint(self.model_dir)

    def test_override_by_filename(self):
        v1 = self._touch('last-v1.ckpt')
        self.assertEqual(resolve_checkpoint(self.model_dir, 'last-v1.ckpt'), v1)
        self.assertEqual(
            resolve_checkpoint(self.model_dir, self.dated),
            os.path.join(self.ckpt_dir, self.dated),
        )

    def test_override_by_full_path(self):
        v1 = self._touch('last-v1.ckpt')
        self.assertEqual(resolve_checkpoint('/nonexistent', v1), v1)

    def test_missing_last_raises(self):
        os.remove(os.path.join(self.ckpt_dir, 'last.ckpt'))
        with self.assertRaises(FileNotFoundError):
            resolve_checkpoint(self.model_dir)

    def test_missing_override_raises(self):
        with self.assertRaises(FileNotFoundError):
            resolve_checkpoint(self.model_dir, 'nope.ckpt')


if __name__ == '__main__':
    unittest.main()
