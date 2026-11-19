import tempfile
import unittest
from pathlib import Path

import torch

from charm_pipeline.common import require_fresh, validate_domains
from charm_pipeline.__main__ import parser, validate_training_splits


class PipelineTests(unittest.TestCase):
    def test_domain_validation(self):
        validate_domains(['Grocery', 'Movies'], 'Toys')
        for sources, target in [(['Grocery', 'Grocery'], 'Toys'),
                                (['Grocery', 'Movies'], 'Movies'),
                                (['../Grocery', 'Movies'], 'Toys')]:
            with self.assertRaises(ValueError):
                validate_domains(sources, target)

    def test_protect_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            require_fresh(directory)
            (directory / 'checkpoint.bin').write_text('old')
            with self.assertRaises(FileExistsError):
                require_fresh(directory)

    def test_nc_requires_and_respects_test_mask(self):
        from types import SimpleNamespace
        from model.eval.eval_magb_graph_zeroshot import set_eval_indices
        with tempfile.TemporaryDirectory() as tmp:
            dataset = SimpleNamespace(labels=torch.tensor([0, 1]), records=[{}, {}],
                                      graph_label_spec={'valid_id2label': {0: 'a', 1: 'b'}},
                                      data_args=SimpleNamespace(graph_data_root=tmp))
            with self.assertRaises(FileNotFoundError):
                set_eval_indices(dataset, -1, 'first', 0, 'test')
            torch.save(torch.tensor([False, True]), Path(tmp) / 'test_mask.pt')
            set_eval_indices(dataset, -1, 'first', 0, 'test')
            self.assertEqual(dataset.sample_indices, [1])

    def test_prevents_one_class_training_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for domain in ['Grocery', 'Movies']:
                (root / domain).mkdir()
                torch.save({'train': {'pos_edges': [(0, 1)] * 3,
                                      'neg_edges': [(0, 2)] * 3}},
                           root / domain / 'lp_zeroshot_split.pt')
            with self.assertRaises(ValueError):
                validate_training_splits(root, ['Grocery', 'Movies'], 3)
            validate_training_splits(root, ['Grocery', 'Movies'], 6)



if __name__ == '__main__':
    unittest.main()
