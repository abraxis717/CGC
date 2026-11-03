#!/usr/bin/env python3
"""Standalone deterministic mechanics tests for the R97W portable runner."""
import copy
import json
import math
import pathlib
import unittest
import torch
import experiment as exp

class TestPortableR97W(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads((pathlib.Path(__file__).resolve().parent / 'SPEC_PORTABLE.json').read_text())
        cls.api = exp.verify_authority(exp.repo_root(), cls.spec)

    def test_parameter_parity(self):
        a = exp.init_state('ECS', 17, self.spec)
        b = exp.init_state('FFN', 17, self.spec)
        self.assertEqual(sum(p.numel() for p in a), 120)
        self.assertEqual(sum(p.numel() for p in b), 120)
        self.assertEqual(tuple(a[0].shape), (3, 5, 8))

    def test_shapes_and_gradients(self):
        support, query, ood = exp.teacher('smooth', 17, 0, 'UNIT', self.spec)
        self.assertEqual(tuple(support[0].shape), (32, 5))
        a = exp.init_state('ECS', 17, self.spec)
        loss = float(exp.mse(exp.predict('ECS', a, query[0], self.api), query[1]))
        self.assertTrue(math.isfinite(loss))
        curve, after = exp.eval_episode('ECS', a, (support, query, ood), self.spec, self.api)
        self.assertEqual(sorted(curve), self.spec['protocol']['eval_steps'])
        self.assertTrue(any(not torch.equal(x, y) for x, y in zip(a, after)))

    def test_synthetic_splits(self):
        a = exp.teacher('smooth', 17, 0, 'TRAIN', self.spec)[0][0]
        b = exp.teacher('smooth', 17, 0, 'TEST_R97W_FINAL_BLIND', self.spec)[0][0]
        self.assertFalse(torch.equal(a,b))

    def test_meta_training_mechanics(self):
        spec = copy.deepcopy(self.spec)
        spec['protocol'].update(train_episodes=5, dev_checkpoints=[0,5], dev_episodes=2,
            test_episodes=2, inner_steps=4, eval_steps=[0,1,2,4],
            forget_probe_episodes=3, progress_every=5)
        states, trace, forgot = exp.training_run('smooth', 17, spec, self.api)
        self.assertEqual(len(trace),2)
        self.assertEqual(len(forgot),3)
        rows, summary = exp.evaluate_group('smooth',17,states,spec,self.api,'UNIT_TEST',2)
        result = exp.summarize_results('smooth',17,rows,summary,forgot,spec)
        self.assertEqual(result['comparisons']['ECS_COLD']['n'], 2)
        self.assertIsInstance(result['learning_after_forgetting_utility'],bool)

if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
