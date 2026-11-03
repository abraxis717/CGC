import unittest
import torch
from cgc import CGC120, analytic_grad, forward, forward_s3, gd_step, mse_loss, project_s3

class TestCGC(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(17)
        self.w = 0.03 * torch.randn(5, 6, dtype=torch.float64, generator=g)
        self.x = 0.2 * torch.randn(12, 5, dtype=torch.float64, generator=g)

    def test_s3_parity(self):
        self.assertTrue(torch.allclose(forward(self.w, self.x),
                                       forward_s3(*project_s3(self.w), self.x),
                                       atol=1e-11, rtol=1e-11))

    def test_analytic_gradient(self):
        w = self.w.detach().clone().requires_grad_(True)
        y = torch.zeros(self.x.shape[0], dtype=torch.float64)
        self.assertTrue(torch.allclose(analytic_grad(w, self.x, y),
                                       torch.autograd.grad(mse_loss(w, self.x, y), w)[0],
                                       atol=1e-11, rtol=1e-11))

    def test_recurrent_differentiability(self):
        w0 = self.w.detach().clone().requires_grad_(True)
        y = torch.zeros(self.x.shape[0], dtype=torch.float64)
        w = w0
        for _ in range(4):
            w = gd_step(w, self.x, y, 0.01)
        gradient = torch.autograd.grad(mse_loss(w, self.x, y), w0)[0]
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.abs().sum()), 0)

    def test_four_channels(self):
        weights = torch.stack([self.w.clone() for _ in range(4)])
        model = CGC120(weights)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 120)
        self.assertEqual(tuple(model(self.x).shape), (12, 4))

if __name__ == '__main__':
    unittest.main()
