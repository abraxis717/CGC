"""Cubic Geometric Core (CGC): research PyTorch reference."""
from .ecsg_e0 import ECSGE0Cell, analytic_grad, forward, forward_s3, gd_step, mse_loss, phi, phip, project_s3, require_float64
from .four_channel import CGC120
