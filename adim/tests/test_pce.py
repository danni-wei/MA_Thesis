import unittest
import torch
from pcetools import PCE, Uniform
from scipy.special import eval_legendre
import numpy as np
from pcetools import config


class TestPCE(unittest.TestCase):
    def test_calculate23(self):
        p = 3
        a1, b1 = -1, 1
        a2, b2 = 0, 1
        pce_coeffs = [1.0, 2.0, -0.5, -3, 0.8, -0.2, 0.7, 0.4, -2, 0.1]
        x1, y1 = 0.3, 0.5
        x2, y2 = -0.6, 0.1

        u = PCE([Uniform(a1, b1), Uniform(a2, b2)], p)
        u.pce_coeffs = torch.tensor(pce_coeffs, dtype=config.dtype)

        def f(x, degree):
            return eval_legendre(degree, x)
        
        def calc_u(x, y):
            x = 2 * (x - a1) / (b1 - a1) - 1
            y = 2 * (y - a2) / (b2 - a2) - 1

            vals = [
                f(x, 3), f(x, 2)*f(y, 1), f(x, 1)*f(y, 2), f(y, 3),
                f(x, 2), f(x, 1)*f(y, 1), f(y, 2),
                f(x, 1), f(y, 1),
                1.0
                    ]
            return np.dot(pce_coeffs, vals)

        u1, u2 = calc_u(x1, y1), calc_u(x2, y2)

        X = torch.tensor([
            [x1, y1],
            [x2, y2]
        ], dtype=config.dtype)

        ux = u(X)
        self.assertAlmostEqual(u1, ux[0].item(), places=7)
        self.assertAlmostEqual(u2, ux[1].item(), places=7)


if __name__ == "main":
    unittest.main()