import numpy as np


def leapfrog(q, p, grad_U, epsilon, L):
    """
    Standard 2nd-order Störmer-Verlet leapfrog integrator.

    Args:
        q: np.array shape (d,)
        p: np.array shape (d,)
        grad_U: callable, grad_U(q) -> np.array shape (d,)
        epsilon: float
        L: int

    Returns:
        q_new, -p_new, n_grad_evals
    """
    q = q.copy()
    p = p.copy()

    p = p - 0.5 * epsilon * grad_U(q)
    for _ in range(L - 1):
        q = q + epsilon * p
        p = p - epsilon * grad_U(q)
    q = q + epsilon * p
    p = p - 0.5 * epsilon * grad_U(q)

    n_grad_evals = L + 1
    return q, -p, n_grad_evals


def yoshida4(q, p, grad_U, epsilon, L):
    """
    Yoshida 4th-order symplectic integrator.
    3 leapfrog sub-steps per Yoshida step with coefficients w1, w0, w1.

    Args:
        q: np.array shape (d,)
        p: np.array shape (d,)
        grad_U: callable, grad_U(q) -> np.array shape (d,)
        epsilon: float
        L: int — number of Yoshida steps

    Returns:
        q_new, -p_new, n_grad_evals
    """
    cbrt2 = 2.0 ** (1.0 / 3.0)
    w1 = 1.0 / (2.0 - cbrt2)
    w0 = -cbrt2 / (2.0 - cbrt2)
    weights = [w1, w0, w1]

    q = q.copy()
    p = p.copy()
    n_grad_evals = 0

    for _ in range(L):
        for wi in weights:
            h = wi * epsilon
            p = p - 0.5 * h * grad_U(q)
            n_grad_evals += 1
            q = q + h * p
            p = p - 0.5 * h * grad_U(q)
            n_grad_evals += 1

    return q, -p, n_grad_evals
