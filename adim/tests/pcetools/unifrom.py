from scipy.special import legendre
from scipy.stats import uniform
from .distribution import Distribution
import torch
from . import config

class Uniform(Distribution):
    def __init__(self, a, b):
        """
        Инициализация равномерного распределения на отрезке [a, b].
        :param a: Нижний предел интервала.
        :param b: Верхний предел интервала.
        """
        self.a = a
        self.b = b

    def polynom_coeffs(self, degree):
        return torch.tensor(legendre(degree).coeffs, dtype=config.dtype)
    
    @property
    def linear_transform_coeffs(self):
        return (2 / (self.b - self.a), -(self.b + self.a) / (self.b - self.a))
    
    def ppf(self, x):
        return uniform.ppf(x, loc=self.a, scale=self.b-self.a)