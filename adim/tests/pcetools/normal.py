from scipy.special import hermite
from scipy.stats import norm
from .distribution import Distribution
import torch
from . import config

class Normal(Distribution):
    def __init__(self, mean, std_dev):
        """
        Инициализация нормального распределения с заданными параметрами.
        :param mean: Среднее значение.
        :param std_dev: Стандартное отклонение.
        """
        self.mean = mean
        self.std_dev = std_dev

    def polynom_coeffs(self, degree):
        return torch.tensor(hermite(degree).coeffs, dtype=config.dtype)
    
    @property
    def linear_transform_coeffs(self):
        return (1 / self.std_dev, -self.mean / self.std_dev)
    
    def ppf(self, x):
        return norm.ppf(x, loc=self.mean, scale=self.std_dev)