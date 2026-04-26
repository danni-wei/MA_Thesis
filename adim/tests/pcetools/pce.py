from .abstract_pce import AbstractPCE
import torch
from . import config
from typing import overload, Iterable, Union, Optional
from .distribution import Distribution
from scipy.stats import qmc

class PCE(AbstractPCE):
    @overload
    def __init__(self, 
                 distributions: Iterable[Distribution], 
                 p: int, 
                 pce_coeffs: Optional[torch.FloatTensor]=None, 
                 names: Optional[Iterable[str]]=None
                 ) -> None:
        ...

    @overload
    def __init__(self, 
                 distributions: Iterable[Distribution], 
                 degrees_sets: Iterable[Iterable[int]], 
                 pce_coeffs: Optional[torch.FloatTensor]=None, 
                 names: Optional[Iterable[str]]=None
                 ) -> None:
        ...

    def __init__(self, 
                distributions: Iterable[Distribution], 
                p: Union[int, Iterable[Iterable[int]]], 
                pce_coeffs: Optional[torch.FloatTensor]=None, 
                names: Optional[Iterable[str]]=None
                ) -> None:
        super().__init__(len(distributions))
        self.distributions = distributions
        self.names = names if names is not None else []

        if isinstance(p, int):
            self.degrees_sets = list(self.generate_degrees(self.vars, p)) 
        else:
            self.degrees_sets = p

        if pce_coeffs is not None:
            self.pce_coeffs = pce_coeffs
        else:
            self.pce_coeffs = torch.zeros(len(self.degrees_sets))

    @staticmethod
    def generate_degrees(m, p):
        def degrees_with_sum(m, s):
            if m == 1:
                yield [s]
                return
            if s == 0:
                yield [0] * m
                return
            for i in range(s+1):
                for right_part in degrees_with_sum(m-1, i):
                    yield [s-i] + right_part
        
        for s in range(p, -1, -1):
            yield from degrees_with_sum(m, s)
        return

    @property
    def components(self):
        return zip(self.pce_coeffs, self.degrees_sets)
    
    def polynom_coeffs(self, var, degree):
        return self.distributions[var].polynom_coeffs(degree)
    
    def linear_transform_coeffs(self, var):
        return self.distributions[var].linear_transform_coeffs
    
    def sample(self, n):
        sampler = qmc.LatinHypercube(self.vars)
        p_samples = sampler.random(n)
        for var in range(self.vars):
            p_samples[:, var] = self.distributions[var].ppf(p_samples[:, var])
        return torch.tensor(p_samples, dtype=config.dtype)