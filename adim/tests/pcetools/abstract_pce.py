from abc import ABC, abstractmethod
import torch
from . import config

class AbstractPCE(ABC):
    def __init__(self, vars):
        self.vars = vars
        self.cache = dict()

    @property
    @abstractmethod
    def components(self):
        pass

    @abstractmethod
    def polynom_coeffs(self, var, degree):
        pass

    @abstractmethod
    def linear_transform_coeffs(self, var):
        pass

    def transform(self, var, x):
        a, b = self.linear_transform_coeffs(var)
        return x * a + b

    def polynom(self, var, degree, x):
        value = self.cache.get((var, degree), None)
        if value is not None:
            return value

        polynom_coeffs = self.polynom_coeffs(var, degree)
        assert len(polynom_coeffs) == degree + 1

        p_features = torch.stack(
            [
                self.transform(var, x) ** (degree - j) 
             for j in range(degree+1)
             ], dim=1)

        value = p_features @ polynom_coeffs
        self.cache[(var, degree)] = value
        return value
    
    def _handle_args(self, args):
        # либо один тензор со всеми данными, либо vars векторов
        assert len(args) == 1 or len(args) == self.vars

        if len(args) == 1:
            assert args[0].dim() == 2 and args[0].size(-1) == self.vars
            return args[0]

        sizes = set()
        for i, arg in enumerate(args):
            args[i] = torch.as_tensor(arg, dtype=config.dtype).reshape(-1)
            sizes.add(args[i].size(-1))

        assert len(sizes) <= 2
        if len(sizes) == 1:
            return torch.stack(args, dim=-1)
        
        n = max(sizes)
        for i, arg in enumerate(args):
            if arg.size(-1) == 1:
                args[i] = args[i] * torch.ones(n, dtype=config.dtype)

        return torch.stack(args, dim=-1)
        
    
    def __call__(self, *args):
        X = self._handle_args(list(args))

        assert X.dtype == config.dtype, (
            f"Входные данные должны иметь тип {config.dtype}"
        )
        s = 0
        for pce_coeff, degrees in self.components:
            phi = torch.stack(
                [
                    self.polynom(var_num, degrees[var_num], X[:, var_num]) 
                    for var_num in range(self.vars)
                ], dim=1)
            s += phi.prod(dim=-1) * pce_coeff
        self.cache.clear()
        return s
    
    def derivative(self, var):
        return Derivative(self, var)


class Derivative(AbstractPCE):
    def __init__(self, pce, dvar):
        super().__init__(pce.vars)
        self.pce = pce
        self.dvar = dvar

    @property
    def components(self):
        for pce_coeff, degrees in self.pce.components:
            if degrees[self.dvar]:
                degrees = list(degrees)
                degrees[self.dvar] -= 1
                yield pce_coeff, degrees
            else:
                yield 0.0, degrees
    
    def polynom_coeffs(self, var, degree):
        if var != self.dvar:
            return self.pce.polynom_coeffs(var, degree)

        coeffs = self.pce.polynom_coeffs(var, degree+1)
        assert len(coeffs) == degree + 2
        degrees = torch.arange(degree + 1, -1, -1)
        coeffs *= degrees
        coeffs *= self.pce.linear_transform_coeffs(var)[0]
        return coeffs[:-1] if len(coeffs) > 1 else coeffs

    def linear_transform_coeffs(self, var):
        return self.pce.linear_transform_coeffs(var)