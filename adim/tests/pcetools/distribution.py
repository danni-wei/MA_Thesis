from abc import ABC, abstractmethod

class Distribution(ABC):
    @abstractmethod
    def __init__(self, *params):
        """
        Конструктор для инициализации параметров распределения.
        """
        pass

    @abstractmethod
    def polynom_coeffs(self, degree):
        """
        Коэффициенты ортогонального полинома степени degree 
        При необходимости значение x нормализуется в зависимости от параметров
        распределения.
        """
        pass
    
    @property
    @abstractmethod
    def linear_transform_coeffs(self):
        pass

    @abstractmethod
    def ppf(self, x):
        pass