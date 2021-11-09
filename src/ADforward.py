import itertools
import numpy as np


class AutoDiff():
    def __init__(self, value, derivative=1):
        self.value = value
        self.derivative = derivative
    
    def __add__(self, other):
        try:
            value = self.value + other.value
            derivative = self.derivative + other.derivative
        except AttributeError:
            value = self.value + other
            derivative = self.derivative
        return AutoDiff(value, derivative)
    
    def __mul__(self, other):
        try:
            value = self.value * other.value
            derivative = self.derivative * other.value + self.value * other.derivative
        except AttributeError:
            value = self.value * other
            derivative = self.derivative * other
        return AutoDiff(value, derivative)
    
    def cos(self):
        value = np.cos(self.value)
        derivative = -np.sin(self.value) * self.derivative
        return AutoDiff(value, derivative)

    def sin(self):
        value = np.sin(self.value)
        derivative = np.cos(self.value) * self.derivative
        return AutoDiff(value, derivative)

    def tan(self):
        value = np.tan(self.value)
        derivative = (1/np.cos(self.value)**2) * self.derivative
        # implement vector multiplication
        return AutoDiff(value, derivative)
    
    def cot(self):
        value = np.cot(self.value)
        derivative = (1/np.sin(self.value)**2) * self.derivative
        # implement vector multiplication
        return AutoDiff(value, derivative)
    
    def exp(self):
        # value = self.value
        # derivative = self.derivative
        value = np.exp(self.value)
        derivative = np.exp(self.value) * self.derivative
        # implement vector exponentiation
        return AutoDiff(value, derivative)

    def log(self):
        if not self.value > 0: raise ValueError("Cannot take ln of non-positive values")
        value = np.log(self.value)
        derivative = 1 / self.value
        return AutoDiff(value, derivative)
    
    def sqrt(self):
        if not self.value > 0: raise ValueError('Cannot take sqrt of non-positive values')
        value = np.sqrt(self.value)
        derivative = 1 / 2 * self.derivative * (self.value ** -0.5)
        return AutoDiff(value, derivative)

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)


# TESTS

initial = 2
x = AutoDiff(initial)

for a, b in itertools.product(range(1, 10), range(1, 10)):
    f = a * np.cos(x) + b
    f = a * np.sin(x) + b
    f = a * np.tan(x) + b
    f = a * np.cot(x) + b
    f = a * np.exp(x) + b
    f = a * np.log(x) + b
    f = a * np.sqrt(x) + b
    print("f", f)
    print("a", a, "b", b)
    print("value", f.value, "derivative", f.derivative)

        
