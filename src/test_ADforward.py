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
    
    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        try:
            value = self.value - other.value
            derivative = self.derivative - other.derivative
        except AttributeError:
            value = self.value - other
            derivative = self.derivative
        return AutoDiff(value, derivative)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __neg__(self,other):

        value=-self.value
        derivative=-self.derivative

        return AutoDiff(value,derivative)

    def __truediv__(self, other):
        try:
            value=self.value/other.value
            derivative=(self.derivative*other.value-self.value*other.derivative)/(other.derivative)**2
        except AttributeError:
            value=self.value/other
            derivative=self.derivative/other
        
        return AutoDiff(value,derivative)

    def __rtruediv__(self, other):

        if self.value==0: raise ZeroDivisionError

        value=other/self.value
        derivative=-other * self.derivative / self.value ** 2

        return AutoDiff(value,derivative)
        
    def __mul__(self, other):
        try:
            value = self.value * other.value
            derivative = self.derivative * other.value + self.value * other.derivative
        except AttributeError:
            value = self.value * other
            derivative = self.derivative * other
        return AutoDiff(value, derivative)
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, p):
        value_base = self.value
        derivative_base = self.derivative
        
        if value_base < 0 and 0<p<1: raise ValueError('Cannot take power of non-positive values')
        if value_base == 0 and p < 1: raise ZeroDivisionError
        
        if isinstance(p,int) or isinstance(p,float):
            float(p)
            value = value_base**p # implement vector multiplication
        
            derivative = p * derivative_base * value_base**(p - 1)
        
        if isinstance(p,AutoDiff):
            
            value=value_base**p.value  # implement vector multiplication
            derivative=value*(p.derivative*np.log(value_base)+p.value*derivative_base/value_base)
            
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
    
    # def power(self, p):
    #     value = self.value
    #     derivative = self.derivative
    #     float(p)
    #     if not value > 0 and p > 1: raise ValueError('Cannot take power of non-positive values')
    #     # if value == 0 and p < 1:
    #     #     raise ZeroDivisionError
    #     value = np.power(value, p)
    #     derivative = np.power((p * self.derivative * value), (p - 1))
    #     return AutoDiff(value, derivative)

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




# TESTS

initial = 2
x = AutoDiff(initial)

for a, b in itertools.product(range(1, 10), range(1, 10)):
    f = a * np.cos(x) + b
    f = a * np.sin(x) + b
    f = a * np.tan(x) + b
    f = a * np.cot(x) + b
    f = a * np.exp(x) + b
    # f = a * np.power(x, 2) + b
    f = a * np.log(x) + b
    f = a * np.sqrt(x) + b
    print("f", f)
    print("a", a, "b", b)
    print("value", f.value, "derivative", f.derivative)

        