
# base class for autodiff
import numpy as np
from autodiff.Dual import *
from autodiff.elementary import *

class AutoDiff():
    def __init__(self, f):
        self.f = f if isinstance(f, list) else [f]

# child class inherits from autodiff
class Forward(AutoDiff):
    def __init__(self, f):
        super().__init__(f)

    def get_value(self):
        l=[]
        for element in self.f:
            l.append([element.value])
        return l

    def get_jacobian(self):
        l=[]
        for element in self.f:
            l.append(element.derivative)
        return np.array(l)