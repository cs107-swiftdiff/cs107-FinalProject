import sys
sys.path.append('../AD')     
sys.path.append('AD')
import numpy as np

from Dual import *
from AD_forward import *
from elementary import *

def test_get_value():
        x1 = Dual(value = 1, derivative=1, index = 0, total = 3)
        y1 = Dual(value = 2, derivative=1, index = 1, total = 3)
        z1 = Dual(value = 6, derivative=1, index = 2, total = 3)
        f1 = 5 * x1 - 2 * y1 + 3 * z1 + 10
        fwd_test = Forward(f1)
        assert fwd_test.get_value() == [[29]]

def test_get_jacobian():
        x1 = Dual(value = 1, derivative=1, index = 0, total = 3)
        y1 = Dual(value = 2, derivative=1, index = 1, total = 3)
        z1 = Dual(value = 6, derivative=1, index = 2, total = 3)
        f1 = 5 * x1 - 2 * y1 + 3 * z1 + 10
        fwd_test = Forward(f1)
        assert (fwd_test.get_jacobian() == np.array([[ 5,-2,3.]])).all()


def test_get_value_multi():

        x1 = Dual(value = 1, derivative=1, index = 0, total = 3)
        y1 = Dual(value = 2, derivative=1, index = 1, total = 3)
        z1 = Dual(value = 6, derivative=1, index = 2, total = 3)
        f1 = 5 * x1 - 2 * y1 + 3 * z1 + 10
        f2 = x1 ** (logistic(y1, 1, 7, 8) - exp(sin(y1) + cos(z1)))
        f3 = log_base(tanh(y1), 3) - ln(tan(sqrt(x1))) 
        fwd_test=Forward([f1,f2,f3])
        # error=np.array([[0.00000001],[0.00000001],[0.00000001]])
        # assert ((fwd_test.get_jacobian()-npß.array([[29],[1],[-0.4763696792908]]))<error).all()
        error=0.00000001 # np.array([[0.00000001],[0.00000001],[0.00000001]])
        assert ((fwd_test.get_value()-np.array([[29],[1],[-0.4763696792908]])).all() < error)

def test_get_jacobian_multi():
        x1 = Dual(value = 1, derivative=1, index = 0, total = 3)
        y1 = Dual(value = 2, derivative=1, index = 1, total = 3)
        z1 = Dual(value = 6, derivative=1, index = 2, total = 3)
        f1 = 5 * x1 - 2 * y1 + 3 * z1 + 10
        f2 = x1 ** (logistic(y1, 1, 7, 8) - exp(sin(y1) + cos(z1)))
        f3 = log_base(tanh(y1), 3) - ln(tan(sqrt(x1))) 
        fwd_test=Forward([f1,f2,f3])
        # error=np.array([[ 0.00000001,0.00000001,0.00000001],[0.0000001,0.0000001,0.0000001],[0.0000001, 0.00000001,0.00000001]])
        error=0.00000001
        der=np.array([[ 5,-2,3],[-6.48484369,0,0],[-1.09975017, 0.06670883,0]])
        assert ((fwd_test.get_jacobian()-der).all()<error)
