import sys
sys.path.append('../AD')     
sys.path.append('AD')
import pytest
from AD.Dual import *
from AD.elementary import *
import numpy as np

def test_exp_dual():
    x = Dual(2, 1)
    y = exp(x)
    assert y.value == np.exp(2)
    assert y.derivative == np.exp(2)

def test_exp_real():
    x_real = exp(3)
    assert x_real == np.exp(3)

def test_sin_dual():
    x = Dual(0.5, 1)
    y = sin(x)
    assert y.value == np.sin(0.5)
    assert y.derivative == np.cos(0.5)

def test_sin_real():
    x_real = sin(0.5)
    assert x_real == np.sin(0.5)
                      
def test_cos_dual():
    x = Dual(0.5, 1)
    y = cos(x)
    assert y.value == np.cos(0.5)
    assert y.derivative == -np.sin(0.5)

def test_cos_real():
    x_real = cos(0.5)
    assert x_real == np.cos(0.5)
    
def test_tan_dual():
    x = Dual(0.5, 1)
    y = tan(x)
    assert y.value == np.tan(0.5)
    assert y.derivative == (1 / np.cos(0.5) ** 2)

def test_tan_real():
    x_real = tan(0.5)
    assert x_real == np.tan(0.5)

def test_sinh_dual():
    x = Dual(1,1)
    y = sinh(x)
    assert y.value == np.sinh(1)
    assert y.derivative == np.cosh(1)

def test_sinh_real():
    x_real = sinh(1)
    assert x_real == np.sinh(1)

def test_cosh_dual():
    x = Dual(1,1)
    y = cosh(x)
    assert y.value == np.cosh(1)
    assert y.derivative == np.sinh(1)

def test_cosh_real():
    x_real = cosh(1)
    assert x_real == np.cosh(1)

def test_tanh_dual():
    x = Dual(0.5,1)
    y = tanh(x)
    yderivative=1 / (np.cosh(0.5)**2)
    assert y.value == np.tanh(0.5)
    assert y.derivative == yderivative

def test_tanh_real():
    x_real = tanh(1)
    assert x_real == np.tanh(1)

def test_ln_dual():
    x = Dual(1,1)
    y = ln(x)
    assert y.value == 0
    assert y.derivative == 1

def test_ln_real():
    x_real = ln(3)
    assert x_real == np.log(3)

def test_logbase_dual():
    x = Dual(3,1)
    y = log_base(x,3)
    assert y.value == 1
    assert y.derivative == 1/3/np.log(3)

def test_logbase_real():
    x_real = log_base(2,2)
    assert x_real == 1

def test_logistic_dual():
    x = Dual(1,1)
    y = logistic(x)
    assert y.value == 1/(1+np.exp(-1))
    assert y.derivative == y.value*(1- y.value)

def test_logistic_real():
    x_real = logistic(1)
    assert x_real == 1/(1+np.exp(-1))

def test_sqrt_dual():
    x = Dual(9,1)
    y = sqrt(x)
    assert y.value == sqrt(9)
    assert y.derivative == 1 / 2 * (9 ** -0.5)

def test_sqrt_real():
    x_real = sqrt(9)
    assert x_real == np.sqrt(9)
