#import sys
#sys.path.append('AutoDiff/src/autodiff')

import pytest
from AD.Node import *
from AD.elementary_node import *
import numpy as np

def test_exp_node():
    x = Node(2)
    y = exp(x)
    y.derivative=1
    assert y.value == np.exp(2)
    assert y.der() == 1
    assert x.der() == np.exp(2)

def test_exp_real():
    x_real = exp(3)
    assert x_real == np.exp(3)

def test_sin_node():
    x = Node(0.5)
    y = sin(x)
    y.derivative=1
    assert y.value == np.sin(0.5)
    assert y.der() == 1
    assert x.der() == np.cos(0.5)

def test_sin_real():
    x_real = sin(0.5)
    assert x_real == np.sin(0.5)
                      
def test_cos_node():
    x = Node(0.5)
    y = cos(x)
    y.derivative=1
    assert y.value == np.cos(0.5)
    assert y.der() == 1
    assert x.der() == -np.sin(0.5)

def test_cos_real():
    x_real = cos(0.5)
    assert x_real == np.cos(0.5)
    
def test_tan_node():
    x = Node(0.5)
    y = tan(x)
    y.derivative=1
    assert y.value == np.tan(0.5)
    assert y.der() == 1
    assert x.der() == (1 / np.cos(0.5) ** 2)

def test_tan_real():
    x_real = tan(0.5)
    assert x_real == np.tan(0.5)

def test_sinh_node():
    x = Node(1)
    y = sinh(x)
    y.derivative=1
    assert y.value == np.sinh(1)
    assert y.der() == 1
    assert x.der() == np.cosh(1)

def test_sinh_real():
    x_real = sinh(1)
    assert x_real == np.sinh(1)

def test_cosh_node():
    x = Node(1)
    y = cosh(x)
    y.derivative=1
    assert y.value == np.cosh(1)
    assert y.der() == 1
    assert x.der() == np.sinh(1)

def test_cosh_real():
    x_real = cosh(1)
    assert x_real == np.cosh(1)

def test_tanh_node():
    x = Node(0.5)
    y = tanh(x)
    y.derivative=1
    xderivative=1 / (np.cosh(0.5)**2)
    assert y.value == np.tanh(0.5)
    assert y.der() == 1
    assert x.der() == xderivative

def test_tanh_real():
    x_real = tanh(1)
    assert x_real == np.tanh(1)

def test_ln_node():
    x = Node(1)
    y = ln(x)
    y.derivative=1
    assert y.value == 0
    assert y.der() == 1
    assert x.der() == 1

def test_ln_real():
    x_real = ln(3)
    assert x_real == np.log(3)

def test_ln_neg():
    with pytest.raises(ValueError):
        y=ln(Node(-1))
    with pytest.raises(ValueError):
        y=ln(-1)  
        
def test_logbase_node():
    x = Node(3)
    y = log_base(x,3)
    y.derivative=1
    xder=round(1/3/np.log(3),10)
    assert y.value == 1
    assert y.der() == 1
    assert round(x.der(),10) == xder

def test_logbase_real():
    x_real = log_base(2,2)
    assert x_real == 1
    
def test_logbase_neg():
    with pytest.raises(ValueError):
        y=log_base(Node(-1),2)
    with pytest.raises(ValueError):
        y=log_base(-1,-2)  
        
def test_logistic_node():
    x = Node(1)
    y = logistic(x)
    y.derivative=1
    assert y.value == 1/(1+np.exp(-1))
    assert y.der() == 1
    assert x.der() == y.value*(1- y.value)

def test_logistic_real():
    x_real = logistic(1)
    assert x_real == 1/(1+np.exp(-1))

def test_sqrt_node():
    x = Node(9)
    y = sqrt(x)
    y.derivative=1
    assert y.value == sqrt(9)
    assert y.der() == 1
    assert x.der() == 0.5 * (9 ** -0.5)

def test_sqrt_real():
    x_real = sqrt(9)
    assert x_real == np.sqrt(9)

def test_sqrt_neg():
    with pytest.raises(ValueError):
        y=sqrt(Node(-1))
    with pytest.raises(ValueError):
        y=sqrt(-1)       