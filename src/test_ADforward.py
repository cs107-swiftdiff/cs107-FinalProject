from ADforward import AutoDiff
import numpy as np
import pytest

## Testing all of the inputs

def test_invalid_derivative():
    with pytest.raises(TypeError):
        x = AutoDiff(0, 'mwahaha')

def test_invalid_value():
    with pytest.raises(TypeError):
        x = AutoDiff('mwahaha', 1)


# Testing __add__

def test_add_constant():
    x = AutoDiff(1)
    y = x + 2
    assert y.value == 3
    assert y.derivative == 1

def test_add():
    x=AutoDiff(1,1)
    y=AutoDiff(2,2)
    z=x+y
    assert z.value == 3
    assert z.derivative == 3

# Testing __radd__
def test_radd_constant():
    x = AutoDiff(1)
    y = 2 + x
    assert y.value == 3
    assert y.derivative == 1

def test_radd():
    x=AutoDiff(1,1)
    y=AutoDiff(2,2)
    z=y+x
    assert z.value==3
    assert z.derivative == 3


# Testing __sub__

def test_sub_constant():
    x = AutoDiff(1, 1)
    y = x-2
    assert y.value == -1
    assert y.derivative == 1
    
def test_sub():
    x = AutoDiff(1, 1)
    y = AutoDiff(2, 2)
    z = y-x
    assert z.value == -1
    assert z.derivative == -1

# Testing __rsub__
def test_rsub_constant():
    x = AutoDiff(1, 1)
    y = 2-x
    assert y.value == 1
    assert y.derivative == -1

def test_rsub():
    x = AutoDiff(1, 1)
    y = AutoDiff(2,2)
    z = y-x
    assert z.value == 1
    assert z.derivative == 1

#Testing __neg__    
def test_neg():
    x = AutoDiff(1, 1)
    y=-x
    assert y.value == -1
    assert y.derivative == -1



# Testing __mul__

def test_mul():
    x = AutoDiff(1, 1)
    y = AutoDiff(2, 2)
    z = x * y
    assert z.value == 2
    assert z.derivative == 4

def test_mul_constant():
    x = AutoDiff(1, 1)
    z = x * 2
    assert z.value == 2
    assert z.derivative == 2

# Testing __rmul__

def test_rmul():
    x = AutoDiff(1, 1)
    y = AutoDiff(2, 2)
    z = y * x
    assert z.value == 2
    assert z.derivative == 4

def test_rmul_constant():
    x = AutoDiff(1, 1)
    y = 2 * x
    assert y.value == 2
    assert y.derivative == 2

# Testing __div__

def test_truediv():
    x = AutoDiff(1, 1)
    y = AutoDiff(2, 2)
    z = x / y
    assert z.value == 0.5
    assert z.derivative == 4

def test_truediv_constant():
    x = AutoDiff(1, 1)
    z = x * 2
    assert z.value == 2
    assert z.derivative == 2


# Testing __rdiv__

def test_rdiv():
    x = AutoDiff(1, 1)
    y = AutoDiff(2, 2)
    z = y * x
    assert z.value == 2
    assert z.derivative == 4

def test_rdiv_constant():
    x = AutoDiff(1, 1)
    y = 2 * x
    assert y.value == 2
    assert y.derivative == 2

def test_rtruediv_zero():
    x = AutoDiff(0, 1)
    with pytest.raises(ZeroDivisionError):
         y=1/x

# Testing __pow__
def test_pow_constant():
    x=AutoDiff(2, 2)
    y=x**2
    assert y.value ==4
    assert y.derivative==8

def test_pow():
    x=AutoDiff(2, 2)
    y=AutoDiff(3, 3) 
    z=x**y
    assert z.value==8
    assert z.derivative== 8*(3*np.log(2)+8*3/2)

def test_pow_zero():
    x=AutoDiff(0,1)
    with pytest.raises(ZeroDivisionError):
        y=x**0.5

def test_pow_valueerror():
        x=AutoDiff(-1,1)
    with pytest.raises(ValueError):
        y=x**0.5

# Testing __cos__
def test_cos():
    x = AutoDiff(0.5, 1)
    y = x.cos()
    assert y.value == np.cos(0.5)
    assert y.derivative == -np.sin(0.5)

# Testing __sin__
def test_sin():
    x = AutoDiff(0.5, 1)
    y = x.sin()
    assert y.value == np.sin(0.5)
    assert y.derivative == np.cos(0.5)
    
# Testing __tan__
def test_tan():
    x = AutoDiff(0.5, 1)
    y = x.tan()
    assert y.value == np.tan(0.5)
    assert y.derivative == 1 / (np.sin(0.5)**2)

# Testing __cot__
def test_cot():
    x = AutoDiff(0.5, 1)
    y = x.cot()
    assert y.value == 1 / np.tan(0.5)
    assert y.derivative == -1 / (np.cos(0.5)**2)

# Testing __exp__
def test_exp():
    x = AutoDiff(0.5, 1)
    y = x.exp()
    assert y.value == np.exp(0.5)
    assert y.derivative == 1 * np.exp(0.5)

# Testing __log__
def test_log():
    x = AutoDiff(0.5, 1)
    y = x.log()
    assert y.val == np.log(0.5)
    assert y.der == 1 / 0.5

def test_log_valueerror():
    x=AutoDiff(-0.5,1)
    with pytest.raises(ValueError):
        y=x.log()

# Testing __sqrt__
def test_sqrt():
    x = AutoDiff (0,1,'x')
    y = x.sqrt()
    assert y.val == np.sqrt(0.5)
    assert y.der == -0.5 / (x**0.5)

def test_sqrt_valueerror():
    x=AutoDiff(-0.5,1)
    with pytest.raises(ValueError):
        y=x.sqrt()

