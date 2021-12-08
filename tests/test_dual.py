import sys
sys.path.append('../AD')     
sys.path.append('AD')
import pytest

from Dual import *
import numpy as np


# Testing __add__
def test_add_constant():
    x = Dual(1,1)
    y = x + 2
    assert y.value == 3
    assert y.derivative == 1

def test_add_dual():
    y = Dual(1,1,index=0,total=2)+Dual(2,2,index=1,total=2)
    assert y.value == 3
    assert (y.derivative == np.array([1,2])).all()

# Testing __radd__
def test_radd_constant():
    x = Dual(1,1)
    y = 2 + x
    assert y.value == 3
    assert y.derivative == 1

def test_radd_dual():
    x=Dual(1,1,index=0,total=2)
    y=Dual(2,2,index=1,total=2)
    z=y+x
    assert z.value==3
    assert (z.derivative == np.array([1,2])).all()


# Testing __sub__

def test_sub_constant():
    x = Dual(1, 1)
    y = x-2
    assert y.value == -1
    assert y.derivative == 1
    
def test_sub_dual():
    x = Dual(1, 1,index=0,total=2)
    y = Dual(2, 2,index=1,total=2)
    z = x-y
    assert z.value == -1
    assert (z.derivative == np.array([1,-2])).all()


# Testing __rsub__
def test_rsub_constant():
    x = Dual(1, 1)
    y = 2-x
    assert y.value == 1
    assert y.derivative == -1

def test_rsub_dual():
    x = Dual(1, 1,index=0,total=2)
    y = Dual(2, 2,index=1,total=2)
    z = y-x
    assert z.value == 1
    assert (z.derivative == np.array([-1,2])).all()

#Testing __neg__    
def test_neg():
    x = Dual(1, 1)
    y=-x
    assert y.value == -1
    assert y.derivative == -1

# Testing __mul__
def test_mul_constant():
    x = Dual(1, 1)
    z = x * 2
    assert z.value == 2
    assert z.derivative == 2

def test_mul_dual():
    x = Dual(1, 1,index=0,total=2)
    y = Dual(2, 2,index=1,total=2)
    z = x * y
    assert z.value == 2
    assert (z.derivative == np.array([2, 2])).all()


# Testing __rmul__
def test_rmul_constant():
    x = Dual(1, 1)
    y = 2 * x
    assert y.value == 2
    assert y.derivative == 2

def test_rmul_dual():
    x = Dual(1, 1,index=0,total=2)
    y = Dual(2, 2,index=1,total=2)
    z = y * x
    assert z.value == 2
    assert (z.derivative == np.array([2, 2])).all()

# Testing __div__
def test_truediv_constant():
    x = Dual(1, 1)
    z = x / 2
    assert z.value == 0.5
    assert z.derivative == 0.5

def test_truediv_dual():
    x = Dual(1, 1,index=0,total=2)
    y = Dual(2, 2,index=1,total=2)
    z = x / y
    assert z.value == 0.5
    assert (z.derivative == np.array([0.5, -0.5])).all()

# Testing __rdiv__
def test_rdiv_constant():
    x = Dual(1, 1)
    y = 2 / x
    assert y.value == 2
    assert y.derivative == -2

def test_rdiv_dual():
    x = Dual(1, 1,index=0,total=2)
    y = Dual(2, 2,index=1,total=2)
    z = y / x
    assert z.value == 2
    assert (z.derivative == np.array([-2, 2])).all()

def test_rtruediv_zero():
    x = Dual(0, 1)
    with pytest.raises(ZeroDivisionError):
         y=1/x

 # Testing __pow__
def test_pow_constant():
    x=Dual(2, 2)
    y=x**2
    assert y.value ==4
    assert y.derivative==8

def test_pow_dual():
    x = Dual(2, 2,index=0,total=2)
    y = Dual(3, 3,index=1,total=2)
    z=x**y
    zder2=8*np.log(2)*3
    assert z.value==8
    assert (z.derivative == np.array([24,zder2])).all()
    
def test_pow_zero():
    x=Dual(0,1)
    with pytest.raises(ZeroDivisionError):
        y=x**0.5

 # Testing __rpow__
def test_rpow_constant():
    x=Dual(2, 2)
    y=2**x
    yder=np.log(2)*4*2
    assert y.value ==4
    assert y.derivative==yder

def test_rpow_dual():
    x = Dual(2, 2,index=0,total=2)
    y = Dual(3, 3,index=1,total=2)
    z=y**x
    zder1=9*np.log(3)*2
    assert z.value==9
    assert (z.derivative == np.array([zder1,18])).all()


## Testing comparsion
def test_lt_constant():
    x = Dual(1,1)
    assert False == (x < 2)
    assert False == (x < 0)

def test_lt_dual():
    x = Dual(1,1)
    y = Dual(2,1)
    assert False == (x < y)
    assert False == (y < x)

def test_le_constant():
    x = Dual(1,1)
    assert False  == (x <= 2)
    assert False == (x <= 1)
    assert False  == (x <= 0)

def test_le_dual():
    x = Dual(1,1)
    y = Dual(1,1)
    z = Dual(2,1)
    assert True  == (x <= y)
    assert True == (x <= z)
    assert False  == (z <= x)

def test_gt_constant():
    x = Dual(2,1)
    assert False  == (x > 1)
    assert False == (1 > x)

def test_gt_dual():
    x = Dual(2,1)
    y = Dual(1,1)
    z = Dual(3,1)
    assert False == (x > y)
    assert False == (x > z)

def test_ge_constant():
    x = Dual(2,1)
    assert False  == (x >= 1)
    assert False  == (x >= 2)
    assert False == (x >= 3)


def test_ge_dual():
    x = Dual(2,1)
    y = Dual(1,1)
    z = Dual(2,1)
    assert True  == (x >= y)
    assert True  == (x >= z)
    assert False == (y >= x)

def test_eq_constant():
    x = Dual(2,1)
    assert False == (x == 2)
    assert False == (x == 1)


def test_eq():
    x = Dual(2,1)
    y = Dual(2,1)
    z = Dual(1,1)
    assert True == (x == y)
    assert False == (x == z)

def test_ne_constant():
    x = Dual(2,1)
    assert True == (x != 1)
    assert True == (x != 2)

    
def test_ne_dual():
    x = Dual(2,1)
    y = Dual(2,1)
    z = Dual(1,1)
    assert True == (x != z)
    assert False == (x != y)
