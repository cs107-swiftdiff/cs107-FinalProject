import sys
sys.path.append('../AD')     
sys.path.append('AD')
import pytest

from Node import Node #change
import numpy as np

# Testing __add__
def test_add_constant():
    x = Node(1)
    y = x + 2
    y.derivative=1
    assert y.value == 3
    assert x.der() == 1
    assert y.der() == 1

def test_add_node():
    x1=Node(1)
    x2=Node(2)
    y = x1+x1+x2
    y.derivative=1
    assert y.value == 4
    assert x1.der() == 2
    assert x2.der() == 1
    assert y.der() == 1

# Testing __radd__
def test_radd_constant():
    x = Node(1)
    y = 2 + x
    y.derivative=1
    assert y.value == 3
    assert y.der() == 1
    assert x.der() == 1

def test_radd_node():
    x1=Node(1)
    x2=Node(2)
    y=x2+x1
    y.derivative=1
    assert y.value==3
    assert y.der()==1
    assert x2.der()==1
    assert x1.der() ==1

# Testing __sub__

def test_sub_constant():
    x = Node(1)
    y = x-2
    y.derivative=1
    assert y.value == -1
    assert y.der() == 1
    assert x.der() == 1
    
def test_sub_node():
    x1 = Node(1)
    x2 = Node(2)
    y = x1-x1-x2
    y.derivative=1
    assert y.value == -2
    assert y.der() == 1
    assert x1.der() == 0
    assert x2.der() == -1

# Testing __rsub__
def test_rsub_constant():
    x = Node(1)
    y = 2-x
    y.derivative=1
    assert y.value == 1
    assert y.der() == 1
    assert x.der() == -1

def test_rsub_node():
    x1 = Node(1)
    x2 = Node(2)
    y = x2-x1
    y.derivative=1
    assert y.value==1
    assert y.der() == 1
    assert x1.der() == -1
    assert x2.der() == 1

#Testing __neg__    
def test_neg():
    x = Node(1)
    y=-x
    y.derivative=1
    assert y.value == -1
    assert y.der() == 1
    assert x.der() == -1

# Testing __mul__
def test_mul_constant():
    x = Node(1)
    y = x * 2
    y.derivative = 1
    assert y.value == 2
    assert y.der() == 1
    assert x.der() == 2

def test_mul_node():
    x1 = Node(1)
    x2 = Node(2)
    y = x1 * x2
    y.derivative=1
    assert y.value == 2
    assert y.der() == 1
    assert x1.der() == 2
    assert x2.der() == 1


# Testing __rmul__
def test_rmul_constant():
    x = Node(1)
    y = 2 * x
    y.derivative=1
    assert y.value == 2
    assert y.der() == 1
    assert x.der() ==2

def test_rmul_node():
    x1 = Node(1)
    x2 = Node(2)
    y = x2 * x1
    y.derivative=1
    assert y.value == 2
    assert y.der() == 1
    assert x2.der() == 1
    assert x1.der() == 2


# Testing __div__
def test_truediv_constant():
    x = Node(1)
    y = x / 2
    y.derivative=1
    assert y.value == 0.5
    assert y.der() == 1
    assert x.der() == 0.5


def test_truediv_node():
    x = Node(1)
    y = Node(2)
    z = x / y
    z.derivative=1
    assert z.value == 0.5
    assert z.der() == 1
    assert x.der() == 0.5
    assert y.der() == -0.25


# Testing __rdiv__
def test_rdiv_constant():
    x = Node(1)
    y = 2 / x
    y.derivative=1
    assert y.value == 2
    assert y.der() == 1
    assert x.der() == -2

def test_rdiv_node():
    x = Node(1)
    y = Node(2)
    z = y / x
    z.derivative=1
    assert z.value == 2
    assert z.der() == 1
    assert y.der() == 1
    assert x.der() == -2


 # Testing __pow__
def test_pow_constant():
    x=Node(2)
    y=x**2
    y.derivative=1
    assert y.value ==4
    assert y.der() == 1
    assert x.der() == 4

def test_pow_node():
    x = Node(2)
    y = Node(3)
    z=x**y
    z.derivative=1
    yder=8*np.log(2)
    assert z.value==8
    assert z.der() == 1
    assert x.der() == 12
    assert y.der() == yder


 # Testing __rpow__
def test_rpow_constant():
    x=Node(2)
    y=2**x
    y.derivative=1
    xder=4*np.log(2)
    assert y.value ==4
    assert y.der() == 1
    assert x.der() ==xder



## Testing comparsion
def test_lt_constant():
    x = Node(1)
    assert False == (x < 2)
    assert False == (x < 0)

def test_lt_node1():
    x = Node(1)
    x.derivative=1
    y = Node(2)
    y.derivative=1
    assert False == (x < y)
    assert False == (y < x)

def test_lt_node2():
    x = Node(1)
    y = Node(2)
    assert True == (x < y)
    assert False == (y < x)

def test_lt_node3():
    x = Node(1)
    x.derivative=1
    y = Node(2)
    y.derivative=2
    assert True == (x < y)
    assert False == (y < x)

def test_lt_node4():
    x = Node(1)
    x.derivative=1
    y = Node(2)
    assert False == (x < y)
    assert False == (y < x)

def test_lt_node5():
    x = Node(1)
    y = Node(2)
    y.derivative=2
    assert False == (x < y)
    assert False == (y < x)


def test_le_constant():
    x = Node(1)
    assert False  == (x <= 2)
    assert False == (x <= 1)
    assert False  == (x <= 0)

def test_le_node1():
    x = Node(1)
    x.derivative=1
    y = Node(2)
    y.derivative=1
    assert True == (x <= y)
    assert False == (y <= x)

def test_le_node2():
    x = Node(1)
    y = Node(2)
    z = Node(1)
    assert True == (x <= y)
    assert False == (y <= x)
    assert True == (x <= z)

def test_le_node3():
    x = Node(1)
    y = Node(2)
    y.derivative=2
    assert False == (x <= y)
    assert False == (y <= x)

def test_le_node4():
    x = Node(1)
    x.derivative=1
    y = Node(2)
    assert False == (x <= y)
    assert False == (y <= x)

def test_gt_constant():
    x = Node(2)
    assert False  == (x > 1)
    assert False == (1 > x)

def test_gt_node1():
    x = Node(1)
    x.derivative=1
    y = Node(2)
    y.derivative=1
    assert False == (x > y)
    assert False == (y > x)

def test_gt_node2():
    x = Node(1)
    y = Node(2)
    assert False == (x > y)
    assert True == (y > x)

def test_gt_node3():
    x = Node(1)
    x.derivative=1
    y = Node(2)
    y.derivative=2
    assert False == (x > y)
    assert True == (y > x)

def test_gt_node4():
    x = Node(1)
    x.derivative=1
    y = Node(2)
    assert False == (x > y)
    assert False == (y > x)


def test_ge_constant():
    x = Node(1)
    assert False  == (x >= 1)
    assert False  == (x >= 2)
    assert False == (x >= 3)

def test_ge_node1():
    x = Node(1)
    x.derivative=1
    y = Node(2)
    y.derivative=1
    assert False == (x >= y)
    assert True == (y >= x)

def test_ge_node2():
    x = Node(1)
    y = Node(2)
    z = Node(1)
    assert False == (x >= y)
    assert True == (y >= x)
    assert True == (x >= z)

def test_ge_node3():
    x = Node(1)
    y = Node(2)
    y.derivative=2
    assert False == (x >= y)
    assert False == (y >= x)

def test_eq_constant():
    x = Node(1)
    assert False == (x == 2)
    assert False == (x == 1)

def test_eq_node1():
    x = Node(1)
    x.derivative=1
    y = Node(2)
    y.derivative=1
    z = Node(1)
    z.derivative=1
    assert True == (x == z)
    assert False == (y == x)


def test_eq_node2():
    x = Node(1)
    y = Node(2)
    z = Node(1)
    assert True == (x == z)
    assert False == (y == x)

def test_eq_node3():
    x = Node(1)
    y = Node(2)
    y.derivative=2
    assert False == (x == y)


def test_ne_constant():
    x = Node(1)
    assert True == (x != 1)
    assert True == (x != 2)

    
def test_ne_node1():
    x = Node(1)
    x.derivative=1
    y = Node(2)
    y.derivative=1
    z= Node(1)
    z.derivative=1
    assert True == (x != y)
    assert False == (z != x)

def test_ne_node2():
    x = Node(1)
    y = Node(2)
    z = Node(1)
    assert True == (x != y)
    assert False == (z != x)

