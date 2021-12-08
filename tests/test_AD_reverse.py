import sys
sys.path.append('../AD')     
sys.path.append('AD')
from Node import *
from AD_reverse import *
from elementary_node import *
def test_get_value():
    x1 = Node(1)
    y1 = Node(2)
    z1 = Node(6)
    def f1(x1,y1,z1): return ( 5 * x1 - 2 * y1 + 3 * z1 + 10)
    test=Reverse(f1,[x1,y1,z1])
    assert  test.get_value()==[[29]]
    
def test_get_jacobian():
    # test single func
    x1 = Node(1)
    y1 = Node(2)
    z1 = Node(6)
    def f1(x1,y1,z1): return ( 5 * x1 - 2 * y1 + 3 * z1 + 10)
    test=Reverse(f1,[x1,y1,z1])
    assert (test.get_jacobian() == np.array([[ 5, -2,3.]])).all()

def test_get_value_multi():
    x1 = Node(1)
    y1 = Node(2)
    z1 = Node(6)
    def f1(x1,y1,z1): return ( 5 * x1 - 2 * y1 + 3 * z1 + 10)
    def f2(x1,y1,z1): return (x1 ** (logistic(y1, 1, 7, 8) - exp(sin(y1) + cos(z1))))
    def f3(x1,y1,z1): return (log_base(tanh(y1), 3) - ln(tan(sqrt(x1))))
    test=Reverse([f1,f2,f3],[x1,y1,z1])
    error=0.00000001 # np.array([[0.00000001],[0.00000001],[0.00000001]])
    assert ((test.get_value()-np.array([[29],[1],[-0.4763696792908]])).all() < error)

def test_get_jacobian_multi():
    x1 = Node(1)
    y1 = Node(2)
    z1 = Node(6)
    # def f1(x1,y1,z1): return ( 5 * x1 - 2 * y1 + 3 * z1 + 10)
    # def f2(x1,y1,z1): return (exp(sin(x1 * y1) + cos(z1)) - x1 ** (logistic(x1 / y1, y1 / x1, 7, 8)))
    # def f3(x1,y1,z1): return (log_base(x1 * tanh(y1) - cosh(z1 * x1) + sinh(z1 / x1), 3) - ln(tan(sqrt(x1)))) 
    
    def f1(x1,y1,z1): return (5 * x1 - 2 * y1 + 3 * z1 + 10)
    def f2(x1,y1,z1): return (x1 ** (logistic(y1, 1, 7, 8) - exp(sin(y1) + cos(z1))))
    def f3(x1,y1,z1): return (log_base(tanh(y1), 3) - ln(tan(sqrt(x1))))
    # test_multiple = Reverse([f1, f2, f3],[x1,y1,z1])
    
    test=Reverse([f1,f2,f3],[x1,y1,z1])
    der=np.array([[ 5,-2,3],[-6.48484369,0,0],[-1.09975017, 0.06670883,0]])
    error=0.00000001 
    print(test.get_jacobian())
    assert ((test.get_jacobian()-der).all()<error)
