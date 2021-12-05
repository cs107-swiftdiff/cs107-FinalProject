from AD.AD_reverse import * 
from AD.Node import * 
from AD.elementary_node import * 

"""
REVERSE MODE

Define the variables of your function in the following format:
<variable> = Node(value = <FLOAT, or INT>)

Value: the initial value of the variable

Define the functions in the following format:
def <function>(variables): return (<the function you want to define>)
"""

demo_state = "R"
while demo_state == "R":

        print("""
        WELCOME TO THE AD REVERSE MODE DEMO!
        \nIn this demo, we will guide you through the following steps of setting up your AD systems (reverse mode):
        \n*Instantiating variables
        \n*Instantiating functions
        \n*Instantiating AD objects
        \nAs well as the following methods you can use from our package:
        \n*Get function value
        \n*Get function jacobian
        \n*Get multiple function jacobians
        """)
        step1 = None
        while step1 != "":
            step1 = input("(press ENTER to continue to STEP #1)")

        print("""
        STEP #1:
        \nDefine the variables of your function in the following format:
        <variable> = Dual(value = <DUAL, FLOAT, or INT>, derivative = <DUAL, FLOAT, or INT>, index = <optional INT>, total = <optional INT>)
        \nExample variables:
        \nx1 = Node(1)
        \ny1 = Node(2)
        \nz1 = Node(6)
        """)
        step2 = None
        while step2 != "":
            step2 = input("(press ENTER to continue to STEP #2)")

        x1 = Node(1)
        y1 = Node(2)
        z1 = Node(6)

        print(">>>x1\n", eval("x1"))
        print(">>>y1\n", eval("y1"))
        print(">>>z1\n", eval("z1"))

        print("""
        STEP #2
        \nInput your function including the variables you defined above.
        \nExample function:
        \ndef f1(x1,y1,z1): return (5 * x1 - 2 * y1 + 3 * z1 + 10)
        """)
        step3 = None
        while step3 != "":
            step3 = input("(press ENTER to continue to STEP #3)")

        def f1(x1,y1,z1): return (5 * x1 - 2 * y1 + 3 * z1 + 10)

        print(">>>f1\n", eval("\nf1"))

        print("""
        STEP #3
        \nCreate an instance of your function as a Forward AD class.
        \nExample usage:
        \ntest = Reverse(f1,[x,y,z])
        """)
        method1 = None
        while method1 != "":
            method1 = input("(press ENTER to continue to METHOD #1)")
        
        test = Reverse(f1,[x1,y1,z1])
        
        print(">>>test\n", eval("\ntest"))

        print("""
        METHOD #1:
        \nYou can use the get_value function to get the value of your function:
        """)

        # funcval = fwdtest.get_value()

        print(">>>test.get_value()\n", eval("test.get_value()"))
        method2 = None
        while method2 != "":
            method2 = input("(press ENTER to continue to METHOD #2)")

        print("""
        METHOD #2:
        \nYou can use the get_jacobian function to get the jacobian of your function:
        """)

        # funcjac = fwdtest.get_jacobian()

        print(">>>test.get_jacobian()\n", eval("test.get_jacobian()"))
        method3 = None
        while method3 != "":
            method3 = input("(press ENTER to continue to METHOD #3)")
        
        def f2(x1,y1,z1): return (x1 ** (logistic(y1, 1, 7, 8) - exp(sin(y1) + cos(z1))))
        def f3(x1,y1,z1): return (log_base(tanh(y1), 3) - ln(tan(sqrt(x1))))
        test_multiple = Reverse([f1, f2, f3],[x1,y1,z1])
        # funcmultjac = fwdtest_multiple.get_jacobian()
        print("""
        METHOD #3:
        \nYou can also use the get_jacobian function to get the jacobian of multiple functions:
        \nExample usage:
        \ndef f2(x1,y1,z1): return (x1 ** (logistic(y1, 1, 7, 8) - exp(sin(y1) + cos(z1))))
        \ndef f3(x1,y1,z1): return (log_base(tanh(y1), 3) - ln(tan(sqrt(x1))))
        \nfwdtest_multiple = Forward([f1, f2, f3])
        """)

        print(">>>test_multiple.get_jacobian()\n", eval("test_multiple.get_jacobian()"))
        finish = None
        while finish != "":
            finish = input("(press ENTER to finish)")

        print("""
        CONGRATULATIONS! YOU HAVE COMPLETED THE AD REVERSE MODE DEMO!
        \nIn case you forget, all information can be found within the AD documentation at github.com
        """)

        demo_state = input("""
            MENU OPTIONS:
            R) Restart
            E) Exit
            \n>>>""")