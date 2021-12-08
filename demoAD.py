import sys
sys.path.append('../AD')     
sys.path.append('AD')
from AD_forward import * 
from Dual import * 
from elementary import * 

"""
FORWARD MODE

Define the variables of your function in the following format:
<variable> = Dual(value = <DUAL, FLOAT, or INT>, derivative = <DUAL, FLOAT, or INT>, index = <optional INT>, total = <optional INT>)

Value: the initial value of the variable
Derivative: the initial derivative of the variable

Optional attributes for multivariable functions:
Index: index of the variable within the multivariate function
Total: the total number of variables within the multivariable function
"""

demo_state = "R"
while demo_state == "R":

    print("""
    WELCOME TO THE AD DEMO!
    \nIn this interactive demo, we will guide you through the following steps of setting up your AD systems:
    \n*Instantiating variables
    \n*Instantiating functions
    \n*Instantiating AD objects
    \nAs well as the following methods you can use from our package:
    \n*Get function value
    \n*Get function jacobian
    \n*Get multiple function jacobians
    """)

    demo_mode = None
    while demo_mode != 1 and demo_mode !=2:
        demo_mode = int(input("""
        PLEASE SELECT YOUR DEMO MODE:
        \n1) Interactive
        \n2) Non-interactive
        \n>>>"""))

    if demo_mode == 1:

        print("""
        STEP #1:
        \nDefine the variables of your function in the following format:
        <variable> = Dual(value = <DUAL, FLOAT, or INT>, derivative = <DUAL, FLOAT, or INT>, index = <optional INT>, total = <optional INT>)
        \nExample variables:
        \nx1 = Dual(value = 1, derivative = 1, index = 0, total = 3)
        \ny1 = Dual(value = 2, derivative = 2, index = 1, total = 3)
        \nz1 = Dual(value = 6, derivative = 4, index = 2, total = 3)
        \nTry defining your own variables (feel free to copy the example if you want to speed up the process):
        """)

        x1 = None
        y1 = None
        z1 = None

        while not isinstance(x1, Dual):
            str_x1 = input("\nx1 =")
            try:
                x1 = eval(str_x1)
                print("You've defined your first variable as: ", x1)
            except:
                print("Variables must follow the Dual class format.")
        while not isinstance(y1, Dual):
            str_y1 = input("\ny1 =")
            try:
                y1 = eval(str_y1)
                print("You've defined your second variable as: ", y1)
            except:
                print("Variables must follow the Dual class format.")
        while not isinstance(z1, Dual):
            str_z1 = input("\nz1 =")
            try:
                z1 = eval(str_z1)
                print("You've defined your third variable as: ", z1)
            except:
                print("Variables must follow the Dual class format.")

        print("""
        STEP #2:
        \nInput your function including the variables you defined above.
        \nExample function:
        \nf1 = 5 * x1 - 2 * y1 + 3 * z1 + 10
        \nTry defining your own function (feel free to copy the example if you want to speed up the process):
        """)

        f1 = None
        while not isinstance(f1, Dual):
            str_f1 = input("\nf1 =")
            try:
                f1 = eval(str_f1)
            except:
                print("Your function is invalid. Please check your syntax.")

        print("""
        STEP #3:
        \nCreate an instance of your function as a Forward AD class.
        \nExample usage:
        \nfwdtest = Forward(f1)
        \nTry instantiating your function (feel free to copy the example if you want to speed up the process):
        """)

        fwdtest = None
        while not isinstance(fwdtest, Forward):
            str_fwdtest = input("\nfwdtest =")
            try:
                fwdtest = eval(str_fwdtest)
            except:
                print("Forward AD instances must follow the Forward AD class format.")

        print("""
        METHOD #1:
        \nYou can use the get_value function to get the value of your function:
        \nExample usage:
        \n>>>fwdtest.get_value()
        \nTry this yourself (feel free to copy the example if you want to speed up the process):
        """)

        str_funcval = None
        while str_funcval != "fwdtest.get_value()":
            str_funcval= input(">>>")
            try:
                funcval = eval(str_funcval)
                print("Function value: ", funcval)
            except:
                print("Try getting the value of your function.")

        print("""
        METHOD #2:
        \nYou can use the get_jacobian function to get the jacobian of your function:
        \nExample usage:
        \n>>>fwdtest.get_jacobian()
        \nTry this yourself (feel free to copy the example if you want to speed up the process):
        """)

        str_funcjac = None
        while str_funcjac != "fwdtest.get_jacobian()":
            str_funcjac= input(">>>")
            try:
                funcjac = eval(str_funcjac)
                print("Function jacobian: ", funcjac)
            except:
                print("Try getting the jacobian of your function.")

        f2 = x1 ** (logistic(y1, 1, 7, 8) - exp(sin(y1) + cos(z1)))
        f3 = log_base(tanh(y1), 3) - ln(tan(sqrt(x1)))
        fwdtest_multiple = Forward([f1, f2, f3])

        print("""
        METHOD #3:
        \nYou can also use the get_jacobian function to get the jacobian of multiple functions:
        \nExample usage:
        \nf2 = x1 ** (logistic(y1, 1, 7, 8) - exp(sin(y1) + cos(z1)))
        \nf3 = log_base(tanh(y1), 3) - ln(tan(sqrt(x1)))
        \nfwdtest_multiple = Forward([f1, f2, f3])
        \n>>>fwdtest_multiple.get_jacobian()
        \nTry this yourself (feel free to copy the example if you want to speed up the process):
        """)

        str_funcmultjac = None
        while str_funcmultjac != "fwdtest_multiple.get_jacobian()":
            str_funcmultjac= input(">>>")
            try:
                funcmultjac = eval(str_funcmultjac)
                print("Function multiple jacobians: ", funcmultjac)
            except:
                print("Try getting the multiple jacobians of your functions.")

    elif demo_mode == 2:

        print("""
        STEP #1:

        \nDefine the variables of your function in the following format:
        <variable> = Dual(value = <DUAL, FLOAT, or INT>, derivative = <DUAL, FLOAT, or INT>, index = <optional INT>, total = <optional INT>)
        \n
        Example variables:
        \nx1 = Dual(value = 1, derivative=1, index = 0, total = 3)
        \ny1 = Dual(value= 2, derivative=1, index = 1, total = 3)
        \nz1 = Dual(value= 6, derivative=1, index = 2, total = 3)
        """)

        x1 = Dual(value = 1, derivative=1, index = 0, total = 3)
        y1 = Dual(value= 2, derivative=1, index = 1, total = 3)
        z1 = Dual(value= 6, derivative=1, index = 2, total = 3)

        print(">>>x1\n", eval("x1"))
        print(">>>y1\n", eval("y1"))
        print(">>>z1\n", eval("z1"))

        print("""
        STEP #2

        \nInput your function including the variables you defined above.

        \n
        Example function:
        \nf1 = 5 * x1 - 2 * y1 + 3 * z1 + 10
        """)

        f1 = 5 * x1 - 2 * y1 + 3 * z1 + 10

        print(">>>f1\n", eval("\nf1"))

        print("""
        STEP #3

        \nCreate an instance of your function as a Forward AD class.

        \n
        Example usage:
        \nfwdtest = Forward(f1)
        """)
        
        fwdtest = Forward(f1)
        
        print(">>>fwdtest\n", eval("\nfwdtest"))

        print("""
        METHOD #1:

        \nYou can use the get_value function to get the value of your function:
        """)

        # funcval = fwdtest.get_value()

        print(">>>fwdtest.get_value()\n", eval("fwdtest.get_value()"))

        print("""
        METHOD #2:

        \nYou can use the get_jacobian function to get the jacobian of your function:
        """)

        # funcjac = fwdtest.get_jacobian()

        print(">>>fwdtest.get_jacobian()\n", eval("fwdtest.get_jacobian()"))


        f2 = x1 ** (logistic(y1, 1, 7, 8) - exp(sin(y1) + cos(z1)))
        f3 = log_base(tanh(y1), 3) - ln(tan(sqrt(x1)))
        fwdtest_multiple = Forward([f1, f2, f3])
        # funcmultjac = fwdtest_multiple.get_jacobian()
        print("""
        METHOD #3:

        \nYou can also use the get_jacobian function to get the jacobian of multiple functions:

        \nExample usage:
        \nf2 = x1 ** (logistic(y1, 1, 7, 8) - exp(sin(y1) + cos(z1)))
        \nf3 = log_base(tanh(y1), 3) - ln(tan(sqrt(x1)))
        \nfwdtest_multiple = Forward([f1, f2, f3])
        """)
        print(">>>fwdtest_multiple.get_value()\n", eval("fwdtest_multiple.get_value()"))
        print(">>>fwdtest_multiple.get_jacobian()\n", eval("fwdtest_multiple.get_jacobian()"))

    print("""
    CONGRATULATIONS! YOU HAVE COMPLETED THE AD DEMO!

    \nIn case you forget, all information can be found within the AD documentation at github.com
    """)

    demo_state = input("""
        MENU OPTIONS:
        R) Restart
        E) Exit
        \n>>>""")




    
