<img src="https://github.com/cs107-swiftdiff/cs107-FinalProject/blob/main/docs/Logo.png" alt="Logo" width="400" height="330">

# Introduction
SwiftDiff software performs automatic differentiation (AD) for the user. AD is widely used across fields of science, engineering, and mathematics. Because the ability to compute derivates is key to research and applications in these fields, developing and implementing methodologies of AD that operate with speed and precision is crucial to enabling progress. Our AD software, SwiftDiff, sequentially evaluates elementary functions, and avoids the complexity of symbolic differentiation and precision issues of numerical differentiation. By overcoming all the setbacks of both finite difference methods and symbolic derivatives, AD is the most efficient and effective method. The system we present implements multiple methods of AD that compute the derivatives of a function in a single flow with machine precision and accuracy.

# Background

**Definition of a Derivative:**

<img src="https://render.githubusercontent.com/render/math?math=\color{gray}\lim_{h\to0}\frac{f(x%2Bh)-f(x)}{h}">

Originally conceptualized by Robert Edwin Wengert in his 1964 paper, *A simple automatic derivative evaluation program,* automatic differentiation has garnered much interest in the computational science, machine learning, and optimization communities, with its various forms being implemented in industry-standard libraries such as TensorFlow. 

Automatic differentiation, also known as algorithmic differentiation, computational differentiation, auto-differentiation, or autodiff, is different from either numerical or symbolic differentiation methods.

[Numerical differentiation (ND)](https://en.wikibooks.org/wiki/Introduction_to_Numerical_Methods/Numerical_Differentiation) is a class of methods that computes derivatives through computing discrete numerical approximations of the derivative. Common ND approaches include finite difference methods, which convert differential equations into an algebraically solvable system of linear equations. However, ND suffers from two main sources of inaccuracy - truncation and roundoff errors - as its precision is dependant on the step size of the derivative calculations. Furthermore there is a tradeoff in error reduction of trunction and roundoff errors, as smaller values of delta reduce truncation error but exacerbate roundoff error due to limited floating point accuracy.

[Symbolic differentiation (SD)](https://www.cs.utexas.edu/users/novak/asg-symdif.html) uses procedural rules to find general solutions to derivatives with respect to a variable. Instead of computing numerical approximations, SD manipulates a given input function to output a new function, ultimately producing a tree of expressions. However, SD methods suffer from computational inefficiency as derivatives can often become incredibly long and complex quickly, making the SD process slow and complicated.


**Dual Number:**

Dual numbers are utilized in our forward mode to represent the value and the derivative part of a variable because of its special property for addition and multiplication.

A dual number is made up of a real part and a dual part, which is writen as <img src="https://render.githubusercontent.com/render/math?math=\color{gray}z = a %2B b\epsilon">, where <img src="https://render.githubusercontent.com/render/math?math=\color{gray}\epsilon ^2 = 0">, but <img src="https://render.githubusercontent.com/render/math?math=\color{gray}\epsilon \ne 0">

Here are the definitions for addition and multiplication dual numbers that we utilize for our forward mode:

<img src="https://render.githubusercontent.com/render/math?math=\color{gray}z_1%2Bz_2=(a_1%2Bb_1\epsilon)%2B(a_2%2Bb_2\epsilon)=(a_1%2Ba_2)%2B(b_1%2Bb_2)\epsilon">
<img src="https://render.githubusercontent.com/render/math?math=\color{gray}z_1z_2 = (a_1 %2B b_1\epsilon)(a_2 %2B b_2\epsilon)=(a_1a_2)%2B(a_1b_1%2Ba_2b_2)\epsilon">

## Components of Automatic Differentiation:

Different from numerical differentiation and symbolic differentiation, automatic differentiation evalutes derivatives by breaking down complex functions into elementary functions to enable simple calculations intermediate values and subsequent efficient computation of the composite derivative.

All functions are compositions of a finite set of elementary operations for which derivatives are known. Combining the derivatives of these elementary functions through the chain rule results in the composite derivative of the function. 

The most simple type of AD is the forward accumulation mode, which applies the chain rule to each elementary operation in the forward primal trace, and then compute the corresponding derivative trace. This allows us to compute the Jacobians, or first-order partial derivatives, of vector-valued functions. Doing so is an efficient way of computing Jacobian-vector products, allowing us to derive the vector product in one forward pass. A computational graph can also complement tracing of the elementary operations by visualizing the relationship between the intermediate variables. 

A Jacobian matrix is simply a matrix of first-order derivatives of a function:

If f was a matrix of multiple functions:
<img src="https://render.githubusercontent.com/render/math?math=\color{gray}\f=\begin{bmatrix}f_1(x,y)\\f_2(x,y)\end{bmatrix}">

The Jacobian matrix would look like:
<img src="https://render.githubusercontent.com/render/math?math=\color{gray}J=\begin{bmatrix}\frac{\partial f_{1}}{\partial x}\frac{\partial f_{1}}{\partial y}\\ \frac{\partial f_{2}}{\partial x}\frac{\partial f_{2}}{\partial y}\end{bmatrix}">

See the following example of a trace table and its corresponding computational graph for the function <img src="https://render.githubusercontent.com/render/math?math=\color{gray}f(x,y)=e^{-(sin(x)-cos(y))^2}">:

| <img src="https://render.githubusercontent.com/render/math?math=\color{gray}Trace"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}Elementary Operation"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}Value">      | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}Elementary Derivation">        | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}\nabla{x}"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}\nabla{y}"> |
|-------|----------------------|------------|-----------------------------|-------------|---------------------------|
| <img src="https://render.githubusercontent.com/render/math?math=\color{gray}x_1">   | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}x"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}\pi/2"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}1"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}1"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}0"> |
| <img src="https://render.githubusercontent.com/render/math?math=\color{gray}y_1">   | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}y"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}\pi/3"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}1"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}0">  | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}1"> |
| <img src="https://render.githubusercontent.com/render/math?math=\color{gray}v_1"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}sin(x)"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}1"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}cos(x_1)\dot{x_1}"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}0"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}0"> |
| <img src="https://render.githubusercontent.com/render/math?math=\color{gray}v_2"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}cos(y)"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}1/2"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}-sin(y_1)\dot{y_1}"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}0"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}-\sqrt{3}/2"> |
| <img src="https://render.githubusercontent.com/render/math?math=\color{gray}v_3"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}v_1-v_2"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}1/2"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}\dot{v_1} - \dot{v_2}"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}0"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}\sqrt{3}/2"> |
| <img src="https://render.githubusercontent.com/render/math?math=\color{gray}v_4"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}v_3^2"> |<img src="https://render.githubusercontent.com/render/math?math=\color{gray}1/4"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}2 v_3 \dot{(v_3)}"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}0"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}\sqrt{3}/2"> |
| <img src="https://render.githubusercontent.com/render/math?math=\color{gray}v_5"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}-v_4"> |<img src="https://render.githubusercontent.com/render/math?math=\color{gray}-1/4"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray} -\dot{v_4}"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}0"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}-\sqrt{3}/2"> |
| <img src="https://render.githubusercontent.com/render/math?math=\color{gray}v_6"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}e^{v_5}"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}e^{-1/4}"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}e^{v_5} \dot{(v_5)}">| <img src="https://render.githubusercontent.com/render/math?math=\color{gray}0"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}-e^{(-1/4)} * \sqrt{3}/2"> |

![Computational Graph of f(x, y)](computational_graph.png?raw=true)

  In reverse mode, partial derivative values are stored at each node within a graph, and the full derivative is only computed using the chain rule during the backward pass. Further explanation on reverse mode is available towards the end of our documentation. 

  These two approaches are useful for different types of problems, with forward mode providing computational advantages in terms of storage and reverse mode providing computational advantages for functions with a large number of inputs. The main technical difference between forward and reverse mode is the point at which the matrix multiplication begins.

# How to Use

## Package Installation
Create a virtual environment:
```
conda -V
conda create -n vir_name
activate vir_name
```
Install the package:
```
pip install swiftdiff==0.0.2
```
Users can also clone this repo:
```
git clone https://github.com/cs107-swiftdiff/cs107-FinalProject
```
Install dependencies:
```
pip install -r requirements.txt
```

## Package Import
In the script, import AD_Forward, Dual, and elementary classes and methods
```
from AD.AD_forward import * 
from AD.Dual import * 
from AD.elementary import * 
```
Optionally, import AD_Reverse, Node and elementary_node classes and methods for reverse mode usage:
```
from AD.AD_reverse import *
from AD.Node import *
from elementary_node import *
```

## Package Usage

### Intialize Variables
Define the variables of your function in the following format:
variable = Dual(value, derivative, index, total)
```
x1 = Dual(value = 1, derivative = 1, index = 0, total = 3)
y1 = Dual(value = 2, derivative = 2, index = 1, total = 3)
z1 = Dual(value = 6, derivative = 4, index = 2, total = 3)
```
Define your function in terms of your variables:
```
f1 = 5 * x1 - 2 * y1 + 3 * z1 + 10
```
Create an instance of your function as a Forward AD class.
```
fwdtest = Forward(f1)
```

### Methods
Get the value of your function:
```
fwdtest.get_value()
```

Get the jacobian (derivative) of your function:
```
fwdtest.get_jacobian()
```
Get the jacobians of multiple functions:
```
f2 = x1 ** (logistic(y1, 1, 7, 8) - exp(sin(y1) + cos(z1)))
f3 = log_base(tanh(y1), 3) - ln(tan(sqrt(x1)))
twdtest_multiple=Forward([f1,f2,f3])
fwdtest_multiple.get_jacobian()
```

See demoAD.py for an interactive tutorial on primary usage.

See demoReverse.py for an interactive tutorial on reverse mode usage.

More information on SwiftDiff methods can be found in the [Implementation](#Implementation) section.

## Demos
SwiftDiff includes two basic demos.
To run the Main AD demo, run the following command:
```
python demoAD.py
```
To run the Reverse AD demo, run the following command:
```
python demoReverse.py
```

# Software Organization

## Directory structure:

```
SwiftDiff/
????????? AD
???   ????????? AD_forward.py
???   ????????? AD_reverse.py
???   ????????? Dual.py
???   ????????? Node.py
???   ????????? __init__.py
???   ????????? elementary.py
???   ????????? elementary_node.py
????????? LICENSE
????????? README.md
????????? demoAD.py
????????? demoReverse.py
????????? docs
???   ????????? Logo.png
???   ????????? backup.md
???   ????????? computational_graph.png
???   ????????? documentation.md
???   ????????? milestone1.md
???   ????????? milestone2.md
???   ????????? milestone2_progress.md
???   ????????? test_coverage.png
????????? old
???   ????????? ADforward.py
???   ????????? test_ADforward.py
????????? requirements.txt
????????? tests
    ????????? test_AD_forward.py
    ????????? test_AD_reverse.py
    ????????? test_Node.py
    ????????? test_dual.py
    ????????? test_elementary.py
    ????????? test_elementary_node.py
```

## Modules:
* **Forward mode module** - implements forward mode AD
* **Reverse mode module** - implements reverse mode AD

### Test Suite Environment:
We use pytest to test the performance and coverage of our code. Using pytest before integrating new changes will ensure that new changes are only merged once they pass the tests/break the code. The test suite is in the test folder, and will contain several tests for the code contained in the AD folder. Our test suite can be run by executing the following in the CLI:
1. ```pip install pytest-cov```
2. ```pytest --cov=AD```

Below is a screenshot of our test coverage:

![test_coverage](test_coverage.png?raw=true)

As you can see - all of our testing for each class was successful - they were all over 90%. This shows that our AD software is robust.

### Distribution:
OMIT(We will distribute our package via PyPI and create a landing page with detailed documentation and download options.)
SwiftDiff is available on [PyPI](https://pypi.org/project/SwiftDiff/) and [Github](https://github.com/cs107-swiftdiff/cs107-FinalProject). See Package Installation for more details.


### Software Packaging:
* Our software is packaged in accordance with the standard [Python Packaging protocol](https://packaging.python.org/tutorials/packaging-projects/).

# Implementation

## Core data structures:
* list
* np.array
* float
* int
* tuple

## Core classes:
* class AutoDiff(function)
  * class Forward(function) - child of AutoDiff class
  * class Reverse(function,variable) - child of AutoDiff class
* class Dual(value, derivative, index, total)
* class Node(value, derivative, children)

## Important attributes:
class AutoDiff:
* ```self.f``` - function

class Forward:
* ```self.f``` - function
 
class Reverse:
* ```self.f``` - function
* ```self.variables``` - variables in the function

class Dual:
* ```self.value``` - value of the variable
* ```self.derivative``` - derivative of the variable
* ```self.index``` - index of the variable inside the function
* ```self.total``` - total number of variables in the function

class Node:
* ```self.value``` - value of the node
* ```self.derivative``` - derivative of the node
* ```self.children``` - children of the node

## Elementary functions
* SwiftDiff overloads elementary functions to accomodate Dual objects and Node objects:
  * ```exp``` - calculates e to the power of the input
  * ```sin``` - calculates the sine of the input
  * ```cos``` - calculates cosine of the input
  * ```tan``` - calculates tangent of the input
  * ```sinh``` - calculates the hyperbolic sine of the input
  * ```cosh``` - calculates the hyperbolic cosine of the input
  * ```tanh``` - calculates the hyperbolic tangent of the input
  * ```ln``` - calculates the log of the input
  * ```log_base``` - calculates the log of the input in a specified base
  * ```logistic``` - calculates the logistic function including the input
  * ```sqrt``` - calculates the square root of the input

## Classes 

### ```Dual(value, derivative, **kwargs)```
  * Inputs:
    * value
    * derivative
    * **kwargs (optional)
      * index
      * total
  * Attributes:
    * ```self.value``` - value of the variable
    * ```self.derivative``` - derivative of the variable
    * ```self.index``` - index of the variable inside the function
    * ```self.total``` - total number of variables in the function
  * Methods: 
    * ```__init__``` - constructs an instance of the Dual class
    * ```__repr__``` - returns class name and attributes of a Dual object
    * ```__add__``` - adds a Dual object to something
    * ```__radd__``` - adds something to a Dual object
    * ```__sub__``` - subtracts something from a Dual object
    * ```__rsub__``` - subtracts a Dual object from something
    * ```__neg__``` - returns the negative of a Dual object
    * ```__mul__``` - multiplies a Dual object and something
    * ```__rmul__``` - multiplies something and a Dual object
    * ```__truediv__``` - divides a Dual object by something
    * ```__rtruediv__``` - divides something by a Dual object
    * ```__pow__``` - raises a Dual object to the power of something
    * ```__rpow__``` - raises something to the power of a Dual object
    * ```__lt__``` - decides if a Dual object is less than something
    * ```__le__``` - decides if a Dual object is less than or equal to something
    * ```__gt__``` - decides if a Dual object is greater than something
    * ```__ge__``` - decides if a Dual object is greater than or equal to something
    * ```__eq__``` - decides if a Dual object is equal to something
    * ```__ne__``` - decides if a Dual object is not equal to something

### ```Node(value)```
  * Inputs:
    * value
  * Attributes:
    * ```self.value``` - value of the variable
    * ```self.derivative``` - derivative of the variable
    * ```self.children``` - list of children of the node
  * Methods: 
    * ```__init__``` - constructs an instance of the Node class
    * ```__repr__``` - returns class name and attributes of a Node object
    * ```der``` - calculates the derivatives of a Node object
    * ```__add__``` - adds a Node object to something
    * ```__radd__``` - adds something to a Node object
    * ```__sub__``` - subtracts something from a Dual object
    * ```__rsub__``` - subtracts a Node object from something
    * ```__neg__``` - returns the negative of a Node object
    * ```__mul__``` - multiplies a Node object and something
    * ```__rmul__``` - multiplies something and a Node object
    * ```__truediv__``` - divides a Node object by something
    * ```__rtruediv__``` - divides something by a Node object
    * ```__pow__``` - raises a Node object to the power of something
    * ```__rpow__``` - raises something to the power of a Node object
    * ```__lt__``` - decides if a Node object is less than something
    * ```__le__``` - decides if a Node object is less than or equal to something
    * ```__gt__``` - decides if a Node object is greater than something
    * ```__ge__``` - decides if a Node object is greater than or equal to something
    * ```__eq__``` - decides if a Node object is equal to something
    * ```__ne__``` - decides if a Node object is not equal to something
    
### ```Forward(f)```
  * Inputs:
    * f
  * Attributes:
    * ```self.f``` - functions from input
  * Methods: 
    * ```get_value``` - calculates the value
    * ```get_jacobian``` - calculates the Jacobian

### ```Reverse(f,variables)```
  * Inputs:
    * f
    * variables
  * Attributes:
    * ```self.f``` - functions from input
    * ```self.variables``` - variables in the functions
  * Methods: 
    * ```get_value``` - calculates the value
    * ```get_jacobian``` - calculates the Jacobian
    
## External dependencies: 
  * [NumPy](https://numpy.org/) - to perform mathematical operations
  * [PyPI](https://pypi.org/) - to publish the package

Checks are implemented throughout these classes to ensure that inputs are valid. Elementary functions are either overloaded or handled via external packages.

# Extension - Reverse Mode AD
## Initial Plan
[From Milestone 2] Experiments with Codex to generate novel AD methods - translating various concepts in AD from natural language into code through OpenAI's Codex.

Codex is a powerful code-generation tool developed by OpenAI as a fine-tuned implementation of their flagship natural language model, GPT3. Integrated into emerging applications and existing development environments, such as Github Copilot, Codex has shown massive potential to revolutionize the software development process through eliminating repetitive work and increasing efficiency.We will use Codex to generate different implementations of autodifferentiation based on natural language prompts. We will evaluate Codex's performance using our test suite on reverse-pass and reverse-mode implementations. Throughout this process, we will examine the impact of different natural language prompts and various model parameters (temperature, top P, frequency penalty, presence penalty, engine type (davinci-codex, cushman-codex), etc.) on the performance of the generated code (measured in terms of precision, speed, coverage, accuracy, etc.). Beyond reverse-mode implementations, we will explore novel approaches to implementing AD using creative Codex prompts.

However, we didn't use Codex because it generates the code for us, but we believe that we need to write our own code for this project. Thus, we decided to implement reverse mode for autodifferentiation.

## New Extension

Reverse mode automatic differentiation builds on the forward mode computational graph by enabling reverse traversal to compute gradients. Reverse mode stores values for all variables in nodes, and computes the gradient in one pass, which is more computationally efficient than forward mode AD.

Forward mode and reverse mode have relative advantages and disadvantages in different types of siutations. While reverse mode has a significantly smaller arithmetic count than forward mode, it also has to store the full computational graph in memory. In contrast, forward mode is more efficient when the number of variables is small, but it becomes far more complex for large numbers of variables. Our implementation of reverse mode allows users to select the appropriate solution for their application. For applications where the number of variables is small, forward mode is more efficient, while for large numbers of variables, reverse mode is more efficient.

Implementation-wise, reverse mode is a two-pass process that works by first running a forward pass through the the elementary functions and storing all partial derivatives without evaluating the chain rule. Then, reverse mode traverses the graph in reverse order and multiplies the current partial derivative by the derivative of the next node. The partial derivatives derived from reverse mode reflect the sensitivity of the output with respect to the intermediate variable v, as follows:

<img src="https://render.githubusercontent.com/render/math?math=\color{gray}v_{j-m}=\frac{\partial{f_i}}{\partial{v_{j-m}}}">

Our reverse mode extension allows users to work with reverse mode AD in a similar fashion to forward mode AD. Users can call the following methods when working with Reverse AutoDiff class objects:
* ```get_value``` - gets the value of the function
* ```get_jacobian``` - gets the Jacobian (derivative) of the function

But be careful that there are some differences in initializing. To use reverse mode:

First, define variables:
```
x1 = Node(1)
y1 = Node(2)
```

Second, define functions:
```
def f1(x1,y1): return (x1+y1)
```
Third, create an instance of your function as a Reverse AD class:
```
test = Reverse(f1,[x1,y1])
```
Forth, use get_value(), and get_jacobian() function to get value and Jacobian
```
test.get_value()
test.get_jacobian()
```

## Future Steps
Autodifferentiation has applications across all industries, including physics, biology, genetics, applied mathematics, optimization, statistics / machine learning, health science, and more.
We believe that our software, if implemented correctly -  can be beneficial to all industries beyond software engineering. 

In addition to applications within these industries, there are several ways in which the SwiftDiff package can be expanded on in the future to include more functionality. We welcome community contribution - feel free to reach out or make a pull request to the repository.

Potential features / expansions:
* Visualization module - allowing users to visualize the workflow of calculations within forward and reverse mode AD
* Statistics Module - aggregating cost, efficiency, and error statistics across multiple cycles of AD using various methods
* ML Integration - integrating SwiftDiff into existing ML systems
* NLG AD Education Tool - using NLG technologies (e.g. [GPT3](https://openai.com/blog/gpt-3-apps/)) to interactively teach users about AD
* AD Builder Assistant - using copilot technologies (e.g. [Codex](https://openai.com/blog/openai-codex/)) to help users build their own AD packages

# Licensing
SwiftDiff uses the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html), 
a copyleft license that makes the complete source code of licensed works and modifications available. This license allows others to copy, distribute, and fork SwiftDiff given the license section remains unchanged.

# Broader Impact and Inclusivity Statement

## Broader Impact
 
As we worked on this project throughout the semester - we found the development of our project to be highly complex and pragmatic; as we explored and researched the fundamentals of Automatic Differentiation to create our software, we constantly thought to ourselves, wouldn't it be nice if there was a program that did this for us? This is precisely why we created our software - in order to provide a quick, easy, and automated method to calculate complex derivatives, and take advantage of the power of machines. However - this is the very foundation to why softwares like this may need to be more careful and purposeful in its applications. Programming a software to automatically perform differentiation is powerful - yet can take away from learning and education. Students like us - building this software - take a lot away from creating this software; we???ve done heavy research, and manually programmed all the code itself. For future students, who may utilize our software, may not even understand the fundamentals of derivatives, differentiation, etc. Instead of learning the true dynamics of differentiation, they can simply use our software to retrieve derivatives efficiently. Therefore - we highlight that our software be used as a tool, not a shortcut. We believe that our software should be utilized with respect and academic honor.
 
## Inclusivity
 
As a completely woman group, we take immense pride and passion in the notion of inclusivity, both within the details of our software, and the software development field as a whole.  As the developers of AutoDiff, we encourage participation and engagement from all different backgrounds, and fully welcome a global/international reach. As the Python Software Diversity Statement highlights - we hope to embody mutual respect, tolerance, and encouragement within our software, and as developers ourselves. We hope that our software is accessible, and utilized by all different types of users.

To embody this inclusivity - us developers have constantly collaborated and contributed our code with the utmost respect and consideration. Through effective communication, persistently checking in, approving each other's pull requests, endless hours of Zoom calls, and in-person meetings, we have practiced maintaining a code of honesty and respect throughout the development of our software. By doing so - we hope that our users can do the same while interacting with our software. 

We are aware, however, that there are limitations in terms of access to our software - and we hope to contribute to a movement in overcoming these barriers and supporting a more inclusive environment for software development. Although this is a macro problem deeply ingrained in the history and development of computer science, we want to do our best to make sure our software is accessible to all different backgrounds of users. 

For those who find parts of our program to be offensive, we will do our best to re-format our code base and ensure that no sensitive names for classes are used. 

For those who hope to contribute to our program itself, we will leave our emails below to make sure full collaboration and contributions can be made by the community as a whole.

For those who are unable to understand our code through the English language, we will do our best to provide translation scripts and integrate them into our code base in the upcoming months.

For those who weren???t able to receive the education to learn how to code, we will be happy to accommodate a possible time to meet, to allow us to explain what each class, function, etc. does in our software. 
 
For those with disabilities, we hope to integrate additional modules, such as audio modules and Braille modules in the upcoming months. 

Although the automation of our software is complete - we will continuously consider all components of our software and improve it to allow full accessibility and inclusivity.

Thank you so much for utilizing our software, and here are our emails:
Nadine Lee nadine_lee@college.harvard.edu
Li Sun lsun@g.harvard.edu
Alice Cai acai@college.harvard.edu

# Resources for Further Exploration
## Educational Resources
1. [Systems Development for Computational Science - Fabian Wermelinger](https://harvard-iacs.github.io/2021-CS107/)
   1. [Lecture 9](https://harvard-iacs.github.io/2021-CS107/lectures/lecture9/presentation/lecture09.pdf)
   2. [Lecture 10](https://harvard-iacs.github.io/2021-CS107/lectures/lecture10/presentation/lecture10.pdf)
   3. [Lecture 11](https://harvard-iacs.github.io/2021-CS107/lectures/lecture11/presentation/lecture11.pdf)
2. [Forward-Mode Automatic Differentiation (AD) via High Dimensional Algebras - Chris Rackauckas](https://mitmath.github.io/18337/lecture8/automatic_differentiation)
3. [Juedes, David W. *A taxonomy of automatic differentiation tools.* No. ANL/CP-74447; CONF-910189-3. Argonne National Lab., IL (United States), 1991.](https://www.osti.gov/servlets/purl/5015838)
4. [Introduction to Numerical Methods/Numerical Differentiation - Wikibooks](https://en.wikibooks.org/wiki/Introduction_to_Numerical_Methods/Numerical_Differentiation)
5. [(SD Example) CS 381K: Symbolic Differentiation - Gordon Novak](https://www.cs.utexas.edu/users/novak/asg-symdif.html)
6. [(AD Educational Video) What is Automatic Differentiation? - Ari Seff](https://www.youtube.com/watch?v=wG_nF1awSSY)
   
## AD Applications
1. [Baydin, Atilim Gunes, et al. "Automatic differentiation in machine learning: a survey." *Journal of Machine Learning Research* 18 (2018).](https://www.jmlr.org/papers/volume18/17-468/17-468.pdf)
2. [Tamayo-Mendoza, Teresa, et al. "Automatic differentiation in quantum chemistry with applications to fully variational Hartree???Fock." ACS central science 4.5 (2018): 559-566.](https://pubs.acs.org/doi/abs/10.1021/acscentsci.7b00586?__cf_chl_jschl_tk__=pmd_S_w295swYDncnRI6Z2Vzy.G3OG7GxeMU4QmznNEKm8E-1634831077-0-gqNtZGzNAjujcnBszQil)
3. [Mordvintsev, Alexander, et al. "Growing neural cellular automata." *Distill* 5.2 (2020): e23.](https://distill.pub/2020/growing-ca/)

## Toolkits & Packages
1. [Database of Tools for AD - autodiff.org](http://www.autodiff.org/?module=Tools)
2. [Paszke, Adam, et al. "Automatic differentiation in pytorch." (2017).](https://openreview.net/pdf?id=BJJsrmfCZ)
3. [A Gentle Introduction to torch.autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
