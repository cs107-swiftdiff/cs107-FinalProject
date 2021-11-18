<img src="https://github.com/cs107-bestorg/cs107-FinalProject/blob/main/docs/Logo.png" alt="Logo" width="400" height="330">

# Introduction
BestOrg software performs automatic differentiation (AD) for the user. AD is widely used across fields of science, engineering, and mathematics. Because the ability to compute derivates is key to research and applications in these fields, developing and implementing methodologies of AD that operate with speed and precision is crucial to enabling progress. Our AD software, BestOrg, sequentially evaluates elementary functions, and avoids the complexity of symbolic differentiation and precision issues of numerical differentiation. By overcoming all the setbacks of both finite difference methods and symbolic derivatives, AD is the most efficient and effective method. The system we present implements multiple methods of AD that compute the derivatives of a function in a single flow with machine precision and accuracy.

# Background

**Definition of a Derivative:**

<img src="https://render.githubusercontent.com/render/math?math=\color{gray}\lim_{h\to0}\frac{f(x + h) - f(x)}{h}">

Originally conceptualized by Robert Edwin Wengert in his 1964 paper, *A simple automatic derivative evaluation program,* automatic differentiation has garnered much interest in the computational science, machine learning, and optimization communities, with its various forms being implemented in industry-standard libraries such as TensorFlow. 

Automatic differentiation, also known as algorithmic differentiation, computational differentiation, auto-differentiation, or autodiff, is different from either numerical or symbolic differentiation methods.

[Numerical differentiation (ND)](https://en.wikibooks.org/wiki/Introduction_to_Numerical_Methods/Numerical_Differentiation) is a class of methods that computes derivatives through computing discrete numerical approximations of the derivative. Common ND approaches include finite difference methods, which convert differential equations into a algebraically solvable system of linear equations. However, ND suffers from two main sources of inaccuracy - truncation and roundoff errors - as its precision is dependant on the step size of the derivative calculations. Furthermore there is a tradeoff in error reduction of trunction and roundoff errors, as smaller values of delta reduce truncation error but exacerbate roundoff error due to limited floating point accuracy.

[Symbolic differentiation (SD)](https://www.cs.utexas.edu/users/novak/asg-symdif.html) uses procedural rules to find general solutions to derivatives with respect to a variable. Instead of computing numerical approximations, SD manipulates a given input function to output a new function, ultimately producing a tree of expressions. Hoewver, SD methods suffer from computational inefficiency as derivatives can often become incredibly long and complex quickly, making the SD process slow and complicated.

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

  In reverse mode, partial derivative values are stored at each node within a graph, and the full derivative is only computed using the chain rule during the backward pass. 

  These two approaches are useful for different types of problems, with forward mode providing computational advantages in terms of storage and reverse mode providing computational advantages for functions with a large number of inputs. The main technical difference between forward and reverse mode is the point at which the matrix multiplication begins.

# How to Use
Users can mainly interact with the BestorgAD package in two ways:
1. Via the BestorgAD user interface (GUI + text UI)
2. Via importing BestorgAD into existing code

## Import the Package
Create a virtual environment:
```
conda -V
conda create -n vir_name
activate vir_name
```
Install the package:
```
conda install -n BestorgAD
```
Users can also clone this repo:
```
git clone https://github.com/cs107-bestorg/cs107-FinalProject
```
Install dependencies:
```
conda install numpy
```
Copy ADforward.py which in AD file into current working directory

In the script, import AutoDiff
```
from ADforward import AutoDiff
```
Define 1-D Scalar Variable:
```
# x = a, in 1-D
>>> x = AutoDiff(a,1)
```
Define function:
```
>>> f(x)
```
Find value and 1-D Jacobian (derivative):
```
# value
>>> f.value

# Jacobian
# 1-D Jacobian is derivative - need to use .derivative
>>> f.derivative
```
Demo: R1 -> R1
Consider the function: $f(x) = 2 x^{4}$ at $x = 2$
```
#define variable
>>> x = AutoDiff(2,1)
#define function
>>> f = 2*x**4
#value
>>> f.value
32
#Jacobian
>>> f.derivative
48
```

## Interacting with the GUI / Text UI
Call the script for the desired interface type:
```
python /UI/GUI.py
```
OR
```
python /UI/textUI.py
```
From here, a GUI or Text UI will appear, where users can input functions, select methodologies (forward mode, reverse mode, etc.), set parameters, and visualize outputs. 

## 
ing the Package into Existing Code
```
import BestorgAD
```

Users can call the following functions after importing our package:
* forward_AD(function, (coordinate)) - performs forward-mode AD
* reverse_AD(function, (coordinate)) - performs reverse-mode AD
* generate_graph(function) - generates a computational graph
* generate_tracetable(function, (coordinate)) - generates a trace table
* (TBD) calculate_statistics(function, (coordinate), [methods]) - calculates statistics across different methods of AD
[More information on BestorgAD methods within classes](#Implementation)

# Software Organization
## Directory structure:
```
cs107-FinalProject/
├── docs
│   ├── milestone1.md
|   └── milestone2_progress
|   └── milestone2.md
├── AD
│   ├── __init__.py
|   ├── ADmain.py
|   ├── ADforward.py
|   ├── ADnovelmethod.py
|   ├── ADgraph.py
|   └── (TBD) ADstatistics.py
├── UI
│   ├── GUI.py
|   └── textUI.py
├── test_ADforward.py
├── setup.py
├── LICENSE
├── README.md
└── requirements.txt
```

## Modules:
* **Forward mode module** - implements forward mode AD
* **Novel AD module** - implements a novel method of AD (TBD as project progresses)
* **UI module** - constructs a user interface for direct interaction
* **Visualization module** - creates visualizations for AD processes
* **Statistical analysis module** - analyzes performance across different instances of vairous AD algorithms

### Test Suite Environment:
We use pytest to test the performance and coverage of our code. Using pytest before integrating new changes will ensure that new changes are only merged once they pass the tests/break the code. The test suite is in the main directory, within test_ADforward.py and will contain several tests for the AD software. Our test suite can be run by executing the following in the CLI:
1. ```pip install pytest-cov```
2. ```pytest --cov=AD  test_ADforward.py```

Below is a screenshot of our tested ADforward.py:

![IMG_3340](https://user-images.githubusercontent.com/85530513/142346583-63fc0b1d-7c7d-46d3-8fe7-2e1754e32e7e.jpeg)

### Distribution Plan:
We will distribute our package via PyPI and create a landing page with detailed documentation and download options.

### Software Packaging:
* Our software will be packaged in accordance with the standard [Python Packaging protocol](https://packaging.python.org/tutorials/packaging-projects/).
* We will not use a framework, as frameworks are generally best used for large-scale projects (e.g. app / web development), but we could use a 
[Python package boilerplate](https://github.com/mtchavez/python-package-boilerplate) instead.

# Implementation

### Current forward-mode AD implementation:

Core data structures:
* NA for forward-mode implementation

Core classes:
* class AutoDiff(self, derivative)

Important attributes:
* self.value
* self.derivative

External dependencies:
* numpy
* (for testing) pytest

Elementary functions
* BestorgAD implements elementary functions as AutoDiff object methods according to the rules of trignometric differentiation.
  * ```__add__```
  * ```__radd__```
  * ```__sub__```
  * ```__rsub__```
  * ```__mul__```
  * ```__rmul__```
  * ```__truediv__```
  * ```__rtruediv__```
  * ```__neg__```
  * ```__pow__```
  * ```sin```
  * ```cos```
  * ```tan```
  * ```exp```
  * ```log```
  * ```sqrt```

## To Implement
We have not yet implemented vector functionality for BestorgAD. In order to accomodate more complex functions, multiple functions, and multiple inputs, we will use the following datastructures: tuples, lists, dictionaries, ndarray, and trees as core basic data structures. BestorgAD custom classes will serve to facilitate the flow of data within the package.
Some examples:
* tuples: (value of function,its derivative) may change in the future
* Lists: store intermediate trace values.
* Dictionaries: match each opearion and its parameters.
* ndarray: do not have specific examples, but may use to store lists
* trees: help to build the structure of computional graph.

## Classes 

### ```inputFunction()```
  * Inputs:
    * function to evaluate derivative of
    * value at which to evaluate the derivative
  * Attributes:
    * ```self.function```
    * ```self.value```
  * Methods: 
    * ```__init__()``` - construct instance 
    * ```decompose()``` - returns a dictionary of elementary operations {'operation':[parameter(s) of the operation]} 

### ```elementaryOperation()```
  * Inputs: 
    * intermediate trace values from previous calculations
    * elementary operation (from inputFunction.decompose() dictionary)
  * Attributes:
    * ```self.input_intertraces``` - [inputed intermediate trace values]
    * ```self.output_intertrace``` - outputed intermediate value
    * ```self.output_interderivation``` - {'variable': symbolic partial derivative value with respect to the variable}
  * Methods:
    * ```__init__``` - construct instance
    * ```evaluate_trace()``` - computes the trace value of the elementary operation and stores that value in self.intermediate_value
    * ```elem_derive()``` - derives the symbolic derivative of the elementary operation and outputs into a newly initiated instance of class elementaryDerivative()
    * Overloaded dunder methods to deal with dual numbers:
      * ```__add__()``` - e.g. ComplexNumber(self.real + other.real, self.imaginary + other.imaginary) 
      * ```__sub__()``` - e.g. ComplexNumber(self.real - other.real, self.imaginary - other.imaginary) 
      * ```__mul__()``` - e.g. ComplexNumber(self.real * other.real - self.imaginary * other.imaginary, self.real * other.imaginary + other.real * self.imaginary) 
      * ```__truediv__()``` - e.g. ComplexNumber((self.real * other.real + self.imaginary * other.imaginary) / (other.real^2 + other.imaginary^2), (self.imaginary * other.real + self.real * other.imaginary) / (other.real^2 + other.imaginary^2)) 
      * Other elementary functions TBD (sin, cos, tan, cot, sqrt, exp, etc.) 
        * e.g. ```__sin__()``` - e.g. sin(self.real) * cosh(self.imaginary) + i cos(self.real) * sinh(self.imaginary))

### ```elementaryDerivation()```
  * Inputs: 
    * intermediate partial derivative values from previous calculations
    * intermediate trace values from previous calculations
    * elementary operation (from elementaryOperation.elem_derive())
  * Attributes:
    * ```self.input_intertraces``` - [inputed intermediate trace values]
    * ```self.output_partials``` - {'variable': numerical partial derivative value with respect to the variable}
  * Methods:
    * ```__init__``` - construct instance
    * ```evaluate_partial()``` - computes the numerical partial derivative of the elementary operation with respect to a specific variable
    * ```elem_derive()``` - derives the symbolic derivative of the elementary operation and outputs into a newly initiated instance of class elementaryDerivative()
    * Overloaded dunder methods to deal with dual numbers:
      * ```__add__()``` - e.g. ComplexNumber(self.real + other.real, self.imaginary + other.imaginary) 
      * ```__sub__()``` - e.g. ComplexNumber(self.real - other.real, self.imaginary - other.imaginary) 
      * ```__mul__()``` - e.g. ComplexNumber(self.real * other.real - self.imaginary * other.imaginary, self.real * other.imaginary + other.real * self.imaginary) 
      * ```__truediv__()``` - e.g. ComplexNumber((self.real * other.real + self.imaginary * other.imaginary) / (other.real^2 + other.imaginary^2), (self.imaginary * other.real + self.real * other.imaginary) / (other.real^2 + other.imaginary^2)) 
      * Other elementary functions TBD (sin, cos, tan, cot, sqrt, exp, etc.) 
        * e.g. ```__sin__()``` - e.g. sin(self.real) * cosh(self.imaginary) + i cos(self.real) * sinh(self.imaginary))

### ```visualAid()```
  * Inputs: 
    * function to generate visual aids for
  * Attributes:
    * ```self.function```
  * Methods:
    * ```__init__``` - construct instance
    * ```generate_graph()``` - generates a computational graph of the input function
    * ```generate_tracetable()``` - generates a trace table for the input function with the following basic format:
  
        | Trace | Elementary Operation | Value | Elementary Derivation | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}\nabla{x}"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}\nabla{y}"> | <img src="https://render.githubusercontent.com/render/math?math=\color{gray}\nabla{etc.}"> |
        |-------|----------------------|-------|-----------------------|-------------|-------------|---------------|
        | ... | ... | ... |  ... | ... | ... | ... |

  * Note: visualAid() serves as a wrapper class that can access the functionality of the methods defined in other classes

Checks will be implemented throughout these classes to ensure that inputs are valid. Elementary functions will either be overloaded or handled via external packages.

### GUI(): 
This class will inherit attributes and methods from an external GUI package ([PyQt5](https://pypi.org/project/PyQt5/)).

## External dependencies: 
  * [NumPy](https://numpy.org/) - to perform mathematical operations
  * [Matplotlib](https://matplotlib.org/) - to generate graphical visualizations
  * [PyQt5](https://pypi.org/project/PyQt5/) - to generate user interfaces
  * [Pandas](https://pandas.pydata.org/) - to generate trace tables
  * [PyPI](https://pypi.org/) - to publish the package


# Licensing
BestorgAD uses the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html), 
a copyleft license that makes the complete source code of licensed works and modifications available. This license allows others to copy, distribute, and fork BestorgAD given the license section remains unchanged.

# Extension
* Experiments with Codex to generate novel AD methods - translating various concepts in AD from natural language into code through OpenAI's Codex

### Proposal:

Codex is a powerful code-generation tool developed by OpenAI as a fine-tuned implementation of their flagship natural language model, GPT3. Integrated into emerging applications and existing development environments, such as Github Copilot, Codex has shown massive potential to revolutionize the software development process through eliminating repetitive work and increasing efficiency.

We will use Codex to generate different implementations of autodifferentiation based on natural language prompts. We will evaluate Codex's performance using our test suite on reverse-pass and reverse-mode implementations. Throughout this process, we will examine the impact of different natural language prompts and various model parameters (temperature, top P, frequency penalty, presence penalty, engine type (davinci-codex, cushman-codex), etc.) on the performance of the generated code (measured in terms of precision, speed, coverage, accuracy, etc.). Beyond reverse-mode implementations, we will explore novel approaches to implementing AD using creative Codex prompts.

## Other Potential Extensions (TBD as project progresses)
* Statistics Module - aggregates cost, efficiency, and error statistics across multiple cycles of AD using various methods
* ML Integration - integrates BestorgAD into existing ML systems
* AD for Neural Cellular Automata - implements BestorgAD in differentiable model of morphogenesis
  * [Example of growing neural cellular automata](https://distill.pub/2020/growing-ca/)
* Novel NLG Differentiation Algorithm - uses NLG technologies (e.g. [GPT3](https://openai.com/blog/gpt-3-apps/)) to compute derivatives
* AD Builder Assistant - uses copilot technologies (e.g. [Codex](https://openai.com/blog/openai-codex/)) to help users build their own AD packages through natural language


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
2. [Tamayo-Mendoza, Teresa, et al. "Automatic differentiation in quantum chemistry with applications to fully variational Hartree–Fock." ACS central science 4.5 (2018): 559-566.](https://pubs.acs.org/doi/abs/10.1021/acscentsci.7b00586?__cf_chl_jschl_tk__=pmd_S_w295swYDncnRI6Z2Vzy.G3OG7GxeMU4QmznNEKm8E-1634831077-0-gqNtZGzNAjujcnBszQil)
3. [Mordvintsev, Alexander, et al. "Growing neural cellular automata." *Distill* 5.2 (2020): e23.](https://distill.pub/2020/growing-ca/)

## Toolkits & Packages
1. [Database of Tools for AD - autodiff.org](http://www.autodiff.org/?module=Tools)
2. [Paszke, Adam, et al. "Automatic differentiation in pytorch." (2017).](https://openreview.net/pdf?id=BJJsrmfCZ)
3. [A Gentle Introduction to torch.autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)



---
