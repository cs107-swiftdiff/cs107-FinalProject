# Introduction
This software performs automatic differentiation (AD) for the user. AD is widely used across fields of science, engineering, and mathematics. Because the ability to compute derivates is key to research and applications in these fields, developing and implementing methodologies of AD that operate with speed and precision is crucial to enabling progress. The system we present implements multiple methods of AD that compute the derivatives of a function in a single flow with machine precision and accuracy.

# Background

**Definition of a Derivative:**

<img src="https://render.githubusercontent.com/render/math?math=\color{gray}\lim_{h\to0}\frac{f(x + h) - f(x)}{h}">

Originally conceptualized by Robert Edwin Wengert in his 1964 paper, *A simple automatic derivative evaluation program,* automatic differentiation has garnered much interest in the computational science, machine learning, and optimization communities, with its various forms being implemented in industry-standard libraries such as TensorFlow. 

Automatic differentiation, also known as algorithmic differentiation, computational differentiation, auto-differentiation, or autodiff, can be thought of as a synethesis of both numerical and symbolic differentiation methods.

[Numerical differentiation (ND)](https://en.wikibooks.org/wiki/Introduction_to_Numerical_Methods/Numerical_Differentiation) is a class of methods that computes derivatives through computing discrete numerical approximations of the derivative. Common ND approaches include finite difference methods, which convert differential equations into a algebraically solvable system of linear equations. However, ND suffers from two main sources of inaccuracy - truncation and roundoff errors - as its precision is dependant on the step size of the derivative calculations. Furthermore there is a tradeoff in error reduction of trunction and roundoff errors, as smaller values of delta reduce truncation error but exacerbate roundoff error due to limited floating point accuracy.

[Symbolic differentiation (SD)](https://www.cs.utexas.edu/users/novak/asg-symdif.html) uses procedural rules to find general solutions to derivatives with respect to a variable. Instead of computing numerical approximations, SD manipulates a given input function to output a new function, ultimately producing a tree of expressions. Hoewver, SD methods suffer from computational inefficiency as derivatives can often become incredibly long and complex quickly, making the SD process slow and complicated.

## Components of Automatic Differentiation:

Synethesizing numerical and symbolic differentiation methods, automatic differentiation evalutes derivatives by breaking down complex functions into elementary functions to enable simple calculations intermediate values and subsequent efficient computation of the composite derivative.

All functions are compositions of a finite set of elementary operations for which derivatives are known. Combining the derivatives of these elementary functions through the chain rule results in the composite derivative of the function. 

The most simple type of AD is the forward accumulation mode, which applies the chain rule to each elementary operation in the forward primal trace, and then compute the corresponding derivative trace. This allows us to compute the Jacobians, or first-order partial derivatives, of vector-valued functions. Doing so is an efficient way of computing Jacobian-vector products, allowing us to derive the vector product in one forward pass. A computational graph can also complement tracing of the elementary operations by visualizing the relationship between the intermediate variables. 

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

## Importing the Package into Existing Code
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
│   ├── milestone1
|   └── milestone2
├── ADpackage
│   ├── __init__.py
|   ├── ADmain.py
|   ├── ADforward.py
|   ├── ADreverse.py
|   ├── ADnovelmethod.py
|   ├── ADgraph.py
|   └── (TBD) ADstatistics.py
├── UI
│   ├── GUI.py
|   └── textUI.py
├── setup.py
├── LICENSE
├── README.md
├── requirements.txt
├──.travis.yml
└──.codecov.yml
```

## Modules:
* **Forward mode module** - implements forward mode AD
* **Reverse mode module** - implements reverse mode AD
* **Novel AD module** - implements a novel method of AD (TBD as project progresses)
* **UI module** - constructs a user interface for direct interaction
* **Visualization module** - creates visualizations for AD processes
* **Statistical analysis module** - analyzes performance across different instances of vairous AD algorithms

### Test Suite Environment:
TravisCI + CodeCov

### Distribution Plan:
We will distribute our package via PyPI and create a landing page with detailed documentation and download options.

### Software Packaging:
* Our software will be packaged in accordance with the standard [Python Packaging protocol](https://packaging.python.org/tutorials/packaging-projects/).
* We will not use a framework, as frameworks are generally best used for large-scale projects (e.g. app / web development), but we could use a 
[Python package boilerplate](https://github.com/mtchavez/python-package-boilerplate) instead.

# Implementation
BestorgAD relies on tuples, lists, dictionaries, ndarray, and trees as core basic data structures. BestorgAD custom classes will serve to facilitate the flow of data within the package.

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
      * ```__add__()```
      * ```__subtract__()```
      * ```__multiply__()```
      * ```__divide__()```
      * Other elementary functions TBD (sin, cos, tan, cot, sqrt, exp, etc.)

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
      * ```__add__()```
      * ```__subtract__()```
      * ```__multiply__()```
      * ```__divide__()```
      * Other elementary functions TBD (sin, cos, tan, cot, sqrt, exp, etc.)

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


# Potential Extensions (TBD as project progresses)
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


### Questions for 107 Staff:
* Should outputs of methods within classes be stored as attributes of those classes? (e.g. [elementaryOperation.elemDerive -> self.output_interderivation](#elementaryOperation()))
