# Introduction to symbolic computation with  MXNet and Julia programming language

MXNet is a framework for machine learning. The popularity of MXNet continues to increase. MXNet provides an eager execution mode and a static graph mode. We explore  a creation of symbolic graph with primitive functions. Besides, Julia uses just in time compiling that make Julia faster than Python. 

## Installation
### installing Julia
We need to download the installer [here](https://julialang.org/downloads/). We need also to add Julia on a path to make it accessible from the shell, the powershell, or the cmd. The executable is located on `c:\users\user\.julia\bin` for Windows. We need to replace `user` by our user's name. We can also download a script [here](https://github.com/abelsiqueira/jill) to install Julia easily on Linux.


### Installing MXNet 
It is very easy to install MXNet. We just import the module `Pkg` and install MXNet.
```julia
import Pkg
Pkg.add("MXNet")
```
### Install Julia Extension on Visual Studio Code
We search the extension of Julia on Visual Studio Code with its id. The id of Julia extension is given by "[julialang.language-julia](https://marketplace.visualstudio.com/items?itemName=julialang.language-julia)". We can download the extension directly from a [market place](https://marketplace.visualstudio.com/items?itemName=julialang.language-julia) as second alternative.

## Basic Operations 
The fast way of mastering anything is to begin with easy examples. We can increase the challenge little by little in each loop of learning. We will view the way of performing graph computation with MXNet.

### Addition
The first step is to import MXNet in current namespace. You can access to objects inside MXNet without each time specifying the name of module.
```julia
using MXNet
```
The second step  is to create two symbolic variables `x` and `y`
```julia
x=mx.Variable(:x)
y=mx.Variable(:y)
```
`:x` is Julia symbol. The variable `x` will be identified by this symbol. The symbol in Julia is like some kind of constant string. The function `mx.Variable` creates symbolic variable. 

The third step is to add the two variables by creating new symbolic node:

```julia 
z= x .+ y
```
The operation `.+` is element-wise addition. 

The fourth step is to create dictionary that will link numerical values to symbolic variables.

```julia
args=Dict(:x => mx.NDArray([4]),
          :y => mx.NDArray([5]))
```
We give a value of four to variable $x$, and a value of five to variable $y$. All operations are tensor oriented. It requires to embed even scalar value in array. `mx.NDArray([4])` creates an array that has only one element. Similarly, `mx.NDArray([5])` creates an array containing only one element.

The fifth step is to create an executor by providing a symbolic expression. The executor will be able to evaluate the symbolic expression with provided value in dictionary. The following line creates an executor:
```julia
executor=mx.bind(z,mx.cpu(),args)
```
The first argument is our symbolic expression that we want to evaluate. The second argument is device that will execute the instructions. In this case, the CPU will execute the instructions. The third argument is the dictionary that links input symbolic nodes to their numerical values.

The sixth step is to evaluate the expression by calling the method `forward` like this:

```julia
outs=mx.forward(executor,is_train=false)
```
The first argument is the executor and the second argument indicates that we make inference only.

The finally step is to print evaluated value on screen. This step is unnecessary if you executing instruction by instruction from Julia command line. However, it is good habit to put all instructions in file because we can come back later. We build also your own library of snippets. Snippets help to accelerate coding on big project because we will simply aggregate the right snippets.
```julia
println(outs)
```
We can verify that the result is nine.

### Multiplication
 The operator `.*`is element wise multiplication. We can use the precedent example and replace the addition operator by multiplication operator.

```julia
using MXNet
x=mx.Variable(:x)
y=mx.Variable(:y)
z=x .* y
args=Dict(:x => mx.NDArray([4]),
          :y => mx.NDArray([5]))
executor=mx.bind(z,mx.cpu(),args)
outs=mx.forward(executor,is_train=false)
println(outs)
```
We can verify the result is twenty.

### Power
As example, we have to compute $y=x^3$. The operation will translate as `y=x .^ 3` in Julia programming language. The operator  `.^` is element wise power by scalar value.

```julia
using MXNet
x=mx.Variable(:x)
y=x .^ 3
args=Dict(:x => mx.NDArray([2]))
executor=mx.bind(y,mx.cpu(),args)
outs=mx.forward(executor,is_train=false)
println(outs)
```
We can verify that the result is eight.

### Gradient
Automatic differentiation is at heart of modern machine learning. A gradient is vector of derivatives. Each component of a gradient represents the derivative of scalar function with respect to one independent variable. We obtain the derivative by applying the chain rule. As example, we make an assumption that we have function $f[g(h(x))]$ of others functions. By applying chain rule, we obtain the derivative of $f$ with respect to x as given:

$\frac{\partial f}{\partial x}=\frac{\partial f}{\partial g} \frac{\partial g}{\partial h} \frac{\partial h}{\partial x}$

The derivative of $f$ with respect to x will be calculated in four steps. We calculate the derivative of $f$ with respect to $g$, the derivative of $g$ with respect to $h$, the derivative of $h$ with respect to $x$. Finally, we will make the product of three terms for obtaining the derivative of $f$ with respect to $x$.  The chain rule brings modularity on calculus of derivatives. We begin by the top function and we finish the bottom function. The chain rule brings also the concept of backward computation. A backward computation is evaluating gradient from output to inputs. The backward method will apply on graph detained by executor. We obtain at end of the gradient with respect to inputs variable. As example, we compute the gradient of $y=2x$ at $x=3$. We define the symbolic expression as:

```julia
using MXNet
x=mx.Variable(:x)
y=2 .* x
```
We link the symbol `:x` to the numerical value 3.
```julia
args=Dict(:x => mx.NDArray([3]))
```
This time, we need to define a dictionary indicating the independent variable. The only independent variable is x. We define a dictionary by following code:

```julia
args_grad=Dict(:x => mx.NDArray([0]))
```
The array requires no special initialization. Therefore, the array contains only one zero. The array will receive the derivative of y with respect to x. We bind the symbolic variable to its numerical values by calling an appropriate method. As previously stated, the function returns an executor.

```julia
executor=mx.bind(y,mx.cpu(),args;args_grad=args_grad)
```
Julia programming language defines a type of args_grad as named argument. The three first argument are imperatives. A semi-colon makes the separation between mandatory arguments and named arguments. Each argument after the semi-colon will generate an error without the name of argument. Even if we compute gradient, it requires to computing the output first. We have to call `forward` function. This time, we will set `is_train=true` to keep record of intermediary values before performing the backward evaluation of gradient.
```julia
outs=mx.forward(executor,is_train=true)
```
Then we call `backward` method to compute gradient. We give to backward method the executor and the derivative of output with respect to itself as arguments. The derivative of output with respect to itself is obviously a matrix that all elements are equal to one.
```julia
mx.backward(executor,mx.ones_like(outs[1]))
```
The backward function computes an element wise product of gradient and second argument of backward function. The second argument has the same shape as the output of function `forward`. The function `mx.ones_like` creates an array containing one with the same shape as its argument. 


The function backward attaches an array of gradients to executor. Because we provided only one independent variable, the array of gradients contains only one element. We can verify on screen the value of array of gradients:

```julia
println(executor.grad_arrays)
```

## Finding a Minimum of Scalar Quadratic Function
A gradient descent is the method of choice in modern machine learning for minimizing a function. We compute a minimum by successively applying a gradient descent. The gradient descent is given by:

$x_{k+1} = x_k- \lambda \left(\frac{\partial f}{\partial x}\right)_{x=x_k}$

We call $\lambda$ learning rate. $\lambda$ is superior to zero.We can verify that gradient descent find a minimum of a function. For a small variation, $f(x_{k+1})$ can be written as 

$f(x_{k+1})= f(x_k)+\left(\frac{\partial f}{\partial x}\right)_{x=x_k}^T (x_{k+1}-x_{k})$

We can rewrite a gradient descent like this:

$x_{k+1} - x_k= \lambda \left(\frac{\partial f}{\partial x}\right)_{x=x_k}$

By substituting $(x_{k+1}-x_k)$ in right side of $f(x_{k+1})$, we obtain 

$f(x_{k+1})= f(x_k)- \lambda \left(\frac{\partial f}{\partial x}\right)_{x=x_k}^T \left(\frac{\partial f}{\partial x}\right)_{x=x_k}$

The second term of right side is positive or equal to zero.

$\left(\frac{\partial f}{\partial x}\right)_{x=x_k}^T \left(\frac{\partial f}{\partial x}\right)_{x=x_k} =\left\|\left(\frac{\partial f}{\partial x}\right)_{x=x_k}\right\|_2^2\geq 0$

As result, the difference between $f(x_{k+1})$ and $f(x_{k})$ will be inferior or equal to zero.

$- \lambda \left(\frac{\partial f}{\partial x}\right)_{x=x_k}^T \left(\frac{\partial f}{\partial x}\right)_{x=x_k}=f(x_{k+1})- f(x_k)\leq 0$

$f(x_{k+1})$ will be inferior or equal to $f(x_k)$. 

$f(x_{k+1}) \leq f(x_k)$


The previous statement is true only if $\left\|x_{k+1}-x_{k}\right\|_2$ stays very small. Therefore, we must choose wisely the value of $\lambda$. At the minimum, 

$\left(\frac{\partial f}{\partial x}\right)_{x=x_m}=0$
 
$x_m$ will be the value that minimize $f(x)$. $f(x_m)$ is the minimum of the function $f(x)$. Actually, the minimum can be inaccessible. We have to find the stop criteria.

The first step, we define the computation graph.

```julia
using MXNet

x=mx.Variable(:x)

y=x .^ 2 + 2 .* x .+ 1
```

The second step is to create the executor. We define the dictionaries that link symbolic nodes, numerical values,and independent variables. We have only one independent variable that we bind its numerical to computational graph.

```julia
args=Dict(:x => mx.zeros(1))

args_grad=Dict(:x => mx.zeros(1))

executor=mx.bind(y,mx.cpu(),args;args_grad=args_grad)
```
The function `mx.zeros` creates an array containing all elements equal to zero.

The third step is  a loop in which we perform forward computation, backward computation, and to update the value of x. We begin by the forward computation:

```julia
outs=mx.forward(executor,is_train=true)
```
We make backward computation by passing also matrix with the same shape as the last output but all elements equal to one:

```julia
mx.backward(executor,mx.ones_like(outs[1]))
```
We can now collect the value of gradient of output with respect to x.
```julia
x_grad=executor.grad_arrays[1]
```
We use a special method that subtract the numerical value bound to variable `:x`. This method does not create a new array. The method simply modifies the existing array. The method performs the subtraction in existing array.

```julia
mx.sub_from!(executor.arg_dict[:x], 0.1 .*  x_grad)
```
We can print the new value of x.
```julia
println("x=",executor.arg_dict[:x])
```
We put all together in the loop. the code look like that:
```julia
for i=1:100
    outs=mx.forward(executor,is_train=true)
    mx.backward(executor,mx.ones_like(outs[1]))
    x_grad=executor.grad_arrays[1]
    mx.sub_from!(executor.arg_dict[:x], 0.1 .*  x_grad)
    println("x=",executor.arg_dict[:x])
end
```
The final line of screen looks like that:

```bash
x=NDArray(Float32[-0.9999999])
```

