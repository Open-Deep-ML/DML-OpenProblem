# OpenProblem

Welcome to the **OpenProblem** repository! This repository contains a collection of coding challenges designed to help you learn and implement various machine learning and deep learning algorithms from scratch. Each problem includes a detailed description, example inputs and outputs, a learning section, starter code, and test cases.


## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Example Problem](#example-problem)
  - [Matrix times Vector (easy)](#matrix-times-vector-easy)

## Introduction

This repository is designed to provide hands-on practice with implementing fundamental algorithms in machine learning and deep learning. By working through these problems, you'll gain a deeper understanding of how these algorithms work and how to implement them from scratch.

## Getting Started

To get started with the problem set, clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/openproblem.git
```
Each problem is contained within a Python dictionary in the problems.py file. You can work on each problem individually by following the provided instructions.

join our [Discord community](https://discord.gg/s4uVTQwk).

## Example Problem

### Matrix times Vector (easy)

**Description**: Write a Python function that takes the dot product of a matrix and a vector. Return -1 if the matrix could not be dotted with the vector.

**Example**:
```python
input: a = [[1,2],[2,4]], b = [1,2]
output: [5, 10]
reasoning: 1*1 + 2*2 = 5; 1*2 + 2*4 = 10
```
### Learn
```html
<h2>Transpose of a Matrix</h2>
Consider a matrix \(M\) and its transpose \(M^T\), where:

Original Matrix \(M\):
\[
M = \begin{pmatrix} 
a & b & c \\ 
d & e & f 
\end{pmatrix}
\]

Transposed Matrix \(M^T\):
\[
M^T = \begin{pmatrix} 
a & d \\ 
b & e \\ 
c & f 
\end{pmatrix}
\]

Transposing a matrix involves converting its rows into columns and vice versa. This operation is fundamental in linear algebra for various computations and transformations.
```


### Starter Code

```python
def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
    return c
```

### Solution
```python
def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
    if len(a[0]) != len(b):
        return -1
    vals = []
    for i in a:
        hold = 0
        for j in range(len(i)):
            hold += (i[j] * b[j])
        vals.append(hold)
    return vals
```


### Test Cases

```pyhton
print(matrix_dot_vector([[1,2,3],[2,4,5],[6,8,9]], [1,2,3])) # Expected output: [14, 25, 49]
print(matrix_dot_vector([[1,2],[2,4],[6,8],[12,4]], [1,2,3])) # Expected output: -1
```


### Contribution:

Contributor: User's Name

profile_url: https://profile.url

On [deep-ml.com](https://www.deep-ml.com/), you can view the contributors for each problem they worked on, along with a link to their chosen profile.
