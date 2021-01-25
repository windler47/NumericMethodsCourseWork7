from typing import Callable

import numpy
from numpy import array as n_array
from numpy.linalg import solve as lg_solve
from collections import namedtuple

TwoVariablesFunction = Callable[[float, float], float]
OneVariableFunction = Callable[[float], float]
ThreeDiagonalCoefficients = namedtuple('ThreeDiagonalCoefficients', ('a', 'b', 'c', 'd', 'P', 'Q'))


class Jacobi2(object):
    def __init__(self,
                 df1_dx1: TwoVariablesFunction, df1_dx2: TwoVariablesFunction,
                 df2_dx1: TwoVariablesFunction, df2_dx2: TwoVariablesFunction
                 ):
        self.df1_dx1 = df1_dx1
        self.df1_dx2 = df1_dx2
        self.df2_dx1 = df2_dx1
        self.df2_dx2 = df2_dx2

    def values_for(self, x1: float, x2: float):
        return n_array([
            [self.df1_dx1(x1, x2), self.df1_dx2(x1, x2)],
            [self.df2_dx1(x1, x2), self.df2_dx2(x2, x2)]
        ])


class NonLinearEquationsSystem2(object):
    def __init__(self, f1: TwoVariablesFunction, f2: TwoVariablesFunction, jacobi_matrix: Jacobi2):
        self.f1 = f1
        self.f2 = f2
        self.W = jacobi_matrix

    def solve_newton(self, p: float, q: float, eps: float = 0.001):
        delta_p, delta_q = 10, 10
        iteration_number = 0
        tp, tq = p, q
        while abs(delta_p) > eps or abs(delta_q) > eps:
            jacobi_values = self.W.values_for(tp, tq)
            function_values = n_array([-self.f1(tp, tq), -self.f2(tp, tq)])
            delta_p, delta_q = lg_solve(jacobi_values, function_values)
            tp += delta_p
            tq += delta_q
            iteration_number += 1
            print(f'Iteration number {iteration_number}: p={tp}; q={tq}; delta_p={delta_p}; delta_q={delta_q}')
            if iteration_number > 100:
                break
        return tp, tq


class BoundaryConditions(object):
    def __init__(self,
                 left_boundary, right_boundary,
                 left_dy, left_y, left_value,
                 right_dy, right_y, right_value,
                 ):
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary

        self.left_dy = left_dy
        self.left_y = left_y
        self.left_value = left_value

        self.right_dy = right_dy
        self.right_y = right_y
        self.right_value = right_value


def thomas_p(a, b, c, prev_p):
    return -c/(b + a*prev_p)


def thomas_q(a, b, d, prev_p, prev_q):
    return (d - a*prev_q)/(b + a*prev_p)


def finite_difference_scheme(p: OneVariableFunction, q: OneVariableFunction, f: OneVariableFunction,
                             boundary_conditions: BoundaryConditions, h: float):
    three_diagonal_matrix = []
    # Left border
    a = 0
    b = boundary_conditions.left_y * h - boundary_conditions.left_dy
    c = boundary_conditions.left_dy
    d = boundary_conditions.left_value * h
    t_p = -c/b
    t_q = d/b
    three_diagonal_matrix.append(ThreeDiagonalCoefficients(a, b, c, d, t_p, t_q))

    h_sq = h**2
    h_hf = h/2
    x = boundary_conditions.left_boundary
    last_step_value = boundary_conditions.right_boundary - 3*h/2
    print(f'x:{x}; P:{t_p}; Q:{t_q}; a:{a}; b:{b}; c:{c}; d:{d}')
    # Iteration
    while x < last_step_value:
        x += h
        a = 1 - p(x)*h_hf
        b = q(x)*h_sq - 2
        c = 1 + p(x)*h_hf
        d = f(x)*h_sq

        t_q = thomas_q(a, b, d, t_p, t_q)
        t_p = thomas_p(a, b, c, t_p)
        print(f'x:{x}; P:{t_p}; Q:{t_q}; a:{a}; b:{b}; c:{c}; d:{d}')
        three_diagonal_matrix.append(ThreeDiagonalCoefficients(a, b, c, d, t_p, t_q))

    # Right border
    a = -boundary_conditions.right_dy
    b = boundary_conditions.right_y * h + boundary_conditions.right_dy
    c = 0
    d = boundary_conditions.right_value * h
    t_q = thomas_q(a, b, d, t_p, t_q)
    t_p = 0
    # print(f'x:{boundary_conditions.right_boundary}; P:{t_p}; Q:{t_q}; a:{a}; b:{b}; c:{c}; d:{d}')
    three_diagonal_matrix.append(ThreeDiagonalCoefficients(a, b, c, d, t_p, t_q))

    # Reverse
    y = []
    y.append((a * three_diagonal_matrix[-1].Q - d) / (-b - a * three_diagonal_matrix[-1].P))
    for i in range(1, len(three_diagonal_matrix)):
        m_element = three_diagonal_matrix[-i]
        y_val = m_element.P * y[-1] + m_element.Q
        y.append(y_val)
    return reversed([y])