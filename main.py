# This is a sample Python script.

# Dina = 10 variant
# Maksimov = 14

from math import sin, cos, exp
from lib import Jacobi2, NonLinearEquationsSystem2, BoundaryConditions, finite_difference_scheme, TwoVariablesFunction, OneVariableFunction

EPSILON = 0.00001
# система уравнений для коеффициентов граничных условий
f1: TwoVariablesFunction = lambda p, q: 2 * p**2 + p - q
f2: TwoVariablesFunction = lambda p, q: sin(p * q - 0.5) + q

# вручную посичтанные частные производные для системы коеффициентов граничных условий
df1_dp: TwoVariablesFunction = lambda p, q: 4 * p + 1
df1_dq: TwoVariablesFunction = lambda p, q: -1

df2_dp: TwoVariablesFunction = lambda p, q: q * cos(p*q - 0.5)
df2_dq: TwoVariablesFunction = lambda p, q: p * cos(p*q - 0.5)

W = Jacobi2(
    df1_dp, df1_dq,
    df2_dp, df2_dq
)

# # y'' + P(x)y' + Q(x)y = F(x)
p_x: OneVariableFunction = lambda x: 2/(exp(x) + 1)     # P(x)
q_x: OneVariableFunction = lambda x: exp(x)/(exp(x+1))  # Q(x)
f_x: OneVariableFunction = lambda x: 0


def main():
    non_linear_equations_system = NonLinearEquationsSystem2(f1, f2, W)
    p, q = non_linear_equations_system.solve_newton(0.5, 0.5, EPSILON)
    print(f'p,q={p}, {q}')
    boundary_conditions = BoundaryConditions(
        0, 1,  # Интервал [0, 1]
        1, 0, 1,  # 1 * y'(0) + 0 * y(0) = 1
        q, p, 1  # q * y'(1) + p * y(1) = 1
    )
    for h in [0.1, 0.01]:
        y_values = finite_difference_scheme(p_x, q_x, f_x, boundary_conditions, h)
        print(f'h={h}; Y={list(y_values)}')


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
