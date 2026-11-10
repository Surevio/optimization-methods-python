# Лабораторная работа №6.3 - Двойственная задача ЗЛП
# АА-23-07, Рихельгоф Владимир, №18, Вариант 1
# Исходная (прямая): min f = -x1-4x2
# x1<=2, x1+2x2>=2, x2<=2, x1+x2<=3, x1,x2>=0
import numpy as np
from scipy.optimize import linprog

# ПРЯМАЯ ЗАДАЧА
# min f = -x1 - 4x2
# Приводим к стандарту: все ограничения вида <= 
# (1) x1 <= 2
# (2) -x1 - 2x2 <= -2   (из x1+2x2 >= 2)
# (3) x2 <= 2
# (4) x1 + x2 <= 3

c_p = np.array([-1., -4.])
A_p = np.array([[ 1.,  0.],
                [-1., -2.],
                [ 0.,  1.],
                [ 1.,  1.]])
b_p = np.array([2., -2., 2., 3.])

res_p = linprog(c_p, A_ub=A_p, b_ub=b_p, bounds=[(0,None),(0,None)], method='highs')
print("=== ПРЯМАЯ ЗАДАЧА (min) ===")
print(f"x* = ({res_p.x[0]:.6f}, {res_p.x[1]:.6f})")
print(f"f* = {res_p.fun:.6f}")

# ФОРМУЛИРОВКА ДВОЙСТВЕННОЙ ЗАДАЧИ
# Прямая: min c^T x,  Ax <= b,  x >= 0
# Двойственная: max b^T y,  A^T y <= c,  y <= 0 (т.к. прямая на min)
# Для задачи min с ограничениями Ax<=b, x>=0:
# Двойственная: max b^T y, A^T y <= c, y <= 0

# A^T y <= c:
# A^T = [[1,-1,0,1],  --> строка для x1: y1 - y2 + y4 <= -1
#         [0,-2,1,1]] --> строка для x2: -2y2 + y3 + y4 <= -4
# b^T y = 2y1 - 2y2 + 2y3 + 3y4
# y1,y2,y3,y4 <= 0

# Перепишем: yi <= 0 -> замена yi = -vi, vi >= 0
# max b^T y = -2v1+2v2-2v3-3v4
# A^T y <= c: 
#   -v1+v2+0-v4 <= -1  -> v1-v2+v4 >= 1
#   2v2-v3-v4 <= -4 -> 2v2-v3-v4+4 <= 0 -> нет
# 
# Стандарт двойственной (через scipy linprog):
# Двойственная задача для МИНИМИЗАЦИИ:
# При прямой: min c^T x, Ax<=b, x>=0
# Двойственная: max b^T y, A^T y <= c, y<=0
# 
# Для scipy: max(-b^T y) = min b^T(-y), пусть z=-y >= 0
# min (-b)^T z = min(-2z1+2z2-2z3-3z4), т.е. коэф = (2,-2,2,3)
# Ограничения A^T y <= c -> -A^T z <= c -> A^T z >= -c
# т.е. -A^T z <= c нет... пересчитаем:
# y <= 0 -> z = -y >= 0. A^T y = -A^T z <= c -> A^T z >= -c.
# В форме ub: -A^T z <= c (уже есть).
# 
# Попробуем напрямую через dual через linprog shadow prices:

res_p2 = linprog(c_p, A_ub=A_p, b_ub=b_p, bounds=[(0,None),(0,None)], method='highs',
                 options={'disp':False})
# Двойственные переменные (теневые цены) - уже в res_p2
dual_vars = res_p2.ineqlin.marginals
print("\n=== ДВОЙСТВЕННЫЕ ПЕРЕМЕННЫЕ (теневые цены) ===")
for i,y in enumerate(dual_vars):
    print(f"  y{i+1} = {y:.6f}")

# Проверка: b^T * y = f* (теорема о дополняющей нежёсткости)
dual_obj = b_p @ dual_vars
print(f"\nb^T * y = {dual_obj:.6f}")
print(f"f* (прямая) = {res_p2.fun:.6f}")
print(f"Совпадают: {abs(dual_obj - res_p2.fun) < 1e-8}")

# ДВОЙСТВЕННАЯ ЗАДАЧА ЯВНО
# Двойственная: max b^T y, A^T y <= c, y <= 0
# Перепишем для scipy (min -b^T y, A^T y <= c, -y <= 0 т.е. y<=0 через bounds)
# A^T = [[1,-1,0,1],[0,-2,1,1]]
At = A_p.T  # (2x4)
c_d = -b_p  # для минимизации: min(-b^T y) = max(b^T y)
# Ограничения: A^T y <= c_p (вектор коэф целевой прямой)
# bounds: y <= 0 -> (None, 0) для каждой переменной

A_d_ub = At  # A^T y <= c_p
b_d_ub = c_p

res_d = linprog(c_d, A_ub=A_d_ub, b_ub=b_d_ub, 
                bounds=[(None,0)]*4, method='highs')
print("\n=== ДВОЙСТВЕННАЯ ЗАДАЧА (max b^T y) ===")
if res_d.status == 0:
    print(f"y* = {res_d.x}")
    print(f"Значение двойственной (max b^T y) = {-res_d.fun:.6f}")
else:
    print("Статус:", res_d.message)
    # Используем теневые цены
    print("Используем теневые цены из прямой задачи:")
    print(f"y* = {dual_vars}")
    print(f"max b^T y = {dual_obj:.6f}")

# ПРОВЕРКА ТЕОРЕМЫ ДВОЙСТВЕННОСТИ
print("\n=== ПРОВЕРКА ТЕОРЕМ ДВОЙСТВЕННОСТИ ===")
x_opt = res_p2.x
y_opt = dual_vars
print(f"1. Теорема о равенстве экстремумов: f(x*) = g(y*)")
print(f"   f(x*) = {c_p @ x_opt:.6f}")
print(f"   g(y*) = b^T y* = {b_p @ y_opt:.6f}")
print(f"   Совпадают: {abs(c_p @ x_opt - b_p @ y_opt) < 1e-6}")

print(f"\n2. Теорема о дополняющей нежёсткости:")
for i in range(4):
    slack = b_p[i] - A_p[i] @ x_opt
    yi = y_opt[i]
    product = slack * yi
    active = abs(slack) < 1e-6
    print(f"   Огр.{i+1}: slack={slack:.4f}, y{i+1}={yi:.4f}, "
          f"slack*y={product:.6f} ≈ 0: {abs(product)<1e-6}  {'(активное)' if active else ''}")

# КАК ПОЛУЧИТЬ РЕШЕНИЕ ПРЯМОЙ ИЗ ДВОЙСТВЕННОЙ
print("\n=== ПЕРЕХОД ОТ ДВОЙСТВЕННОЙ К ПРЯМОЙ ===")
print("Из условий дополняющей нежёсткости:")
print("  Если yi* ≠ 0 -> i-е ограничение прямой активно (ограничение-равенство)")
active_constraints = [i for i in range(4) if abs(y_opt[i]) > 1e-6]
print(f"  Активные ограничения (yi≠0): {[i+1 for i in active_constraints]}")
print("  Решение прямой = пересечение активных ограничений")
if len(active_constraints) >= 2:
    i1, i2 = active_constraints[0], active_constraints[1]
    A_sys = np.array([A_p[i1], A_p[i2]])
    b_sys = np.array([b_p[i1], b_p[i2]])
    try:
        x_from_dual = np.linalg.solve(A_sys, b_sys)
        print(f"  Система: ограничения {i1+1} и {i2+1} как равенства")
        print(f"  x* = {x_from_dual} -> f = {c_p @ x_from_dual:.4f}")
    except:
        print("  Система вырождена, берём из одного активного ограничения")

print("\nВсе вычисления завершены.")
