# Лабораторная работа №7. Численные методы решения задач математического программирования
# Рихельгоф Владимир, АА-23-07, вариант 3
# f(x1,x2,x3) = x1*x2 + x2*x3 -> max
# ограничения: x1 + x2 = 4, x2 + x3 = 4
# ответ аналитически (метод Лагранжа): x* = (2, 2, 2), f* = 8

import numpy as np
from numpy.linalg import inv
from scipy.optimize import minimize


def f(x):
    return x[0] * x[1] + x[1] * x[2]


# ищем минимум -f, это то же самое что максимум f
def F(x):
    return -f(x)


def gradF(x):
    return -np.array([x[1], x[0] + x[2], x[1]])


def g1(x):
    return x[0] + x[1] - 4


def g2(x):
    return x[1] + x[2] - 4


# 1. scipy, метод SLSQP
constraints = [{"type": "eq", "fun": g1},
               {"type": "eq", "fun": g2}]
res = minimize(F, [1, 2, 1], method="SLSQP", constraints=constraints)
print("SLSQP:", np.round(res.x, 4), "f =", round(f(res.x), 4), "nfev =", res.nfev)


# 2. метод барьерных функций
# равенство g(x) = 0 заменяем на -eps <= g(x) <= eps и уменьшаем eps и r
def barrier_method(x0, eps0=0.5, r0=1.0, r_factor=0.5, eps_factor=0.5, maxiter=26):
    x = np.array(x0, dtype=float)
    r, eps = r0, eps0
    nfev = 0
    for k in range(maxiter):
        def Fb(x):
            a = [eps - g1(x), eps + g1(x), eps - g2(x), eps + g2(x)]
            if min(a) <= 0:
                return 1e10  # вышли за барьер
            return F(x) - r * sum(np.log(a))
        rr = minimize(Fb, x, method="Nelder-Mead")
        x = rr.x
        nfev += rr.nfev
        viol = max(abs(g1(x)), abs(g2(x)))
        print("  итерация %2d: r=%.4f eps=%.4f x=%s f=%.4f нарушение=%.2e"
              % (k + 1, r, eps, np.round(x, 4), f(x), viol))
        r *= r_factor
        eps *= eps_factor
    return x, nfev


print("\nМетод барьерных функций:")
xb, nb = barrier_method([2.1, 2.1, 2.1])
print("Барьерные функции:", np.round(xb, 4), "f =", round(f(xb), 4), "nfev =", nb)


# 3. метод проекции градиента
A = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
b = np.array([4.0, 4.0])
P = np.eye(3) - A.T @ inv(A @ A.T) @ A   # проектор направления


def project(v):
    return v + A.T @ inv(A @ A.T) @ (b - A @ v)


# шаг подбираем дихотомией: делим пополам, пока F не уменьшится
def line_search(x, d, alpha=1.0):
    nf = 0
    while alpha > 1e-10:
        nf += 1
        if F(project(x + alpha * d)) < F(x):
            return alpha, nf
        alpha /= 2
    return 0.0, nf


def gradient_projection(x0, eps=1e-6, maxiter=1000):
    x = project(np.array(x0, dtype=float))
    nfev = 0
    for k in range(maxiter):
        d = -P @ gradF(x)          # проекция антиградиента
        if np.linalg.norm(d) < eps:
            return x, k, nfev
        alpha, nf = line_search(x, d)
        nfev += nf
        x = project(x + alpha * d)
    return x, maxiter, nfev


xp, kp, npf = gradient_projection([1, 2, 1])
print("\nПроекция градиента:", np.round(xp, 4), "f =", round(f(xp), 4), "итераций =", kp, "nfev =", npf)
