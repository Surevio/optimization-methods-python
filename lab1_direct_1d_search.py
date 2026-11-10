# Лабораторная работа №1 - Прямые методы одномерной минимизации
# АА-23-07, Рихельгоф Владимир, №18 по списку, Вариант 1
# Функция: f(x) = x^3 - 3*sin(x), x ∈ [0, 1]
# Методы: 1) Поразрядный поиск (два прохода), 2) Метод парабол, 3) Метод ломанных

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import os
os.makedirs("images", exist_ok=True)  # сюда сохраняются графики

# Целевая функция (задание 1)
def f(x):
    return x**3 - 3*np.sin(x)

a, b = 0, 1   # отрезок

# 2. Встроенные функции scipy
res_brent = optimize.minimize_scalar(f, bounds=(a, b), method='bounded')
x_brent = res_brent.x
print("=== Задание 2: Встроенные функции scipy ===")
print(f"scipy (bounded Brent): x = {x_brent:.10f}, f(x) = {f(x_brent):.10f}")

res_golden = optimize.minimize_scalar(f, bounds=(a, b), method='bounded', options={'xatol': 1e-10})
x_golden = res_golden.x
print(f"scipy (golden/bounded): x = {x_golden:.10f}, f(x) = {f(x_golden):.10f}")

# 3.1.1 Метод поразрядного поиска (два прохода, p - количество интервалов)
def porazryadny_dva_prohoda(f, a, b, eps, p):
    # Метод поразрядного поиска: два прохода.
    # p - количество интервалов разбиения начального отрезка.
    count_f = 0  # счётчик вычислений функции

    # Первый проход
    h = (b - a) / p
    x_vals = [a + i * h for i in range(p + 1)]
    f_vals = [f(x) for x in x_vals]
    count_f += len(x_vals)

    # Найдём точку минимума на первом проходе
    idx_min = np.argmin(f_vals)
    x_min1 = x_vals[idx_min]

    # Сужаем отрезок: [x_min1 - h, x_min1 + h]
    a2 = max(a, x_min1 - h)
    b2 = min(b, x_min1 + h)

    # Второй проход
    h2 = (b2 - a2) / p
    x0 = a2
    f0 = f(x0)
    count_f += 1

    while True:
        x1 = x0 + h2
        if x1 > b2:
            break
        f1 = f(x1)
        count_f += 1
        if f1 < f0:
            x0 = x1
            f0 = f1
        else:
            h2 = h2 / 2
            if h2 < eps:
                break

    approximations = [x_min1, x0]
    return x0, f0, count_f, approximations


# 3.1.2 Метод парабол
def parabolic_method(f, a, b, eps):
    # Метод парабол (квадратичная интерполяция).
    count_f = 0
    x1 = a
    x2 = (a + b) / 2
    x3 = b
    f1, f2, f3 = f(x1), f(x2), f(x3)
    count_f += 3
    approximations = []

    while abs(b - a) > eps:
        denom = (x2 - x1) * (f2 - f3) - (x2 - x3) * (f2 - f1)
        if abs(denom) < 1e-14:
            break
        x_star = x2 - 0.5 * ((x2 - x1)**2 * (f2 - f3) - (x2 - x3)**2 * (f2 - f1)) / denom
        # Ограничим x_star отрезком
        x_star = max(a, min(b, x_star))
        f_star = f(x_star)
        count_f += 1
        approximations.append(x_star)

        # Обновляем тройку точек
        if x_star < x2:
            if f_star < f2:
                x3, f3 = x2, f2
                x2, f2 = x_star, f_star
            else:
                x1, f1 = x_star, f_star
        else:
            if f_star < f2:
                x1, f1 = x2, f2
                x2, f2 = x_star, f_star
            else:
                x3, f3 = x_star, f_star

        a = min(x1, x2, x3)
        b = max(x1, x2, x3)

    return x2, f2, count_f, approximations


# 3.1.3 Метод ломанных (Piyavsky)
def metod_lomanyh(f, df_approx, a, b, eps):
    # Метод ломанных (метод Пиявского).
    # Использует константу Липшица L.
    # Классическая реализация: поддерживаем список интервалов.
    count_f = 0

    # 1. Оценим константу Липшица на равномерной сетке
    N = 50
    xs = np.linspace(a, b, N)
    ys = [f(x) for x in xs]
    count_f += N
    L = 0
    for i in range(N - 1):
        slope = abs(ys[i+1] - ys[i]) / ((b - a) / (N - 1))
        if slope > L:
            L = slope
    L = max(L * 1.5, 1e-6)

    # 2. Инициализация: два конца отрезка
    points = [(a, f(a)), (b, f(b))]
    count_f += 2
    approximations = []

    for _ in range(5000):
        # Для каждого интервала [x_i, x_{i+1}] вычислим значение нижней огибающей в центре
        points.sort(key=lambda p: p[0])
        best_val = float('inf')
        best_x = a

        for i in range(len(points) - 1):
            xi, fi = points[i]
            xj, fj = points[i + 1]
            # Минимум нижней огибающей на этом интервале
            x_mid = (xi + xj) / 2 + (fi - fj) / (2 * L)
            x_mid = np.clip(x_mid, xi, xj)
            lower = (fi + fj) / 2 - L * (xj - xi) / 2
            if lower < best_val:
                best_val = lower
                best_x = x_mid
                best_interval = (xi, fi, xj, fj)

        xi, fi, xj, fj = best_interval
        # Точка разбиения
        x_new = (xi + xj) / 2 + (fi - fj) / (2 * L)
        x_new = np.clip(x_new, xi + 1e-15, xj - 1e-15)
        f_new = f(x_new)
        count_f += 1
        approximations.append(x_new)

        points.append((x_new, f_new))

        # Критерий остановки: длина лучшего интервала мала
        if (xj - xi) < eps:
            break

    # Вернём точку с минимальным значением
    best_pt = min(points, key=lambda p: p[1])
    return best_pt[0], best_pt[1], count_f, approximations


# Тесты методов
eps = 1e-4
print("\n=== Задание 3: Реализованные методы (eps=1e-4) ===")

xp, fp, cp, ap = porazryadny_dva_prohoda(f, a, b, eps, p=10)
print(f"Поразрядный поиск (2 прохода): x = {xp:.8f}, f(x) = {fp:.8f}, вычислений: {cp}")

xpar, fpar, cpar, apar = parabolic_method(f, a, b, eps)
print(f"Метод парабол:                  x = {xpar:.8f}, f(x) = {fpar:.8f}, вычислений: {cpar}")

xlom, flom, clom, alom = metod_lomanyh(f, None, a, b, eps)
print(f"Метод ломанных:                 x = {xlom:.8f}, f(x) = {flom:.8f}, вычислений: {clom}")


# 4. Таблица: зависимость числа вычислений от точности eps = 10^-k, k=2..6
print("\n=== Задание 4: Число вычислений функции при разных eps ===")
print(f"{'eps':<12} {'Поразрядный':>14} {'Парабол':>10} {'Ломанных':>10}")
counts_poisk, counts_par, counts_lom = [], [], []
eps_list = [10**(-k) for k in range(2, 7)]

for eps_k in eps_list:
    _, _, c1, _ = porazryadny_dva_prohoda(f, 0, 1, eps_k, p=10)
    _, _, c2, _ = parabolic_method(f, 0, 1, eps_k)
    _, _, c3, _ = metod_lomanyh(f, None, 0, 1, eps_k)
    counts_poisk.append(c1)
    counts_par.append(c2)
    counts_lom.append(c3)
    print(f"1e-{int(-np.log10(eps_k)):<9} {c1:>14} {c2:>10} {c3:>10}")


# График 1: функция и точки приближений
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
x_plot = np.linspace(a, b, 500)
y_plot = f(x_plot)

titles = ["Поразрядный поиск\n(два прохода)", "Метод парабол", "Метод ломанных"]
results = [(xp, fp, ap), (xpar, fpar, apar), (xlom, flom, alom)]

for i, (ax, title, (xmin, fmin, approx)) in enumerate(zip(axes, titles, results)):
    ax.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x)')
    if approx:
        xapp = approx[-min(5, len(approx)):]
        ax.scatter(xapp, [f(xa) for xa in xapp], color='orange', s=60, zorder=5, label='Приближения')
    ax.scatter([xmin], [fmin], color='red', s=100, zorder=6, marker='*', label=f'Минимум\nx={xmin:.5f}')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('f(x) = x³ - 3sin(x), x ∈ [0, 1]', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('images/lab1_methods.png', dpi=120, bbox_inches='tight')
plt.close()
print("\nГрафик методов сохранён: plot_methods.png")


# График 2: число вычислений от eps
fig2, ax2 = plt.subplots(figsize=(8, 5))
k_vals = list(range(2, 7))
ax2.plot(k_vals, counts_poisk, 'o-', label='Поразрядный поиск', color='blue')
ax2.plot(k_vals, counts_par, 's-', label='Метод парабол', color='green')
ax2.plot(k_vals, counts_lom, '^-', label='Метод ломанных', color='red')
ax2.set_xlabel('k (ε = 10⁻ᵏ)')
ax2.set_ylabel('Число вычислений функции')
ax2.set_title('Зависимость числа вычислений от точности')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/lab1_counts.png', dpi=120, bbox_inches='tight')
plt.close()
print("График числа вычислений сохранён: plot_counts.png")


# 5. Минимум f(x) = e^x - 1 - x - x²/2 - x³/6 на [-5, 5]
def f2(x):
    return np.exp(x) - 1 - x - x**2/2 - x**3/6

print("\n=== Задание 5: f(x) = eˣ - 1 - x - x²/2 - x³/6 на [-5, 5], eps=1e-4 ===")
xp2, fp2, cp2, _ = porazryadny_dva_prohoda(f2, -5, 5, 1e-4, p=20)
xpar2, fpar2, cpar2, _ = parabolic_method(f2, -5, 5, 1e-4)
xlom2, flom2, clom2, _ = metod_lomanyh(f2, None, -5, 5, 1e-4)
x_ref2 = optimize.minimize_scalar(f2, bounds=(-5, 5), method='bounded').x
print(f"scipy.brent:        x = {x_ref2:.8f}, f(x) = {f2(x_ref2):.8f}")
print(f"Поразрядный поиск:  x = {xp2:.8f}, f(x) = {fp2:.8f}, вычислений: {cp2}")
print(f"Метод парабол:      x = {xpar2:.8f}, f(x) = {fpar2:.8f}, вычислений: {cpar2}")
print(f"Метод ломанных:     x = {xlom2:.8f}, f(x) = {flom2:.8f}, вычислений: {clom2}")

# График f2
fig3, ax3 = plt.subplots(figsize=(9, 5))
x3 = np.linspace(-5, 5, 500)
ax3.plot(x3, f2(x3), 'b-', linewidth=2, label='f(x)')
ax3.scatter([xp2], [fp2], color='blue', s=100, zorder=5, label=f'Поразрядный x={xp2:.5f}')
ax3.scatter([xpar2], [fpar2], color='green', s=100, zorder=5, label=f'Парабол x={xpar2:.5f}')
ax3.scatter([xlom2], [flom2], color='red', s=100, zorder=5, label=f'Ломанных x={xlom2:.5f}')
ax3.scatter([x_ref2], [f2(x_ref2)], color='black', s=120, zorder=6, marker='*', label=f'scipy x={x_ref2:.5f}')
ax3.set_title('f(x) = eˣ - 1 - x - x²/2 - x³/6, x ∈ [-5, 5]')
ax3.set_xlabel('x')
ax3.set_ylabel('f(x)')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/lab1_f2.png', dpi=120, bbox_inches='tight')
plt.close()
print("График f2 сохранён: plot_f2.png")


# 6.1 Метод ломанных для f(x) = cos(x)/x², x ∈ [1, 12]
def f6(x):
    return np.cos(x) / x**2

print("\n=== Задание 6.1: f(x) = cos(x)/x², x ∈ [1, 12], метод ломанных ===")
xlom6, flom6, clom6, alom6 = metod_lomanyh(f6, None, 1, 12, 1e-4)
x_ref6 = optimize.minimize_scalar(f6, bounds=(1, 12), method='bounded').x
print(f"scipy.brent:     x = {x_ref6:.8f}, f(x) = {f6(x_ref6):.8f}")
print(f"Метод ломанных:  x = {xlom6:.8f}, f(x) = {flom6:.8f}, вычислений: {clom6}")

fig4, ax4 = plt.subplots(figsize=(9, 5))
x4 = np.linspace(1, 12, 500)
ax4.plot(x4, f6(x4), 'b-', linewidth=2, label='f(x)')
ax4.scatter([xlom6], [flom6], color='red', s=120, zorder=6, marker='*',
            label=f'Ломанных x={xlom6:.5f}')
ax4.set_title('f(x) = cos(x)/x², x ∈ [1, 12] - Метод ломанных')
ax4.set_xlabel('x')
ax4.set_ylabel('f(x)')
ax4.legend()
ax4.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/lab1_f6.png', dpi=120, bbox_inches='tight')
plt.close()
print("График f6 сохранён: plot_f6.png")

print("\n=== Все графики сохранены ===")
