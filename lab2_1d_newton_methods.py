# Лабораторная работа №2 - Методы одномерной минимизации с производными
# АА-23-07, Рихельгоф Владимир, №18 по списку, Вариант 1
# Методы: средней точки, Ньютона, Ньютона-Рафсона
# Основная функция лаб1: f(x)=x³-3sin(x), x∈[0,1]
# Функция задания 4: f(x)=x*arctg(x)-0.5*ln(1+x²)
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import os
os.makedirs("images", exist_ok=True)  # сюда сохраняются графики

# Основная функция (вариант 1)
def f(x):   return x**3 - 3*np.sin(x)
def df(x):  return 3*x**2 - 3*np.cos(x)
def d2f(x): return 6*x + 3*np.sin(x)
a, b = 0.0, 1.0

# Функция задания 4
def g(x):   return x*np.arctan(x) - 0.5*np.log(1 + x**2)
def dg(x):  return np.arctan(x)
def d2g(x): return 1.0 / (1 + x**2)

# 2.1.1 Метод средней точки
def midpoint(df, a, b, eps):
    cf, cd = 0, 0
    approx = []
    while (b - a) > eps:
        mid = (a + b) / 2
        val = df(mid); cd += 1
        approx.append(mid)
        if val > 0:
            b = mid
        else:
            a = mid
    x = (a + b) / 2
    return x, cf, cd, approx

# 2.1.2 Метод Ньютона
def newton(f, df, d2f, x0, eps, maxiter=1000):
    cf, cd = 0, 0
    approx = [x0]
    x = x0
    for _ in range(maxiter):
        d1 = df(x);  cd += 1
        d2 = d2f(x); cd += 1
        if abs(d2) < 1e-14:
            break
        x_new = x - d1 / d2
        approx.append(x_new)
        if abs(x_new - x) < eps:
            x = x_new
            break
        x = x_new
    cf += 1  # одно вычисление f для финального значения
    return x, cf, cd, approx

# 2.1.3 Метод Ньютона-Рафсона (с регулировкой шага)
def newton_raphson(f, df, d2f, x0, eps, maxiter=1000):
    cf, cd = 0, 0
    approx = [x0]
    x = x0
    for _ in range(maxiter):
        d1 = df(x);  cd += 1
        d2 = d2f(x); cd += 1
        if abs(d2) < 1e-14:
            break
        x_hat = x - d1 / d2
        d1_hat = df(x_hat); cd += 1
        # Шаг по НР: tau = d1² / (d1² + d1_hat²)
        denom = d1**2 + d1_hat**2
        tau = d1**2 / denom if denom > 1e-14 else 1.0
        x_new = x - tau * d1 / d2
        approx.append(x_new)
        if abs(x_new - x) < eps:
            x = x_new
            break
        x = x_new
    cf += 1
    return x, cf, cd, approx

# Запуск на основной функции
eps = 1e-4
x0 = 0.5  # начальное приближение
xref = optimize.minimize_scalar(f, bounds=(a,b), method='bounded').x
print(f"scipy: x*={xref:.8f}, f={f(xref):.8f}")

xm, cfm, cdm, apm = midpoint(df, a, b, eps)
xn, cfn, cdn, apn = newton(f, df, d2f, x0, eps)
xr, cfr, cdr, apr = newton_raphson(f, df, d2f, x0, eps)
print(f"Средняя точка:   x={xm:.8f}, f={f(xm):.8f}, cf={cfm}, cd={cdm}")
print(f"Ньютон:          x={xn:.8f}, f={f(xn):.8f}, cf={cfn}, cd={cdn}")
print(f"Ньютон-Рафсон:   x={xr:.8f}, f={f(xr):.8f}, cf={cfr}, cd={cdr}")

# Таблица по eps
print("\neps   | Ср.точка (cd) | Ньютон (cd) | Нью-Рафс (cd)")
data = {}
for k in range(2, 7):
    ek = 10**(-k)
    _, _, c1, _ = midpoint(df, a, b, ek)
    _, _, c2, _ = newton(f, df, d2f, x0, ek)
    _, _, c3, _ = newton_raphson(f, df, d2f, x0, ek)
    data[k] = (c1, c2, c3)
    print(f"1e-{k}  | {c1:>13} | {c2:>11} | {c3:>13}")

# Графики
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
xp = np.linspace(a, b, 400)
yp = f(xp)
methods = [("Средняя точка", xm, apm), ("Ньютон", xn, apn), ("Ньютон-Рафсон", xr, apr)]
for ax, (title, xmin, app) in zip(axes, methods):
    ax.plot(xp, yp, 'b-', lw=2)
    shown = app[-min(6, len(app)):]
    ax.scatter(shown, [f(x) for x in shown], color='orange', s=60, zorder=5, label='Приближения')
    ax.scatter([xmin], [f(xmin)], color='red', s=120, marker='*', zorder=6, label=f'x*={xmin:.5f}')
    ax.set_title(title); ax.set_xlabel('x'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.suptitle('f(x) = x³ - 3sin(x), x ∈ [0,1]', fontweight='bold')
plt.tight_layout()
plt.savefig('images/lab2_methods.png', dpi=120, bbox_inches='tight')
plt.close()

# График зависимости числа производных от eps
fig2, ax2 = plt.subplots(figsize=(8, 5))
ks = list(range(2, 7))
ax2.plot(ks, [data[k][0] for k in ks], 'o-', label='Средняя точка')
ax2.plot(ks, [data[k][1] for k in ks], 's-', label='Ньютон')
ax2.plot(ks, [data[k][2] for k in ks], '^-', label='Ньютон-Рафсон')
ax2.set_xlabel('k (ε=10⁻ᵏ)'); ax2.set_ylabel('Число вычислений производных')
ax2.set_title('Зависимость числа вычислений от точности'); ax2.legend(); ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('images/lab2_counts.png', dpi=120, bbox_inches='tight')
plt.close()

# Задание 4: f(x)=x*arctg(x)-0.5*ln(1+x²)
print("\n=== Задание 4: g(x)=x*arctg(x)-0.5*ln(1+x²) ===")
# Производная: g'(x) = arctg(x), обнуляется при x=0 -> минимум x*=0
# g''(x) = 1/(1+x²) > 0 -> это минимум
# Диапазон сходимости Ньютона: вся ось (g'' > 0 везде)
gref = optimize.minimize_scalar(g, bounds=(-5, 5), method='bounded').x
print(f"scipy: x*={gref:.8f}, g={g(gref):.10f}")

# Для Ньютона: анализ диапазона сходимости
print("Проверка сходимости Ньютона с разных начальных приближений:")
converged = []
for x0_test in np.linspace(-10, 10, 41):
    xconv, _, _, _ = newton(g, dg, d2g, x0_test, 1e-8, maxiter=200)
    ok = abs(xconv) < 1e-4
    if x0_test in [-10, -5, -1, 0, 1, 5, 10]:
        print(f"  x0={x0_test:6.1f} -> x*={xconv:.6f}, {'OK' if ok else 'FAIL'}")
    converged.append((x0_test, ok))

xgm, cfgm, cdgm, apgm = midpoint(dg, -5, 5, 1e-4)
xgn, cfgn, cdgn, apgn = newton(g, dg, d2g, 1.0, 1e-4)
xgr, cfgr, cdgr, apgr = newton_raphson(g, dg, d2g, 1.0, 1e-4)
print(f"Средняя точка:  x={xgm:.8f}, g={g(xgm):.8f}")
print(f"Ньютон:         x={xgn:.8f}, g={g(xgn):.8f}")
print(f"Ньютон-Рафсон:  x={xgr:.8f}, g={g(xgr):.8f}")

# Графики задания 4
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
xp3 = np.linspace(-5, 5, 400)
yp3 = g(xp3)
methods3 = [("Средняя точка", xgm, apgm), ("Ньютон", xgn, apgn), ("Ньютон-Рафсон", xgr, apgr)]
for ax, (title, xmin, app) in zip(axes3, methods3):
    ax.plot(xp3, yp3, 'b-', lw=2)
    shown = app[-min(6, len(app)):]
    ax.scatter(shown, [g(x) for x in shown], color='orange', s=60, zorder=5, label='Приближения')
    ax.scatter([xmin], [g(xmin)], color='red', s=120, marker='*', zorder=6, label=f'x*={xmin:.4f}')
    ax.set_title(title); ax.set_xlabel('x'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.suptitle('g(x) = x·arctg(x) - ½ln(1+x²)', fontweight='bold')
plt.tight_layout()
plt.savefig('images/lab2_task4.png', dpi=120, bbox_inches='tight')
plt.close()

print("\nВсе графики сохранены.")
