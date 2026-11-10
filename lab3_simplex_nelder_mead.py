# Лабораторная работа №3 - Прямые методы многомерной минимизации
# АА-23-07, Рихельгоф Владимир, №18, Вариант 1
# f(x) = 64x1²+126x1x2+64x2²-10x1+30x2+13
# Методы: регулярный симплекс, Нелдер-Мид
# Задание 5: функция Розенброка, x0=(-1,1), ε=1e-3
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import os
os.makedirs("images", exist_ok=True)  # сюда сохраняются графики

# Целевая функция (вариант 1)
def f(x):
    x1, x2 = x[0], x[1]
    return 64*x1**2 + 126*x1*x2 + 64*x2**2 - 10*x1 + 30*x2 + 13

# Аналитическое решение: grad=0 -> 128x1+126x2=10, 126x1+128x2=-30
A = np.array([[128., 126.],[126., 128.]])
b = np.array([10., -30.])
x_exact = np.linalg.solve(A, b)
print(f"Аналитическое решение: x*={x_exact}, f={f(x_exact):.8f}")

# scipy для раздела 2
res_scipy = optimize.minimize(f, [0, 0], method='Nelder-Mead', options={'xatol':1e-10,'fatol':1e-10,'maxiter':10000})
print(f"scipy Nelder-Mead: x*={res_scipy.x}, f={res_scipy.fun:.8f}, nfev={res_scipy.nfev}")

# Метод регулярного симплекса
def regular_simplex(f, x0, step, eps, maxiter=5000):
    # Регулярный симплекс: начальный симплекс строится как правильный
    # (все рёбра равной длины step). Операции: отражение, растяжение, сжатие.
    # Коэффициенты: alpha=1, gamma=2, beta=0.5.
    n = len(x0)
    nfev = 0
    # Правильный симплекс по формуле Спендли
    p = step * (np.sqrt(n+1) - 1 + n) / (n * np.sqrt(2))
    q = step * (np.sqrt(n+1) - 1)     / (n * np.sqrt(2))
    vertices = []
    for i in range(n+1):
        v = np.array(x0, dtype=float)
        for j in range(n):
            if j == i-1:
                v[j] += p
            else:
                v[j] += q
        vertices.append(v)

    fvals = [f(v) for v in vertices]
    nfev += len(vertices)
    history = [vertices[np.argmin(fvals)].copy()]

    alpha, gamma, beta = 1.0, 2.0, 0.5

    for iteration in range(maxiter):
        order = np.argsort(fvals)
        vertices = [vertices[i] for i in order]
        fvals    = [fvals[i]    for i in order]

        history.append(vertices[0].copy())
        if np.std(fvals) < eps:
            break

        centroid = np.mean(vertices[:-1], axis=0)
        worst = vertices[-1]

        xr = centroid + alpha*(centroid - worst)
        fr = f(xr); nfev += 1

        if fr < fvals[0]:
            xe = centroid + gamma*(xr - centroid)
            fe = f(xe); nfev += 1
            vertices[-1], fvals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fvals[-2]:
            vertices[-1], fvals[-1] = xr, fr
        else:
            if fr < fvals[-1]:
                xc = centroid + beta*(xr - centroid)
            else:
                xc = centroid + beta*(worst - centroid)
            fc = f(xc); nfev += 1
            if fc < min(fr, fvals[-1]):
                vertices[-1], fvals[-1] = xc, fc
            else:
                # Редукция
                best = vertices[0]
                for i in range(1, len(vertices)):
                    vertices[i] = best + beta*(vertices[i] - best)
                    fvals[i] = f(vertices[i]); nfev += 1

    best_idx = np.argmin(fvals)
    return vertices[best_idx], fvals[best_idx], nfev, iteration+1, history

# Метод Нелдера-Мида
def nelder_mead(f, x0, step, eps, maxiter=5000):
    # Нелдер-Мид: начальный симплекс строится смещением по каждой оси.
    # Адаптивные коэффициенты по Gao & Han (2012): alpha=1, gamma=1+2/n, beta=0.75-1/(2n), delta=1-1/n.
    # Ключевое отличие от регулярного: коэффициенты и форма симплекса адаптируются к n.
    n = len(x0)
    nfev = 0
    vertices = [np.array(x0, dtype=float)]
    for i in range(n):
        v = np.array(x0, dtype=float)
        v[i] += step
        vertices.append(v)

    fvals = [f(v) for v in vertices]
    nfev += len(vertices)
    history = [vertices[np.argmin(fvals)].copy()]

    alpha = 1.0
    gamma = 1.0 + 2.0/n
    beta  = 0.75 - 1.0/(2*n)
    delta = 1.0 - 1.0/n

    for iteration in range(maxiter):
        order = np.argsort(fvals)
        vertices = [vertices[i] for i in order]
        fvals    = [fvals[i]    for i in order]

        history.append(vertices[0].copy())
        if np.std(fvals) < eps:
            break

        centroid = np.mean(vertices[:-1], axis=0)
        worst = vertices[-1]

        xr = centroid + alpha*(centroid - worst)
        fr = f(xr); nfev += 1

        if fr < fvals[0]:
            xe = centroid + gamma*(xr - centroid)
            fe = f(xe); nfev += 1
            vertices[-1], fvals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fvals[-2]:
            vertices[-1], fvals[-1] = xr, fr
        else:
            if fr < fvals[-1]:
                xc = centroid + beta*(xr - centroid)
                fc = f(xc); nfev += 1
                vertices[-1], fvals[-1] = (xc, fc) if fc <= fr else (xr, fr)
            else:
                xc = centroid + beta*(worst - centroid)
                fc = f(xc); nfev += 1
                if fc < fvals[-1]:
                    vertices[-1], fvals[-1] = xc, fc
                else:
                    best = vertices[0]
                    for i in range(1, len(vertices)):
                        vertices[i] = best + delta*(vertices[i] - best)
                        fvals[i] = f(vertices[i]); nfev += 1

    best_idx = np.argmin(fvals)
    return vertices[best_idx], fvals[best_idx], nfev, iteration+1, history

# Запуск
x0 = np.array([0.0, 0.0])
eps = 1e-4

xs, fs, nfs, its, hs = regular_simplex(f, x0, step=1.5, eps=eps)
xn, fn, nfn, itn, hn = nelder_mead(f, x0, step=1.5, eps=eps)

print(f"\n=== Основная функция, x0=(0,0), eps={eps} ===")
print(f"Регулярный симплекс: x*=({xs[0]:.6f}, {xs[1]:.6f}), f={fs:.8f}, iter={its}, nfev={nfs}")
print(f"Нелдер-Мид:          x*=({xn[0]:.6f}, {xn[1]:.6f}), f={fn:.8f}, iter={itn}, nfev={nfn}")

# Таблица
print("\nk  | Симпл.: iter / nfev | НМ: iter / nfev")
table_data = {}
for k in range(2, 6):
    ek = 10**(-k)
    _, _, nf1, it1, _ = regular_simplex(f, x0, 1.5, ek)
    _, _, nf2, it2, _ = nelder_mead(f, x0, 1.5, ek)
    table_data[k] = (it1, nf1, it2, nf2)
    print(f"{k}  | {it1:>5} / {nf1:<6} | {it2:>5} / {nf2:<6}")

# Построение графиков
def plot_path(f, history, title, filename, x_sol=None):
    hx = np.array([h[0] for h in history])
    hy = np.array([h[1] for h in history])
    cx = x_sol[0] if x_sol is not None else hx[-1]
    cy = x_sol[1] if x_sol is not None else hy[-1]
    r  = max(np.ptp(hx), np.ptp(hy), 2.0) * 0.7
    x1r = np.linspace(cx-r, cx+r, 300)
    x2r = np.linspace(cy-r, cy+r, 300)
    X1, X2 = np.meshgrid(x1r, x2r)
    Z = f([X1, X2])
    fig, ax = plt.subplots(figsize=(7,6))
    cp = ax.contourf(X1, X2, Z, levels=25, cmap='viridis', alpha=0.75)
    ax.contour(X1, X2, Z, levels=25, colors='white', linewidths=0.4, alpha=0.5)
    plt.colorbar(cp, ax=ax)
    ax.plot(hx, hy, 'w-', lw=1.0, alpha=0.7)
    ax.scatter(hx[::max(1,len(hx)//20)], hy[::max(1,len(hy)//20)],
               c='white', s=20, zorder=4, alpha=0.8)
    ax.plot(hx[0], hy[0], 'gs', ms=10, zorder=5, label=f'Старт ({hx[0]:.2f}, {hy[0]:.2f})')
    ax.plot(hx[-1], hy[-1], 'ro', ms=10, zorder=5, label=f'Финиш ({hx[-1]:.4f}, {hy[-1]:.4f})')
    if x_sol is not None:
        ax.plot(x_sol[0], x_sol[1], 'y*', ms=14, zorder=6, label=f'Точное ({x_sol[0]:.4f}, {x_sol[1]:.4f})')
    ax.set_xlabel('x₁'); ax.set_ylabel('x₂'); ax.set_title(title)
    ax.legend(fontsize=8, loc='upper left')
    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches='tight')
    plt.close()

plot_path(f, hs, 'Регулярный симплекс\nf=64x₁²+126x₁x₂+64x₂²-10x₁+30x₂+13',
          'images/lab3_simplex.png', x_exact)
plot_path(f, hn, 'Метод Нелдера-Мида\nf=64x₁²+126x₁x₂+64x₂²-10x₁+30x₂+13',
          'images/lab3_nelder.png', x_exact)

# scipy визуализация
fig0, ax0 = plt.subplots(figsize=(7,6))
r0 = 3.0
x1r = np.linspace(x_exact[0]-r0, x_exact[0]+r0, 300)
x2r = np.linspace(x_exact[1]-r0, x_exact[1]+r0, 300)
X1, X2 = np.meshgrid(x1r, x2r)
Z = f([X1, X2])
cp0 = ax0.contourf(X1, X2, Z, levels=30, cmap='plasma', alpha=0.75)
ax0.contour(X1, X2, Z, levels=30, colors='white', linewidths=0.4, alpha=0.5)
plt.colorbar(cp0, ax=ax0)
ax0.plot(res_scipy.x[0], res_scipy.x[1], 'r*', ms=14,
         label=f'scipy NM: ({res_scipy.x[0]:.4f}, {res_scipy.x[1]:.4f})')
ax0.plot(x_exact[0], x_exact[1], 'y^', ms=10,
         label=f'Аналит.: ({x_exact[0]:.4f}, {x_exact[1]:.4f})')
ax0.set_xlabel('x₁'); ax0.set_ylabel('x₂')
ax0.set_title('Линии уровня и решение scipy\nf=64x₁²+126x₁x₂+64x₂²-10x₁+30x₂+13')
ax0.legend(); plt.tight_layout()
plt.savefig('images/lab3_scipy.png', dpi=120, bbox_inches='tight')
plt.close()

# Задание 5: функция Розенброка
def rosenbrock(x):
    return 100*(x[0]**2 - x[1])**2 + (x[0]-1)**2

x0_rb = np.array([-1.0, 1.0])
eps_rb = 1e-3

xrs, frs, nfrs, itrs, hrs = regular_simplex(rosenbrock, x0_rb, step=0.5, eps=eps_rb, maxiter=20000)
xrn, frn, nfrn, itrn, hrn = nelder_mead(rosenbrock, x0_rb, step=0.5, eps=eps_rb, maxiter=20000)
res_rb_scipy = optimize.minimize(rosenbrock, x0_rb, method='Nelder-Mead',
                                  options={'xatol':1e-8,'fatol':1e-8,'maxiter':50000})

print(f"\n=== Функция Розенброка, x0=(-1,1), eps=1e-3 ===")
print(f"scipy NM:            x*=({res_rb_scipy.x[0]:.6f},{res_rb_scipy.x[1]:.6f}), f={res_rb_scipy.fun:.2e}, nfev={res_rb_scipy.nfev}")
print(f"Регулярный симплекс: x*=({xrs[0]:.6f},{xrs[1]:.6f}), f={frs:.6f}, iter={itrs}, nfev={nfrs}")
print(f"Нелдер-Мид:          x*=({xrn[0]:.6f},{xrn[1]:.6f}), f={frn:.6f}, iter={itrn}, nfev={nfrn}")
print(f"Точное решение:       (1.000000, 1.000000), f=0")

plot_path(rosenbrock, hrs,
          'Регулярный симплекс - функция Розенброка\nx₀=(-1,1), точное x*=(1,1)',
          'images/lab3_rb_simplex.png', np.array([1.0,1.0]))
plot_path(rosenbrock, hrn,
          'Нелдер-Мид - функция Розенброка\nx₀=(-1,1), точное x*=(1,1)',
          'images/lab3_rb_nelder.png', np.array([1.0,1.0]))

print("\nВсе графики сохранены.")
