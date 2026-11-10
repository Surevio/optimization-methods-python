# Лабораторная работа №4 - Градиентные методы многомерной минимизации
# АА-23-07, Рихельгоф Владимир, №18, Вариант 1
# f(x) = 64x1²+126x1x2+64x2²-10x1+30x2+13
# Методы: наискорейший спуск, метод Марквардта
# Задание 4: функция Розенброка, x0=(-1,1), eps=1e-3
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import minimize_scalar
import os
os.makedirs("images", exist_ok=True)  # сюда сохраняются графики

# Целевая функция (вариант 1)
def f(x):
    x1, x2 = x[0], x[1]
    return 64*x1**2 + 126*x1*x2 + 64*x2**2 - 10*x1 + 30*x2 + 13

def grad_f(x):
    x1, x2 = x[0], x[1]
    return np.array([128*x1 + 126*x2 - 10,
                     126*x1 + 128*x2 + 30])

def hess_f(x):
    return np.array([[128., 126.],
                     [126., 128.]])

# Аналитическое решение
x_exact = np.linalg.solve(np.array([[128.,126.],[126.,128.]]), np.array([10.,-30.]))
print(f"Аналитика: x*={x_exact}, f={f(x_exact):.8f}")

# Одномерная минимизация (метод парабол - лучший по лаб.1)
def parabolic_1d(phi, a, b, eps=1e-8):
    # Метод парабол для нахождения шага вдоль направления.
    x1, x2, x3 = a, (a+b)/2, b
    f1, f2, f3 = phi(x1), phi(x2), phi(x3)
    for _ in range(1000):
        denom = (x2-x1)*(f2-f3) - (x2-x3)*(f2-f1)
        if abs(denom) < 1e-14:
            break
        xs = x2 - 0.5*((x2-x1)**2*(f2-f3)-(x2-x3)**2*(f2-f1))/denom
        xs = np.clip(xs, a, b)
        fs = phi(xs)
        if abs(b-a) < eps:
            break
        if xs < x2:
            if fs < f2: x3,f3=x2,f2; x2,f2=xs,fs
            else:        x1,f1=xs,fs
        else:
            if fs < f2: x1,f1=x2,f2; x2,f2=xs,fs
            else:        x3,f3=xs,fs
        a,b = min(x1,x2,x3), max(x1,x2,x3)
    return x2

# Метод наискорейшего спуска
def steepest_descent(f, grad, x0, eps, maxiter=10000):
    # На каждой итерации: направление d = -grad f(x).
    # Шаг alpha находится точной одномерной минимизацией методом парабол
    # вдоль d: min_{alpha>=0} f(x + alpha*d).
    # Критерий остановки: ||grad|| < eps.
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    nf, ng = 1, 1

    for it in range(maxiter):
        g = grad(x); ng += 1
        if np.linalg.norm(g) < eps:
            break
        d = -g
        # Одномерный поиск: phi(alpha) = f(x + alpha*d)
        phi = lambda a: f(x + a*d)
        # Границы поиска: берём достаточно широкий диапазон
        alpha = parabolic_1d(phi, 0, 2.0)
        nf += 20  # примерное число вычислений в parabolic_1d
        x = x + alpha * d
        history.append(x.copy())
        nf += 1

    return x, f(x), nf, ng, it+1, history  # steepest

# Метод Марквардта (Levenberg-Marquardt)
def marquardt(f, grad, hess, x0, eps, maxiter=1000):
    # Итерация: x_{k+1} = x_k - (H + mu*I)^{-1} * grad
    # mu подбирается: если f уменьшилась - mu /= 2, иначе mu *= 2.
    # Начальное mu = 10 * trace(H(x0)) / n.
    # Критерий остановки: ||grad|| < eps.
    x = np.array(x0, dtype=float)
    n = len(x)
    H0 = hess(x)
    mu = 10.0 * np.trace(H0) / n
    history = [x.copy()]
    nf, ng, nh = 1, 1, 1

    for it in range(maxiter):
        g = grad(x); ng += 1
        if np.linalg.norm(g) < eps:
            break
        H = hess(x); nh += 1
        # Решаем (H + mu*I) * p = -g
        A = H + mu * np.eye(n)
        p = np.linalg.solve(A, -g)
        x_new = x + p
        f_new = f(x_new); nf += 1

        if f_new < f(x):
            x = x_new
            mu = max(mu / 2, 1e-12)
        else:
            mu = mu * 2

        history.append(x.copy())

    return x, f(x), nf, ng, nh, it+1, history

# Запуск на основной функции
x0 = np.array([0.0, 0.0])
eps = 1e-4

xs, fs, nfs, ngs, its, hs = steepest_descent(f, grad_f, x0, eps)
xm, fm, nfm, ngm, nhm, itm, hm = marquardt(f, grad_f, hess_f, x0, eps)

print(f"\n=== Основная функция, x0=(0,0), eps={eps} ===")
print(f"Наискорейший спуск: x*=({xs[0]:.6f},{xs[1]:.6f}), f={fs:.8f}, iter={its}, nf={nfs}, ng={ngs}")
print(f"Марквардт:          x*=({xm[0]:.6f},{xm[1]:.6f}), f={fm:.8f}, iter={itm}, nf={nfm}, ng={ngm}, nh={nhm}")

# Таблица по eps
print("\nk  | НС: iter/nf/ng | Марк.: iter/nf/ng/nh")
table_data = {}
for k in range(2, 6):
    ek = 10**(-k)
    _, _, f1, g1, it1, _ = steepest_descent(f, grad_f, x0, ek)
    _, _, f2, g2, h2, it2, _ = marquardt(f, grad_f, hess_f, x0, ek)
    table_data[k] = (it1, f1, g1, it2, f2, g2, h2)
    print(f"{k}  | {it1:>4}/{f1:>5}/{g1:>4}   | {it2:>4}/{f2:>5}/{g2:>4}/{h2:>4}")

# Визуализация
def plot_path(f, history, title, filename, x_sol=None):
    hx = np.array([h[0] for h in history])
    hy = np.array([h[1] for h in history])
    cx = x_sol[0] if x_sol is not None else hx[-1]
    cy = x_sol[1] if x_sol is not None else hy[-1]
    r = max(max(abs(hx-cx)), max(abs(hy-cy)), 1.5) * 1.3
    x1r = np.linspace(cx-r, cx+r, 300)
    x2r = np.linspace(cy-r, cy+r, 300)
    X1,X2 = np.meshgrid(x1r, x2r)
    Z = f([X1,X2])
    fig, ax = plt.subplots(figsize=(7,6))
    cp = ax.contourf(X1,X2,Z,levels=25,cmap='viridis',alpha=0.75)
    ax.contour(X1,X2,Z,levels=25,colors='white',linewidths=0.4,alpha=0.5)
    plt.colorbar(cp, ax=ax)
    step = max(1, len(hx)//30)
    ax.plot(hx, hy,'w-',lw=1.0,alpha=0.7)
    ax.scatter(hx[::step],hy[::step],c='white',s=20,zorder=4,alpha=0.8)
    ax.plot(hx[0],hy[0],'gs',ms=10,zorder=5,label=f'Старт ({hx[0]:.1f},{hy[0]:.1f})')
    ax.plot(hx[-1],hy[-1],'ro',ms=10,zorder=5,label=f'Финиш ({hx[-1]:.4f},{hy[-1]:.4f})')
    if x_sol is not None:
        ax.plot(x_sol[0],x_sol[1],'y*',ms=14,zorder=6,label=f'Точное ({x_sol[0]:.4f},{x_sol[1]:.4f})')
    ax.set_xlabel('x₁'); ax.set_ylabel('x₂'); ax.set_title(title)
    ax.legend(fontsize=8,loc='upper left')
    plt.tight_layout()
    plt.savefig(filename,dpi=120,bbox_inches='tight')
    plt.close()

plot_path(f, hs, 'Наискорейший спуск\nf=64x₁²+126x₁x₂+64x₂²-10x₁+30x₂+13',
          'images/lab4_steepest.png', x_exact)
plot_path(f, hm, 'Метод Марквардта\nf=64x₁²+126x₁x₂+64x₂²-10x₁+30x₂+13',
          'images/lab4_marquardt.png', x_exact)

# Задание 4: функция Розенброка
def rosenbrock(x):
    return 100*(x[0]**2 - x[1])**2 + (x[0]-1)**2

def grad_rb(x):
    x1,x2 = x[0],x[1]
    return np.array([400*x1*(x1**2-x2) + 2*(x1-1),
                    -200*(x1**2-x2)])

def hess_rb(x):
    x1,x2 = x[0],x[1]
    return np.array([[1200*x1**2 - 400*x2 + 2, -400*x1],
                     [-400*x1, 200.]])

x0_rb = np.array([-1.0, 1.0])
eps_rb = 1e-3

xrs, frs, nfrs, ngrs, itrs, hrs = steepest_descent(rosenbrock, grad_rb, x0_rb, eps_rb, maxiter=50000)
xrm, frm, nfrm, ngrm, nhrm, itrm, hrm = marquardt(rosenbrock, grad_rb, hess_rb, x0_rb, eps_rb, maxiter=5000)
res_rb = optimize.minimize(rosenbrock, x0_rb, method='L-BFGS-B',
                            jac=grad_rb, options={'gtol':1e-10})

print(f"\n=== Функция Розенброка, x0=(-1,1), eps=1e-3 ===")
print(f"scipy L-BFGS-B:     x*=({res_rb.x[0]:.6f},{res_rb.x[1]:.6f}), f={res_rb.fun:.2e}, nfev={res_rb.nfev}")
print(f"Наискорейший спуск: x*=({xrs[0]:.6f},{xrs[1]:.6f}), f={frs:.6f}, iter={itrs}, nf={nfrs}, ng={ngrs}")
print(f"Марквардт:          x*=({xrm[0]:.6f},{xrm[1]:.6f}), f={frm:.6f}, iter={itrm}, nf={nfrm}, ng={ngrm}")

plot_path(rosenbrock, hrs, 'Наискорейший спуск - функция Розенброка\nx₀=(-1,1), точное x*=(1,1)',
          'images/lab4_rb_steepest.png', np.array([1.0,1.0]))
plot_path(rosenbrock, hrm, 'Метод Марквардта - функция Розенброка\nx₀=(-1,1), точное x*=(1,1)',
          'images/lab4_rb_marquardt.png', np.array([1.0,1.0]))

print("\nВсе графики сохранены.")
