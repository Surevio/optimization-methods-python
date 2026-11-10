# Лабораторная работа №5 - Методы случайного поиска
# АА-23-07, Рихельгоф Владимир, №18, Вариант 1
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import os
os.makedirs("images", exist_ok=True)  # сюда сохраняются графики

def f(x):
    x1,x2 = x[0],x[1]
    return 64*x1**2 + 126*x1*x2 + 64*x2**2 - 10*x1 + 30*x2 + 13

x_exact = np.linalg.solve([[128.,126.],[126.,128.]], [10.,-30.])
print(f"Аналитика: x*={x_exact}, f={f(x_exact):.6f}")

# Адаптивный метод случайного поиска
def adaptive_random_search(f, x0, eps, delta0=2.0, alpha=0.95, beta=1.05,
                            maxiter=100000, seed=42):
    # Адаптивный случайный поиск (метод с изменяемым шагом).
    # Успех: delta *= beta (поощрение).
    # Неудача: delta *= alpha (сжатие).
    # Критерий останова: delta < eps (шаг стал меньше точности).
    rng = np.random.default_rng(seed)
    x = np.array(x0, dtype=float)
    delta = delta0
    fx = f(x)
    nfev = 1
    history = [x.copy()]
    n = len(x)

    for it in range(maxiter):
        e = rng.standard_normal(n)
        e /= np.linalg.norm(e)
        x_new = x + delta * e
        f_new = f(x_new); nfev += 1

        if f_new < fx:
            x = x_new
            fx = f_new
            delta = min(delta * beta, delta0 * 5)
        else:
            delta *= alpha

        history.append(x.copy())
        if delta < eps:
            break

    return x, fx, nfev, it+1, history

# Случайный поиск с возвратом
def random_search_return(f, x0, eps, delta0=2.0, gamma=0.95, maxiter=100000, seed=42):
    # Случайный поиск с возвратом при неудаче + шаг в противоположном направлении.
    # Успех: принять новую точку (delta не меняется).
    # Неудача прямого шага: пробуем -e (возврат).
    # Неудача обоих: delta *= gamma (сжатие).
    # Критерий: delta < eps.
    rng = np.random.default_rng(seed)
    x = np.array(x0, dtype=float)
    delta = delta0
    fx = f(x)
    nfev = 1
    history = [x.copy()]
    n = len(x)

    for it in range(maxiter):
        e = rng.standard_normal(n)
        e /= np.linalg.norm(e)

        x_fwd = x + delta * e
        f_fwd = f(x_fwd); nfev += 1

        if f_fwd < fx:
            x, fx = x_fwd, f_fwd
        else:
            x_bwd = x - delta * e
            f_bwd = f(x_bwd); nfev += 1
            if f_bwd < fx:
                x, fx = x_bwd, f_bwd
            else:
                delta *= gamma

        history.append(x.copy())
        if delta < eps:
            break

    return x, fx, nfev, it+1, history

# Запуск на основной функции
x0 = np.array([0.0, 0.0])
eps = 1e-4

xa, fa, nfa, ita, ha = adaptive_random_search(f, x0, eps)
xr, fr, nfr, itr, hr = random_search_return(f, x0, eps)

print(f"\n=== Основная функция, x0=(0,0), eps={eps} ===")
print(f"Адаптивный:  x*=({xa[0]:.6f},{xa[1]:.6f}), f={fa:.8f}, iter={ita}, nfev={nfa}")
print(f"С возвратом: x*=({xr[0]:.6f},{xr[1]:.6f}), f={fr:.8f}, iter={itr}, nfev={nfr}")
print(f"Аналитика:   x*=({x_exact[0]:.6f},{x_exact[1]:.6f}), f={f(x_exact):.8f}")

# Таблица (усредняем по 5 запускам)
print("\nk  | Адапт: iter/nfev | Возврат: iter/nfev")
table_data = {}
for k in range(2, 6):
    ek = 10**(-k)
    its_a,nfs_a,its_r,nfs_r = [],[],[],[]
    for seed in range(5):
        _,_,nf1,it1,_ = adaptive_random_search(f, x0, ek, seed=seed)
        _,_,nf2,it2,_ = random_search_return(f, x0, ek, seed=seed)
        its_a.append(it1); nfs_a.append(nf1)
        its_r.append(it2); nfs_r.append(nf2)
    ia,na = int(np.mean(its_a)), int(np.mean(nfs_a))
    ir,nr = int(np.mean(its_r)), int(np.mean(nfs_r))
    table_data[k] = (ia,na,ir,nr)
    print(f"{k}  | {ia:>8}/{na:<8} | {ir:>8}/{nr:<8}")

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
    step = max(1,len(hx)//40)
    ax.plot(hx,hy,'w-',lw=0.7,alpha=0.5)
    ax.scatter(hx[::step],hy[::step],c='white',s=15,zorder=4,alpha=0.7)
    ax.plot(hx[0],hy[0],'gs',ms=10,zorder=5,label=f'Старт ({hx[0]:.1f},{hy[0]:.1f})')
    ax.plot(hx[-1],hy[-1],'ro',ms=10,zorder=5,label=f'Финиш ({hx[-1]:.4f},{hy[-1]:.4f})')
    if x_sol is not None:
        ax.plot(x_sol[0],x_sol[1],'y*',ms=14,zorder=6,
                label=f'Точное ({x_sol[0]:.4f},{x_sol[1]:.4f})')
    ax.set_xlabel('x₁'); ax.set_ylabel('x₂'); ax.set_title(title)
    ax.legend(fontsize=8,loc='upper left')
    plt.tight_layout()
    plt.savefig(filename,dpi=120,bbox_inches='tight')
    plt.close()

plot_path(f, ha,'Адаптивный случайный поиск\nf=64x₁²+126x₁x₂+64x₂²-10x₁+30x₂+13',
          'images/lab5_adaptive.png', x_exact)
plot_path(f, hr,'Случайный поиск с возвратом\nf=64x₁²+126x₁x₂+64x₂²-10x₁+30x₂+13',
          'images/lab5_return.png', x_exact)

# Задание 4: функция Розенброка
def rosenbrock(x):
    return 100*(x[0]**2-x[1])**2+(x[0]-1)**2

x0_rb = np.array([-1.0,1.0])
eps_rb = 1e-3

xra,fra,nfra,itra,hra = adaptive_random_search(rosenbrock, x0_rb, eps_rb,
                                                delta0=1.0, maxiter=500000)
xrr,frr,nfrr,itrr,hrr = random_search_return(rosenbrock, x0_rb, eps_rb,
                                               delta0=1.0, maxiter=500000)
res_rb = optimize.minimize(rosenbrock, x0_rb, method='Nelder-Mead',
                            options={'xatol':1e-8,'fatol':1e-8})

print(f"\n=== Функция Розенброка, x0=(-1,1), eps=1e-3 ===")
print(f"scipy NM:     x*=({res_rb.x[0]:.6f},{res_rb.x[1]:.6f}), f={res_rb.fun:.2e}, nfev={res_rb.nfev}")
print(f"Адаптивный:   x*=({xra[0]:.6f},{xra[1]:.6f}), f={fra:.6f}, iter={itra}, nfev={nfra}")
print(f"С возвратом:  x*=({xrr[0]:.6f},{xrr[1]:.6f}), f={frr:.6f}, iter={itrr}, nfev={nfrr}")

plot_path(rosenbrock, hra,'Адаптивный поиск - функция Розенброка\nx₀=(-1,1)',
          'images/lab5_rb_adaptive.png', np.array([1.0,1.0]))
plot_path(rosenbrock, hrr,'Случайный поиск с возвратом - функция Розенброка\nx₀=(-1,1)',
          'images/lab5_rb_return.png', np.array([1.0,1.0]))

print("\nВсе графики сохранены.")
