# Лабораторная работа №6 - Решение ЗЛП графическим методом
# АА-23-07, Рихельгоф Владимир, №18, Вариант 1 (задача №1)
# f(x1,x2) = -x1 - 4x2 -> max/min
# x1 <= 2, x1+2x2 >= 2, x2 <= 2, x1+x2 <= 3, x1,x2 >= 0
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
import os
os.makedirs("images", exist_ok=True)  # сюда сохраняются графики

# Определение задачи
# f = -x1 - 4x2  (ищем max и min)
c = np.array([-1., -4.])  # коэффициенты целевой функции

# Ограничения в виде A_ub @ x <= b_ub:
# x1 <= 2
# -x1 - 2x2 <= -2  (из x1+2x2 >= 2)
# x2 <= 2
# x1 + x2 <= 3
# x1 >= 0, x2 >= 0 (через bounds)
A_ub = np.array([
    [ 1.,  0.],   # x1 <= 2
    [-1., -2.],   # x1+2x2 >= 2  -> -x1-2x2 <= -2
    [ 0.,  1.],   # x2 <= 2
    [ 1.,  1.],   # x1+x2 <= 3
])
b_ub = np.array([2., -2., 2., 3.])
bounds = [(0, None), (0, None)]

# scipy для max (минимизируем -f = x1+4x2)
res_min = linprog(c,  A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
res_max = linprog(-c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

f_min = res_min.fun       # минимум f
x_min = res_min.x
f_max = -res_max.fun      # максимум f
x_max = res_max.x

print(f"Минимум: f={f_min:.4f} при x1={x_min[0]:.4f}, x2={x_min[1]:.4f}")
print(f"Максимум: f={f_max:.4f} при x1={x_max[0]:.4f}, x2={x_max[1]:.4f}")

# Вершины допустимой области
# Найдём все вершины пересечением прямых ограничений
def find_vertices():
    # Перебор всех пар прямых, поиск точек пересечения внутри ОДР.
    lines = [
        ([1., 0.], 2.),    # x1 = 2
        ([-1., -2.], -2.), # x1+2x2 = 2
        ([0., 1.], 2.),    # x2 = 2
        ([1., 1.], 3.),    # x1+x2 = 3
        ([1., 0.], 0.),    # x1 = 0
        ([0., 1.], 0.),    # x2 = 0
    ]
    vertices = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            A = np.array([lines[i][0], lines[j][0]])
            b = np.array([lines[i][1], lines[j][1]])
            try:
                x = np.linalg.solve(A, b)
                if x[0] >= -1e-9 and x[1] >= -1e-9:  # x1,x2 >= 0
                    # Проверка всех ограничений
                    ok = True
                    for a_row, b_val in zip(A_ub, b_ub):
                        if np.dot(a_row, x) > b_val + 1e-9:
                            ok = False; break
                    if ok:
                        vertices.append(x)
            except np.linalg.LinAlgError:
                pass
    # Убрать дубликаты
    unique = []
    for v in vertices:
        if not any(np.linalg.norm(v-u) < 1e-9 for u in unique):
            unique.append(v)
    return unique

vertices = find_vertices()
print("\nВершины допустимой области:")
for i, v in enumerate(vertices):
    print(f"  V{i+1}: ({v[0]:.4f}, {v[1]:.4f}),  f={np.dot(c, v):.4f}")

# Упорядочим вершины по углу для построения многоугольника
verts_arr = np.array(vertices)
center = verts_arr.mean(axis=0)
angles = np.arctan2(verts_arr[:,1]-center[1], verts_arr[:,0]-center[0])
order = np.argsort(angles)
verts_ordered = verts_arr[order]

# Построение графика
fig, ax = plt.subplots(figsize=(9, 8))

x1 = np.linspace(-0.3, 3.5, 400)

# Заливка допустимой области
poly = plt.Polygon(verts_ordered, closed=True,
                   facecolor='lightblue', edgecolor='navy', linewidth=1.5, alpha=0.5,
                   label='Допустимая область')
ax.add_patch(poly)

# Линии ограничений
colors_lines = ['#e74c3c','#e67e22','#27ae60','#8e44ad','#2980b9','#16a085']
labels_lines = ['x₁ = 2','x₁ + 2x₂ = 2','x₂ = 2','x₁ + x₂ = 3','x₁ = 0','x₂ = 0']
# x1=2
ax.axvline(2, color=colors_lines[0], lw=1.8, ls='--', label=labels_lines[0])
# x1+2x2=2 -> x2=(2-x1)/2
ax.plot(x1, (2-x1)/2, color=colors_lines[1], lw=1.8, ls='--', label=labels_lines[1])
# x2=2
ax.axhline(2, color=colors_lines[2], lw=1.8, ls='--', label=labels_lines[2])
# x1+x2=3 -> x2=3-x1
ax.plot(x1, 3-x1, color=colors_lines[3], lw=1.8, ls='--', label=labels_lines[3])

# Вершины с подписями
fvals = [np.dot(c, v) for v in vertices]
for i, (v, fv) in enumerate(zip(vertices, fvals)):
    ax.plot(v[0], v[1], 'ko', ms=8, zorder=6)
    ax.annotate(f'V{i+1}({v[0]:.1f},{v[1]:.1f})\nf={fv:.1f}',
                xy=v, xytext=(v[0]+0.08, v[1]+0.08), fontsize=9,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

# Отметить min и max
ax.plot(x_min[0], x_min[1], 'rv', ms=14, zorder=7, label=f'min f={f_min:.2f}')
ax.plot(x_max[0], x_max[1], 'g^', ms=14, zorder=7, label=f'max f={f_max:.2f}')

# Линии уровня целевой функции: -x1-4x2=C -> x2=(-C-x1)/4
for C, ls in [(-2,'dotted'),(-4,'dotted'),(-6,'dotted'),(f_min,'solid'),(f_max,'solid')]:
    x2_level = (-C - x1)/4
    lbl = f'f = {C:.1f}' if C in [f_min,f_max] else None
    lw = 2.0 if C in [f_min,f_max] else 0.8
    col = 'red' if C==f_min else ('green' if C==f_max else 'gray')
    ax.plot(x1, x2_level, color=col, lw=lw, ls=ls, label=lbl, alpha=0.8)

# Стрелка градиента (направление убывания f = -x1-4x2 - вниз-влево)
ax.annotate('', xy=(0.5, 0.5), xytext=(1.0, 1.5),
            arrowprops=dict(arrowstyle='->', color='purple', lw=2.5))
ax.text(0.55, 0.45, '∇f', fontsize=13, color='purple', fontweight='bold')

ax.set_xlim(-0.3, 3.5); ax.set_ylim(-0.3, 3.0)
ax.set_xlabel('x₁', fontsize=13); ax.set_ylabel('x₂', fontsize=13)
ax.set_title('ЗЛП: f(x₁,x₂)=-x₁-4x₂ -> extr\nГрафический метод', fontsize=13)
ax.axhline(0, color='black', lw=0.8); ax.axvline(0, color='black', lw=0.8)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/lab6_graph.png', dpi=130, bbox_inches='tight')
plt.close()

print("\nГрафик сохранён.")

# Вывод значений функции в вершинах
print("\n--- Значения f в вершинах (для отчёта) ---")
for i, v in enumerate(vertices):
    print(f"V{i+1}=({v[0]:.2f},{v[1]:.2f}):  f = {np.dot(c,v):.4f}")
print(f"\nМинимум: f_min = {f_min:.4f}  при ({x_min[0]:.4f}, {x_min[1]:.4f})")
print(f"Максимум: f_max = {f_max:.4f}  при ({x_max[0]:.4f}, {x_max[1]:.4f})")
