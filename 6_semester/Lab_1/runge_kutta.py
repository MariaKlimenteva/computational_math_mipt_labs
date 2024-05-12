import numpy as np
import matplotlib.pyplot as plt
import mplcursors

# Правые части системы ОДУ
def f(t, y):
    k1 = 0.04
    k2 = 3*10**7
    k3 = 10**4

    dy1 = -k1 * y[0]**2 + k3 * y[1] * y[2]
    dy2 = k1 * y[0]**2 - k2 * y[1]**2 - k3 * y[1] * y[2]
    dy3 = k2 * y[1]**2

    return np.array([dy1, dy2, dy3])

# Метод Рунге-Кутты
def rk2(f, t, y, h):
    c1 = 1/3
    c2 = 2/3
    a11 = 1/2
    a12 = 0
    a21 = -1/2
    a22 = 2
    b1 = -1/2
    b2 = 3/2

    k1 = f(t + c1*h, y)
    k2 = f(t + c2*h, y + h*a21*k1 + h*a11*k1)

    return y + h*(b1*k1 + b2*k2)

# Основная программа
def solve_rkf(f, t0, y0, tf, h):
    t_values = [t0]
    y_values = [y0]

    t = t0
    y = y0

    while t < tf:
        h = min(h, tf - t)  # Подстраховка, чтобы шаг не превышал конечное время tf
        y = rk2(f, t, y, h)
        t += h

        t_values.append(t)
        y_values.append(y)

    return np.array(t_values), np.array(y_values)

# Начальные условия
t0 = 0
y0 = np.array([1, 0, 0])
tf = 0.3
h = 0.01

# Решение системы ОДУ
t_values, y_values = solve_rkf(f, t0, y0, tf, h)

# Визуализация результатов
plt.plot(t_values, y_values[:,0], label='A', linestyle='-', marker='o')
plt.plot(t_values, y_values[:,1], label='B', linestyle='--', marker='s')
plt.plot(t_values, y_values[:,2], label='C', linestyle=':', marker='^')
plt.xlabel('t')
plt.ylabel('Концентрация')
plt.title('Динамика концентрации веществ')
plt.legend()
plt.grid(True)

# Включение интерактивных меток
mplcursors.cursor(hover=True)

plt.show()