import numpy as np
import matplotlib.pyplot as plt

# Константы
k1 = 0.04
k2 = 3107
k3 = 104

# Начальные условия
y1_0 = 1
y2_0 = 0
y3_0 = 0

# Функция системы ОДУ
def f(t, y):
  y1, y2, y3 = y
  return np.array([-k1 * y1 + k3 * y2 * y3,
                   k1 * y1 - k3 * y2 * y3 - k2 * y2,
                   k2 * y2])

# Параметры интегрирования
t_start = 0
t_end = 0.3
dt = 0.001

# Коэффициенты метода Рунге-Кутты
c_2 = 1/2
b_2 = 1

# Неявный метод Рунге-Кутты
def rk2_implicit(f, t, y, dt):
  k1 = dt * f(t, y)
  k2 = dt * f(t + c_2 * dt, y + k1 * c_2)
  y_new = y + b_2 * k1 + (1 - b_2) * k2
  return y_new

# Решение системы ОДУ
t = np.arange(t_start, t_end, dt)
y1 = np.zeros(len(t))
y2 = np.zeros(len(t))
y3 = np.zeros(len(t))

y1[0] = y1_0
y2[0] = y2_0
y3[0] = y3_0

for i in range(1, len(t)):
  y_new = rk2_implicit(f, t[i-1], [y1[i-1], y2[i-1], y3[i-1]], dt)
  y1[i] = y_new[0]
  y2[i] = y_new[1]
  y3[i] = y_new[2]

# Построение графиков
plt.plot(t, y1, label="A")
plt.plot(t, y2, label="B")
plt.plot(t, y3, label="C")
plt.xlabel("t")
plt.ylabel("Концентрация")
plt.legend()
plt.show()