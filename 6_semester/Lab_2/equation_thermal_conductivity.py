import numpy as np
import matplotlib.pyplot as plt

# Параметры задачи
a_S = 40
a_out = 500
h = 1/20
Nx = int(2/h)   # Количество точек по x
Ny = int(2/h)   # Количество точек по y
dt = 1.5625e-7  # Шаг по времени
tolerance = 1e-4  # Точность для критерия остановки
error = []

# Создание сетки
x = np.arange(-1, 1+h, h)
y = np.arange(-1, 1+h, h)
X, Y = np.meshgrid(x, y)
print("len X, Y: ", len(X), len(Y))

# Инициализация решения
U = np.zeros((Nx, Ny))

# Определение области S
S_mask = (X >= -0.75) & (X <= -0.25) & (Y >= 0.25) & (Y <= 0.75)

# Граничные условия
U[:, 0] = 0  # D1
U[:, -1] = 0  # D2
U[0, :] = 0  # D3
U[-1, :] = 0  # D4
U[int(Nx/2):, -1] = 1  # N1 - производная по x = 1
U[0:int(Nx/2), 0] = -1  # N2 - производная по y = -1 

def build_matrix(U):
    A = np.zeros((Nx*Ny, Nx*Ny))
    b = np.zeros(Nx*Ny)
    
    for i in range(Nx):
        for j in range(Ny):
            k = i + j * Nx
            a_ij = a_S if S_mask[i, j] else a_out
            a_i1j = a_S if S_mask[i+1, j] else a_out
            a_im1j = a_S if S_mask[i-1, j] else a_out
            a_ij1 = a_S if S_mask[i, j+1] else a_out
            a_ijm1 = a_S if S_mask[i, j-1] else a_out

            if i == 0 or i == Nx - 1 or j == 0 or j == Ny - 1:  # Границы D1-D4
                A[k, k] = 1
                b[k] = 0
            elif i == Nx - 1:  # N1: u_x = 1
                A[k, k] = 1
                A[k, k - 1] = -1
                b[k] = h
            elif j == 0:  # N2: u_y = -1
                A[k, k] = 1
                A[k, k + Nx] = -1
                b[k] = -h
            else:  # Внутренние точки
                A[k, k] = 1 + dt * (a_i1j + a_im1j + a_ij1 + a_ijm1) / h**2
                A[k, k - 1] = -dt * (a_im1j + a_ij) / (2 * h**2)
                A[k, k + 1] = -dt * (a_i1j + a_ij) / (2 * h**2)
                A[k, k - Nx] = -dt * (a_ijm1 + a_ij) / (2 * h**2)
                A[k, k + Nx] = -dt * (a_ij1 + a_ij) / (2 * h**2)
                b[k] = U[i, j] + 0.5 * dt * a_ij * (
                    (U[i + 1, j] - 2 * U[i, j] + U[i - 1, j]) / h**2 +
                    (U[i, j + 1] - 2 * U[i, j] + U[i, j - 1]) / h**2
                ) 
    
    if np.any(A):
        print("Матрица содержит ненулевые элементы.")
    else:
        print("Матрица полностью нулевая.")
    return A, b

t = 0
while True:
    A, b = build_matrix(U)
    print(len(U))
    
    print("Детерминант матрицы: ", np.linalg.det(A))
    print(b)


    U_new = np.linalg.solve(A, b).reshape((Nx, Ny))
    print("U_new: ", len(U_new))
    print(np.max(np.abs(U_new - U)) - tolerance)
    error.append(np.max(np.abs(U_new - U)))
    # Проверка критерия остановки
    if np.max(np.abs(U_new - U)) < tolerance:
        break

    U = U_new
    t += dt


print(f"Установившееся решение достигнуто за {t:.3f} секунд.")
X = X[:-1, :]
Y = Y[:-1, :]
X = X[:, :-1]
Y = Y[:, :-1]
print(len(X))
print(len(U))
plt.contourf(X, Y, U, levels=20)
plt.colorbar()
plt.title(f"Установившееся решение (t = {t:.3f})")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("2D graph.png")

# График сходимости
iters = list(range(0, len(error)))
print(iters)
plt.figure()
plt.plot(iters, error)
plt.title("Сходимость решения")
plt.xlabel("Итерация")
plt.ylabel("Максимальная ошибка")
plt.savefig("graph.png")