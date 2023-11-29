import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

from scipy.sparse.linalg import cg

# Размер матрицы
n = 1000

# Относительная невязка
relative_eps = 1e-4
    
def create_matrix(alpha, betta):
    # Создание пустой матрицы
    matrix = np.zeros((n, n))

    # Заполнение главной диагонали
    np.fill_diagonal(matrix, 2 + alpha)
    matrix[0][0] = 2 + betta

    # Заполнение диагоналей выше и ниже главной диагонали
    for i in range(1, n):
        matrix[i, i - 1] = -1
        matrix[i - 1, i] = -1
    
    return matrix

def create_right_parts():
    # Заполнение правых частей
    column = [0] * 1000  # создаем список из 1000 нулей
    column[495:506] = [1] * 11  # заменяем элементы с индексами от 495 до 505 на 1
    f = np.array(column)
    return f

def calculate_eigenvalue_of_matrix(alpha, betta):
    # Вычислим собственные числа матрицы
    matrix = create_matrix(alpha, betta)
    eigen_values = linalg.eigvals(matrix)
    lamda_min = min(eigen_values)
    lamda_max = max(eigen_values)
    return lamda_max, lamda_min
    
def find_optimal_iterative_parameter(alpha, betta):
    # Выбор оптимального итерационного параметра
    lamda_max, lamda_min = calculate_eigenvalue_of_matrix(alpha, betta)
    tau = 2/(lamda_max + lamda_min)
    q = (lamda_max-lamda_min)/(lamda_max+lamda_min)
    print(q)
    
    return tau

def multiply(matrix, vector):
    result = [0] * n 
    sum = 0
    for i in range(n):
        # row = matrix[i]
        sum = 0
        for j in range(n):
            # if row[j] != 0:
            sum += matrix[i][j] * vector[j]
        result[i] = sum
    return np.array(result)

    
def iterative_solver_no_precond(matrix, f, tau):
    # Выбираем начальное приближение x0
    x0 = np.array(n * [0])
    # Инициализируем переменные для хранения невязки итераций
    x = x0
    r0 = f - matrix.dot(x)#multiply(matrix, x) #
    r_norm0 = np.linalg.norm(r0)
    k = 0
    nevyaska = []
    
    while True:
        x_new = x + tau * (f - matrix.dot(x)) # multiply(matrix, x)
        
        r = f - matrix.dot(x_new) # multiply(matrix, x_new)
        r_norm = np.linalg.norm(r)
        nevyaska.append(r_norm / r_norm0)  # Сохраняем относительную невязку
        if r_norm / r_norm0 < relative_eps:
            break
        x = x_new
        k += 1
        
    return nevyaska, k, x

def iterative_solver_precon(matrix, f):
    # Создаем диагональный предобуславливатель
    D = np.diag(np.diag(matrix))

    # Вычисляем матрицу T и вектор c для метода простых итераций
    T = np.dot(np.linalg.inv(D), matrix)
    c = np.dot(np.linalg.inv(D), f)
    I = np.eye(n)

    # Начальное приближение
    x0 = np.array(n * [0])

    # Метод простых итераций
    x = x0
    r0 = f - multiply(matrix, x) #matrix.dot(x)
    r_norm0 = np.linalg.norm(r0)
    k = 0
    nevyaska = []

    while True:
        x_new = ((I - T)).dot(x) + c
        r = f - matrix.dot(x_new)
        r_norm = np.linalg.norm(r)
        nevyaska.append(r_norm / r_norm0)  # Сохраняем относительную невязку
        if r_norm / r_norm0 < relative_eps:
            break
        x = x_new
        k += 1
        
    return nevyaska, k, x

def CG(A, b):
    num_iters = 0
    nevyaska = []

    def callback(xk):
        nonlocal num_iters
        num_iters += 1
        nevyaska.append(np.linalg.norm(b - A.dot(xk)))

    x, status = cg(A, b, tol=1e-4, callback=callback)
    return x, status, num_iters, nevyaska

def main():
    # Значения альфа и бетта
    alpha = 0.01
    betta = 10
    
    matrix = create_matrix
    f = create_right_parts
    tau = find_optimal_iterative_parameter
    
    nevyaska_1, k_1, x_1 = iterative_solver_no_precond(matrix(alpha, betta), f(), tau(alpha, betta))
    
    nevyaska_2, k_2, x_2 = iterative_solver_precon(matrix(alpha, betta), f())
    
    x_3, status, k_3, nevyaska_3 = CG(matrix(alpha, betta), f())
    

    
    # Построение графика
    plt.plot(range(k_1 + 1), nevyaska_1, marker='2', color='c', markersize=0.5, label = 'simple iteration, no precond')
    plt.plot(range(k_2 + 1), nevyaska_2, marker='2', color='r', markersize=0.5, label = 'diagonal precond')
    plt.plot(range(k_3), nevyaska_3, marker='v', color='b', markersize=0.5, label = 'CG')
    plt.yscale('log')  # Логарифмическая шкала по оси y для удобства отображения
    plt.xlabel('Номер итерации')
    plt.ylabel('Относительная невязка')
    plt.title('Зависимость относительной невязки от номера итерации')
    plt.grid(True)
    plt.legend()
    plt.show()
        
    
    # Построение графика
    plt.semilogy(range(n), x_3, marker='2', color='c', markersize=0.5, label = 'CG')
    plt.semilogy(range(n), x_1, marker='o', color='b', markersize=0.5, label = 'simple iterations, no precond')
    plt.semilogy(range(n), x_2, marker='v', color='r', markersize=0.5, label = 'simple iterations, diag precond')
    # plt.yscale('log') 
    # plt.semilogy()
    plt.xlabel('Component index')
    plt.ylabel('Solution x')
    plt.title('Зависимость x(component index)')
    plt.grid(True)
    plt.legend()
    plt.show()


    
main()