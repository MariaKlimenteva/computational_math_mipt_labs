import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.sparse import csr_matrix
from scipy.sparse import diags

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
        
    return nevyaska, k

def iterative_solver_precon(matrix, f):
    # Создаем диагональный предобуславливатель
    print(matrix)
    D = np.diag(np.diag(matrix))

    # Вычисляем матрицу T и вектор c для метода простых итераций
    T = np.dot(np.linalg.inv(D), matrix)
    c = np.dot(np.linalg.inv(D), f)
    I = np.eye(n)

    # Начальное приближение
    x0 = np.array(n * [0])

    # Метод простых итераций
    x = x0
    r0 = matrix.dot(x)
    r_norm0 = np.linalg.norm(r0)
    k = 0
    nevyaska = []
    # for i in range(0, n):
    while True:
        x_new = (I - np.dot(T, x)) + c
        r = matrix.dot(x_new)
        r_norm = np.linalg.norm(r)
        nevyaska.append(r_norm / r_norm0)  # Сохраняем относительную невязку
        if r_norm / r_norm0 < relative_eps:
            break
        x = x_new
        k += 1
        
    return nevyaska, k  
    
def make_graph(nevyaska, k):
    # Построение графика
    plt.plot(range(k + 1), nevyaska, marker='2', color='c', markersize=0.5, label = 'simple iteration, no precond')
    plt.yscale('log')  # Логарифмическая шкала по оси y для удобства отображения
    plt.xlabel('Номер итерации')
    plt.ylabel('Относительная невязка')
    plt.title('Зависимость относительной невязки от номера итерации')
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    # Значения альфа и бетта
    alpha = 0.01
    betta = 10
    
    matrix = create_matrix
    f = create_right_parts
    tau = find_optimal_iterative_parameter
    
    nevyaska_1, k_1 = iterative_solver_no_precond(matrix(alpha, betta), f(), tau(alpha, betta)) # использование callbacks
    make_graph(nevyaska_1, k_1)
    
    # nevyaska_2, k_2 = iterative_solver_precon(matrix(alpha, betta), f())
    # make_graph(nevyaska, k)
    
main()