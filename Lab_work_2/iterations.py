import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.sparse import csr_matrix 

# Размер матрицы
n = 1000

# Относительная невязка
relative_eps = 1e-4
    
def create_matrix():
    # Значения альфа и бетта
    alpha = 0.01
    betta = 10

    # Создание пустой матрицы
    matrix = np.zeros((n, n))

    # Заполнение главной диагонали
    np.fill_diagonal(matrix, 2 + alpha)
    matrix[0][0] = 2 + betta

    # Заполнение диагоналей выше и ниже главной диагонали
    for i in range(1, n):
        matrix[i, i - 1] = -1
        matrix[i - 1, i] = -1
        
    # print(matrix)
    return matrix

def create_right_parts():
    # Заполнение правых частей
    column = [0] * 1000  # создаем список из 1000 нулей
    column[495:506] = [1] * 11  # заменяем элементы с индексами от 495 до 505 на 1
    f = np.array(column)
    return f

def calculate_eigenvalue_of_matrix():
    # Вычислим собственные числа матрицы
    matrix = create_matrix()
    eigen_values = linalg.eigvals(matrix)
    lamda_min = min(eigen_values)
    lamda_max = max(eigen_values)
    return lamda_max, lamda_min
    
def find_optimal_iterative_parameter():
    # Выбор оптимального итерационного параметра
    lamda_max, lamda_min = calculate_eigenvalue_of_matrix()
    tau = 2/(lamda_max + lamda_min)
    
    return tau

# def multiply_matrix_by_vector():
#     
    
def iterative_solver(matrix, f, tau):

    # Выбираем начальное приближение x0
    x0 = np.array(n * [0])
    # Инициализируем переменные для хранения невязки итераций
    x = x0
    r0 = f - matrix.dot(x)
    r_norm0 = np.linalg.norm(r0)
    k = 0
    residuals = []
    
    while True:
        x_new = x + tau * (f - matrix.dot(x))
        r = f - matrix.dot(x_new)
        r_norm = np.linalg.norm(r)
        residuals.append(r_norm / r_norm0)  # Сохраняем относительную невязку
        if r_norm / r_norm0 < relative_eps:
            break
        x = x_new
        k += 1
        
    return residuals, k
        

    
def make_graph(residuals, k):
    # Построение графика
    plt.plot(range(k + 1), residuals, marker='2', color='c', markersize=0.5, label = 'simple iteration, no precond')
    plt.yscale('log')  # Логарифмическая шкала по оси y для удобства отображения
    plt.xlabel('Номер итерации')
    plt.ylabel('Относительная невязка')
    plt.title('Зависимость относительной невязки от номера итерации')
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    matrix = create_matrix()
    f = create_right_parts()
    tau = find_optimal_iterative_parameter()
    
    residuals, k = iterative_solver(matrix, f, tau)
    
    make_graph(residuals, k)
    
main()