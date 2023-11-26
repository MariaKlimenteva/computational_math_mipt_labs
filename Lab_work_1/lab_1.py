import matplotlib as mp
mp.use('TkAgg', force=True)

import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.sparse.linalg import splu
import scipy.sparse as sp
#########################################################################################
# Задание 1 (аналитическое решение)
#########################################################################################
# 6 = x1, 7 = x2, 8 = x3, 11 = x4, 12 = x5, 13 = x6, 16 = x7, 17 = x8, 18 = x9

c = 0.1
h = 1/4
u_1 = u_5 = u_10 = u_15 = u_2 = u_3 = u_14 = u_9 = u_19 = u_22 = u_23 = u_21 = u_20 = 0

# Система из 9 уравнений 
# -u_1  - u_11  - u_7  - u_5  + 4*u_6  + c*u_6*h**2  = h**2         # Узел 6
# -u_2  - u_12  - u_8  - u_6  + 4*u_7  + c*u_7*h**2  = h**2         # Узел 7
# -u_3  - u_13  - u_9  - u_7  + 4*u_8  + c*u_8*h**2  = h**2         # Узел 8
# -u_6  - u_16  - u_12 - u_10 + 4*u_11 + c*u_11*h**2 = h**2         # Узел 11
# -u_7  - u_13  - u_17 - u_11 + 4*u_12 + c*u_12*h**2 = h**2         # Узел 12
# -u_8  - u_12  - u_18 - u_14 + 4*u_13 + c*u_13*h**2 = h**2         # Узел 13
# -u_11 - u_15  - u_21 - u_17 + 4*u_16 + c*u_16*h**2 = h**2         # Узел 16
# -u_12 - u_18  - u_16 - u_22 + 4*u_17 + c*u_17*h**2 = h**2         # Узел 17
# -u_13 - u_17  - u_19 - u_23 + 4*u_18 + c*u_18*h**2 = h**2         # Узел 18

# Составим матрицу по системе уравнений
A = np.array([[4 + c * h**2, -1, 0, -1, 0, 0, 0, 0, 0],
             [-1, 4 + c * h**2, -1, 0, -1, 0, 0, 0, 0],
             [0, -1, 4 + c * h**2, 0, 0, -1, 0, 0, 0],
             [-1, 0, 0, 4 + c * h**2, -1, 0, -1, 0, 0],
             [0, -1, 0, -1, 4 + c * h**2, -1, 0, -1, 0],
             [0, 0, -1, 0, -1, 4 + c * h**2, 0, 0, -1],
             [0, 0, 0, -1, 0, 0, 4 + c * h**2, -1, 0],
             [0, 0, 0, 0, -1, 0, -1, 4 + c * h**2, -1],
             [0, 0, 0, 0, 0, -1, 0, -1, 4 + c * h**2]])

# Составим столбец правых частей
b = np.array([h**2, h**2, h**2, h**2, h**2, h**2, h**2, h**2, h**2])

# Решение системы уравнений
x = np.linalg.solve(A,b)
# print("Столбец решений для задания 1:\n", x)

# Визуализация матрицы А
# plt.spy(A)
# plt.title("Spy plot of matrix A")
# plt.xlabel("Column index")
# plt.ylabel("Row index")
# plt.show()

#########################################################################################
# Задание 2
#########################################################################################
# пересчет номера узла
def recalculate_node_number(N_x, N_y, i, j):  
    return N_y * i + j

# проверка является ли границей
def is_boundary(N_x, N_y, i, j):
    if(i < 0 or j < 0 or i >= N_x or j >= N_y): return 1
    return 0

# Заполнение списка string + проверка, является ли узел граничным
def fill_list(N_x, N_y, i, j):
    string = [0] * (N_x*N_y)                         
    if is_boundary(N_x, N_y, i + 1, j) == 0:
        string[recalculate_node_number(N_x, N_y, i + 1, j)] = -1      # если не граничная
    if is_boundary(N_x, N_y, i, j + 1) == 0:
        string[recalculate_node_number(N_x, N_y, i, j + 1)] = -1
    if is_boundary(N_x, N_y, i - 1 ,j) == 0:
        string[recalculate_node_number(N_x, N_y, i - 1, j)] = -1
    if is_boundary(N_x, N_y, i, j - 1) == 0:
        string[recalculate_node_number(N_x, N_y, i, j - 1)] = -1
    string[recalculate_node_number(N_x, N_y, i, j)] = 4 + 1e-10 * N_x*N_y   # в центр кладем 4
    return string

# Генерирует матрицу А размерами (N_x - 1)(N_y - 1) на (N_x - 1)(N_y - 1), где N_x, N_y заданные размеры сетки
def generate_matrix(N_x, N_y):
    N_x -= 1
    N_y -= 1
    A = np.zeros((N_x*N_y, N_x*N_y))
    for i in range(N_x):
        for j in range(N_y):
            A[recalculate_node_number(N_x, N_y, i, j)] = fill_list(N_x, N_y, i, j)
    return A

# Геренирует столбец правых частей
def generating_right_parts(N_x, N_y):
    N_x -= 1
    N_y -= 1
    b = np.zeros(N_x*N_y)
    for i in range(N_x):
        for j in range(N_y):
            b[recalculate_node_number(N_x, N_y, i, j)] = 1/(N_x*N_y)
    return b 

def LU_decomposition(A):
    L = np.array([[0.0] * len(A)] * len(A))
    U = np.array([[0.0] * len(A)] * len(A))
    
    for i in range(len(A)):
        L[i][i] = 1.0
    for i in range(len(A)):
        for j in range(len(A)):
            if(i <= j):
                sum = 0.0
                for k in range(i):
                    sum += L[i][k] * U[k][j]
                U[i][j] = A[i][j] - sum
            else:
                sum = 0.0
                for k in range(j):
                    sum += L[i][k] * U[k][j]
                L[i][j] = (A[i][j] - sum) / U[j][j]    
    return [L, U]
            
def get_solution_from_LU(L, U, f):
    x = np.array([0.0] * len(L))
    y = np.array([0.0] * len(L))
    
    y = np.linalg.solve(L,f)    
    x = np.linalg.solve(U, y)
    
    return [x, y]

def solution_visualisation(size, res):
    field = np.zeros((size + 1, size + 1))
    for i in range (size - 1):
        for j in range (size - 1):
            field[i + 1][j + 1] = res[i + j * (size - 1)] 
    print(field)
    fig, ax = plt.subplots()
    x, y = np.meshgrid(np.linspace(0, 1, size+1), np.linspace(0, 1, size+1))
    
    im = ax.pcolormesh(x, y, field, shading='nearest', cmap='Spectral_r')
    fig.colorbar(im, ax=ax, label = 'value')
    ax.plot(x, y, marker='', color='k', linestyle=None, alpha=0.5, linewidth=0.)
    plt.show()

    
    
def check_time():
    time_to_solve = []
    number_of_nodes = []
    
    for i in range(2, 50):
        number_of_nodes.append(i)
    
    for i in number_of_nodes:
        start_time = time.time()
        A = generate_matrix(i, i)
        lu = splu(A)
        x = lu.solve(generating_right_parts(i, i))
        
        end_time = time.time()
        time_solving = end_time - start_time
        time_to_solve.append(time_solving)
    return [time_to_solve, number_of_nodes]

def print_graph():
    time_to_solve = check_time()[0]
    number_of_nodes = check_time()[1]
    plt.plot(number_of_nodes, time_to_solve)
    plt.xlabel('Число узлов')
    plt.ylabel('Время решения, с')
    plt.title('Зависимость времени решения СЛАУ от числа узлов')
    plt.show()
    
def main():
    # №1 (решение с помощью функций)
    # nodes_1_x = nodes_1_y = 4
    # A = generate_matrix(nodes_1_x, nodes_1_y)
    # print("Вид матрицы А: \n", A)

    # Визуализация
    # print("Визуализация матрицы с помощью spy\n")
    # plt.spy(A)
    
    
    # plt.matshow(A, cmap='Oranges', vmin=-1, vmax=1)
    # plt.show()

    # b = generating_right_parts(nodes_1_x, nodes_1_y)
    # print("Столбец правых частей: \n", b)
    
    # №2
    nodes_2_x = nodes_2_y = 120
    
    A = generate_matrix(nodes_2_x, nodes_2_y)
    # plt.matshow(A, cmap='Oranges', vmin=-1, vmax=1)
    # plt.show()
    
    lu = splu(sp.csc_matrix(A))
    x = lu.solve(generating_right_parts(nodes_2_x, nodes_2_y))
    print(generating_right_parts(nodes_2_x, nodes_2_y))
    print(x)

    solution_visualisation(nodes_2_x, x)

    
    # print_graph()

main()
