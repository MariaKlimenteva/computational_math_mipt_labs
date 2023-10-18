import matplotlib as mp
mp.use('TkAgg', force=True)

import matplotlib.pyplot as plt
import numpy as np
import time

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
print("Столбец решений для задания 1:\n", x)

# Визуализация матрицы А
plt.spy(A)
plt.title("Spy plot of matrix A")
plt.xlabel("Column index")
plt.ylabel("Row index")
# plt.show()

#########################################################################################
# Задание 2
#########################################################################################
# пересчет номера узла
def recalculate_node_number(size, i, j):  
    return size * i + j

# проверка является ли границей
def is_boundary(h, i, j):
    if(i < 0 or j < 0 or i >= h or j >= h): return 1
    return 0

# Заполнение списка string + проверка, является ли узел граничным
def fill_list(h, i, j):
    string = [0] * (h**2)                         
    if is_boundary(h, i + 1, j) == 0:
        string[recalculate_node_number(h, i + 1, j)] = -1      # если не граничная
    if is_boundary(h, i, j + 1) == 0:
        string[recalculate_node_number(h, i, j + 1)] = -1
    if is_boundary(h, i - 1 ,j) == 0:
        string[recalculate_node_number(h, i - 1, j)] = -1
    if is_boundary(h, i, j - 1) == 0:
        string[recalculate_node_number(h, i, j - 1)] = -1
    string[recalculate_node_number(h, i, j)] = 4 + 0.1 * h**2   # в центр кладем 4
    return string

# Сздает матрицу A размера (size-1)^2 x (size-1)^2 и заполняет ее значениями,
# используя функцию fill_list, возвращает заполненную матрицу A
def generate_matrix(size):
    size = size - 1 
    A = np.zeros((size**2, size**2))
    for i in range(size):
        for j in range(size):
            A[recalculate_node_number(size, i, j)] = fill_list(size, i, j)
    return A

# Геренирует столбец правых частей
def generating_right_parts(size):
    size = size - 1
    b = np.zeros(size**2)
    for i in range(size):
        for j in range(size):
            b[recalculate_node_number(size, i, j)] = 1 / size**2 
            
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
    print("\n\n\ny:\n", y)
    
    x = np.linalg.solve(U, y)
    print("\n\n\nx:\n", x)
    
    return [x, y]
    
def solution_visualisation(size, res):
    S = size + 1
    x = np.arange(0, 1, 1 / S)
    for i in range(size + 1):
        x[i] = i / size
    y = -x
    z = np.zeros((S, S))
    for i in range(1, size):
        for j in range(1, size):
            z[i][j] = res[(size - 1) * (i - 1) + (j - 1)]
    plt.contourf(x, y, z, levels = 100, cmap = mp.cm.magma_r)
    plt.show()
    
def check_time():
    time_to_solve = []
    number_of_nodes = []
    for i in range(1, 100):
        number_of_nodes.append(i)
        start_time = time.time()
        L = LU_decomposition(generate_matrix(i))[0]
        U = LU_decomposition(generating_right_parts(i))[1]
        x = get_solution_from_LU(L, U, generating_right_parts(i))
        end_time = time.time()
        time_to_solve.append(end_time - start_time)
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
    result = generate_matrix(4)
    print("Вид матрицы А: \n", result)

    # Визуализация
    print("Визуализация матрицы с помощью spy\n")
    plt.spy(result)
    plt.show()

    result = generating_right_parts(4)
    print("Столбец правых частей: \n", result)
    
    # №2
    ## Я не поняла какой брать столбец правых частей, поэтому взяла его из задания 1
    ## LU - разложение
    l = LU_decomposition(generate_matrix(4))[0]
    print("\n\nМатрица L: \n", l)
    
    u = LU_decomposition(generate_matrix(4))[1]
    print("\n\nМатрица U: \n", u)
    
    x = get_solution_from_LU(l, u, result)[0]
    
    plt.spy(u)
    plt.title("Spy plot of matrix L")
    plt.xlabel("Column index")
    plt.ylabel("Row index")
    plt.show()
    solution_visualisation(4, x)
    
    # print_graph()
    
main()
