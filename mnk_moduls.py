#import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve

def rejections_list(X, Y, approx_y):
    rejection_list = []
    for index in range(len(X)):
        rejection_list.append((approx_y[index] - Y[index])**2)
    #rejection_list.sort(reverse=True)
    return rejection_list

def enjection_filter(X, Y, rejection_list, k):    
    #находим отклонения, похожие на выбросы
    enjections = []
    md = np.median(rejection_list)
    #print(f"Медиана: {md}")
    std = np.std(rejection_list)
    #print(f"Стандартное отклонение: {md}")
    for index in range(len(rejection_list)):
        if (rejection_list[index] - md > k*std):
            enjections.append(index)

    #формируем списки данных без выбросов
    X_after = []
    Y_after = []

    for i in range(len(X)):
        if i not in enjections:
            X_after.append(X[i])
            Y_after.append(Y[i])
    
    return X_after, Y_after, enjections
    

#заполнение квадратной матрицы и столбца свободных членов
def readmatrix(X, Y, ap, n):
#n - степень полинома, Y, X - списки координат заданных точек, ap - колличество точек таблицы
    #квадр. матрица
    matr = []
    sv_chl = []
    for i in range(0, n+1):
        sums_list = []
        for j in range(0, n+1):
            sums = 0
            for k in range(ap):
                sums += (X[k])**(i+j)
            sums_list.append(sums)
        matr.append(sums_list)
    
    #столбец св. членов
    for ii in range(n+1):
        sums_2 = 0
        for k in range(ap):
            sums_2 +=  Y[k] * X[k] ** ii
        sv_chl.append(sums_2)

    return matr, sv_chl


#функция замены строк в матрице, если на главной диагонали есть нули
def diagonal(matr, sv_chl, n):
    temp = 0
    for i in range(n+1):
        if (matr[i][i] == 0):
            for j in range(n+1):
                if (j == i): continue
                if (matr[j][i] != 0 and matr[i][j] != 0):
                    for k in range(n+1):
                        temp = matr[j][k]
                        matr[j][k] = matr[i][k]
                        matr[i][k] = temp
                    temp = sv_chl[j]
                    sv_chl[j] = sv_chl[i]
                    sv_chl[i] = temp
                    break

#заполнение столбца неизвестных коэффициентов
def find_unknow_coef(matr, sv_chl, n):
    
    #приведение матрицы к треугольному виду
    for k in range(n+1):
        for i in range(k+1, n+1):
            M = matr[i][k] / matr[k][k]
            for j in range(k, n+1):
                matr[i][j] -= M*matr[k][j]
            sv_chl[i] -= M * sv_chl[k]


    #unknow_coef = [0]*(len(sv_chl))
    #unknow_coef[n] = sv_chl[n]/matr[n][n]
    
    #столбец неизвестных коэффициентов
    '''
    matr_new = []
    for row in matr:
        row_list = []
        for i in row:
            row_list.append(float(i))
        matr_new.append(row_list)
    
    sv_chl = [float(x) for x in sv_chl]
    unknow_coef = solve(matr_new, sv_chl)
    '''
    unknow_coef = [0]*n
    unknow_coef.append(sv_chl[n]/matr[n][n])
    for i in range(n - 1, -1, -1):
        s = 0
        for j in range(i, n+1):
            s +=  unknow_coef[j]*matr[i][j]
        
        unknow_coef[i] = (sv_chl[i] - s)/matr[i][i]
    
    return unknow_coef
    

#значение полинома при заданном x
def aprox(unknow_coef, X, n, ap):
    
    y = []
    for i in range(ap):
        y_ = 0
        for j in range(n+1):
            y_ += unknow_coef[j]*X[i]**j
        y.append(y_)
    return y

def squares_method(X, Y, ap, n):

    matr, sv_chl = readmatrix(X, Y, ap, n)
    #print(f"Квадратная матрица: {matr}")
    #print(f"Столбец свободных членов: {sv_chl}")
    diagonal(matr, sv_chl, n)
    unknow_coef = find_unknow_coef(matr, sv_chl, n)

   # print(f"неизвестные коэффициенты: {unknow_coef}")
    y = aprox(unknow_coef, X, n, ap)
    #print(f"Значения аппроксимирующего полинома: {y}")

    '''
    plt.figure(1)
    plt.scatter(X, Y)
    plt.plot(X, y)
    plt.show()
    '''

    return y
'''
X = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Y = [5, 10, 4, 0, 7, 10, 11, 1, 13, 7]
ap = len(X)
n = 3
squares_method(X, Y, ap, n)
'''