#импорт необходимых библиотек и модулей
import numpy as np
from statistics import mean
import mnk_moduls
import tensorflow as tf
import math
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS


"""
Функция isolation_forest реализует алгоритм поиска аномалий Isolation Forest,
принимает: список с значениями моментов времени time_list 
            и список с соответствующими показателями датчика уровня топлива dut_list
возвращает: список с моментами времени зафиксированных выбросов,
            список с соответствующими значениями аномальных показателей уровня топлива,
            список с значениями scores, характеризующих аномальность точки
            и значение процента зафиксированных выбросов в общем объёме переданных данных
"""
def isolation_forest(time_list, dut_list):
    # приведение списка значений уровня топлива к массиву нормированных значений
    dut_array = np.array(dut_list)
    dut_array = dut_array.reshape(-1, 1)
    scaler = StandardScaler()
    scaled_dut = scaler.fit_transform(dut_array)

    # создание модели Isolation Forest с количеством деревьев n_estimators=100
    isolation_forest_model = IsolationForest(n_estimators=100, contamination='auto')
    isolation_forest_model.fit(scaled_dut)# обучение модели
    anomaly_status_list = isolation_forest_model.predict(scaled_dut)# фиксация аномальных значений
    scores = isolation_forest_model.score_samples(scaled_dut)# расчёт показателей аномальности scores

    # формирование списков зафиксированных аномальных значений
    enjections_dut = []
    enjections_time = []
    for i in range(len(anomaly_status_list)):
        if anomaly_status_list[i] == -1:
            enjections_dut.append(dut_list[i])
            enjections_time.append(time_list[i])

    enj_proc = len(enjections_dut) /len(dut_list) * 100# процент зафиксированных выбросов

    return enjections_time, enjections_dut, scores, enj_proc


"""
Функция mnk реализует алгоритм поиска аномалий на основе стандартного отклонения и  
                                                    аппроксимации методом наименьших квадратов,
принимает: список с значениями моментов времени time_list, 
           список с соответствующими показателями датчика уровня топлива dut_list,
           параметр k, определяющий порог фиксации значения как выброса

возвращает: список с моментами времени зафиксированных выбросов,
            список с соответствующими значениями аномальных показателей уровня топлива
            и значение процента зафиксированных выбросов в общем объёме переданных данных
"""
def mnk(time_list, dut_list, k):
    approx_y = mnk_moduls.least_squares(time_list, dut_list, 1)# аппроксимация исходных данных
    rejection_list = mnk_moduls.rejections_list(time_list, dut_list, approx_y)# расчёт отклонений точек от аппроксимирующего полинома
    time_after, dut_after, enjections = mnk_moduls.enjection_filter(time_list, dut_list, rejection_list, k)# детекция выбросов на основе отклонений

    # формирование списков зафиксированных аномальных значений
    enjections_dut = []; enjection_time = []
    for i in enjections:
        enjections_dut.append(dut_list[i])
        enjection_time.append(time_list[i])

    enjection_proccents = len(enjections)/len(time_list) * 100# процент зафиксированных выбросов

    return enjection_time, enjections_dut, enjection_proccents


"""
Функция knn реализует алгоритм k-ближайших соседей
принимает: список с значениями моментов времени time_list, 
           список с соответствующими показателями датчика уровня топлива dut_list,
           метод (threshold или std (стандартное отклонение))
           параметр k, определяющий попрог фиксации значения как выброса

возвращает: список с моментами времени зафиксированных выбросов,
            список с соответствующими значениями аномальных показателей уровня топлива,
            список с рассчитанными средними расстояниями до k соседей для каждой точки
            и значение процента зафиксированных выбросов в общем объёме переданных данных
"""
def knn(time_list, dut_list, method, thrashold=90, k=2.5):
    dut_list = np.array(dut_list)
    dut_list = dut_list.reshape(-1, 1)
    # создание и обучение модели k-ближайших соседей с k равным половине точек исследуемой выборки
    knn = NearestNeighbors(n_neighbors=(len(dut_list)//2))
    knn.fit(dut_list)

    # формирование списка средних расстояний
    dist, idxs = knn.kneighbors(dut_list)
    avg_list = [mean(sublist) for sublist in dist]

    # threshold, медиана и стандартное отклонение
    threshold = np.percentile(avg_list, thrashold)
    md = np.median(avg_list)
    std = np.std(avg_list)

    # формирование списков зафиксированных аномальных значений
    enjections_dut = []
    enjections_time = []
    for i in range(len(avg_list)):
        # если выбран метод threshold
        if (method == "threshold"):
            if (avg_list[i] > threshold):
                enjections_dut.append(dut_list[i])
                enjections_time.append(time_list[i])
        # если выбран метод на основе стандартного отклонения
        elif (method == "std"):
            if (avg_list[i] - md > k*std):
                enjections_dut.append(dut_list[i])
                enjections_time.append(time_list[i])
    enjection_proccent = len(enjections_dut)/len(dut_list) * 100# процент зафиксированных выбросов

    return enjections_time, enjections_dut, avg_list, enjection_proccent

#DBSCAN
def dbscan(time_list, dut_list, epsi, min_samples_):
    dbscan = DBSCAN(eps=epsi, min_samples=min_samples_)
    dut_array = np.array(dut_list)
    dut_array = dut_array.reshape(-1, 1)
    dbscan.fit(dut_array)
    labels = dbscan.fit_predict(dut_array)
    enjections_dut = []
    enjections_time = []
    values_null = []
    time_null = []
    for i in range(len(labels)):
        if labels[i] == -1:
            enjections_dut.append(dut_list[i])
            enjections_time.append(time_list[i])
        if labels[i] == 0:
            values_null.append(dut_list[i])
            time_null.append(time_list[i])
    enj_proc = len(enjections_dut) /len(dut_list) * 100

    return enjections_time, enjections_dut, enj_proc, time_null, values_null, labels

"""
Функция dbscan_auto реализует алгоритм dbscan с автоматическим подбором размеров кластера и величины эпсилон-окрестности
принимает: список с значениями моментов времени time_list, 
           список с соответствующими показателями датчика уровня топлива dut_list,

возвращает: список с моментами времени зафиксированных выбросов,
            список с соответствующими значениями аномальных показателей уровня топлива,
            список с привязкой каждой точки к определенному кластеру
            и значение процента зафиксированных выбросов в общем объёме переданных данных
"""
def dbscan_auto(time_list, dut_list):
    # Преобразуем входящий список в numpy массив
    dut_array = np.array(dut_list)
    dut_array = dut_array.reshape(-1, 1)
    # Подбор оптимального значения минимального размера кластера и эпсилон-окрестности
    optics = OPTICS(metric='euclidean', xi=0.7, n_jobs=-1)
    optics.fit(dut_array)
    # Получаем оптимальные параметры eps и min_samples
    eps = optics.get_params()['xi']
    min_samples = optics.get_params()['min_samples']
    # Создание и обучение модели алгоритма DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(dut_array)
    labels = dbscan.fit_predict(dut_array)# список с привязкой точек к кластерам

    # формирование списков зафиксированных аномальных значений
    enjections_dut = []
    enjections_time = []
    for i in range(len(labels)):
        if labels[i] == -1:
            enjections_dut.append(dut_list[i])
            enjections_time.append(time_list[i])
    enj_proc = len(enjections_dut) / len(dut_list) * 100# процент зафиксированных выбросов

    return enjections_time, enjections_dut, labels, enj_proc

"""
Функция combo_methods реализует совместное применение четырёх алгоритмов поиска аномалий:
            mnk, isolation_forest, knn и dbscan
            За итоговые выбросы принимаются точки, принятые за аномальные как минимум в 3-ёх из 4-ёх алгоритмов.
Принимает: список с значениями моментов времени time_list, 
           список с соответствующими показателями датчика уровня топлива dut_list,

Возвращает: список с моментами времени зафиксированных выбросов,
            список с соответствующими значениями аномальных показателей уровня топлива
            и значение процента зафиксированных выбросов в общем объёме переданных данных
"""
def combo_methods(time, dut):
    #Функция, сравнивающая результаты 4-ёх алгоритмов, принимает за выбросы точки, принятые за аномальные в как минимум n алгоритмах
    def methods_comparer(time1, enj1, time2, enj2, time3, enj3, time4, enj4, n):
        time_res = []
        enj_res = []
        # Формируем пары (время, значение уровня топлива) из переданных списков
        pairs = list(zip(time1 + time2 + time3 + time4, enj1 + enj2 + enj3 + enj4))
        pairs1 = list(zip(time1, enj1))
        pairs2 = list(zip(time2, enj2))
        pairs3 = list(zip(time3, enj3))
        pairs4 = list(zip(time4, enj4))
        res_pairs = []
        # Проходим по каждой точке
        for pair in pairs:
            counter = 0
            for i in [pairs1, pairs2, pairs3, pairs4]:
                if pair in i: counter += 1# если значение аномальное то увеличиваем счётчик counter
            # Если counter >= n (т.е. точка выброс как минимум в n алгоритмах), то принимаем её за аномальную
            if (counter >= n and pair not in res_pairs):
                res_pairs.append(pair)
                time_res.append(pair[0])
                enj_res.append(pair[1])
        return time_res, enj_res

    time_mnk, enj_mnk, enj_proc_mnk = mnk(time, dut, k=2)# считаем выбросы mnk
    time_isf, enj_isf, scores_isf, enj_proc_isf = isolation_forest(time_list=time, dut_list=dut)# считаем выбросы isolation_forest
    time_kn, enj_kn, avg_kn, enj_proc_kn = knn(time_list=time, dut_list=dut, method="std", k=2)# считаем выбросы knn
    time_clr, enj_clr, enj_proc_clr, labels_clr = dbscan_auto(time_list=time, dut_list=dut)# считаем выбросы dbscan

    # Считаем итоговые выбросы на основе результатов 4-ёх алгоритмов
    time_res, dut_res = methods_comparer(time_mnk, enj_mnk,
                                         time_isf, enj_isf,
                                         time_kn, enj_kn,
                                         time_clr, enj_clr, n=3)
    proc = len(dut_res) / len(dut) * 100# процент зафиксированных выбросов

    return time_res, dut_res, proc

"""
Функция neural_network реализует детекцию аномалий на основе обученной RNN LSTM модели
Принимает: список с значениями моментов времени time_list, 
           список с соответствующими показателями датчика уровня топлива и скорости  - dut_list и speed_list
        
Возвращает: список с моментами времени зафиксированных выбросов,
            список с соответствующими значениями аномальных показателей уровня топлива
            и значение процента зафиксированных выбросов в общем объёме переданных данных
"""
def neural_network(time, dut, speed):

    # Функция расчёта расстояния между двумя точками
    def distance(point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    # Функция для переданной точки point формирует список расстояний до k ближайших точек из списка points
    def k_nearest_points(point, points, k):
        distances = [distance(point, p) for p in points]
        distances.sort()
        return distances[0:k]

    # Функция детекции аномалий на основе реальных и предсказанных данных
    def neighbour_enjections(t, time, dut, real_delta_, predict_delta_, k, method):
        real_delta = [list(x) for x in zip(real_delta_, t)]# список с реальными значениями
        predict_delta = [list(x) for x in zip(predict_delta_, t)]# список с предсказанными LSTM моделью значений

        # формирование списка средних расстоний avg_dist каждой точки до k ближайших точек предсказанной выборки
        avg_dist = []
        for i in range(1, len(dut)):
            dist_list = k_nearest_points(real_delta[i - 1], predict_delta, k)
            avg_dist.append(sum(dist_list) / len(dist_list))

        md = np.median(avg_dist)# медиана списка средних расстояний
        std = np.std(avg_dist)# стандартное отклонение списка средних расстояний

        # формирование списков с аномальными значениями
        enjctions_dut = []
        enjections_time = []
        for i in range(1, len(dut)):
            # метод детекции на основе стандартного отклонения
            if (method == "avg"):
                if (avg_dist[i - 1] - md > 2 * std):
                    enjctions_dut.append(dut[i])
                    enjections_time.append(time[i])
            # метод на основе порога threshold
            elif (method == "thrashold"):
                # если расстояние точки до предсказанной выборки больше 5, то принимаем точку за выброс
                if (avg_dist[i - 1] > 5):
                    enjctions_dut.append(dut[i])
                    enjections_time.append(time[i])

        return enjections_time, enjctions_dut

    # функция формирования из исходных реальных значений списков ускорений и модулей колебаний уровня топлива
    def test_transfom(time, dut, speed):
        accelerations_test = []
        deltaDut_test = []
        for j in range(1, len(time)):
            if time[j] != time[j - 1]:
                accelerations_test.append(((speed[j] - speed[j - 1]) / 3.6 / (time[j] - time[j - 1])))
            else:
                accelerations_test.append((speed[j] - speed[j - 1]) / 3.6)

            deltaDut_test.append(abs(dut[j] - dut[j - 1]))

        return accelerations_test, deltaDut_test

    # функция предсказания LSTM моделью значений модуля колебания между соседними точками
    def neural_predict(time, dut, speed):
        model = tf.keras.models.load_model('model_1.h5')# загружаем обученную модель
        accelerations, deltaDut = test_transfom(time, dut, speed)# формируем данные в виде списков ускорений и колебаний уровня топлива
        # формируем входной массив ускорений
        input_array = np.asarray(accelerations)
        input_array = np.expand_dims(input_array, axis=0)
        # предсказание значений модулей колебаний уровня топлива между соседними точками
        predict_delta = model.predict(input_array)
        predict_list = list(predict_delta)[0]
        time_list = [i for i in range(0, len(accelerations))]
        return time_list, deltaDut, predict_list

    # формируем выходные значения нейронной сети
    t, real_delta, predict_list = neural_predict(time=time, dut=dut, speed=speed)
    # формируем списки выбросов на основе выходных значений нейронной сети и реальных значений
    enjections_time, enjctions_dut = neighbour_enjections(t, time, dut, real_delta,
                                                            predict_list, 10, method="thrashold")
    proc = len(enjctions_dut) / len(dut) * 100# процент зафиксированных выбросов

    return t, real_delta, predict_list, enjections_time, enjctions_dut, proc

# Функция производит детекцию аномальности зафиксированного процента выбросов
def check_anomaly(enj_proc, method, p):
    # Нормальные проценты выбросов для каждого алгоритма
    normal_dict = {0: 5.2, 1: 5.74, 2: 6.06, 3: 27.07, 4: 7, 5: 4.2}

    # производим детекцию аномалий на основе превышения нормального процента выбросов в p раз
    ride_anomaly_value = []
    ride_anomaly_multy = []
    ride_anomaly_num = []
    trouble = 0
    for i in range(len(enj_proc)):
        if enj_proc[i] > p * normal_dict[method]:
            ride_anomaly_num.append(i + 1)
            ride_anomaly_value.append(enj_proc[i])
            ride_anomaly_multy.append(round(enj_proc[i] / normal_dict[method], 2))

    proc = len(ride_anomaly_num)/len(enj_proc)# процент аномальных поездок за день
    # статус аномалии trouble (0, 0.5 или 1) в зависимости от  процента аномальных поездок за день (proc)
    if proc >= 0.5: trouble = 1
    elif (proc > 0 and proc < 0.5): trouble = 0.5

    return ride_anomaly_num, ride_anomaly_value, ride_anomaly_multy, trouble