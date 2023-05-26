import numpy as np 
import pandas as pd

#функция перевода значений ДУТ в литры
def litres_conversion(data):
    #таблица перевода (https://forum.gurtam.com/viewtopic.php?id=7450&p=0)
    litres_table = pd.DataFrame({'litres': [0,  6, 11, 17, 22, 28, 33, 39, 45, 50], 'mV': [2900, 2600, 2300, 1900, 1600, 1200, 900, 500, 100, 0]})
    for i in range(len(data.DUT)):
        count = False
        for j in range(len(litres_table) - 1):
            if (data.DUT[i] <= litres_table.mV[j]) and (data.DUT[i] >= litres_table.mV[j + 1]):
                count = True
                k = (litres_table.mV[j+1] - litres_table.mV[j])/(litres_table.litres[j+1] - litres_table.litres[j])
                b = litres_table.mV[j] - k*litres_table.litres[j]
        
        data.DUT[i] = (data.DUT[i] - b)/k
        #Заменено из-за предупреждения
        # data = data.copy()
        # data.DUT[i] = (data.DUT[i] - b)/k

        if count == False:
            data.DUT[i] = data.DUT[i - 1]

#функция разделяет поездку на 4 части и для каждой устарняет значения, отклоняющиеся от среднего больше стандартного отклонения
def enjection_filter(data, rides_list):
    #рассматриваем каждую поездку из списка поездок
    for ride in rides_list:
        #расчёт среднего(s) и стандартного отклонения(sr) для каждой из 4х равных частей поездки
        s_1 = np.std(list(data.DUT[ride[0]:ride[len(ride)//4]+1]))
        sr_1 = np.mean(list(data.DUT[ride[0]:ride[len(ride)//4]+1]))
        s_2 = np.std(list(data.DUT[ride[len(ride)//4]+1:ride[len(ride)//2]+1]))
        sr_2 = np.mean(list(data.DUT[ride[len(ride)//4]+1:ride[len(ride)//2]+1]))
        s_3 = np.std(list(data.DUT[ride[len(ride)//2]+1:ride[3*len(ride)//4]+1]))
        sr_3 = np.mean(list(data.DUT[ride[len(ride)//2]+1:ride[3*len(ride)//4]+1]))
        s_4 = np.std(list(data.DUT[ride[3*len(ride)//4]+1:ride[-1]+1]))
        sr_4 = np.mean(list(data.DUT[ride[3*len(ride)//4]+1:ride[-1]+1]))
        count = 0
        for i in ride:
            #первая четверть поездки
            if count < len(ride)/4:
                s = s_1
                sr = sr_1
            #вторая четверть поездки
            if (count > len(ride)/4) and (count < len(ride)/2):
                s = s_2
                sr = sr_2
            #третья четверть
            if (count > len(ride)/2) and (count < 3*len(ride)/4):
                s = s_3
                sr = sr_3
            #4я четверть
            if (count > 3*len(ride)/4) and (count < len(ride)):
                s = s_4
                sr = sr_4
            #|данное значение - среднее| > стандартного отклонения -> данное значение - выброс и мы приравниваем его среднему
            if abs(data.DUT[i] - sr) > s:
                data.DUT[i] = sr
            count+=1
#фильтр бегущего среднего от +-n значений для поездок
def rides_mean_filter(data, rides_list, n):
    #анализируем каждую поездку
    for ride in rides_list:
        #переменные, которые нужны в тех случая когда расстояние от текущей точки, от которой считается среднее, до конца(начала) поездки меньше n
        #в этих случая, соответсвенно, надо брать меньше значений с одной строны
        begin_n = 0
        end_n = n
        #список усреднённых значений ДУТ текущей поездки
        dut_one_ride_list = []

        #считаем сумму и далее средне +-n значений от текущего
    dut_rides_list = []
    for ride in rides_list:
        begin_n = 0
        end_n = n
        dut_one_ride_list = []
        for i in range(len(ride)):
            sum_dut = 0
            for j in ride[i - begin_n: i + end_n + 1]:
                sum_dut = sum_dut + data.DUT[j]
            mean_dut = sum_dut/(begin_n + end_n + 1)
            #увеличиваем данную переменную при удалении от начала поездки
            if i < n:
                begin_n += 1
            #уменьшаем данную переменную при приближении к началу поездки
            if i > len(ride) - n - 2:
                end_n =- 1
            dut_one_ride_list.append(mean_dut)#обновляем список со средними
        #присваиваем столбцу ДУТ датафрейма значения из списка
        data.DUT[ride[0]:ride[-1]+1] = dut_one_ride_list
    
    