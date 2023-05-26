from pandas import Series
import pandas as pd
import numpy as np 
import datetime
from geopy.distance import geodesic
from scipy import stats
import my_func_modul


#функция расчёта пробега за поездку, возвращает дистанцию в метрах
def distance_ride(data, ride):
    distance = 0
    #выделяем датафрейм с координатам данной поездки, сглаживаем координаты, убирая выбросы
    df_dist = data[['Y', 'X']][ride[0]:ride[-1]][(np.abs(stats.zscore(data[['Y', 'X']][ride[0]:ride[-1]])) < 1).all(axis=1)]
    #нумеруем индексы полученного датафрейма с нуля
    df_dist.index = Series(list(range(0, len(df_dist))))
    #проходим по датафрейму с координатами, считаем расстояние между текущей 
    # и предыдущей координатой, суммируем результат с итоговой дистанцие
    for i in range(len(df_dist)):
        if i > 0:
            distance += geodesic((df_dist.Y[i], df_dist.X[i]), (df_dist.Y[i - 1], df_dist.X[i - 1])). m
    #возвращаем пробег за поездку в метрах
    return distance

#функция получения первого и последнего значения ДУТ поездки для аппроксимации, используя среднее первых и последних n значений
# возвращает усреднённое первое и последнее значение, и их индексы
def mean_for_rides(data, ride, n):
    if len(ride) >= n:
            first_dut_list = []
            last_dut_list = []
            #заполняем список с первыми n значениями ДУТ поездки
            for i in ride[:n]:
                first_dut_list.append(data.DUT[i])
            #заполняем список с последними n значениями ДУТ поездки
            for j in ride[len(ride) - n:]:
                last_dut_list.append(data.DUT[j])
            #среднее полученных списков
            first_dut_ride = np.mean(first_dut_list)#первое значение поездки
            last_dut_ride = np.mean(last_dut_list)#поеледнее значение поездки 
    else:
        #если длина списка меньше n, то берёс мреднее от 3х значений
        first_dut_ride = np.mean([data.DUT[ride[0]],data.DUT[ride[1], data.DUT[ride[2]]]])
        last_dut_ride = np.mean([data.DUT[ride[-1]],data.DUT[ride[-2],], data.DUT[ride[-3]]])
    #индексы полученных значений
    first_index_ride = ride[0]
    last_index_ride = ride[-1]
    return first_dut_ride, last_dut_ride, first_index_ride, last_index_ride

#функция производит детекцию заправки или слива, сравнивая первое значение ДУТ текущей поездки с последним значением ДУТ предыдущей
def d_value(first_this, last_previous):
    #порог, выше которого - аномалия(слив либо заправка)
    thrash_dvalue = 5
    zapravka = False
    sliv = False
    dvalue = first_this - last_previous
    if abs(dvalue) > thrash_dvalue:
        #значение ДУТ увеличилось -> заправка
        if dvalue > 0:
            zapravka = True
        #Значение ДУТ уменьшилось -> слив
        if dvalue < 0:
            sliv = True
    #возвращаем значения разницы, заправки и сива
    return dvalue, zapravka, sliv

#функция расчёта и коррекции расхода топлива 
#на входе усреднённые первое и последнее изначения ДУТ текущей поездки, а также их моменты времени 
def avg_fuel(first_this, last_this, first_time, last_time):
    thrash_avg_time = 0.004167#порог расхода топлива в секунду

    avg_fuel = first_this - last_this#расход топлива за поездку (первое - последнее значение) (л)
    avg_fuel_time = avg_fuel/(last_time - first_time)#расход топлива в секунду

    #если отрицательный расход топлива за поездку, в поездке проиходит рост уровня топлива -> аномалия 
    if (avg_fuel < 0):
        #меняем последнее значение ДУТ поездки на зеркально противоположное 
        # относительно первого значения ДУТ(чтобы уровень топлива уменьшался)
        last_this = 2*first_this - last_this
        avg_fuel = first_this - last_this#пересчитываем расход
        avg_fuel_time = avg_fuel/(last_time - first_time)#пересчитываем расход в секунду (л/с)
    
    #если расход в секунду больше порога (происходит аномально быстрый рост уровня топлива)
    if abs(avg_fuel_time) > thrash_avg_time:
        #если происходит возрастание уровня топлива
        if avg_fuel < 0:
            last_this = first_this#делаем прямую горизонтальной, компинсируя аномальный рост. (т.е. расход становится равным нулю)
            avg_fuel = first_this - last_this#пересчитываем расход
        #если же аномальный спад уровня топлива
        else:
            #понижаем спад до порогового и пересчитываем расход
            avg_fuel_time = thrash_avg_time
            last_this = first_this - avg_fuel_time*(last_time - first_time)
            avg_fuel = first_this - last_this

    #возвращаем скорректированные (в случае аномалии) первое и последнее значения поездки, расход (л) и расход в секунду
    return first_this, last_this, avg_fuel

#Расчёт времени стоянок, на входе - датафрейм за день и список списков индексов поездок 
def time_of_parcking(data, rides_list):
    parking_index = []
    #df = pd.DataFrame({'Продолжительность стоянки (ч)': [], 'Начало стоянки': [], 'Конец стоянки': []})
    rides_index = my_func_modul.listmerge(rides_list)#объединяем список поездок в один 
    #составляем список индексов стоянок, т.е. индексов, не входящих в список индексов поездок
    for i in range(len(data)):
        if i not in rides_index:
            parking_index.append(i)
    #list_parking_index = split(parking_index)#разделяем на списки по отдельным стоянкам
    #разделяем на списки по отдельным стоянкам и удаляем стоянки с одним значение
    list_parking_index = [i for i in my_func_modul.split(parking_index) if (len(i) > 1)]
    parking_parametres = []
    #расчёт времени стоянки, начало и конца для каждой стоянки 
    for parking in list_parking_index:
        parking_parametres.append([
            (data.unixtimestamp[parking[-1]] - data.unixtimestamp[parking[0]])/3600, data.Time[parking[0]], data.Time[parking[-1]]
            ])
    
    #возвращаем таблицу с стоянками
    return pd.DataFrame(parking_parametres, columns=('Продолжительность стоянки (ч)', 'Начало стоянки', 'Конец стоянки'))

#функция с общими расчётами, формирующая отчёт за день, на входе - датафрейм за день и список с индексами поездок           
def general_calculate_func(data, rides_list):
    #пороговое минимальное и максимальное значение расхода топлива на 100 км
    thrash_charge_fuel_min = 6
    thrash_charge_fuel_max = 15
    #крайнее значение ДУТ с предыдущей поездки, в начале берётся первое значение с данной поездки. 
    # В идеале - брать это значение с последней поездки предыдущего дня, чтобы детектить сливы и заправки, которые могли произойти за ночь
    last_previous = data.DUT[rides_list[0][0]]
    last_previous_index = 0
    #средний расход, пробег, моточасы, средняя скорость за день
    all_avgfuel = 0
    day_distance = 0
    day_time_rides = 0
    avg_speed_day = 0
    #номер анализируемой поездки
    ride_num = 0
    #список с заправками и сливами
    d_value_list = []
    #идём по каждой поездки из списка
    for ride in rides_list:
        ride_num += 1 

        #получаем среднее первое и последнее значение поездки, и их индексы в общем датафрейме
        first_this_dut, last_this_dut, first_this_index, last_this_index = mean_for_rides(data, ride, 10)
        #по полученным значениям рассчитываем расход (в литрах) и, в случае аномалии, корретируем значения на нормальные и пересчитываем
        first_this_dut, last_this_dut, avgfuel = avg_fuel(first_this_dut, last_this_dut, data.unixtimestamp[ride[0]], data.unixtimestamp[ride[-1]])
        #проверяем на запраку и слив, которые могли произойти во время стоянки (сравниваем первое значение данной поездки с последним предыдущей. 
        # если был слив или заправка, dvalue(разница) будет больше порога)
        dvalue, zapravka, sliv = d_value(first_this_dut, last_previous)
        #считаем пробег за поездку
        distance = distance_ride(data, ride)
        #считаем расход (л/100 км)
        mean_charge_fuel = ((avgfuel)/(distance/1000)) * 100
        #моточасы за поездку (мин)
        time_of_ride = (data.unixtimestamp[ride[-1]] - data.unixtimestamp[ride[0]])/60
        #средняя скорость за поездку
        avg_speed = np.mean([data.Speed[ride[0]:ride[-1]]])
        
        #проверка и коррекция, в случае аномалии, расхода (л/100 км)
        
        #если расход не входит в пороговый интервал 
        if (mean_charge_fuel < thrash_charge_fuel_min) or (mean_charge_fuel > thrash_charge_fuel_max):
            
            #пересчитываем расход (л/100 км) через среднюю скорость за поездку
            mean_charge_fuel = 80/avg_speed + 2.5 + 0.000002 * avg_speed ** 3
            #пресчитваем расход (л)
            avgfuel = (mean_charge_fuel * distance/1000)/100
        
        #формируем отчёт за поездку
        print(f'Поездка №{ride_num}')

        #выводим в отчёте сообщение о запраке или сливе
        if zapravka:
            print(f'Заправлено перед поездкой {dvalue}')
            d_value_list.extend(['Заправка', '{} - {}'.format(data.Time[last_previous_index], data.Time[ride[0]]), dvalue])
        elif sliv:
            print(f'Слито перед поездкой {dvalue}')
            d_value_list.extend(['Слив', '{} - {}'.format(data.Time[last_previous_index], data.Time[ride[0]]), dvalue])
        else: 
            print('Заправок и сливов перед поездкой не зафиксировано')
        
        #формируем и выводим таблицу с основными параметрами за поездку
        result_list = [avgfuel, distance/1000, time_of_ride, data.Time[ride[0]], data.Time[ride[-1]], avg_speed, mean_charge_fuel]
        #таблица с параметрами за поездку
        df_results_ride = pd.DataFrame([result_list], columns = ['|Потрачено литров|', '|Пробег (км)|', '|Длительность (мин)|', '|Начало|', '|Конец|', '|Средняя скорость (км/ч)|', '|Расход (л/км)|'])
        print(df_results_ride)
        print('\n')

        #считаем параметры за день
        avg_speed_day += avg_speed
        day_time_rides += (time_of_ride)
        all_avgfuel += avgfuel
        day_distance += (distance/1000)

        #корректируем значения ДУТ стоянок
        my_func_modul.enable_off_filter(data, ride, last_previous, last_previous_index)
        #обновляем индекс и значение последнего с крайней поездки
        last_previous = last_this_dut
        last_previous_index = last_this_index
###############
    #коррекция значений ДУТ последней стоянки, нужно только для корректного график ДУТ 
    if (len(data) - rides_list[-1][-1] > 0):
        for j in range(last_previous_index, len(data)):
            data.DUT[j] = last_previous
#################
    #средний расход за день(л/100 км)
    day_mean_charge_fuel = (all_avgfuel * 100)/(day_distance)
    #таблица с данными по стоянкам 
    df_parcking = time_of_parcking(data, rides_list)

    #формируем отчёт за день
    print('\nОтчёт за день: ')
    print(f'Потрачено за день: {all_avgfuel} литров')
    print(f'Пробег за день: {day_distance} км')
    print('Моточасы {} часов'.format(datetime.timedelta(seconds =day_time_rides*60)))
    print('Средняя скорость за день {} км/ч'.format(avg_speed_day/ride_num))
    print(f'Средний расход за день: {day_mean_charge_fuel} л/100км')

    print('\nЗаправки/Сливы:')
    if d_value_list != []:
        print(pd.DataFrame([d_value_list], columns=('Заправка/Слив','Время', 'Заправлено/слито (л)')))
    else:
        print('Не обнаружено')

    print('\nСтоянки за день:')
    print(df_parcking)