import numpy as np
from pandas import Series
from geopy.distance import geodesic
from scipy import stats
import pandas as pd

import DUT_filters_modul
import my_func_modul
import datetime
import matplotlib.pyplot as plt

# Функция удаляет выбросы из общего датафрейма данных для корректности дальнейшего отчёта
def enjection_filter(data_day, rides_list, enj):
    filterd_rides = []
    count = 0
    for ride in rides_list:
        f_ride = []
        for i in ride:
            if not (data_day.unixtimestamp[i] in enj[count][0] and data_day.DUT[i] in enj[count][1]):
                f_ride.append(i)
        filterd_rides.append(f_ride)
        count += 1
    return filterd_rides

# функция получения первого и последнего значения ДУТ поездки для аппроксимации, используя среднее первых и последних n значений
# возвращает усреднённое первое и последнее значение, и их индексы
def mean_for_rides(data, ride, n):
    if len(ride) >= n:
            first_dut_list = []
            last_dut_list = []
            # заполняем список с первыми n значениями ДУТ поездки
            for i in ride[:n]:
                first_dut_list.append(data.DUT[i])
            # заполняем список с последними n значениями ДУТ поездки
            for j in ride[len(ride) - n:]:
                last_dut_list.append(data.DUT[j])
            # среднее полученных списков
            first_dut_ride = np.mean(first_dut_list)# первое значение поездки
            last_dut_ride = np.mean(last_dut_list)# последнее значение поездки
    else:
        # если длина списка меньше n, то берём среднее от 3х значений
        first_dut_ride = np.mean([data.DUT[ride[0]],data.DUT[ride[1], data.DUT[ride[2]]]])
        last_dut_ride = np.mean([data.DUT[ride[-1]],data.DUT[ride[-2],], data.DUT[ride[-3]]])
    # индексы полученных значений
    first_index_ride = ride[0]
    last_index_ride = ride[-1]
    return first_dut_ride, last_dut_ride, first_index_ride, last_index_ride


# функция расчёта и коррекции расхода топлива
# на входе усреднённые первое и последнее изначения ДУТ текущей поездки, а также их моменты времени
def avg_fuel(first_this, last_this, first_time, last_time):
    thrash_avg_time = 0.004167 # порог расхода топлива в секунду

    avg_fuel = first_this - last_this# расход топлива за поездку (первое - последнее значение) (л)
    avg_fuel_time = avg_fuel / (last_time - first_time)# расход топлива в секунду

    # если отрицательный расход топлива за поездку, в поездке происходит рост уровня топлива -> аномалия
    if (avg_fuel < 0):
        # меняем последнее значение ДУТ поездки на зеркально противоположное
        # относительно первого значения ДУТ (чтобы уровень топлива уменьшался)
        last_this = 2 * first_this - last_this
        avg_fuel = first_this - last_this  # пересчитываем расход
        avg_fuel_time = avg_fuel / (last_time - first_time)  # пересчитываем расход в секунду (л/с)

    # если расход в секунду больше порога (происходит аномально быстрый рост уровня топлива)
    if abs(avg_fuel_time) > thrash_avg_time:
        # если происходит возрастание уровня топлива
        if avg_fuel < 0:
            last_this = first_this  # делаем прямую горизонтальной, компенсируя аномальный рост. (т.е. расход становится равным нулю)
            avg_fuel = first_this - last_this  # пересчитываем расход
        # если же аномальный спад уровня топлива
        else:
            # понижаем спад до порогового и пересчитываем расход
            avg_fuel_time = thrash_avg_time
            last_this = first_this - avg_fuel_time * (last_time - first_time)
            avg_fuel = first_this - last_this

    # возвращаем скорректированные (в случае аномалии) первое и последнее значения поездки, расход (л) и расход в секунду
    return first_this, last_this, avg_fuel

#функция производит детекцию заправки или слива, сравнивая первое значение ДУТ текущей поездки с последним значением ДУТ предыдущей
def d_value(first_this, last_previous):
    #порог, выше которого - аномалия (слив либо заправка)
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

#функция расчёта пробега за поездку, возвращает дистанцию в метрах
def distance_ride(data, ride):
    distance = 0
    #выделяем датафрейм с координатами данной поездки, сглаживаем координаты, убирая выбросы
    df_dist = data[['Y', 'X']][ride[0]:ride[-1]][(np.abs(stats.zscore(data[['Y', 'X']][ride[0]:ride[-1]])) < 1).all(axis=1)]
    #нумеруем индексы полученного датафрейма с нуля
    df_dist.index = Series(list(range(0, len(df_dist))))
    # проходим по датафрейму с координатами, считаем расстояние между текущей
    # и предыдущей координатой, суммируем результат с итоговой дистанцией
    for i in range(len(df_dist)):
        if i > 0:
            distance += geodesic((df_dist.Y[i], df_dist.X[i]), (df_dist.Y[i - 1], df_dist.X[i - 1])). m
    #возвращаем пробег за поездку в метрах
    return distance


# Расчёт времени стоянок, на входе - датафрейм за день и список списков индексов поездок
def time_of_parcking(data, rides_list):
    parking_index = []
    rides_index = my_func_modul.listmerge(rides_list)  # объединяем список поездок в один
    # составляем список индексов стоянок, т.е. индексов, не входящих в список индексов поездок
    for i in range(len(data)):
        if i not in rides_index:
            parking_index.append(i)
    # разделяем на списки по отдельным стоянкам и удаляем стоянки с одним значением
    list_parking_index = [i for i in my_func_modul.split(parking_index) if (len(i) > 1)]
    parking_parametres = []
    # расчёт времени стоянки, начало и конца для каждой стоянки
    for parking in list_parking_index:
        parking_parametres.append([
            (data.unixtimestamp[parking[-1]] - data.unixtimestamp[parking[0]]) / 3600, data.Time[parking[0]],
            data.Time[parking[-1]]
        ])

    # возвращаем таблицу с стоянками
    return pd.DataFrame(parking_parametres,
                        columns=('Продолжительность стоянки (ч)', 'Начало стоянки', 'Конец стоянки'))

def general_calculate_func(data,
                           rides_list_,
                           enj):

    rides_list = enjection_filter(data, rides_list_, enj)
    thrash_charge_fuel_min = 6
    thrash_charge_fuel_max = 15
    last_previous = data.DUT[rides_list[0][0]]
    last_previous_index = 0
    all_avgfuel = 0
    day_distance = 0
    day_time_rides = 0
    avg_speed_day = 0
    # номер анализируемой поездки
    ride_num = 0
    # список с заправками и сливами
    avg_dut_list = [["№ Поездки", "Начало", "Конец", "Портрачено (л)", "Расход (л/км)"]]
    d_value_list = [["№ Поездки", "Детекция изменения", "Промежуток времени", "Слито/заправлено (л)"]]
    run_list = [["№ Поездки", "Начало", "Конец", "Время в пути", "Пробег (км)", "Ср.ск-ть(км/ч)"]]
    # идём по каждой поездки из списка
    for ride in rides_list:
        ride_num += 1
        # получаем среднее первое и последнее значение поездки, и их индексы в общем датафрейме
        first_this_dut, last_this_dut, first_this_index, last_this_index = mean_for_rides(data, ride, 10)

        # по полученным значениям рассчитываем расход (в литрах) и, в случае аномалии, корретируем значения на нормальные и пересчитываем
        first_this_dut, last_this_dut, avgfuel = avg_fuel(first_this_dut, last_this_dut, data.unixtimestamp[ride[0]],
                                                          data.unixtimestamp[ride[-1]])

        # проверяем на запраку и слив, которые могли произойти во время стоянки (сравниваем первое значение данной поездки с последним предыдущей).
        # если был слив или заправка (dvalue(разница) будет больше порога)
        dvalue, zapravka, sliv = d_value(first_this_dut, last_previous)
        # считаем пробег за поездку
        distance = distance_ride(data, ride)
        # считаем расход (л/100 км)
        mean_charge_fuel = ((avgfuel) / (distance / 1000)) * 100
        # моточасы за поездку (мин)
        time_of_ride = (data.unixtimestamp[ride[-1]] - data.unixtimestamp[ride[0]]) / 60
        # средняя скорость за поездку
        avg_speed = np.mean([data.Speed[ride[0]:ride[-1]]])
        # проверка и коррекция, в случае аномалии, расхода (л/100 км)
        # если расход не входит в пороговый интервал
        if (mean_charge_fuel < thrash_charge_fuel_min) or (mean_charge_fuel > thrash_charge_fuel_max):
            # пересчитываем расход (л/100 км) через среднюю скорость за поездку
            mean_charge_fuel = 80 / avg_speed + 2.5 + 0.000002 * avg_speed ** 3
            # пресчитваем расход (л)
            avgfuel = (mean_charge_fuel * distance / 1000) / 100

        # Обновляем список для таблицы расхода топлива
        avg_dut_list.append([ride_num, data.Time[ride[0]], data.Time[ride[-1]], round(avgfuel, 2), round(mean_charge_fuel, 2)])

        # Обновляем список для заправок и сливов топлива
        if zapravka:
            d_value_list.append(
                [ride_num, 'Заправка', '{} - {}'.format(data.Time[last_previous_index], data.Time[ride[0]]), dvalue])
        elif sliv:
            d_value_list.append([ride_num, 'Слив', '{} - {}'.format(data.Time[last_previous_index], data.Time[ride[0]]), dvalue])
        else:
            d_value_list.append([ride_num, 'Не зафиксировано', ' - ', ' - '])

        # Обновляем список для таблицы пробега
        run_list.append([ride_num, data.Time[ride[0]], data.Time[ride[-1]], round(time_of_ride, 2), round(distance / 1000, 2), round(avg_speed, 2)])

        # считаем параметры за день
        avg_speed_day += avg_speed
        day_time_rides += time_of_ride
        all_avgfuel += avgfuel
        day_distance += (distance / 1000)

        # корректируем значения ДУТ стоянок
        my_func_modul.enable_off_filter(data, ride, last_previous, last_previous_index)

        # обновляем индекс и значение последнего с крайней поездки
        last_previous = last_this_dut
        last_previous_index = last_this_index

    # коррекция значений ДУТ последней стоянки
    if (len(data) - rides_list[-1][-1] > 0):
        for j in range(last_previous_index, len(data)):
            data.DUT[j] = last_previous

    # средний расход за день(л/100 км)
    day_mean_charge_fuel = (all_avgfuel * 100) / day_distance
    # таблица с данными по стоянкам
    df_parcking = time_of_parcking(data, rides_list)

    day_report = (("Показатель", "Значение"),
                  ("Потрачено (л)", round(all_avgfuel, 2)),
                  ("Пробег (км)", round(day_distance, 2)),
                  ("Моточасы (ч)", datetime.timedelta(seconds =day_time_rides*60)),
                  ("Средняя скорость (км/ч)", round((avg_speed_day / ride_num), 2)),
                  ("Средний расход топлива", round(day_mean_charge_fuel, 2)))

    # Данные для формирования итоговых таблиц
    fuel_table = tuple(tuple(sublist) for sublist in avg_dut_list)
    d_table = tuple(tuple(sublist) for sublist in d_value_list)
    run_table = tuple(tuple(sublist) for sublist in run_list)

    # График изменения уровня топлива за день
    figure_avg = plt.figure(figsize=(12, 6))
    axes = figure_avg.add_subplot()
    axes.scatter(my_func_modul.time_converter(data.unixtimestamp), data.DUT)
    axes.grid(True)
    axes.set_xlabel("Время")
    axes.set_ylabel("Уровень топлива, л")
    axes.set_title("Динамика уровня топлива за день")
    axes.set_ylim(0, 50)


    return [day_report, fuel_table, d_table, run_table, figure_avg]
