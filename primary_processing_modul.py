import pandas as pd
from geopy.distance import geodesic
import datetime
import my_func_modul


#функция для разделения датасета по дням. возвращает датафрейм n-го дня
def days_divide(data, day):
    #списки со значениеями дата+время и дистанция от предыдущей точки
    datetime_list = []
    distance_list = [0]

    #проходим по кажой точке 
    for i in range(len(data.unixtimestamp)):
        #Конвертируем Unix-time в читаемый формат, составляем список с датой + временем 
        datetime_list.append(datetime.datetime.fromtimestamp(data.unixtimestamp[i]))
        if i > 0:
            #составляем список дистанций между текущей и предыдущей координатой в метрах
            distance_list.append(geodesic((data.Y[i], data.X[i]), (data.Y[i - 1], data.X[i - 1])). m)
            
    data['datetime'] = datetime_list#столбец даты и времени
    data["Date"] = data['datetime'].dt.date#столбец даты отдельно
    data["Time"] = data['datetime'].dt.time#столбец времени отдельно
    data['Distance'] = distance_list#столбец с дистанцией между соседними точками
    #print(data)
    data["Date"] = pd.to_datetime(data["Date"]) 
    if (day == -1):
        data_day = input('Данные за день: ')#ввод необходимого для отчёта дня по дате
    else: 
        data_day = day
    return data[data['Date'] == data_day].reset_index(drop=True)


#функция детекции поездок. возвращает список с списками индексов поездок 
def ride_on_detection(data):
    notsort_list_of_rides = []#список со всеми индексами тех точек, в которых зафиксирована поездка
    distance_ = 1
    for i in range(len(data.DUT)):
        
        if (i > 1) and (i < len(data.DUT) - 1):
            #условие поездки:
            #1.если включено зажигание в текущей и предыдущей точке
            #2.меняется координата по сравнению с предыдущей точкой 
            #3.Средняя скорость текущего и двух соседних точек больше порога (5)
            #DUT = 42.599999999999994 литров эквиваленто 260 мВ, что соответствует дефолтному значению ДУТ при выключенном зажигании (по крайней мере на этом дачике)
            distance_ = geodesic((data.Y[i], data.X[i]), (data.Y[i - 1], data.X[i - 1])). m
            if (
                ((data.Enable[i] == 1) and ((data.Enable[i - 1] == 1) 
            or (data.Enable[i - 2] == 1)) and (data.DUT[i] != 0) and (data.DUT[i] != 42.599999999999994)) 
            or ((distance_ > 0) and (data.DUT[i] != 0) and (data.DUT[i] != 42.599999999999994)) 
            or ((data.Speed[i] + data.Speed[i - 1] + data.Speed[i + 1])/3 > 5 and (data.DUT[i] != 0) and (data.DUT[i] != 42.599999999999994))
            ):
            #добавляем индекс в список поездок
                notsort_list_of_rides.append(i)
        if i == 0:
            if data.Enable[i] == 1:
                notsort_list_of_rides.append(i)

    #если поездки были
    if len(notsort_list_of_rides) != 0:
        sort_list = my_func_modul.split(notsort_list_of_rides)#разбиваем список на список списков, т.е напр. [[5,6,7,8,9],[31,32,33,34], ...]
        new_sort_list = [i for i in sort_list if (len(i) > 10)]#удаляем слишком короткие поездки (меньше 10ти точек)
        
        return new_sort_list#возвращаем отсортированный по отдельным поездкам список
    else:
        return []
