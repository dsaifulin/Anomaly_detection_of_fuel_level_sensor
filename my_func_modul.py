import matplotlib.pyplot as plt
import random
import numpy as np
import datetime


def time_converter(unix_list):
    datetime_list = []
    for unix_time in unix_list:
        datetime_list.append(datetime.datetime.fromtimestamp(unix_time))
    return datetime_list
def date_transform(day, month, year):
    day_str = str(day); month_str = str(month); year_str = str(year)
    if day < 10:
        day_str = f"0{day}"
    if month < 10:
        month_str = f"0{month}"
    return year_str + "-" + month_str + "-" + day_str

def random_enjections(dut, procEnjections):
    nEnjections = int(procEnjections * len(dut))
    #minD = min(dut); maxD = max(dut)
    minD = 0; maxD = 50
    random_list = [np.random.randint(minD, maxD) for i in range(nEnjections)]
    print(random_list)
    for random_value in random_list:
        random_index = random.randint(0, len(dut))
        for i_dut in range(1, len(dut)):
            if (i_dut == random_index):
                dut[i_dut] = random_value
    return dut


#разбивает список на список списков 
def split(l):
    res = [[l[0]]]
    last = l[0]
    for i in l[1:]:
        if i - last == 1:
            res[-1].append(i)
        else:
            res.append([i])
        last = i
    return res
#разбивает список списков на один список
def listmerge(lstlst):
    all=[]
    for lst in lstlst:
      all.extend(lst)
    return all

#функция выводит точки поездок за день 
def rides_show(data_day, rides_list):
    for i in rides_list:
        time = []
        for j in i:
            #список с индексами поездки
            time.append(data_day.unixtimestamp[j])
        #print(time)
        #график поездки ввиде точек, зависимость (время, ДУТ)
        plt.scatter(time, list(data_day.DUT[i[0]:i[-1]+1]))

#фильтр для приведения значений ДУТ во время стоянки к корректным (по умолчанию при выключенном зажигании у этого автомобиля - 260)
#идея в том, что во время стоянок уровень топлива не меняется. И для каждой стоянки мы берём значение уровня топлива последнего с прошлой поездки
#данный фильтр нужен только для корректного представления данных на графике и в фильтрации поездок не играет никакой роли
#на входе - датафрейм по дню, список с индексами последующей поездки, крайнее значение ДУТ последней поездки и индекс этого значения в общем датафрейме
def enable_off_filter(data, ride, last_previous, last_previous_index):
    for i in range(last_previous_index, ride[0]):#c крайнего предыдущей поездки до первого последующей
        data.DUT.loc[i] = last_previous#значение ДУТ = крайнему значению ДУТ последней поездки