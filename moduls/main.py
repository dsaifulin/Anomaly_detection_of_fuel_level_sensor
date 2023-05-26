import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'
import speed_limitsHERE
import primary_processing_modul
import my_func_modul
import DUT_filters_modul
import day_parameter_calculation_modul


def main():
    
    #загружаем данные за n дней по авто
    mounth_df = pd.read_json('../for_month_Х545СН178.json')
    #именуем колонки датафрейма 
    mounth_df.columns = ['Enable','DUT', 'unixtimestamp','Speed', 'Y', 'X']
    #получаем нужный день
    data_day = primary_processing_modul.days_divide(mounth_df, -1)
    #конвертируем мв ДУТ в литры w
    DUT_filters_modul.litres_conversion(data_day)
    #получаем поездки за день ввиде списка списков индексов общего датафрейма за день
    rides_list_day = primary_processing_modul.ride_on_detection(data_day)
    #если поездки были
    if rides_list_day != []:
        #фильтруем выбросы по поездке в данных ДУТ
       # DUT_filters_modul.enjection_filter(data_day, rides_list_day)
        #бегущее среднее по поездке
        DUT_filters_modul.rides_mean_filter(data_day, rides_list_day, 5)
        #print(data_day)
        ######################
        import squares_method_main
        plt.style.use('ggplot')
        plt.figure()
        plt.plot(squares_method_main.time_converter(data_day.unixtimestamp[rides_list_day[0][0]:rides_list_day[0][-1]]), 
                    data_day.DUT[rides_list_day[0][0]:rides_list_day[0][-1]])
        plt.xlabel("Дата и время поездки")
        plt.ylabel("Уровень топлива, л")
        plt.grid()
        plt.ylim(5, 40)
        plt.show()
        #######################
        #формируем отчёт
        day_parameter_calculation_modul.general_calculate_func(data_day, rides_list_day)
        print('\nПревышения скорости:')
        speed_limitsHERE.speed_limits_with_threads_func(data_day, rides_list_day)#выводим превышения скорости или сообщение об их отсутствии
        #выводим графики 
        #1 график
        plt.figure(1)
        my_func_modul.rides_show(data_day, rides_list_day)#отдельные поездки в виде точек 
        plt.plot(data_day.unixtimestamp, data_day.DUT)#линия изменения уровня топлива/
        plt.ylim([0, 50])
        #2 график
        plt.figure(2)
        plt.plot(data_day.unixtimestamp, data_day.Speed)#скорость от времени
        plt.show()

main()