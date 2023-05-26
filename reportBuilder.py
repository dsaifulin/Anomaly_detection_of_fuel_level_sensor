import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import DUT_filters_modul
import primary_processing_modul
import enjMethods as enj
import my_func_modul as myf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from fpdf import FPDF
import day_report_fuel

def report_builder_to_pdf(day="2022-03-02",
                  method=0):
    method_str = None
    df = pd.read_json('for_month_Х545СН178.json')
    df.columns = ['Enable','DUT', 'unixtimestamp','Speed', 'Y', 'X']
    data_day = primary_processing_modul.days_divide(df, day)
    DUT_filters_modul.litres_conversion(data_day)
    rides_list_day = primary_processing_modul.ride_on_detection(data_day)
    if rides_list_day == []:
        print("Тут должно выскочить окошко")
        return -1
    ride_counter = 1
    figures = []; rides_strs = []; proc_strs = []; res = []
    if rides_list_day != []:
        for ride in rides_list_day:
            time = list(data_day.unixtimestamp[ride[0]:ride[-1]])
            dut = list(data_day.DUT[ride[0]:ride[-1]])
            dut = myf.random_enjections(dut, 0)
            #Комбинация методов
            if method == 1:
                time_res, dut_res, proc = enj.combo_methods(time, dut)
                method_str = "Совметная работа методов"
            #МНК
            elif method == 2:
                time_res, dut_res, proc = enj.mnk(time, dut, k=2)
                method_str = "Применение аппроксимации и стандртного отклонения"
            #ISF
            elif method == 3:
                time_res, dut_res, scores_isf, proc = enj.isolation_forest(time, dut)
                method_str = "Isolation Forest"
            #KNN
            elif method == 4:
                time_res, dut_res, scores_knn, proc = enj.knn(time, dut, method="std", k=2)
                method_str = "KNN (k-ближайших соседей)"
            #DBSCAN
            elif method == 5:
                time_res, dut_res, scores_dbscan, proc = enj.dbscan_auto(time, dut)
                method_str = "DBSCAN"
            #LSTM
            elif method == 0:
                speed = list(data_day.Speed[ride[0]:ride[-1]])
                t, real_delta, predict_list, time_res, dut_res, proc = enj.neural_network(time, dut, speed)
                method_str = "RN LSTM"
            res.append([time_res, dut_res])

            #Для таблицы выбросов
            rides_strs.append(ride_counter); proc_strs.append(round(proc, 2))
            #Графики детекция аномалий
            figure = plt.figure(figsize=(12, 6))
            axes = figure.add_subplot()
            axes.scatter(myf.time_converter(time), dut, c='blue')
            axes.scatter(myf.time_converter(time_res), dut_res, c='red')
            axes.grid(True)
            axes.set_xlabel("Время")
            axes.set_ylabel("Показание датчика уровня топлива")
            axes.set_title(
                f"{day}\nПоездка №{ride_counter}\nКоличество выбросов: {round(proc, 2)} %")
            figures.append(figure)
            ride_counter += 1

    table_proc = tuple([tuple(["№ Поездки", "Процент выбросов"])] + list(map(tuple, zip(*[rides_strs, proc_strs]))))
    anomaly_num, anomaly_value, anomaly_multy, trouble_status = enj.check_anomaly(proc_strs, method, 2)
    anomaly_status = [anomaly_num, anomaly_value, anomaly_multy, trouble_status]
    day_report_lists = day_report_fuel.general_calculate_func(data_day, rides_list_day, res)

    return [figures, table_proc, method_str, method, anomaly_status, day_report_lists]

def graphs_pdf(figures, file_name):
    pdfFile = PdfPages(f"{file_name}.pdf")
    for figure in figures:
        pdfFile.savefig(figure)
    pdfFile.close()


def text_pdf(day, result, file_name,
             fuelReport=False,
             fuelChangeReport=False,
             rangeReport=False
             ):

    def table_builder(pdf, col_width, row_height, TABLE_DATA):
        x = (pdf.w - col_width * len(TABLE_DATA[0])) / 2
        pdf.set_fill_color(159, 214, 83)
        headers = TABLE_DATA[0]
        pdf.set_font("Body", 'B', size=9)
        pdf.set_x(x)
        for header in headers:
            pdf.cell(col_width, row_height, str(header), border=1, fill=True)
        pdf.ln(row_height)

        # Reset font style for data rows
        pdf.set_font("Body", '', size=9)

        # Render table data
        for row in TABLE_DATA[1:]:
            pdf.set_x(x)
            for datum in row:
                pdf.cell(col_width, row_height, str(datum), border=1)
            pdf.ln(row_height)

    def draw_horizontal_line(y):
        pdf.set_line_width(0.5)  # Задать толщину линии
        pdf.set_draw_color(105, 105, 105)  # Задать цвет линии (RGB)
        pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)

    pdf = FPDF()
    col_width = pdf.w / 4.5
    row_height = pdf.font_size * 1.5
    pdf.add_page()

    pdf.add_font('Body', '', 'font/DejaVuSans.ttf')
    pdf.add_font('Body', 'B', 'font/DejaVuSans-Bold.ttf')

    pdf.set_fill_color(159, 214, 83)
    pdf.set_font("Body", '', size=16)
    pdf.cell(0, 10, f"Отчёт за {day}", align='C', border=1, fill=True)
    pdf.ln(20)
    pdf.set_font("Body", '', size=11)
    pdf.set_left_margin(15)
    pdf.cell(0, 10, f"За анализируемый день было совершено {len(result[1]) - 1} поездок.")
    pdf.ln(7)
    pdf.cell(0, 10, f"Анализ исправности датчика уровня топлива проведён алгоритмом:")
    pdf.ln(7)
    pdf.cell(0, 10, f"\"{result[2]}\"")
    pdf.ln(7)
    pdf.set_font("Body", 'B', size=11)
    pdf.cell(0, 10, f"Процент выбросов в каждой поездке:")
    pdf.ln(10)
    proc_table = result[1]
    table_builder(pdf, pdf.w/3, row_height, proc_table)
    pdf.ln(10)
    pdf.set_font("Body", '', size=11)
    if (result[4][3] == 0):
        pdf.cell(0, 10, f"Аномалий в работе датчика уровня топлива не обнаружено, работа стабильная.")
    elif (result[4][3] == 0.5):
        pdf.cell(0, 10, f"Работа в пределах нормы, однако выявлены поездки с аномальным поведением.")
        pdf.ln(15)
        table_proc = tuple([tuple(["№ Поездки", "Процент выбросов", "Превышение нормы"])] +
                           list(map(tuple, zip(*[result[4][0], result[4][1], result[4][2]]))))
        table_builder(pdf, 1.3*col_width, row_height, table_proc)
    elif (result[4][3] == 1):
        pdf.cell(0, 10, f"Аномальное поведение датчика уровня топлива. Рекомендуется диагностика!")
        pdf.ln(15)
        table_proc = tuple([tuple(["№ Поездки", "Процент выбросов", "Превышение нормы"])] + list(
            map(tuple, zip(*[result[4][0], result[4][1], result[4][2]]))))
        table_builder(pdf, col_width*1.3, row_height, table_proc)
    pdf.cell(15)
    report = result[5]
    pdf.set_font("Body", 'B', size=11)
    pdf.cell(0, 10, "Отчёт по основным показателям за день:")
    pdf.ln(10)
    table_builder(pdf, pdf.w / 3, row_height, report[0])
    pdf.ln(7)
    line_y = pdf.get_y()
    draw_horizontal_line(y=line_y)
    pdf.ln(7)
    if (fuelReport):
        pdf.set_font("Body", 'B', size=11)
        pdf.cell(0, 10, f"Отчёт по расходу топлива:")
        pdf.ln(10)
        table_builder(pdf, pdf.w / 6, row_height, report[1])
        pdf.ln(15)
    if (fuelChangeReport):
        pdf.set_font("Body", 'B', size=11)
        pdf.cell(0, 10, f"Отчёт по сливам и заправкам топлива:")
        pdf.ln(10)
        table_builder(pdf, pdf.w / 4.5, row_height, report[2])
        pdf.ln(15)
    if (rangeReport):
        pdf.set_font("Body", 'B', size=11)
        pdf.cell(0, 10, f"Отчёт по пробегу:")
        pdf.ln(10)
        table_builder(pdf, pdf.w / 7, row_height, report[3])
        pdf.ln(15)

    pdf.output(f"{file_name}.pdf")


