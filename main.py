#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Импорт необходимых библиотек и модулей
import sys
from PyQt5.QtCore import QDate
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QHeaderView, QMessageBox, QProgressBar
from PyQt5 import uic, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItemModel, QStandardItem
import reportBuilder
import my_func_modul as myf

# Класс основного окна настройки формируемого отчёта
class App(QWidget):

    # Метод инициализации класса
    def __init__(self):
        self.start()
        self.report()

    # Метод загружает интерфейс окна и настраивает обработчики нажатий для кнопок "Сбросить настройки" и "?"
    def start(self):
        self.ui = uic.loadUi("settings.ui")# загрузка интерфейса
        self.ui.reset_button.clicked.connect(lambda: self.reset_settings())# обработчик нажатия на "Сбросить настройки"
        self.ui.manualMethod.clicked.connect(lambda: self.open_manual_window())# обработчик нажатия на "?"
        self.ui.show()# запуск окна

    # Метод настраивает обработчик нажатия для кнопки "Сформировать отчёт"
    def report(self):
        self.ui.report_button.clicked.connect(lambda: self.report_building())# обработчик нажатия на "Сформировать отчёт"

    # Метод установки значений виджетов окна к значениям по умолчанию
    def reset_settings(self):
        self.ui.boxOfMethods.setCurrentIndex(0)# метод по умолчанию
        self.ui.dateChoose.setDate(QDate(2022, 3, 1))# дата по умолчанию
        # сброс дополнительной настройки отчёта
        self.ui.check_dut.setChecked(False)
        self.ui.check_delta_dut.setChecked(False)
        self.ui.check_run.setChecked(False)

    # Метод расчёта отчёта при нажатии на кнопку "Сформировать отчёт"
    def report_building(self):

        date = self.ui.dateChoose.date()# получение даты отчёта
        day = date.day(); month = date.month(); year = date.year()
        day_str = myf.date_transform(day, month, year)# запрашиваемая дата в виде строки
        method_index = self.ui.boxOfMethods.currentIndex()# получение индекса метода расчёта
        method_text = self.ui.boxOfMethods.currentText()# получение названия метода расчёта
        result = reportBuilder.report_builder_to_pdf(day=day_str, method=method_index)# расчёт настроенного отчёта

        # в запрашиваемый день были поездки
        if (result != -1):

            # открытие окна отчёта и передача в него соответствующих параметров
            result[0].append(result[5][4])
            setBoxStatus = [self.ui.check_dut.isChecked(),
                            self.ui.check_delta_dut.isChecked(),
                            self.ui.check_run.isChecked()]
            self.open_report_window(day_str, result, setBoxStatus)

        # поездок в запрашиваемый день не было
        else:

            # вывод окна с сообщением об отсутствии поездок в запрашиваемый день
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Warning)
            error_dialog.setWindowTitle("Ошибка")
            error_dialog.setText("В запрашиваемый день поездок не было!\nПожалуйста, выберите другой день.")
            error_dialog.setIcon(QMessageBox.Warning)
            error_dialog.exec_()

    # Метод создания окна пользовательской справки
    def open_manual_window(self):
        self.manual_window = Manual()

    # Метод создания окна сформированного отчёта
    def open_report_window(self, day_of_report, result, setBoxStatus):
        self.report_window = ReportWindow(day_of_report, result, setBoxStatus)# создаём окно сформрованного отчёта

# Класс окна сформированного отчёта
class ReportWindow(QWidget):

    # Метод инициализации класса
    def __init__(self, day, result, setBoxStatus):
        self.set_box_status = setBoxStatus
        self.day_of_report = day
        self.result = result
        self.start()

    # Метод загружает и запускает интерфейс окна, настраивает обработчики нажатий для кнопок
    def start(self):
        self.ui = uic.loadUi("report.ui")# загрузка интерфейса из ui модели
        self.ui.day_of_report.setText(f"Отчёт за {self.day_of_report}")# отображения дня сформированного отчёта
        self.ui.graphics_pdf_btn.clicked.connect(lambda: self.graphs_pdf_build())# обработчик нажатия кнопки "Графики"
        self.ui.text_pdf_btn.clicked.connect(lambda: self.text_pdf_building())# обработчик нажатия кнопки "Текстовый отчёт"
        self.report_page_build(result=self.result)# вывод результата анализа датчика в окне
        self.ui.show()# запуск интерфейса

    # Метод формирует краткую сводку об исправности датчика в окне отчёта
    def report_page_build(self, result):
        # формирование таблицы с расчитанными выбросами
        df_table_proc = pd.DataFrame(result[1][1:], columns=result[1][0])
        anomaly_status = result[4]
        model = QStandardItemModel()# создание модели таблицы
        model.setRowCount(df_table_proc.shape[0])# задание количества строк
        model.setColumnCount(df_table_proc.shape[1])# задание количества столбцов
        # заполнение таблицы данными
        for i in range(df_table_proc.shape[0]):
             for j in range(df_table_proc.shape[1]):
                 item = QStandardItem(str(df_table_proc.iloc[i, j]))
                 model.setItem(i, j, item)
        # именование колонок
        model.setHeaderData(0, Qt.Horizontal, "Поездка")
        model.setHeaderData(1, Qt.Horizontal, "Процент выбросов, %")
        self.ui.enj_table.setModel(model)
        header = self.ui.enj_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        anomaly_rides = ", ".join(str(item) for item in anomaly_status[0])# аномальные поездки

        # если зафиксировано малое количество поездок с аноальным поведением, то выводим соответствующее сообщение
        if anomaly_status[3] == 0.5:
            self.ui.result_text.setText(f"<b>Работа датчика в пределах нормы.</b>"
                                        f"<br>Обратите внимание на поездки <u>{anomaly_rides}</u>, зафиксировано аномальное поведение!")
            new_style_sheet = '''QLabel {
                                color: #333;
                                font-size: 17px;
                                font-family: Arial, sans-serif;
                                border: none;
                                border-radius: 5px;
                                background-color: rgb(173, 255, 47);
                                padding: 5px;
                            }'''
            self.ui.result_text.setStyleSheet(new_style_sheet)

        # если зафиксировано большое количество поездок с аноальным поведением, то выводим соответствующее сообщение
        elif anomaly_status[3] == 1:

            self.ui.result_text.setText(f"<b>Возможна неисправность датчика уровня топлива!</b>"
                                        f"<br>В данных поездок <u>{anomaly_rides}</u>, зафиксировано большое количество аномалий.")
            new_style_sheet = """QLabel {
                                color: white;
                                font-size: 17px;
                                font-family: Arial, sans-serif;
                                border: none;
                                border-radius: 5px;
                                background-color: rgb(255, 0, 0);
                                padding: 5px;
                            }"""
            self.ui.result_text.setStyleSheet(new_style_sheet)

    # Метод формирования pdf-отчёта с графиками
    def graphs_pdf_build(self):
        file_name = f'graphs_{self.day_of_report}'
        reportBuilder.graphs_pdf(figures=self.result[0], file_name=file_name)# формируем pdf отчёт
        save_path, _ = QFileDialog.getSaveFileName(None, 'Сохранить PDF', '', 'PDF Files (*.pdf)')# открытие диалогового окна сохранения в нужное место

        # если место сохранения выбрано, то сохраняем
        if save_path:
            import shutil
            shutil.move(f"{file_name}.pdf", save_path)

    # Метод формирования pdf-отчёта с текстом и таблицами
    def text_pdf_building(self):
        file_name = f'text_{self.day_of_report}'
        # формируем pdf отчёт
        reportBuilder.text_pdf(self.day_of_report, self.result, file_name,
                               fuelReport=self.set_box_status[0],
                               fuelChangeReport=self.set_box_status[1],
                               rangeReport=self.set_box_status[2]
                               )
        save_path, _ = QFileDialog.getSaveFileName(None, 'Сохранить PDF', '', 'PDF Files (*.pdf)')# открытие диалогового окна сохранения в нужное место

        # если место сохранения выбрано, то сохраняем
        if save_path:
            import shutil
            shutil.move(f"{file_name}.pdf", save_path)

# Класс окна пользовательской справки
class Manual(QWidget):

    # Метод инициализации класса
    def __init__(self):
        self.ui = uic.loadUi("manual.ui")# загрузка интерфейса
        self.ui.show()# запуск интерфейса

if __name__== '__main__':
    app = QApplication(sys.argv)# создание экземпляра приложения
    app.setWindowIcon(QtGui.QIcon('icon.ico'))# установка иконки приложения
    main = App()# создание экземпляра окна настройки отчёта
    app.exec_()# запуск главного цикла событий приложения