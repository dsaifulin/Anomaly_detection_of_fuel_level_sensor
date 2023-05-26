#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
from PyQt5.QtCore import QDate
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QHeaderView, QMessageBox, QProgressBar
from PyQt5 import uic, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItemModel, QStandardItem
import reportBuilder
import my_func_modul as myf

class App(QWidget):
    def __init__(self):
        self.start()
        self.report()
    def start(self):
        self.ui = uic.loadUi("settings.ui")
        self.ui.reset_button.clicked.connect(lambda: self.reset_settings())
        self.ui.manualMethod.clicked.connect(lambda: self.open_manual_window())
        self.ui.show()
    def report(self):
        self.ui.report_button.clicked.connect(lambda: self.report_building())

    def reset_settings(self):
        self.ui.boxOfMethods.setCurrentIndex(0)
        self.ui.dateChoose.setDate(QDate(2022, 3, 1))
        self.ui.check_dut.setChecked(False)
        self.ui.check_delta_dut.setChecked(False)
        self.ui.check_run.setChecked(False)

    def report_building(self):

        date = self.ui.dateChoose.date()
        day = date.day(); month = date.month(); year = date.year()
        day_str = myf.date_transform(day, month, year)
        method_index = self.ui.boxOfMethods.currentIndex()
        method_text = self.ui.boxOfMethods.currentText()
        result = reportBuilder.report_builder_to_pdf(day=day_str, method=method_index)

        if (result != -1):
            result[0].append(result[5][4])
            setBoxStatus = [self.ui.check_dut.isChecked(),
                            self.ui.check_delta_dut.isChecked(),
                            self.ui.check_run.isChecked()]
            self.open_report_window(day_str, result, setBoxStatus)
        else:
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Warning)
            error_dialog.setWindowTitle("Ошибка")
            error_dialog.setText("В запрашиваемый день поездок не было!\nПожалуйста, выберите другой день.")
            error_dialog.setIcon(QMessageBox.Warning)
            error_dialog.exec_()
    def open_manual_window(self):
        self.manual_window = Manual()
    def open_report_window(self, day_of_report, result, setBoxStatus):
        self.report_window = ReportWindow(day_of_report, result, setBoxStatus)

class ReportWindow(QWidget):
    def __init__(self, day, result, setBoxStatus):
        #self.setFixedSize(905, 657)
        self.set_box_status = setBoxStatus
        self.day_of_report = day
        self.result = result
        self.start()
    def start(self):
        self.ui = uic.loadUi("report.ui")
        self.ui.day_of_report.setText(f"Отчёт за {self.day_of_report}")
        self.ui.graphics_pdf_btn.clicked.connect(lambda: self.graphs_pdf_build())
        self.ui.text_pdf_btn.clicked.connect(lambda: self.text_pdf_building())
        self.report_page_build(result=self.result)
        self.ui.show()
    def report_page_build(self, result):
        df_table_proc = pd.DataFrame(result[1][1:], columns=result[1][0])
        anomaly_status = result[4]
        model = QStandardItemModel()
        model.setRowCount(df_table_proc.shape[0])
        model.setColumnCount(df_table_proc.shape[1])
        for i in range(df_table_proc.shape[0]):
             for j in range(df_table_proc.shape[1]):
                 item = QStandardItem(str(df_table_proc.iloc[i, j]))
                 model.setItem(i, j, item)
        model.setHeaderData(0, Qt.Horizontal, "Поездка")
        model.setHeaderData(1, Qt.Horizontal, "Процент выбросов, %")
        self.ui.enj_table.setModel(model)
        header = self.ui.enj_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        anomaly_rides = ", ".join(str(item) for item in anomaly_status[0])
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
    def graphs_pdf_build(self):
        file_name = f'graphs_{self.day_of_report}'
        reportBuilder.graphs_pdf(figures=self.result[0], file_name=file_name)
        save_path, _ = QFileDialog.getSaveFileName(None, 'Сохранить PDF', '', 'PDF Files (*.pdf)')
        if save_path:
            import shutil
            shutil.move(f"{file_name}.pdf", save_path)

    def text_pdf_building(self):
        file_name = f'text_{self.day_of_report}'
        reportBuilder.text_pdf(self.day_of_report, self.result, file_name,
                               fuelReport=self.set_box_status[0],
                               fuelChangeReport=self.set_box_status[1],
                               rangeReport=self.set_box_status[2]
                               )
        save_path, _ = QFileDialog.getSaveFileName(None, 'Сохранить PDF', '', 'PDF Files (*.pdf)')
        if save_path:
            import shutil
            shutil.move(f"{file_name}.pdf", save_path)

class Manual(QWidget):
    def __init__(self):
        self.ui = uic.loadUi("manual.ui")
        self.ui.show()

if __name__== '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon('icon.ico'))
    main = App()
    app.exec_()