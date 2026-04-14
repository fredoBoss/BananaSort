from PyQt5.QtWidgets import QApplication, QTableWidgetItem,QWidget, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
from PyQt5 import uic
import sys
import json

from arduinoComms import arduinoCommunication

def startArduino():
    global arduino
    arduino = arduinoCommunication("COM9",9600)
    print("Serial Communication: OK")

class weightCalibration(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("old/ui/calibration.ui",self)
        self.ui.setWindowTitle("Banana Sorter Application")
        startArduino()
        # self.showMaximized()
        # self.timer=QTimer()
        # self.timer.timeout.connect(self.runClassify)
        self.readCalibration()
        self.ui.pushButton.clicked.connect(self.readWeight)
        self.ui.pushButton_2.clicked.connect(self.rotateNext)
        self.ui.pushButton_3.clicked.connect(self.saveCalibration)
        
        # self.ui.btnStop.clicked.connect(self.stopTimer)

    def readCalibration(self):
        try:
            with open("calibration.json", "r") as f:
                data = json.load(f)
                print(data['plates'])
                for i, plate in enumerate(data['plates']):
                    print(plate['id'])
                    print(plate['offset'])
                    if i == 0:
                        self.ui.lineEdit.setText(str(plate['offset']))
                    elif i == 1:
                        self.ui.lineEdit_2.setText(str(plate['offset']))
                    elif i == 2:
                        self.ui.lineEdit_3.setText(str(plate['offset']))
                    elif i == 3:
                        self.ui.lineEdit_4.setText(str(plate['offset']))
                    elif i == 4:
                        self.ui.lineEdit_5.setText(str(plate['offset']))

        except FileNotFoundError:
            print("Calibration file not found. Please save calibration first.")
        
    def hideUi(self):
        self.close()

    def readWeight(self):
        arduino.restart()
        self.weightRes = arduino.reqWeight()
        print("Weight: ", self.weightRes)
        self.ui.label_9.setText(str(self.weightRes))

    def rotateNext(self):
        arduino.restart()
        arduino.reqRotateNext()
        print("Rotated to next tray")

    def saveCalibration(self):
        plate1 = self.ui.lineEdit.text()
        plate2 = self.ui.lineEdit_2.text() 
        plate3 = self.ui.lineEdit_3.text()
        plate4 = self.ui.lineEdit_4.text()  
        plate5 = self.ui.lineEdit_5.text()
        try:

            with open("calibration.json", "w") as f:
                data = {
                    "plates": [
                        {"id": "1", "offset": plate1},
                        {"id": "2", "offset": plate2},
                        {"id": "3", "offset": plate3},
                        {"id": "4", "offset": plate4},
                        {"id": "5", "offset": plate5}
                    ]
                }
                json.dump(data, f, indent=4)
                success = QMessageBox()
                success.setWindowTitle("Success")
                success.setText("Calibration data saved successfully!")
                success.setIcon(QMessageBox.Information)
                success.exec_()

        except Exception as e:
            print("Error saving calibration data:", e)

if __name__ == "__main__":
    
    app = QApplication(sys.argv)
    window = weightCalibration()
    window.show()
    sys.exit(app.exec())
    