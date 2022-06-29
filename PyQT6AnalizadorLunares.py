"""
-------------------------------------------
Analizador de Lunares
Autor: Carlos Mir Martínez
Fecha: 29/06/2022
-------------------------------------------
"""

import sys
from PyQt6 import *
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import keras

# Modelo de la Red Neuronal Convolucional Entrenado con más de 90% de precisión.
Melanoma = keras.models.load_model('C:/Users/USUARIO/Desktop/FotosPyQT6/ClasificadorMelanomaRNC.h5')

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.center()

    def initUI(self):
        myFont=QFont()
        myFont.setBold(True)
        
        self.setWindowTitle("Analizador de Lunares")

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('Informe')

        impMenu = QMenu('Redactar', self)
        impAct = QAction('Redactar nuevo informe', self)
        impMenu.addAction(impAct)

        fileMenu.addMenu(impMenu)

        pagelayout = QVBoxLayout()
        button_layout = QHBoxLayout()
        self.stacklayout = QStackedLayout()

        pagelayout.addLayout(button_layout)
        pagelayout.addLayout(self.stacklayout)

        btn01 = QPushButton("Seleccionar Imagen")
        btn01.pressed.connect(self.activate_tab_1)
        btn01.clicked.connect(self.showDialog)
        button_layout.addWidget(btn01)

        btn02 = QPushButton("Analizar Imagen")
        btn02.pressed.connect(self.activate_tab_2)
        btn02.clicked.connect(self.analyze)
        button_layout.addWidget(btn02)

        btn03 = QPushButton("Salir")
        btn03.pressed.connect(self.activate_tab_3)
        btn03.clicked.connect(QApplication.instance().quit)
        button_layout.addWidget(btn03)
        
        global path
        path = QLabel("Ruta Imagen seleccionada: \n", self)
        path.setStyleSheet("border: 1px solid black;"
                           "border-radius: 4px;")
        path.setFont(myFont)
        path.setGeometry(125, 70, 400, 50)
        
        global Imagen
        Imagen = QLabel("",self)
        Imagen.setStyleSheet("border: 1px solid black;"
                             "border-radius: 4px;")
        Imagen.setGeometry(125, 170, 400, 300)
        
        global texto
        texto = QTextEdit("",self)
        texto.setReadOnly(True)
        texto.setFont(myFont)
        texto.setGeometry(125, 135, 400, 40)
    
        name = QLabel(" Autor: Carlos Mir Martínez\n Fecha: 29/06/2022",self)
        name.setStyleSheet("border: 1px solid black;"
                           "border-radius: 4px;")
        name.setFont(myFont)
        name.setGeometry(125, 475, 400, 50)
        
        widget = QWidget()
        widget.setLayout(pagelayout)
        self.setCentralWidget(widget)

        self.setGeometry(700, 700, 600, 600)
    
    def activate_tab_1(self):
        self.stacklayout.setCurrentIndex(0)

    def activate_tab_2(self):
        self.stacklayout.setCurrentIndex(1)

    def activate_tab_3(self):
        self.stacklayout.setCurrentIndex(2)
        
    def closeEvent(self, event):

        reply = QMessageBox.question(self, 'Alerta',
                    "¿Seguro que quieres cerrar el programa?", QMessageBox.StandardButton.Yes |
                    QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:

            event.accept()
        else:

            event.ignore()
            
    def showDialog(self):
        try:
            texto.setHtml("")
            global fname
            fname = QFileDialog.getOpenFileName(self, 'Open file', 'C:/Users/USUARIO/Desktop/Pacientes')
            path.setText("Imagen seleccionada: \n" + str(fname[0]))
            if fname[0]:
                pixmap = QPixmap(fname[0])
                a = pixmap.scaled(400,300)
                Imagen.setPixmap(a) 
        except:
            texto.setHtml("<font color='red' size='4' ><red>ERROR\n</font>")
    def center(self):

        qr = self.frameGeometry()
        cp = self.screen().availableGeometry().center()
    
        qr.moveCenter(cp)
        self.move(qr.topLeft())
         
    def analyze(self):
        
        img = tf.keras.utils.load_img(fname[0], target_size=[64, 64])
        x = tf.keras.utils.img_to_array(img)
        x = np.array(x) /255
        data = x.reshape(1, 64, 64, 3)
        prediction = Melanoma.predict(data)
        prediction1 = prediction > 0.9
        prediction = prediction * 100
        
        try:
            if prediction1:
                
                texto.setHtml("<font color='green' size='5' ><red>ES UN LUNAR\n</font>")
            else:
                
                texto.setHtml("<font color='red' size='5' ><red>NO ES UN LUNAR, CONSULTA AL MÉDICO\n</font>")
        except:
            texto.setHtml("<font color='red' size='5' ><red>ERROR\n</font>")
def main():

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()

if __name__ == '__main__':
    main()

