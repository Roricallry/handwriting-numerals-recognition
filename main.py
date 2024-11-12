# python3.12
import os
import sys
from handwriting import MainWindow
from PyQt5.QtWidgets import QApplication
import Init
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    file_paths = {
        "models": "os.makedirs(\"models\", exist_ok=True)",
        "models/knn_model.pkl": "Init.Init_KNN()",
        "models/svm_model.pkl": "Init.Init_SVM()",
        "models/cnn_model.h5": "Init.Init_CNN()",
        "models/rf_model.pkl": "Init.Init_RF()",
        "models/FCN_model.h5": "Init.Init_FCN()",
    }
    for file_path in file_paths:
        if os.path.exists(file_path):
            pass
        else:

            eval(file_paths[file_path])

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
