# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyparsing import line
# from sklearn.linear_model import LogisticRegression
from Model import LogisticRegressionUsingGD
# from sklearn.metrics import accuracy_score


def load_data(path, header):
    marks_df = pd.read_csv(path, header=header)
    return marks_df

def linEq(x1,x2,y1,y2):
    grad = (y2-y1)/(x2-x1)
    c = y1 - (grad * x1)
    return grad, c

if __name__ == "__main__":
    # load the data from the file
    print("==========================================================================================")
    print("Pilih data yang ingin digunakan sebagai acuan (Nb. Secara default akan memilih pilihan 1):")
    print("1. Tekanan Darah dan Denyut Jantung Maksimal")
    print("2. Tekanan Darah dan Kolesterol")
    print("3. Kolesterol dan Denyut Jantung Maksimal")

    user_data_choice = input("Pilih sesuai nomor : ")

    if(user_data_choice == '3'):
        data_path = "heart3"
        faktor_x = "Kolesterol"
        faktor_y = "Denyut Jantung Maksimal"
    elif(user_data_choice == '2'):
        data_path = "heart2"
        faktor_x = "Tekanan Darah"
        faktor_y = "Kolesterol"
    else:
        data_path = "heart4"
        faktor_x = "Tekanan Darah"
        faktor_y = "Denyut Jantung Maksimal"
    
    user_input_x = float(input(f"Input nilai {faktor_x}: "))
    user_input_y = float(input(f"Input nilai {faktor_y}: "))

    print("==========================================================================================\n")
        
    data = load_data(f"data\{data_path}.txt", None)

    # X = feature values, all the columns except the last column
    X = np.array(data.iloc[:, :-1])

    # y = target values, last column of the data frame
    y = np.array(data.iloc[:, -1])

    # filter out the applicants that got admitted
    admitted = data.loc[y == 1]

    # filter out the applicants that din't get admission
    not_admitted = data.loc[y == 0]

    # plots
    plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Terdiagnosa')
    plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Tidak Terdiagnosa')

    # preparing the data for building the model
    X = np.c_[np.ones((X.shape[0], 1)), X]
    y = y[:, np.newaxis]
    theta = np.zeros((X.shape[1], 1))

    # Logistic Regression from scratch using Gradient Descent
    model = LogisticRegressionUsingGD()
    model.fit(X, y, theta)
    accuracy = model.accuracy(X, y.flatten())
    parameters = model.w_
    print("\n==========================================================================================")
    print("Dataset Model ini memberikan akurasi sebesar {}".format(accuracy))
    print("Dataset Model ini memberikan hasil berupa fungsi linear dengan metode Logistic Regressions")
    # print(parameters)

    # plotting the decision boundary
    # As there are two features
    # wo + w1x1 + w2x2 = 0
    # x2 = - (wo + w1x1)/(w2)

    x_values = [np.min(X[:, 1] - 2), np.max(X[:, 2] + 2)]
    y_values = - (parameters[0] + np.dot(parameters[1], x_values)) / parameters[2]
    grad, c = linEq(x_values[0], y_values[0], x_values[1], y_values[1])
    y_predic = grad * user_input_x + c

    if(y_predic <= user_input_y):
        print("\nBerdasarkan hasil regresi dataset model\nKamu terindikasi memiliki penyakit jantung!")
    else:
        print("\nBerdasarkan hasil regresi dataset model\nKamu tidak memiliki gejala yang mengindikasi penyakit jantung!")
    print("==========================================================================================")

    plt.scatter(user_input_x, user_input_y, s=50, label='User Input')
    plt.plot(x_values, y_values, label='Decision Boundary')
    plt.xlabel(f'{faktor_x}')
    plt.ylabel(f'{faktor_y}')
    plt.legend()
    plt.show()