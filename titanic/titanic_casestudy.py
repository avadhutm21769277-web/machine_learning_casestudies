

#################################################################################

#   required libraries  :

#################################################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import tkinter as tk
from tkinter import messagebox



#################################################################################

#   function name 1      :       load_data()
#   input                :       filename.csv    file
#   output               :       data load into the dataframe (df)

#################################################################################


def load_data(dataset_name):
    df=pd.read_csv(dataset_name)
    return df


##################################################################################
#   function name 2     :       number_and_name_of_headers()
#   input               :       dataframe as df
#   output              :       number and names of headers

##################################################################################

def number_and_name_of_headers(df):
    headers=df.columns[:]
    return headers

##################################################################################

#   function name 3     :       datatypes_of_headers
#   input               :       dataframe as df
#   output              :       datatypes of all headers

##################################################################################

def datatypes_of_headers(df):
    datatypes=df.dtypes
    return datatypes


##################################################################################

#   function name 4     :       feature_headers()
#   input               :       dataframe as df
#   output              :       name of feature headers

##################################################################################

def feature_headers(df):
    independent_variables=df.columns[0:9]
    return independent_variables



#################################################################################

#   function name   5   :       target_header()
#   input               :       dataframe as df
#   output              :       name of target variable

#################################################################################

def target_header(df):
    dependent_variable=df.columns[9:]
    return dependent_variable



#################################################################################

#   function name   6   :   drop_unnecessary_columns()
#   input               :   dataframe as df
#   output              :   updated dataframe as df


#################################################################################

def drop_unnecessary_columns(df):
    df=df.drop(columns=['Passengerid','zero'],inplace=False)
    if df.isnull().sum().any():
        print("Missing valus")
        df = df.fillna(df.mean(numeric_only=True))
    return df


#################################################################################

#   function name   7   :   split_data_vertically()
#   input               :   updated dataframe as df
#   output              :   X , Y   i.e feature header , target header

#################################################################################

def split_data_vertically(df):
    X=df.drop(columns='Survived',axis=1)
    Y=df['Survived']
    return X,Y


#################################################################################

#   funnction name  8   :   split_train_test()
#   input               :   vertically splitted parts   X,Y
#   Output              :   four parts  X_train,X_test,Y_train,Y_test

#################################################################################

def split_train_test(X,Y):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.5,random_state=42)
    return X_train,X_test,Y_train,Y_test


##################################################################################

#   function name   9   :   using_standerd_scaler()
#   input               :   X_train,X_test
#   output              :   scaled x_train and scaled X_test

##################################################################################

def using_standard_scaler(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler,X_train_scaled, X_test_scaled



##################################################################################

#   function name   10  :   logistic_regression_model()
#   input               :   X_train_scaled,X_test_scaled_Y_train,Y_test
#   output              :   model



##################################################################################

def logistic_regression_model(X_train_scaled,X_test_scaled,Y_train,Y_test):
    model=LogisticRegression(random_state=42,warm_start=True,solver="liblinear")
    model.fit(X_train_scaled,Y_train)
    return model

#################################################################################

#   function name   11  :   predict_model()
#   input               :

#################################################################################

def predict_model(trained_model,X_train_scaled,X_test_scaled,Y_train,Y_test):
    Y_pred=trained_model.predict(X_test_scaled)
    return Y_pred

################################################################################

#   function name   12  :   calculate_accuracy()
#   input               :

################################################################################

def calculate_accuracy(Y_test,Y_pred):
    accuracy=accuracy_score(Y_test,Y_pred)
    return accuracy

############################################################################

#   function name       :   save_model()
#   input               :   trained model and filename.pkl file
#   output              :   .pkl file

############################################################################


def save_model(trained_model,name_of_file):
    joblib.dump(trained_model,name_of_file)
    return trained_model,name_of_file


##############################################################################

#   function name   :   load_model()
#   input           :   filename.pkl file

##############################################################################


def load_model(name_of_file):
    loaded_model=joblib.load(name_of_file)
    return loaded_model


##############################################################################

#   function name   :   predict_model
#   input           :   loaded_model


##############################################################################

def predict_model_loaded(loaded_model):
    new_data = pd.DataFrame([{
        "Age": 26,
        "Fare": 7,
        "Sex": 1,
        "sibsp": 25,
        "Parch": 0,
        "Pclass": 3.5,
        "Embarked": 2,

    }])





    # Prediction
    prediction = loaded_model.predict(new_data)

    # resu
    if prediction[0] == 1:
        print("non_survived possibility")
    else:
        print("survived possibility")



#################################################################################
#   GUI Function
#################################################################################

def run_gui(trained_model,scaler):
    root = tk.Tk()
    root.title("Titanic Survival Prediction")
    root.geometry("350x300")

    # Labels & Entries
    labels = ["Age", "Fare", "Sex (1=Male, 0=Female)", "sibsp", "Parch", "Pclass", "Embarked"]
    entries = {}

    for i, label in enumerate(labels):
        tk.Label(root, text=label).grid(row=i, column=0, pady=5, sticky="w")
        entry = tk.Entry(root)
        entry.grid(row=i, column=1)
        entries[label] = entry

    def on_predict():
        try:
            new_data = pd.DataFrame([{
                "Age": float(entries["Age"].get()),
                "Fare": float(entries["Fare"].get()),
                "Sex": int(entries["Sex (1=Male, 0=Female)"].get()),
                "sibsp": int(entries["sibsp"].get()),
                "Parch": int(entries["Parch"].get()),
                "Pclass": float(entries["Pclass"].get()),
                "Embarked": int(entries["Embarked"].get()),
            }])
            new_data_scaled = scaler.transform(new_data)
            prediction = trained_model.predict(new_data_scaled)

            if prediction[0] == 1:
                messagebox.showinfo("Result", "Passenger likely SURVIVED")
            else:
                messagebox.showinfo("Result", "Passenger likely DID NOT SURVIVE")

        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")

    tk.Button(root, text="Predict", command=on_predict).grid(row=len(labels), columnspan=2, pady=10)

    root.mainloop()


####################################################################################################

#   function name   :   main()
#   description     :   calling differnt user definrd function

###################################################################################################


def main():
    print("GUI trained model for TITANIC casestudy  !!!!!!")

    #   1>  load the dataset    :

    dataset_name='MarvellousTitanicDataset.csv'
    df=load_data(dataset_name)
    print(df.head())



    #   2>  number headers  :

    headers=number_and_name_of_headers(df)
    print("dataset headers  :",headers)




    #   3>  datatypes of headers    :

    datatypes=datatypes_of_headers(df)
    print("datatypes of headers :",datatypes)
    ####----------->>>>         AS DATATYPES OF ALL HEADERS CONTAINS NUMERIC VALUES SO WE DONT NEED TO ENCODE THE  DATA



    #   4>  deciding feature headers    :
    independent_variables=feature_headers(df)
    print("feature variables are    :",independent_variables)



    #   5> deciding target header       :
    dependent_variable=target_header(df)
    print("taeget header    :",dependent_variable)


    #   6> drop unnecessary columns:
    #   As dataset column[0] contains passanger id ,'zero' -colums which contains only zero ,  it would not affect output so we drop that column

    df=drop_unnecessary_columns(df)
    print(df.head())


    #   7>  splitting dataset vertically into feature header and target header i.e into two parts

    X,Y=split_data_vertically(df)
    print(X.shape)
    print(Y.shape)

    #   8>  splitting dataset into 4 parts for training and testing perpose

    X_train,X_test,Y_train,Y_test=split_train_test(X,Y)

    print("Shape of X_train",X_train.shape)
    print("Shape of X_test",X_test.shape)
    print("Shape of Y_train",Y_train.shape)
    print("Shape of Y_test",Y_test.shape)


    #   8>   using standers scaler

    scaler,X_train_scaled,X_test_scaled=using_standard_scaler(X_train,X_test)


    #   9>  algorithm selection

    trained_model=logistic_regression_model(X_train_scaled,X_test_scaled,Y_train,Y_test)

    #   10> testing the data

    Y_pred=predict_model(trained_model,X_train_scaled,X_test_scaled,Y_train,Y_test)

    #   11> calculating accuracy

    accuracy = calculate_accuracy(Y_test, Y_pred)
    print("Accuracy score   :", accuracy * 100, "%")

    #   12>

    saved_model,name_of_file=save_model(trained_model,'titanic.pkl')
    print("model save success")

    #   13> load model

    loaded_model=load_model(name_of_file)
    print("loaded sucess")

    #   14> predict_model

    predict_model_loaded(loaded_model)

    #   15> GUI

    run_gui(loaded_model,scaler)



if __name__=="__main__":
    main()