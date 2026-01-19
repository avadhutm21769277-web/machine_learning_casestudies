





##############################################################

#   Required Libraries  :

##############################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import tkinter as tk
from tkinter import messagebox




##############################################################

#   function name       :   load_csv()
#   input               :   csv_name
#   output              :   dataframe as df

##############################################################


def load_csv(csv_name):
    df=pd.read_csv(csv_name)
    print("csv loaded success")
    return df





################################################################

#   Function name       :   checking_headers()
#   input               :   dataframe as df
#   output              :   number_of_variables with their names


################################################################


def checking_headers(df):
    number_of_variables=df.columns[:]
    return number_of_variables



##################################################################

#    Function name      :   check_for_datatypes
#    input              :   dataframe as df
#    output             :   dataypes of all variables



##################################################################

def check_for_datatypes(df):
    datatypes=df.dtypes
    return datatypes



###################################################################

#  function name    :   feature_headers()
#   input           :   dataframe as df
#   output          :   name of independent headers


###################################################################

def feature_headers(df):
    independent_variables=df.columns[0:8]
    return independent_variables


####################################################################

#   function name       :   target_header()
#   input               :   dataframe as df
#   output              :   name of the target header


####################################################################

def target_header(df):
    dependent_header=df.columns[8:]
    return dependent_header



##########################################################################

#   function name      :    check_null_values()
#   input              :    dataframe as df
#   output             :    number of null values respect with all headers

##########################################################################


def checking_null_values(df):
    null_values=df.isnull().sum()
    return null_values



###########################################################################

#   function_name   :   checking_zero_values()
#   input           :   dataframe as df
#   output          :   number of zero values with respect to all headers

###########################################################################

def checking_zero_values(df):
    zero_values=(df==0).sum()
    return zero_values


###########################################################################

#   function name   :   replece_zero_values_by_mean()
#   input           :   dataframe as df
#   output          :   updated dataframe as df


##########################################################################


def replece_zero_values_by_mean(df):
    df['Pregnancies']=df['Pregnancies'].replace(0,df['Pregnancies'].mean())
    df['Glucose']=df['Glucose'].replace(0,df['Glucose'].mean())
    df['BloodPressure']=df['BloodPressure'].replace(0,df['BloodPressure'].mean())
    df['SkinThickness']=df['SkinThickness'].replace(0,df['SkinThickness'].mean())
    df['Insulin']=df['Insulin'].replace(0,df['Insulin'].mean())
    df['BMI']=df['BMI'].replace(0,df['BMI'].mean())
    df['DiabetesPedigreeFunction']=df['DiabetesPedigreeFunction'].replace(0,df['DiabetesPedigreeFunction'].mean())
    df['Age']=df['Age'].replace(0,df['Age'].mean())
    return df



###########################################################################

#   functuon name   :   splitting_vertically()
#   input           :   updated dataframe as df
#   output          :   X,Y   verically two parts


###########################################################################

def splitting_vertically(df):
    X=df.drop('Outcome',axis=1)
    Y=df['Outcome']
    return X,Y



###########################################################################

#   function name   :   split_dataset()
#   input           :   X_train,Y_train,train_percentage,random_state
#   output          :   dataset after splitting

############################################################################

def split_dataset(X,Y):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.7,random_state=42)
    return X_train,X_test,Y_train,Y_test




############################################################################

#   function name   :   train_data() by using standerdscaler
#   input           :   X_train,X_test
#   output          :   scaled train data


############################################################################


def train_data(X_train,X_test):
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    return X_train_scaled,X_test_scaled



############################################################################


#   function name   :    build_pipeline()
#   use             :    cleaning process,algorithm slelection




############################################################################

def build_pipeline():
    pipeline = Pipeline(steps=[
        ("rf", RandomForestClassifier(n_estimators=150,bootstrap=True, random_state=42))
    ])
    return pipeline




############################################################################

#   function name   :   train_pipeline()
#   input           :   pipeline,X_train_scaled,Y_train
#   output          :   trained_pipeline


############################################################################


def train_pipeline(pipeline,X_train_scaled,Y_train):
    pipeline.fit(X_train_scaled,Y_train)
    return pipeline


##########################################################################

#   function name   :   calculate_accuracy()
#   input           :   trained_model
#   output          :   accuracy of the trained model


##########################################################################


def calculate_accuracy(trained_model,X_test_scaled,Y_test):
    y_pred=trained_model.predict(X_test_scaled)
    accuracy=accuracy_score(Y_test,y_pred)
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

def predict_model(loaded_model):
    new_data = pd.DataFrame([{
        "Pregnancies": 2,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 25,
        "Insulin": 80,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.35,
        "Age": 35
    }])

    # Prediction
    prediction = loaded_model.predict(new_data)

    # resu
    if prediction[0] == 1:
        print("Diabetic possibility")
    else:
        print("Non-Diabetic possibility")



########################################################################

def run_gui(loaded_model):
    # Prediction function
    def predict_diabetes():
        try:
            data = {
                "Pregnancies": float(entry_pregnancies.get()),
                "Glucose": float(entry_glucose.get()),
                "BloodPressure": float(entry_bp.get()),
                "SkinThickness": float(entry_skin.get()),
                "Insulin": float(entry_insulin.get()),
                "BMI": float(entry_bmi.get()),
                "DiabetesPedigreeFunction": float(entry_dpf.get()),
                "Age": float(entry_age.get())
            }
            new_data = pd.DataFrame([data])
            prediction = loaded_model.predict(new_data)

            if prediction[0] == 1:
                messagebox.showwarning("Result", "Diabetic possibility")
            else:
                messagebox.showinfo("Result", " Non-Diabetic possibility")
        except:
            messagebox.showerror("Error", "fill information correctly !!!!!!!!!!!!!")

    # GUI Window
    root = tk.Tk()
    root.title("🩺 Diabetes Prediction System")
    root.geometry("450x600")
    root.configure(bg="#f2f2f2")

    title_label = tk.Label(root, text="Diabetes Prediction", font=("Arial", 20, "bold"), fg="darkblue", bg="#f2f2f2")
    title_label.pack(pady=15)

    form_frame = tk.Frame(root, bg="#f2f2f2")
    form_frame.pack(pady=10)

    labels = [
        "Pregnancies", "Glucose", "Blood Pressure",
        "Skin Thickness", "Insulin", "BMI",
        "Diabetes Pedigree Function", "Age"
    ]

    entries = []
    for i, label_text in enumerate(labels):
        lbl = tk.Label(form_frame, text=label_text, font=("Arial", 12), bg="#f2f2f2")
        lbl.grid(row=i, column=0, sticky="w", padx=10, pady=5)
        entry = tk.Entry(form_frame, font=("Arial", 12), width=15)
        entry.grid(row=i, column=1, padx=10, pady=5)
        entries.append(entry)

    # Assign to variables
    global entry_pregnancies, entry_glucose, entry_bp, entry_skin, entry_insulin, entry_bmi, entry_dpf, entry_age
    entry_pregnancies, entry_glucose, entry_bp, entry_skin, entry_insulin, entry_bmi, entry_dpf, entry_age = entries

    predict_btn = tk.Button(root, text="Predict Diabetes", command=predict_diabetes,
                            font=("Arial", 14, "bold"), bg="green", fg="white", padx=10, pady=5)
    predict_btn.pack(pady=20)

    root.mainloop()

#######################################################################



##############################################################################################################

#   function name       :       main()
#   calling user_defined_function


##############################################################################################################


def main():

    print("Industrial format diabetic case study    !!!!")


    #   1>      load the csv

    csv_name='diabetes.csv'
    df=load_csv(csv_name)
    print(df.head)


    #   2>      Checking the variables

    number_of_variables=checking_headers(df)
    print("Name and number of all headers   :",number_of_variables)

    #   3>      datatypes of all variables

    datatypes=check_for_datatypes(df)
    print("Datatypes of headers :",datatypes)

    ####    as all variables contains numeric values so we dont need to encode the data


    #   4>      deciding feature headers  :

    independent_variable=feature_headers(df)
    print("Feature headers      :",independent_variable)


    #   5>      deciding target header     :

    dependent_variable=target_header(df)
    print("Target header    :",dependent_variable)


    #   6>      checking for null values    :

    result_of_null_values=checking_null_values(df)
    print("Null Values  :",result_of_null_values)
    #######     there are no null values presented in the dataset


    #   7>      checking for zero values in the dataset :

    zero_values=checking_zero_values(df)
    print("zero values  :",zero_values)
    #######     there are zero values presented in dataset


    #   8>      filling the zero values by mean of respective header except target header

    df=replece_zero_values_by_mean(df)
    print("cheking zero values of updated datafrane :",df.isnull().sum())


    #   9>      Splitting the data vertically as X and Y  :

    X,Y=splitting_vertically(df)
    print(X.shape)
    print(Y.shape)
    #######     data has splitted vertically success



    #   10>     splitting the data into four parts i.e X_train,X_test,Y_train,Y_test by using standerdscaler()


    X_train,X_test,Y_train,Y_test=split_dataset(X,Y)

    print("Shape of X_train  :",X_train.shape)
    print("Shape of X_test  :",X_test.shape)
    print("Shape of Y_train  :",Y_train.shape)
    print("Shape of Y_test  :",Y_test.shape)


    #   11>     train the data by using standers scaler :


    X_train_scaled,X_test_scaled=train_data(X_train,X_test)

    print("Shape of X_train  :",X_train_scaled.shape)
    print("Shape of X_test  :",X_test_scaled.shape)
    print("Shape of Y_train  :",Y_train.shape)
    print("Shape of Y_test  :",Y_test.shape)


    #   12>     Build pipeline----cleaning the data and alorithms selection

    pipeline=build_pipeline()

    #   13>     trained_model

    trained_model=train_pipeline(pipeline,X_train_scaled,Y_train)


    #   14> Calculate accuracy

    accuracy=calculate_accuracy(trained_model,X_test_scaled,Y_test)
    print(accuracy*100,"%")

    #   15> save model

    saved_model,name_of_file=save_model(trained_model,'diabetes.pkl')

    #   16> load model

    loaded_model=load_model(name_of_file)
    print("loaded sucess")

    #   17> predict_model

    predict_model(loaded_model)

    #   18> GUI

    run_gui(loaded_model)



if __name__=="__main__":
    main()