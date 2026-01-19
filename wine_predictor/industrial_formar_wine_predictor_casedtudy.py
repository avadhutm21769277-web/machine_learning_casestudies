


################################################################################

#   REQUIRED LIBRARIES  :

################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import joblib




##################################################################################

#       function name 1  :   load_data()
#       input            :   file_name.csv file
#       output           :   dataframe as df
#       description      :   converting csv file into dataframe (rows and columns)

##################################################################################

def load_data(csv_filename):
    df=pd.read_csv(csv_filename)
    return df




##################################################################################

#       function name 2  :   name_and_number_of_header()
#       input            :   daframe as df
#       output           :   name of headers and number of headers
#       description      :   is gives the information regarding headers

##################################################################################


def name_of_headers(df):
    headers=df.columns[:]
    return headers



#########################################################################################

#       function name 3  :   feature_header_of_dataset()
#       input            :   daframe as df
#       output           :   feature_header
#       description      :   is gives the names of input headers used to learn the model

##########################################################################################


def feature_header_of_dataset(df):
    feature_headers=df.columns[1:13]
    return feature_headers



#########################################################################################

#       function name 4  :   target_header_of_dataset()
#       input            :   daframe as df
#       output           :   target_header
#       description      :   is gives the name of column name to predict the model

##########################################################################################

def target_header_of_dataset(df):
    target_header=df.columns[0:1]
    return target_header


#########################################################################################

#       function name 5  :   datatypes_of_headers()
#       input            :   daframe as df
#       output           :   datatypes info
#       description      :   is gives the name of column name and datatypes

##########################################################################################

def datatypes_of_headers(df):
    datatypes=df.dtypes
    return datatypes


#########################################################################################

#       function name 6  :   shuffle_the_data()
#       input            :   daframe as df
#       output           :   shuffeled data as df
#       description      :   if target_header arranged in sorted format,
#                            we need to shuffle data to train the model correctly

##########################################################################################

def shuffle_the_data(df):
    df=df.sample(frac=1).reset_index(drop=True)
    return df



#########################################################################################

#       function name 7  :   statistical_info()
#       input            :   daframe as df
#       output           :   statistical_info
#       description      :   it gives information regarding : count,mean,standerd daviation
#                            min,max, quarter distribution

##########################################################################################

def statistical_info(df):
    stat_info=df.describe()
    return stat_info


#########################################################################################

#       function name 8  :   checking_null_values()
#       input            :   daframe as df
#       output           :   number of null values
#       description      :   check the null values presented in dataset and gives number of
#                            total null values with respect to each header

##########################################################################################

def checking_null_values(df):
    null_values=df.isnull().sum()
    return null_values


#########################################################################################

#       function name 9  :   boxplot_of_dataset()
#       input            :   daframe as df
#       output           :   png file of boxplot
#       description      :   boxplot is useful to find any outliers presents in dataset
#                            feature spreding, check data balanced or not?

##########################################################################################

def boxplot_of_dataset(df):
    plt.figure(figsize=(12,8))
    df.plot(kind="box", subplots=True, layout=(4,4), sharex=False, sharey=False, figsize=(15,10))
    plt.suptitle("Boxplots of Features")
    plt.savefig("boxplot.png")
    plt.close()
    print("boxplot of dataset created succesfully")




#################################################################################

#   function name  10   :   split_data_vertically()
#   input               :   updated dataframe as df
#   output              :   X , Y   i.e feature header data, target header data

#################################################################################

def split_data_vertically(df,target_header):
    X=df.drop(columns=target_header,axis=1)
    Y=df[target_header]
    return X,Y


#################################################################################

#   function name  11   :   split_data_into_four_parts()
#   input               :   X,Y , vertically splitted data
#   output              :   X_train,X_test,Y_train,Y_test

#################################################################################

def split_data_into_four_parts(X,Y):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.7,random_state=42)
    return X_train,X_test,Y_train,Y_test


##################################################################################

#   function name  12   :   using_standerd_scaler()
#   input               :   X_train,X_test
#   output              :   scaled x_train and scaled X_test

##################################################################################


def using_standerd_scaler(X_train,X_test):
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    return scaler,X_train_scaled,X_test_scaled


##################################################################################

#   function name  13   :   knn_classifier()
#   input               :   X_train_scaled,X_test_scaled,Y_train,Y_test
#   output              :   accuracy score of knn model

##################################################################################

def knn_classifier(X_train_scaled,X_test_scaled,Y_train,Y_test):

    #    accuracy=[]
    #   for k in range(1,30,1):
    #        knn_model=KNeighborsClassifier(n_neighbors=k,weights="uniform")            #   it is for tuning purpose
    #        knn_model.fit(X_train_scaled,Y_train)
    #        y_predict=knn_model.predict(X_test_scaled)
    #        accuracy_of_knn=accuracy_score(Y_test,y_predict)
    #        accuracy.append(accuracy_of_knn*100)
    #    print(accuracy)
    #    print(max(accuracy))


    knn_model=KNeighborsClassifier(n_neighbors=7,weights="uniform")
    knn_model.fit(X_train_scaled,Y_train)
    y_predict=knn_model.predict(X_test_scaled)
    accuracy_of_knn=accuracy_score(Y_test,y_predict)
    print("Accuracy by knn_model    :",accuracy_of_knn*100,"%")
    cnf=confusion_matrix(Y_test,y_predict)
    print(cnf)
    return knn_model,accuracy_of_knn



##################################################################################

#   function name  13   :   decision_tree_classifier()
#   input               :   X_train_scaled,X_test_scaled,Y_train,Y_test
#   output              :   accuracy score of decision tree model

##################################################################################



def decision_tree_model_train(X_train_scaled,X_test_scaled,Y_train,Y_test):



    decision_tree_model=DecisionTreeClassifier(max_depth=32,random_state=42)
    decision_tree_model.fit(X_train_scaled,Y_train)
    y_predict=decision_tree_model.predict(X_test_scaled)
    accuracy_of_decision_tree=accuracy_score(Y_test,y_predict)
    print("Accuracy by decision tree    :",accuracy_of_decision_tree)
    print("Accuracy by knn_model    :",accuracy_of_decision_tree*100,"%")
    cnf=confusion_matrix(Y_test,y_predict)
    print(cnf)
    return decision_tree_model,accuracy_of_decision_tree




############################################################################

#   function name       :   save_model()
#   input               :   trained model and filename.pkl file
#   output              :   .pkl file

############################################################################

def save_model(knn_model,KNN_accuracy,decision_tree_model,decision_tree_accuracy,name_of_file):
    if KNN_accuracy > decision_tree_accuracy:
        best_model = knn_model
        joblib.dump(best_model, name_of_file)
        print("KNN saved as", name_of_file)
    else:
        best_model = decision_tree_model
        joblib.dump(best_model, name_of_file)
        print("Decision Tree saved as", name_of_file)
    return best_model, name_of_file


##############################################################################

#   function name   :   load_model()
#   input           :   filename.pkl file

##############################################################################


def load_model(name_of_file):
    loaded_model=joblib.load(name_of_file)
    return loaded_model




##############################################################################

#   function name   :   predict_model
#   input           :   best_model


##############################################################################

def predict_model(loaded_model, scaler):
    new_wine = {
        'Alcohol': 13.5,
        'Malic acid': 2.3,
        'Ash': 2.4,
        'Alcalinity of ash': 18.5,
        'Magnesium': 100,
        'Total phenols': 2.5,
        'Flavanoids': 2.0,
        'Nonflavanoid phenols': 0.3,
        'Proanthocyanins': 1.5,
        'Color intensity': 5.0,
        'Hue': 1.0,
        'OD280/OD315 of diluted wines': 3.0,
        'Proline': 1000
    }

    # Convert dict to DataFrame
    new_data = pd.DataFrame([new_wine])

    # Scale features
    new_data_scaled = scaler.transform(new_data)

    # Predict class
    predicted_class = loaded_model.predict(new_data_scaled)
    print("Predicted Wine Class:", predicted_class[0])






##################################################################################

#       function name   :   main()
#       description     :   calling to all user_defined functions

#################################################################################





def main():

    print("WINE PREDICTOR MODEL by using KNN and RANDOM FOREST")

    #   1>  load_data()

    csv_filename='WinePredictor.csv'
    df=load_data(csv_filename)
    print("dataframe created sucess !!")
    print(df.head)


    #   2>  name_and_numer_of_headers()

    headers=name_of_headers(df)
    print("HEADERS ARE  :",headers)


    #   3>  feature_header_of_dataset()

    feature_headers=feature_header_of_dataset(df)
    print("feature headers are  :",feature_headers)

    #   4>  target_header_of_dataset()

    target_header=target_header_of_dataset(df)
    print("target header    :",target_header)

    #   5>  datatypes off headers()

    datatypes=datatypes_of_headers(df)
    print("datatypes    :",datatypes)
    #   As all headres having datatypes in numeric format so we dont need to encode the data



    #   6>  shuffle_the_data
    df=shuffle_the_data(df)
    print(df.head)
    print("Shuffled success")


    #   7>  statistical_info()
    stat_info=statistical_info(df)
    print("statistical information  :",stat_info)



    #   8>  checking_null_values()

    null_values=checking_null_values(df)
    print("null values  :",null_values)
    #    there are no null values present in the dataset



    #   9>  boxplot_of_dataset()

    boxplot_of_dataset(df)


    #   10> split_data_vertically()

    X,Y=split_data_vertically(df,target_header)
    print(X.shape)
    print(Y.shape)

    #   11> split_data_into_four_parts()

    X_train,X_test,Y_train,Y_test=split_data_into_four_parts(X,Y)
    print("shape of X_train :",X_train.shape)
    print("shape of X_test :",X_test.shape)
    print("shape of Y_train :",Y_train.shape)
    print("shape of Y_test :",Y_test.shape)


    #   12> using_standerd_scaler()

    scaler,X_train_scaled,X_test_scaled=using_standerd_scaler(X_train,X_test)



    #   13> knn_classifier()

    knn_model,KNN_accuracy=knn_classifier(X_train_scaled,X_test_scaled,Y_train,Y_test)

    #   14> decision_tree_classifier()

    decision_tree_model,decision_tree_accuracy=decision_tree_model_train(X_train_scaled,X_test_scaled,Y_train,Y_test)

    #   15> save_model()



    best_model,name_of_file=save_model(knn_model,KNN_accuracy,decision_tree_model,decision_tree_accuracy,"best_model.pkl")


    #   16> load model

    loaded_model=load_model(name_of_file)
    print("loaded sucess")


    #   17> predict_model()

    predict_model(loaded_model,scaler)


if __name__=="__main__":
    main()