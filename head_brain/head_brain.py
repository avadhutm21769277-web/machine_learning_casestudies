
################################################################################

#   REQUIRED LIBRARIES  :

################################################################################

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
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

#       function name 2  :   name_of_headers()
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
    feature_headers=df.columns[:3]
    return feature_headers



#########################################################################################

#       function name 4  :   target_header_of_dataset()
#       input            :   daframe as df
#       output           :   target_header
#       description      :   is gives the name of column name to predict the model

##########################################################################################

def target_header_of_dataset(df):
    target_header=df.columns[3:]
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
#       description      :   few columns info arranged in sorted format,
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
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.5,random_state=42)
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

#   function name  13   :   linear_regression()
#   input               :   X_train_scaled,X_test_scaled,Y_train,Y_test
#   output              :   accuracy score of decision tree model

##################################################################################



def linear_regression(X_train_scaled,X_test_scaled,Y_train,Y_test):



    lrmodel=LinearRegression(n_jobs=15,copy_X=False,fit_intercept=True,)
    lrmodel.fit(X_train_scaled,Y_train)
    y_predict=lrmodel.predict(X_test_scaled)
    r2_value=r2_score(Y_test,y_predict)

    mean_sq_error=mean_squared_error(Y_test,y_predict)

    return lrmodel,r2_value,mean_sq_error


############################################################################

#   function name       :   save_model()
#   input               :   trained model and filename.pkl file
#   output              :   .pkl file

############################################################################

def save_model(lrmodel,filename):
    saved_model=joblib.dump(lrmodel,filename)
    return saved_model,filename

##############################################################################

#   function name   :   load_model()
#   input           :   filename.pkl file

##############################################################################
def load_model(filename):
    loaded_model=joblib.load(filename)
    return loaded_model

##############################################################################

#   function name   :   predict_new_data()
#   input           :   loaded model


##############################################################################

def predict_new_data(scaler,loaded_model):
        new_data=pd.DataFrame({
        'Gender':[1],
        'Age Range':[1],
        'Head Size(cm^3)':[4008]
    })

        new_scaled_data=scaler.transform(new_data)
        prediction=loaded_model.predict(new_scaled_data)
        return prediction












##################################################################################

#       function name   :   main()
#       description     :   calling to all user_defined functions
#                           i.e main workflow

#################################################################################




def main():
    print("HEAD_BRAIN MODEL")

    #   1>  load_data()

    csv_filename='MarvellousHeadBrain.csv'
    df=load_data(csv_filename)
    print("dataframe created sucess !!")
    print(df.head())


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

    #   13> linear_regression()
    lrmodel,r2_value,mean_sq_error=linear_regression(X_train_scaled,X_test_scaled,Y_train,Y_test)
    print("R2 score by regression model    :",r2_value)
    print(mean_sq_error)

    #   14> save_model()
    filename='head_brain.pkl'
    saved_model,filename=save_model(lrmodel,filename)
    print("model saved success")

    #   16> load_model()
    loaded_model=load_model(filename)
    print("model loaded succesfully !!!!")

    #   17> predict_new_data()
    prediction=predict_new_data(scaler,loaded_model)
    print("predicted answer of brain size is",prediction)




if __name__=="__main__":
    main()