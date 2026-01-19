
################################################################################

#   REQUIRED LIBRARIES  :

################################################################################
line='___'*30
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import joblib




##################################################################################

#       function name 1  :   load_data()
#       input            :   file_name.csv file
#       output           :   dataframe as df
#       description      :   converting csv file into dataframe (rows and columns)

##################################################################################

def load_data(dataset):
    df=pd.read_csv(dataset)
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
    feature_headers=df.drop(df.columns[4],axis=1).columns.tolist()
    return feature_headers


#########################################################################################

#       function name 4  :   target_header_of_dataset()
#       input            :   daframe as df
#       output           :   target_header
#       description      :   is gives the name of column name to predict the model

##########################################################################################

def target_header_of_dataset(df):
    target_header=df.columns[4:5]
    return target_header

#########################################################################################

#       function name 4  :   remove_unnecessory_columns()
#       input            :   daframe as df
#       output           :   updated dataframe as df
#       description      :   removing un-necessary columns

##########################################################################################

def remove_unnecessory_columns(df):
    df=df.drop(columns='Date ')
    return df

#########################################################################################

#       function name 4  :   split_data()
#       input            :   updated daframe as df
#       output           :   X as feature header, Y as target header
#       description      :   this function split dataset into two parts
#                               i.e  feature headers and target header

##########################################################################################

def split_data(df):
    X=df.drop(columns='Close ')
    Y=df['Close ']
    print("data splitted into feature header and targget header success !!")
    print(line)
    return  X,Y


#########################################################################################

#       function name 4  :   checking_null()
#       input            :   updated daframe as df
#       output           :   check whethere null values presents or not
#       description      :

##########################################################################################

def  checking_null(df):
    print(df.isnull().sum())
    ##  there are no null values presents in the dataset



#########################################################################################

#       function name 4  :   train_data()
#       input            :   X,Y
#       output           :   X_train,X_test,Y_train,Y_test
#       description      :   this functions gives 4 parts of data for training and testing perpose

##########################################################################################


def train_data(X,Y):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
    return X_train,X_test,Y_train,Y_test


#########################################################################################

#       function name 4  :   scaled_data()
#       input            :   X_train,X_test
#       output           :   X_train_scaled,X_test_scaled
#       description      :   this function gives scaled data

##########################################################################################


def scaled_data(X_train,X_test):
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    return scaler,X_train_scaled,X_test_scaled












#########################################################################################

#       function name 4  :   model_selection()
#       input            :   X_train,X_test,Y_train,Y_test
#       output           :   accuracy of the model
#       description      :


##########################################################################################

def model_selection(X_train_scaled,X_test_scaled,Y_train,Y_test):
    model=LinearRegression(n_jobs=15)
    model.fit(X_train_scaled,Y_train)
    Y_pred=model.predict(X_test_scaled)
    r2_value=r2_score(Y_test,Y_pred)
    print("r2 score",r2_value)
    return model,r2_value



############################################################################

#   function name       :   save_model()
#   input               :   trained model and filename.pkl file
#   output              :   .pkl file

###########################################################################

def save_model(model,filename):
    joblib.dump(model,filename)
    return filename



##############################################################################

#   function name   :   load_model()
#   input           :   filename.pkl file

##############################################################################
def load_model(filename):
    loaded_model=joblib.load(filename)
    return loaded_model


##############################################################################

#   function name   :   predict_new_data()
#   input           :   filename.pkl file

##############################################################################

def predict_new_data(scaler,loaded_model):

    new_data=pd.DataFrame(

        {
            'Open ':[57872.85],
            'High ':[58224.00],
            'Low ':[57872.85],
            'Shares Traded ':[43334000],
            'Turnover (₹ Cr)':[2387.96],






        }




    )

    new_scaled_data=scaler.transform(new_data)
    prediction=loaded_model.predict(new_scaled_data)
    return prediction















##################################################################################

#       function name   :   main()
#       description     :   calling to all user_defined functions
#                           i.e main workflow

#################################################################################

def main():
    dataset='NIFTY BANK-16-10-2024-to-16-10-2025.csv'
    print(line)
    print("banknifty analyser...")
    print(line)

    print(line)

    #   1>  load_data

    df=load_data(dataset)
    print(df.head())
    print(line)


    #   2>  name_of_headers
    headers=name_of_headers(df)
    print(headers)
    print(line)

    #   3>  feature_header_of_dataset

    feature_headers=feature_header_of_dataset(df)
    print("feature headers are  :")
    print(feature_headers)
    print(line)


    #   4>  target_header_of_dataset
    target_header=target_header_of_dataset(df)
    print("target headers   :")
    print(target_header)
    print(line)

    #   5   >   remove_unnecessory_columns
    df=remove_unnecessory_columns(df)
    print(df.head())
    print(df.shape)
    print(line)

    #   6>  split_data

    X,Y=split_data(df)
    print(line)
    print("shape of feature headers",X.shape)
    print("shape of target header",Y.shape)
    print(line)

    #   7>  checking_null

    checking_null(df)


    #   9>  train_data

    X_train,X_test,Y_train,Y_test=train_data(X,Y)
    print("shape of X_train",X_train.shape)
    print("shape of X_test",X_test.shape)
    print("shape of Y_train",Y_train.shape)
    print("shape of Y_test",Y_test.shape)


    #   8> scaled data
    scaler,X_train_scaled,X_test_scaled=scaled_data(X_train,X_test)
    print("scaled success")
    print(line)

    #  9> model_selection()

    model,r2_value=model_selection(X_train_scaled,X_test_scaled,Y_train,Y_test)

    #   10> save_model

    filename=save_model(model,'banknify.pkl')

    #   16> load_model()
    loaded_model=load_model(filename)
    print("model loaded succesfully !!!!")

    #   17> predict_new_data()
    prediction=predict_new_data(scaler,loaded_model)
    print("predicted closed candle at price may be at",prediction)



if __name__=="__main__":
    main()

    ################################################################################


    ##  AUTHORESED BY : AVADHUT YASHWANT MOTE    (AI/ML TRAINEE)


    ################################################################################