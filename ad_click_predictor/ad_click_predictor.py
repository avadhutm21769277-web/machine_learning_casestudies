

#################################################################################

#   required libraries  :

#################################################################################

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import joblib


#################################################################################

#   function name 1    :   load_data()
#   input              :   filename.csv file
#   output             :   dataframe as df
#   description        :   converting csv file into dataframe

#################################################################################

def load_data(dataset):
    df=pd.read_csv(dataset)
    return df

#################################################################################

#   function name 2    :   header_of_dataset()
#   input              :   dataframe as df
#   output             :   all headers
#   description        :   it gives name of all headers i.e feature headers and target
#                          header

#################################################################################

def header_of_dataset(df):
    headers=df.columns[:]
    return headers


##################################################################################

#   function name 3     :       feature_headers()
#   input               :       dataframe as df
#   output              :       name of feature headers
#   description         :       it gives name of input headers/  independent variables

##################################################################################

def feature_headers_of_dataset(df):
    feature_headers=df.columns[0:8]
    return feature_headers


#################################################################################

#   function name   4   :       target_header()
#   input               :       dataframe as df
#   output              :       name of target variable
#   description         :       it gives name of variable/ dependent variable

#################################################################################

def target_header_of_dataset(df):
    target_header=df.columns[8:]
    return target_header

##################################################################################

#   function name 5     :       datatypes_of_headers
#   input               :       dataframe as df
#   output              :       datatypes of all headers

##################################################################################

def datatypes_of_headers(df):
    datatypes=df.dtypes
    return datatypes

#################################################################################

#   function name   6   :   drop_unnecessary_columns()
#   input               :   dataframe as df
#   output              :   updated dataframe as df


#################################################################################

def drop_unnecessary_columns(df):
    df=df.drop(columns=['id','full_name'],inplace=False)
    return df

#################################################################################

#   function name   7   :   replece_null_values()
#   input               :   dataframe as df
#   output              :   dataframe as df where null vallues filled by mean values


#################################################################################

def replece_null_values(df):
    print(df.isnull().sum())
    #   if we remove null values then dataset became too small ###
    #   we replace the null values by string-"missing"
    df=df.replace(np.nan,'missing')
    return df

#################################################################################

#   function name   8   :   encoding_by_map()
#   input               :   dataframe as df
#   output              :   updated dataframe as df with encoding


#################################################################################

def encoding_by_map(df):

    df['age'] = df['age'].replace("missing", -1)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')   #   object dataype converts into integer


    df['gender']=df['gender'].map({'Male':1,'Female':2,'Non-Binary':3,'missing':-1})


    #   device_type :    Desktop    Mobile  Tablet        missing
    df['device_type']=df['device_type'].map({'Desktop':1,'Mobile':2,'Tablet':3,'missing':-1})



    #   ad_position :   Top     Side    Bottom  missing
    df['ad_position']=df['ad_position'].map({'Top':1,'Side':2,'Bottom':3,'missing':-1})



    #   browsing_history    :   Shopping    Education   Entertainment   Social Media    News    missing
    df['browsing_history']=df['browsing_history'].map({'Shopping':1,'Education':2,'Entertainment':3,'Social Media':4,'News':5,'missing':-1})

    #   time_of_day :   Afternoon   Night   Evening     Morning     missing
    df['time_of_day']=df['time_of_day'].map({'Afternoon':1,'Night':2,'Evening':3,'Morning':4,'missing':-1})
    return df



#################################################################################

#   function name   8   :   boxplot_visualisation()
#   input               :   dataframe as df
#   output              :   updated dataframe as df with encoding


#################################################################################

def boxplot_visualisation(df):

    plt.figure(figsize=(8,5))
    sns.boxplot(df)
    plt.title("Boxplot of ad_click")
    plt.savefig("boxplot_age.png")
    plt.close()


#################################################################################

#   function name  10   :   split_data_vertically()
#   input               :   updated dataframe as df
#   output              :   X , Y   i.e feature header data, target header data

#################################################################################

def split_data_vertically(df,target_header):
    X=df.drop(columns=target_header,axis=1)
    Y=df[target_header].values.ravel()
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

#   function name  13   :   logistic_regression_classifier()
#   input               :   X_train_scaled,X_test_scaled,Y_train,Y_test
#   output              :   accuracy score of model

##################################################################################

def logistic_regression_classifier(X_train_scaled,X_test_scaled,Y_train,Y_test):


        model=LogisticRegression(C=105.8,max_iter=60,n_jobs=37,random_state=42,penalty='l2',solver='lbfgs')
        model.fit(X_train_scaled,Y_train)
        Y_predict=model.predict(X_test_scaled)
        model_accuracy=accuracy_score(Y_test,Y_predict)
        print(model_accuracy*100)
        return model,model_accuracy


############################################################################

#   function name       :   save_model()
#   input               :   trained model and filename.pkl file
#   output              :   .pkl file

############################################################################

def save_model(model,filename):
    saved_model=joblib.dump(model,filename)
    return saved_model,filename


##############################################################################

#   function name   :   load_model()
#   input           :   filename.pkl file

##############################################################################

def load_model(filename):
    loaded_model=joblib.load(filename)
    return loaded_model


##############################################################################

#   function name   :   predict_model
#   input           :   loaded model


##############################################################################

def predict_new_data(scaler,loaded_model,filename):

    loaded_model = load_model(filename)

    new_data = pd.DataFrame({
    'age': [28],                    # integer
    'gender': [1],                  # Male=1, Female=2, Non-Binary=3
    'device_type': [2],             # Desktop=1, Mobile=2, Tablet=3
    'ad_position': [1],             # Top=1, Side=2, Bottom=3
    'browsing_history': [3],        # Shopping=1, Education=2, Entertainment=3,social_media=4,news=5
    'time_of_day': [2]              # Afternoon=1, Night=2, Evening=3, Morning=4
})

    new_dataframe=pd.DataFrame(new_data)
    new_scaled_data=scaler.transform(new_data)
    prediction = loaded_model.predict(new_scaled_data)
    return prediction


















def main():
    print("CASESTUDY AD_CLICK_PREDICTOR")

    #   1>  load_data()
    df=load_data("ad_click_dataset.csv")
    print("dataframe created")
    print(df.head())
    print(df.shape)

    #   2>  header_of_dataset()
    headers=header_of_dataset(df)
    print("headers   \n :",headers)


    #   3>  feature_headers_of_dataset()
    feature_headers=feature_headers_of_dataset(df)
    print("feature headers are  \n  :",feature_headers)

    #   4>  target_header_of_dataset()
    target_header=target_header_of_dataset(df)
    print("target  variable is  \n  :",target_header)

    #   5>  datatypes_of_headers()
    datatypes=datatypes_of_headers(df)
    print("Dataypes of all headers  \n  :",datatypes)

    #   6>  drop_unnecessary_columns()
    df=drop_unnecessary_columns(df)
    print(df.head())
    print("unnecessery columns removed success")

    #   7>  drop_null_rows()
    df=replece_null_values(df)
    print(df.shape)
    print(df.head())
    print("null values repleced successfully    !!!!")

    #   8> encoding_by_map()
    #   encoding by using map function
    df=encoding_by_map(df)
    print(df.head())
    print(df.isnull().sum())
    print(df.dtypes)
    print(df)


    #   9>  boxplot_visualisation()

    boxplot_visualisation(df)
    print("boxplot of ad click dataset has been saved into current directory !!!")

    #   10> split_data_vertically()

    X,Y=split_data_vertically(df,target_header)
    print("dataset splitted into two parts i.e  feature header & target_header")

    #   11> split_data_into_four_parts()

    X_train,X_test,Y_train,Y_test=split_data_into_four_parts(X,Y)
    print("shape of X_train :",X_train.shape)
    print("shape of X_test :",X_test.shape)
    print("shape of Y_train :",Y_train.shape)
    print("shape of Y_test :",Y_test.shape)

    #   12> using_standerd_scaler()

    scaler,X_train_scaled,X_test_scaled=using_standerd_scaler(X_train,X_test)
    print("scaled success")


    #   13> logistic_regression_classifier()

    model,model_accuracy=logistic_regression_classifier(X_train_scaled,X_test_scaled,Y_train,Y_test)

    #   14> save_model()
    saved_model,filename=save_model(model,'adclick.pkl')

    #   15> load_model()
    loaded_model=load_model(filename)

    #   16> predict_new_data()


    prediction = predict_new_data(scaler,loaded_model,filename )
    print("Predicted Ad Click for new user:", prediction[0])






if __name__=="__main__":
    main()