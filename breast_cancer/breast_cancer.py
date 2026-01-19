




#################################################################################################

#       Required Python Packages        :

#################################################################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split









###################################################################################################

#           FILE PATHS  :

###################################################################################################



INPUT_PATH="breast-cancer-wisconsin.data"

OUTPUT_PATH="breast-cancer-wisconsin.csv"



###################################################################################################

#   Headers :

###################################################################################################

HEADERS=["CodeNumber","ClumpThickness","UniformityCellSize","UniformityCellShape","MarginalAdhesion","SingleEpithelialCellSize","BareNuclei","BlandChromatin","NormalNucleoli","Mitoses","CancerType"]




###################################################################################################

#       Function name       :       read_data()
#       Description         :       Read the data into pandas dataframe
#       Input               :       Path of csv file
#       Output              :       Gives the data
#       Author              :       Avadhut Yashwant Mote
#       Date                :

###################################################################################################

def read_data(path):

    df=pd.read_csv(path,header=None)
    return df




###################################################################################################

#       Function name       :       get_headers()
#       Description         :       dataset headers
#       Input               :       dataset
#       Output              :       dataset headers
#       Author              :       Avadhut Yashwant Mote
#       Date                :

###################################################################################################

def get_headers(dataset):
    return dataset.columns[:]



###################################################################################################

#       Function name       :       add_headers()
#       Description         :       add the headers to the dataset
#       Input               :       dataset
#       Output              :       updated dataset
#       Author              :       Avadhut Yashwant Mote
#       Date                :

###################################################################################################


def add_headers(dataset,headers):
    dataset.columns=headers
    return dataset










###################################################################################################################################

#       Function name   :   handle_missing_values()
#       Description     :   filtering missing values from the dataset
#       Input           :   dataset with missing values
#       Output          :   dataset by removing missing values/fill the missing values by mean of respective headers if required
#       Author          :   Avadhut Yashwant Mote
#       Date            :

#####################################################################################################################################


def handle_missing_values(df,feature_headers):
    print(df.isnull().sum())

    #   count of zero values    :
    count=df[df==0].count()
    print("Count of Zero Values :",count)

    #   count of '?'    :
    question_count=(df[feature_headers]=="?").sum()
    print(f"count of ? in {feature_headers}",question_count)
    feature_with_question_mark=question_count[question_count>0].index
    for feature in feature_with_question_mark:
        df[feature]=df[feature].replace("?",np.nan)
        df[feature]=pd.to_numeric(df[feature],errors="coerce")
        mean_value=df[feature].mean()
        df[feature].fillna(mean_value,inplace=True)
    return df






###################################################################################################################################

#       Function name   :   statastical_information()
#       Description     :   give the stastical information regarding updated dataset
#       Input           :   updated dataset
#       Output          :   statastical info like standerd daviation, mean,median etc
#       Author          :   Avadhut Yashwant Mote
#       Date            :

#####################################################################################################################################


def statastical_information(df):
    print("Statastical info :")
    print(df.describe())




######################################################################################################################################

#   Function name   :   split_dataset
#   Description     :   split the dataset into train percentage
#   input           :   dataset with related information
#   output          :   dataset after splitting
#   Author          :   Avadhut Yashwant Mote

######################################################################################################################################


def split_dataset(dataset,train_percentage,feature_header,target_header,random_state=42):

    X_train,X_test,Y_train,Y_test=train_test_split(dataset[feature_header],dataset[target_header],train_size=train_percentage,random_state=random_state,stratify=dataset[target_header])
    return X_train,X_test,Y_train,Y_test




























######################################################################################################


#       function name   :   main

#       Description     :   Main function where execution starts

#       Author          :   Avadhut Yashwant Mote

#       Date            :


######################################################################################################






def main():

    #   converting data file into csv

    #data_file_to_csv()


    #   Load csv

    dataset=read_data(OUTPUT_PATH)
    print("dataframe created sucessfully")

    HEADERS=get_headers(dataset)
    print(HEADERS)

    Added_headers_into_dataset=add_headers(dataset,HEADERS)
    print(Added_headers_into_dataset)

    updated_dataset=handle_missing_values(dataset,HEADERS)



    statastical_information(updated_dataset)

    feature_headers=HEADERS[1:-1]

    target_headers=HEADERS[-1:]

    print(feature_headers)
    print(target_headers)

    X_train,X_test,Y_train,Y_test=split_dataset(dataset,0.7,feature_headers,target_headers)
    print("Shape of X_train :",X_train.shape)
    print("Shape of X_test :",X_test.shape)
    print("Shape of Y_train :",Y_train.shape)
    print("Shape of Y_test :",Y_test.shape)












if __name__=="__main__":
    main()