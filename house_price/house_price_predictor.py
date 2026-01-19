
################################################################################

#   REQUIRED LIBRARIES  :

################################################################################

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import joblib





####################################################################################

#       function name 1  :   load_data()
#       input            :   file_name.csv file
#       output           :   dataframe as df
#       description      :   converting csv file into dataframe (rows and columns)

####################################################################################

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
    feature_headers=df.columns[1:11]
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

#       function name 4  :   preprocessing_pipeline()
#       input            :   daframe as df
#       output           :   target_header
#       description      :   feature datatypes, missing null values-scaler,one hot encoding

##########################################################################################

def preprocessing_pipeline(df):


    #   splitting data  :   feature header and target header

    fearure_headers=feature_header_of_dataset(df)
    target_header=target_header_of_dataset(df)
    X=df[fearure_headers]
    Y=df[target_header]



    #   identify datatypes  :   catagorigral and numeric

    numeric_features=X.select_dtypes(include=['int64','float64']).columns.tolist()
    catagorical_feature=X.select_dtypes(include=['object']).columns.tolist()
    print("numeric_features :",numeric_features)
    print("catagorical_feature  : ",catagorical_feature)


    #   creating transformers       :

    numeric_transformers = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='mean')),    # filling null values by mean
    ('scaler', StandardScaler())

    ])


    # Categorical pipeline
    categorical_transformers = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),   # filling null values by most frequent
    ('onehot', OneHotEncoder(handle_unknown='ignore'))

    ]
    )

    #   combine in column transformer   :

    preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformers, numeric_features),
        ('cat', categorical_transformers, catagorical_feature)
    ]
)

    print(df.head())


    X_transformed = preprocessor.fit_transform(X)

    # Get one-hot encoded column names
    encoded_columns = list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(catagorical_feature))
    all_columns = numeric_features + encoded_columns

    # Convert to DataFrame
    X_transformed_df = pd.DataFrame(X_transformed, columns=all_columns)
    print(X_transformed_df.head())





    return X_transformed_df,Y,preprocessor




#########################################################################################

#       function name 4  :   model_pipeline()
#       input            :   daframe as df
#       output           :   target_header
#       description      :   feature datatypes, missing null values-scaler,one hot encoding

##########################################################################################

def model_pipeline(X_tranformed_df,Y):

    X_train,X_test,Y_train,Y_test=train_test_split(X_tranformed_df,Y,test_size=0.7,random_state=42)

    reg_model_pipeline=Pipeline(

        steps=
        [
            ('model',LinearRegression())

        ]

    )

    reg_model_pipeline.fit(X_train,Y_train)
    Y_pred=reg_model_pipeline.predict(X_test)

    r2_value=r2_score(Y_test,Y_pred)
    print("R^2 values   :",r2_value)
    return reg_model_pipeline,r2_value



    ############################################################################

#   function name       :   save_model()
#   input               :   trained model and filename.pkl file
#   output              :   .pkl file

############################################################################

def save_model(model,filename):
    saved_model=joblib.dump(model,filename)
    return saved_model,filename


##############################################################################



##############################################################################

#   function name   :   load_model()
#   input           :   filename.pkl file

##############################################################################

def load_model(filename):
    loaded_model=joblib.load(filename)
    return loaded_model



def predict_new_data(loaded_model,preprocessor):
    new_data = pd.DataFrame(
        {
            'area': [9960],
            'bedrooms': [3],
            'bathrooms': [2],
            'stories': [2],
            'mainroad': ['yes'],
            'guestroom': ['no'],
            'basement': ['yes'],
            'hotwaterheating': ['no'],
            'airconditioning': ['yes'],
            'parking': [2],
            'prefarea': ['yes'],
            'furnishingstatus': ['semi-furnished']
        }
    )

    # Transform and predict directly
    X_new_transformed = preprocessor.transform(new_data)
    predicted_price = loaded_model.predict(X_new_transformed)
    print("Predicted Price:", predicted_price[0])




    def proce_prediction():
        pass











##################################################################################

#       function name   :   main()
#       description     :   calling to all user_defined functions
#                           i.e main workflow

#################################################################################

def main():
    print("hous eprice predictor MODEL REGRESSION BASED")

    dataset='Housing.csv'

        #   1>  load_data()

    df=load_data(dataset)
    print("datafeame created")
    print(df.head())
    print(df.shape)


        #   2>  name_of_headers()

    headers=name_of_headers(df)
    print("headers are  :",headers)



    #   3>  feature_header_of_dataset()

    feature_headers=feature_header_of_dataset(df)
    print("feature headers are  :",feature_headers)


        #   4>  target_header_of_dataset()

    target_header=target_header_of_dataset(df)
    print("target header    :",target_header)

    #   5>  preprocessing_pipeline()
    X_tranformed_df,Y,preprocessor=preprocessing_pipeline(df)

    print(X_tranformed_df.head())
    print(Y.shape)


    #   6> model_pipeline()

    model,r2value=model_pipeline(X_tranformed_df,Y)

    #   7> save_model()
    saved_model,filename=save_model(model,'houseprice.pkl')
    print("model saved!!!!")


     #   8> load_model()
    loaded_model=load_model(filename)
    print("model loaded suceess")


    # 9> predict_new_data()
    predict_new_data(loaded_model, preprocessor)

    #proce preduction

    print("this program has been executed succesfully")



if __name__=="__main__":
    main()




