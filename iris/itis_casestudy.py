


################################################################################

#   REQUIRED LIBRARIES  :

################################################################################


import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score,confusion_matrix



from sklearn.tree import DecisionTreeClassifier
import joblib





##################################################################################

#       function name 1  :   load_data()
#       input            :   file_name.csv file
#       output           :   dataframe as df
#       description      :   converting csv file into dataframe (rows and columns)

##################################################################################

def load_data(dataset_name):
    df=pd.read_csv(dataset_name)
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
    feature_headers=df.columns[0:4]
    return feature_headers


#########################################################################################

#       function name 4  :   target_header_of_dataset()
#       input            :   daframe as df
#       output           :   target_header
#       description      :   is gives the name of column name to predict the model

##########################################################################################

def target_header_of_dataset(df):
    target_headers=df.columns[4:]
    return target_headers


#########################################################################################

#       function name 4  :   preprocessing_pipeline()
#       input            :   daframe as df
#       output           :   target_header
#       description      :   feature datatypes, missing null values-scaler,one hot encoding

##########################################################################################

def preprocessing_pipeline(df,feature_headers,target_header):

    #   1>  shuffle the data

    df=df.sample(frac=1,random_state=42).reset_index(drop=True)
    print(df.head())
    print("shuffled success")



    #   2> split data into feature headers and target header i.e X,Y

    X=df[feature_headers]
    Y=df[target_header]
    print(X.shape)
    print(Y.shape)


    #   3>  identify datatypes  :   numeric and catagorical

    numeric_features=X.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_features=X.select_dtypes(include=['object']).columns.tolist()
    print("numerical features   :",numeric_features)
    print("categorical features :",categorical_features)




    #   4>  creating transformers   :

    numeric_tranformer=Pipeline(

        steps=
        [
        ('imputer',SimpleImputer(strategy='mean')),
        ('scaler',StandardScaler())

        ]
    )


    categorical_tranformer=Pipeline(
        steps=[

            ('imputer',SimpleImputer(strategy='most_frequent')),
            ('onehot',OneHotEncoder(handle_unknown='ignore'))
        ]

    )

    #   5>  preprocessor

    preprocessor = ColumnTransformer([
        ('num', numeric_tranformer, numeric_features),
        ('cat', categorical_tranformer, categorical_features)
    ])

    return X,Y,preprocessor



################################################################################
#   function name   :   Train_model()
#   input           :   X,Y,preprocessor
#   output          :
#   descreption     :
################################################################################
def train_models(X, Y, preprocessor):

    if Y.dtypes[0] == 'object':
        le = LabelEncoder()
        Y_encoded = le.fit_transform(Y.values.ravel())
    else:
        Y_encoded = Y.values.ravel()


    X_train,X_test,Y_train,Y_test=train_test_split(X,Y_encoded,test_size=0.3,random_state=42)


    knn_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier(n_neighbors=7))
    ])
    knn_pipeline.fit(X_train, Y_train)
    Y_pred_knn = knn_pipeline.predict(X_test)
    print("KNN Accuracy:", accuracy_score(Y_test, Y_pred_knn))


    dt_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(random_state=42,max_depth=30))
    ])
    dt_pipeline.fit(X_train, Y_train)
    Y_pred_dt = dt_pipeline.predict(X_test)
    print("Decision Tree Accuracy:", accuracy_score(Y_test, Y_pred_dt))

    return knn_pipeline,dt_pipeline





################################################################################
#  Train models
################################################################################


def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

def load_model(filename):
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model



################################################################################
#  Predict new data

#   result
#   0-setosa
#   1-versicolor
#   2-virginica

################################################################################
def predict_new_data(model):
    new_data = pd.DataFrame({
        'sepal.length': [5.1],
        'sepal.width': [3.5],
        'petal.length': [1.4],
        'petal.width': [0.2]
    })
    prediction = model.predict(new_data)
    print("Predicted class:", prediction[0])
































##################################################################################

#       function name   :   main()
#       description     :   calling to all user_defined functions
#                           i.e main workflow

#################################################################################


def main():
    print("Iris predictor by classification")

    #   1>  load_data()
    dataset_name='iris.csv'
    df=load_data(dataset_name)
    print(df.head())
    print("Shape    :",df.shape)

    #   2>  name_of_headers()
    headers=name_of_headers(df)
    print("ALL HEADERS ARE:",headers)

    #   3>  feature_header_of_dataset()
    feature_headers=feature_header_of_dataset(df)
    print("FEATURE HEADERS ARE  :",feature_headers)

    #   4>  target_header_of_dataset()
    target_header=target_header_of_dataset(df)
    print("TARGET HEADERS ARE   :",target_header)

    #   5>  preprocessing_pipeline()
    X,Y,preprocessor=preprocessing_pipeline(df,feature_headers,target_header)

    #   6>  Train_model()

    knn_pipeline, dt_pipeline=train_models(X,Y,preprocessor)



    #     Save models
    save_model(knn_pipeline, 'knn_iris.pkl')
    save_model(dt_pipeline, 'dt_iris.pkl')

    # 6> Load models from disk
    loaded_knn = load_model('knn_iris.pkl')
    loaded_dt = load_model('dt_iris.pkl')



    # 7> Predict new data using loaded models
    predict_new_data(loaded_knn)
    predict_new_data(loaded_dt)








if __name__=="__main__":
    main()