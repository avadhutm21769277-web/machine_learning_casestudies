################################################################################

#   REQUIRED LIBRARIES  :

################################################################################

line="__________"*12

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,VotingClassifier
from sklearn.metrics import accuracy_score,classification_report,roc_auc_score, confusion_matrix


##################################################################################

#       function name 1  :   load_data()
#       input            :   file_name.csv file
#       output           :   dataframe as df
#       description      :   converting csv file into dataframe (rows and columns)

##################################################################################

def load_data(dataset_name):
    try:
        df=pd.read_csv(dataset_name)
        return df
    except FileNotFoundError:
        print(f"Error: The file '{dataset_name}' was not found.")
        return None

##################################################################################

#       function name 2  :   investigate_data_leakage()
#       input            :   dataframe as df
#       output           :   None (prints analysis)
#       description      :   Checks for features that might be leaking target info

##################################################################################

def investigate_data_leakage(df):
    print("Investigating potential data leakage...")
    print("-" * 50)

    # Check if Credit_Worthiness perfectly aligns with Status
    if 'Credit_Worthiness' in df.columns:
        print("Relationship between Credit_Worthiness and Status:")
        print(pd.crosstab(df['Credit_Worthiness'], df['Status']))
        print("-" * 50)
        # A perfect separation here is a huge red flag for leakage.

    # Check dtir1
    if 'dtir1' in df.columns:
        print("Descriptive stats of dtir1 for each Status:")
        print(df.groupby('Status')['dtir1'].describe())
        print("-" * 50)

    # Check business_or_commercial
    if 'business_or_commercial' in df.columns:
        print("Relationship between business_or_commercial and Status:")
        print(pd.crosstab(df['business_or_commercial'], df['Status']))
        print("-" * 50)


#########################################################################################

#       function name 3  :   preprocessing_pipeline()
#       input            :   dataframe as df
#       output           :   X_transformed_df, Y, preprocessor
#       description      :   Cleans data, handles missing values, and prepares features for modeling

##########################################################################################

def preprocessing_pipeline(df):
    # --- CRITICAL STEP: AVOIDING DATA LEAKAGE ---
    # We must drop columns that are determined *after* the loan decision.
    # 'ID' is just an identifier. 'year' might be relevant but let's drop it for simplicity.
    # 'Credit_Worthiness', 'dtir1', 'business_or_commercial' are likely post-decision features.
    # We are keeping 'Status' as our target Y.
    columns_to_drop = ['ID', 'year', 'Credit_Worthiness', 'dtir1', 'business_or_commercial']
    df_cleaned = df.drop(columns=columns_to_drop)
    print(f"Dropped columns to prevent leakage: {columns_to_drop}")
    print("Shape after dropping columns:", df_cleaned.shape)
    print(line)

    # Shuffle the data to ensure randomness
    df_cleaned = df_cleaned.sample(frac=1, random_state=42).reset_index(drop=True)
    print("Data shuffled successfully.")

    # --- ROBUST FEATURE AND TARGET SELECTION ---
    # Define target and features using column names, not indices.
    target_header = 'Status'
    feature_headers = [col for col in df_cleaned.columns if col != target_header]

    X = df_cleaned[feature_headers]
    Y = df_cleaned[target_header]
    print(f"Target variable: '{target_header}'")
    print(f"Number of features: {len(feature_headers)}")
    print(X.shape)
    print(Y.shape)
    print(line)

    # Identify datatypes in feature headers
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    print("Numeric Features:", numeric_features)
    print("Categorical Features:", categorical_features)
    print(line)

    # Create transformers for numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    print("Numeric transformer created.")

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    print("Categorical transformer created.")

    # Combine transformers into a preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # Fit and transform the data
    X_transformed = preprocessor.fit_transform(X)

    # Get feature names after one-hot encoding
    encoded_columns = list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features))
    all_columns = numeric_features + encoded_columns

    # Convert to DataFrame for better readability
    X_transformed_df = pd.DataFrame(X_transformed, columns=all_columns)
    print("Preprocessing complete. Transformed feature data head:")
    print(X_transformed_df.head())

    return X_transformed_df, Y, preprocessor


#########################################################################################

#       function name 4  :   model_pipeline()
#       input            :   X_transformed_df, Y
#       output           :   trained model, accuracy
#       description      :   Trains and evaluates the machine learning model

##########################################################################################

def model_pipeline(X_transformed_df, Y):
    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X_transformed_df, Y, test_size=0.2, random_state=42, stratify=Y)
    # stratify=Y ensures the proportion of defaults is the same in train and test sets

    # Base Models
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=7, random_state=42)

    # Ensemble Model
    ensemble = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')

    # Train the model
    print("Training the ensemble model...")
    ensemble.fit(X_train, Y_train)
    print("Training complete.")

    # Make predictions
    Y_pred = ensemble.predict(X_test)

    # --- EVALUATION ---
    # Accuracy
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"\nEnsemble Model Accuracy: {accuracy*100:.2f}%")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(Y_test, Y_pred))

    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(Y_test, Y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # ROC-AUC Score
    if hasattr(ensemble, "predict_proba"):
        roc_auc = roc_auc_score(Y_test, ensemble.predict_proba(X_test)[:, 1])
        print(f"ROC-AUC Score: {roc_auc:.4f}")

    # Cross-Validation for a more robust estimate
    cv_scores = cross_val_score(ensemble, X_transformed_df, Y, cv=5, scoring='roc_auc')
    print("\nCross-Validation ROC-AUC scores:", cv_scores)
    print("Mean CV ROC-AUC:", cv_scores.mean())

    return ensemble, accuracy


#########################################################################################

#       function name 5  :   find_leaking_feature()
#       input            :   dataframe as df
#       output           :   None (prints detailed analysis)
#       description      :   Brute-force check to find the exact feature causing leakage

##########################################################################################

def find_leaking_feature(df):
    print("DEBUGGING: Checking each feature against the target 'Status' to find leaks...")
    print("="*60)
    target = 'Status'

    # We need to drop the obvious leaks first to focus on the remaining ones
    columns_to_ignore = ['ID', 'year', 'Credit_Worthiness', 'dtir1', 'business_or_commercial']
    df_debug = df.drop(columns=columns_to_ignore)

    potential_leaks = []
    for col in df_debug.columns:
        if col == target:
            continue

        print(f"\n--- Analyzing Feature: '{col}' ---")

        if df_debug[col].dtype == 'object':
            # For categorical features, use crosstab
            print(pd.crosstab(df_debug[col], df_debug[target]))
            # A simple check for perfect separation
            if df_debug.groupby(col)[target].nunique().max() == 1:
                print(f"*** POTENTIAL LEAK DETECTED in '{col}'! It perfectly separates the classes. ***")
                potential_leaks.append(col)
        else:
            # For numerical features, use describe
            print(df_debug.groupby(target)[col].describe())
            # A simple check: if min/max of one class don't overlap with the other at all
            desc = df_debug.groupby(target)[col].describe()
            # This is a heuristic, not foolproof, but a good indicator
            if (desc.loc[0, 'min'] > desc.loc[1, 'max']) or (desc.loc[1, 'min'] > desc.loc[0, 'max']):
                 print(f"*** POTENTIAL LEAK DETECTED in '{col}'! Its ranges do not overlap. ***")
                 potential_leaks.append(col)

    print("\n" + "="*60)
    if potential_leaks:
        print("SUMMARY: The following features are likely leaking information and should be dropped:")
        for leak in potential_leaks:
            print(f"- {leak}")
    else:
        print("SUMMARY: No obvious leaking features found with this simple check. The issue might be more subtle.")
    print("="*60)



##################################################################################

#       function name   :   main()
#       description     :   Main workflow to run the entire process

#################################################################################

def main():
    print(line)
    print("LOAN DEFAULT PREDICTOR CASESTUDY (Improved Version)")
    print(line)

    # 1. Load Data
    dataset_name = 'Loan_Default.csv'
    df = load_data(dataset_name)
    if df is None:
        return # Stop if data loading fails

    print("Data loaded successfully.")
    print(df.head())
    print("Original data shape:", df.shape)
    print(line)

    # 2. Investigate Data Leakage (for understanding)
    investigate_data_leakage(df)
    print(line)

    # 3. Preprocess Data (with leakage prevention)
    X_transformed_df, Y, preprocessor = preprocessing_pipeline(df)
    print(line)

    # 4. Train and Evaluate Model
    model, accuracy = model_pipeline(X_transformed_df, Y)
    print(line)

    print(f"\nFinal Model Accuracy on Test Set: {accuracy*100:.2f}%")
    print("This accuracy is more realistic as the model was trained on data available at the time of application.")


if __name__=="__main__":
    main()