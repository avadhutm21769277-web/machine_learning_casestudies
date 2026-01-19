

################################################################################
# REQUIRED LIBRARIES
################################################################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

################################################################################
# Load Dataset
################################################################################
def load_data(dataset_name):
    df = pd.read_csv(dataset_name)
    df.drop_duplicates(inplace=True)  # Remove duplicates
    return df

################################################################################
# Preprocessing
################################################################################
def preprocessing_pipeline(df):
    # Remove unnecessary / leakage columns
    df = df.drop(columns=['ID', 'year', 'Credit_Worthiness', 'dtir1'])

    X = df.drop(columns=['Status'])
    Y = df['Status']

    # Identify numeric / categorical
    numeric_features = X.select_dtypes(['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(['object']).columns.tolist()

    # Transformers
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    X_transformed = preprocessor.fit_transform(X)

    # Column names after one-hot encoding
    encoded_cols = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
    all_columns = numeric_features + list(encoded_cols)
    X_transformed_df = pd.DataFrame(X_transformed, columns=all_columns)

    return X_transformed_df, Y, preprocessor

################################################################################
# Ensemble Model Pipeline
################################################################################
def model_pipeline(X_transformed_df, Y):
    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X_transformed_df, Y, test_size=0.2, random_state=42, shuffle=True)

    # Base models
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=7, random_state=42)

    ensemble = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')

    # Train
    ensemble.fit(X_train, Y_train)

    # Predict
    Y_pred = ensemble.predict(X_test)

    # Metrics
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"Ensemble Model Accuracy: {accuracy*100:.2f}%")
    print(classification_report(Y_test, Y_pred))
    print("ROC-AUC:", roc_auc_score(Y_test, ensemble.predict_proba(X_test)[:,1]))

    # Cross-validation
    cv_scores = cross_val_score(ensemble, X_transformed_df, Y, cv=5, scoring='roc_auc')
    print("CV ROC-AUC scores:", cv_scores)
    print("Mean CV ROC-AUC:", cv_scores.mean())

    return ensemble, accuracy

################################################################################
# Main
################################################################################
def main():
    dataset_name = 'Loan_Default.csv'
    df = load_data(dataset_name)

    X_transformed_df, Y, preprocessor = preprocessing_pipeline(df)
    ensemble, accuracy = model_pipeline(X_transformed_df, Y)

if __name__ == "__main__":
    main()
