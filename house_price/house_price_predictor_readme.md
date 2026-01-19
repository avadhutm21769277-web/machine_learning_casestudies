##################################		 House Price Predictor (Regression-Based)		#######################################################

##	 Project Overview----->>>

This project is a **house price prediction model** built using **machine learning regression algorithms**. 
It uses historical housing data to predict house prices based on features like area, bedrooms, bathrooms, stories, and amenities.


################################################################################################################################

##	Key Features------>>>

- Load and inspect dataset
- Preprocess features (handle missing values, scaling, one-hot encoding)
- Train regression model (Linear Regression)
- Save and load trained model using `joblib`
- Predict house price for new input data

#################################################################################################################################

## 	Required Libraries	----->>>


------------
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
------------

#################################################################################################################################

		Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```



## Project Structure
```
house_price_predictor/
├── Housing.csv              # Dataset
├── house_price_predictor.py # Main Python script
├── houseprice.pkl           # Saved trained model (after running code)
└── README.md                # This file


#################################################################################################################################```

## ##	##	##	------->>>>	Functions

### 1.load_data(dataset)
- **Input:** CSV file path  
- **Output:** pandas DataFrame  
- **Description:** Reads CSV and returns dataframe.

### 2️. name_of_headers(df)
- **Input:** DataFrame  
- **Output:** List of column names  
- **Description:** Returns headers of dataset.

### 3️. feature_header_of_dataset(df)
- **Input:** DataFrame  
- **Output:** List of feature headers  
- **Description:** Returns feature columns used for training.

### ️4. target_header_of_dataset(df)
- **Input:** DataFrame  
- **Output:** Target column name  
- **Description:** Returns target column used for prediction.

### 5️. preprocessing_pipeline(df)
- **Input:** DataFrame  
- **Output:** Transformed feature DataFrame, target column, preprocessor object  
- **Description:** Handles missing values, scales numeric features, one-hot encodes categorical features.

### 6️.model_pipeline(X, Y)
- **Input:** Transformed features and target  
- **Output:** Trained regression model, R² score  
- **Description:** Splits dataset into train/test, fits Linear Regression, calculates R².

### 7️.save_model(model, filename)
- **Input:** Trained model, filename  
- **Output:** Saved `.pkl` file  
- **Description:** Saves model to disk using `joblib`.

### 8️.load_model(filename)
- **Input:** Filename of saved model  
- **Output:** Loaded model object  
- **Description:** Loads previously saved model.

### 9️.predict_new_data(loaded_model, preprocessor)
- **Input:** Loaded model, preprocessor  
- **Output:** Predicted house price  
- **Description:** Takes new input features, transforms them, predicts price.

#################################################################################################################################


## How to Run
1. Place `Housing.csv` in the same folder as the script.  
2. Run the script:
```bash
python house_price_predictor.py
```
3. Script workflow:
   - Loads dataset
   - Preprocesses features
   - Trains regression model
   - Saves and loads model
   - Predicts price for new data

## Example
```python
Predicted Price: 1234567.89

#################################################################################################################################

## Notes
- Ensure **new input data columns** match the **train dataset features**.  
- You can replace Linear Regression with other regression models like **Random Forest**, **Gradient Boosting**, or **Decision Tree Regressor** for better accuracy.

#################################################################################################################################

AUTHOR	:	Avadhut yashwwant mote