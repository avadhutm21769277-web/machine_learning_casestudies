#####################################			Advertising Predictor Regression Model			######################################################



############		 Project Description	------------>>>>


This project is a **regression-based model** to predict sales of products based on advertising budget allocated for **TV, Radio, and Newspaper**.
The dataset used is `Advertising.csv`.

The model uses **Linear Regression** with **standard scaling** to predict sales and also provides **feature importance** for input variables.

########################################################################################################################################################################


############		 Workflow	----------------->>>>


1. Load dataset (`load_data()`)  
2. Display headers and data information (`name_of_headers()`, `datatypes_of_headers()`)  
3. Check for missing values (`checking_null_values()`)  
4. Statistical summary (`statistical_info()`)  
5. Visualize dataset using boxplots (`boxplot_of_dataset()`)  
6. Remove unnecessary columns (`remove_unnecessary_colums()`)  
7. Split dataset into features and target (`split_into_feature_and_target()`)  
8. Split into training and test sets (`split_dataset()`)  
9. Scale features using `StandardScaler` (`scaled_data()`)  
10. Train Linear Regression model (`linear_regression()`)  
11. Save and load the trained model (`save_model()`, `load_model()`)  
12. Predict new data (`predict_new_data()`)  
13. Display feature importance (`feature_importance()`)



############		 Functions	--------------->>>>>>>>>>


####	Data Loading & Inspection


-load_data(dataset)` – loads CSV into a DataFrame  
-name_of_headers(df)` – returns column names  
-datatypes_of_headers(df)` – returns column data types  
-checking_null_values(df)` – returns number of nulls per column  
-statistical_info(df)` – statistical summary of numeric columns  

#####	 Data Visualization	

-boxplot_of_dataset(df)` – saves a boxplot image (`boxplot.png`) for numeric features  

#####	 Preprocessing

-remove_unnecessary_colums(df)` – removes unwanted columns like `Unnamed: 0`  
-split_into_feature_and_target(df)` – splits into X (features) and Y (target)  
-split_dataset(X,Y)` – splits dataset into training and test sets  
-scaled_data(X_train,X_test)` – scales features using StandardScaler  

######	 Model Training & Evaluation

-linear_regression(X_train_scaled,X_test_scaled,Y_train,Y_test)` – trains Linear Regression, returns model, R² score, and MSE  
-save_model(lrmodel, filename)` – saves model to a `.pkl` file  
-load_model(filename)` – loads model from a `.pkl` file  

### Prediction & Analysis
-predict_new_data(scaler, loaded_model)` – predicts sales for new input data  
-feature_importance(lrmodel, feature_headers)` – shows coefficients of each feature  

###################################################################################################################################################################################


#################################################################################
######		 Requirements	_______>>>>

- Python 3.x  
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  
- joblib  



#####		Install requirements using pip	_________>>>



pip install pandas numpy matplotlib seaborn scikit-learn joblib
```
#################################################################################

#####	 Usage	____________>>>

1. Place `Advertising.csv` in the project folder  
2. Run `advertisement.py`:
	python advertisement.py
3. Outputs:
   - Console prints for data inspection, R² score, and predictions  
   - boxplot.png` saved in the project folder  
   - advertising_model.pkl` saved  

4. Feature importance is displayed in console showing impact of each input feature on sales.  

###################################################################################


## Notes
- R² score ~0.8 on test data, indicating good model performance  
- Newspaper feature may have minimal impact on sales  
- Adjust `test_size` in `split_dataset()` to change train/test ratio  
- New predictions can be made by modifying `predict_new_data()` inputs


####################################################################################

##	AUTHOR	:	Avadhut Yashwant Mote