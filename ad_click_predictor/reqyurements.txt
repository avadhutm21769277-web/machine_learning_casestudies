############################			 Ad Click Predictor			######################################################

This project is a **Machine Learning pipeline** for predicting whether a user will click on an ad based on demographic and behavioral features. 
The pipeline includes data preprocessing, encoding, scaling, model training, evaluation, and saving/loading the trained model.
 It also supports predicting new user data.


#############################################################################################################################################


######		 Features		:--->>>

* Load CSV dataset and inspect headers and datatypes
* Handle missing values
* Encode categorical features
* Visualize dataset using boxplots
* Split dataset into training and testing sets
* Scale features using `StandardScaler`
* Train `Logistic Regression` model
* Save and load trained model (`.pkl`)
* Predict new user ad clicks


###############################################################################################################################################



######		 Dataset		:---->>>

* The dataset should be in CSV format (`ad_click_dataset.csv`)
* Features include: `age`, `gender`, `device_type`, `ad_position`, `browsing_history`, `time_of_day`
* Target variable: `ad_click` (0 = no click, 1 = click)

################################################################################################################################################



######		 Installation		:----->>>

1. Clone this repository:

```bash
git clone https://github.com/your-username/ad-click-predictor.git
cd ad-click-predictor
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python ad_click_predictor.py
```

The script will:

1. Load the dataset
2. Preprocess and encode features
3. Train a logistic regression model
4. Save the trained model
5. Load the model and predict a new user

#############################################################################################################################################


---

##### Predicting New Users

Example of predicting a new user:

```python
new_user = {
    'age': [28],
    'gender': [1],
    'device_type': [2],
    'ad_position': [1],
    'browsing_history': [3],
    'time_of_day': [2]
}

prediction = predict_new_data(scaler, loaded_model)
print("Predicted Ad Click for new user:", prediction[0])
```

-#####################################################################################################################################################


##########	functions used:		________>>>>


Functions Used

	load_data(dataset): Load CSV file into pandas DataFrame

	header_of_dataset(df): Return all dataset headers

	feature_headers_of_dataset(df): Return feature (input) headers

	target_header_of_dataset(df): Return target (dependent) header

	datatypes_of_headers(df): Check datatype of each column

	drop_unnecessary_columns(df): Drop irrelevant columns like id and full_name

	replece_null_values(df): Replace null/NaN values with 'missing'

	encoding_by_map(df): Encode categorical features into numeric values

	boxplot_visualisation(df): Save boxplot for dataset visualization

	split_data_vertically(df, target_header): Split dataset into features (X) and target (Y)

	split_data_into_four_parts(X, Y): Split data into training and testing sets

	using_standerd_scaler(X_train, X_test): Scale features using StandardScaler

	logistic_regression_classifier(X_train_scaled, X_test_scaled, Y_train, Y_test): Train Logistic Regression model and return accuracy

	save_model(model, filename): Save trained model as .pkl file

	load_model(filename): Load saved model

	predict_new_data(scaler, loaded_model): Predict ad click for new user input

############################################################################################################################################################
--

##	AUTHOR	:	AVADHUT YASHWANT MOTE
