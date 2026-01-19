################################################################	 Wine Predictor Project		##############################################################




#####	 Overview	#####

This project implements a Wine Class Prediction Model using K-Nearest Neighbors (KNN) and Decision Tree Classifier. The model predicts the type of wine based on various chemical properties of the wine samples.



#####	Dataset		#####

* File Name: WinePredictor.csv

* Features:

  * Alcohol
  * Malic acid
  * Ash
  * Alcalinity of ash
  * Magnesium
  * Total phenols
  * Flavanoids
  * Nonflavanoid phenols
  * Proanthocyanins
  * Color intensity
  * Hue
  * OD280/OD315 of diluted wines
  * Proline

* Target: Wine Class

* The dataset does not contain null values. All features are numerical.




#####	 Features of the Project	####

1. Load CSV data into a Pandas DataFrame.
2. Explore headers, datatypes, and statistical info.
3. Shuffle the data to prevent bias.
4. Visualize data using boxplots to check for outliers.
5. Split data into training and testing sets.
6. Standardize features using StandardScaler.
7. Train KNN and Decision Tree Classifier models.
8. Compare accuracy of both models.
9. Save the best model as a .pkl file using joblib.
10. Load the saved model and make predictions on new samples.



#####	 Libraries Required	#####

* pandas
* numpy
* matplotlib
* scikit-learn (sklearn)
* joblib

Install required libraries using:


pip install pandas numpy matplotlib scikit-learn joblib
```

##### 	How to Run	#####

1. Place WinePredictor.csv in the project directory.
2. Run the main Python script	:	python industrial_formar_wine_predictor_casedtudy.py	


3. The script will:

   * Train both models.
   * Save the best model (best\_model.pkl).
   * Load the saved model.
   * Predict wine class for a new sample.

#####	 Output	#####

* Confusion matrix and accuracy for both models.
* Boxplot visualization saved as boxplot.png.
* Saved best model file: best\_model.pkl.
* Predicted wine class for new data displayed in console.

#####	 Functions	#####

* load\_data(): Load CSV as DataFrame.
* name\_of\_headers(): Get column headers.
* feature\_header\_of\_dataset(): Get feature headers.
* target\_header\_of\_dataset(): Get target column header.
* datatypes\_of\_headers(): Display column datatypes.
* shuffle\_the\_data(): Shuffle dataset.
* statistical\_info(): Display statistical info.
* checking\_null\_values(): Check for null values.
* boxplot\_of\_dataset(): Save boxplot visualization.
* split\_data\_vertically(): Split into features and target.
* split\_data\_into\_four\_parts(): Train-test split.
* using\_standerd\_scaler(): Standardize features.
* knn\_classifier(): Train and evaluate KNN.
* decision\_tree\_model\_train(): Train and evaluate Decision Tree.
* save\_model(): Save best model.
* load\_model(): Load saved model.
* predict\_model(): Predict wine class for new data.

## Author

* Developed by: AVADHUT YASHWANT MOTE
* Date: Sept/2025
