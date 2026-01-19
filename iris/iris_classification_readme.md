#################################					 Iris Flower Classification			#############################################################



This project is a Machine Learning case study on the famous **Iris dataset**.  
It uses **K-Nearest Neighbors (KNN)** and **Decision Tree Classifier** to predict the species of iris flowers based on their features.


#######################################################################################################################################################################################



####################----->>>>>>> Dataset	:

The dataset `iris.csv` contains the following columns:

| Feature         | Description                     |
|-----------------|---------------------------------|
| sepal.length    | Sepal length in cm              |
| sepal.width     | Sepal width in cm               |
| petal.length    | Petal length in cm              |
| petal.width     | Petal width in cm               |
| variety         | Target species (`setosa`, `versicolor`, `virginica`) |

#############################################################################################



###################----->>>>>>>>> Project Structure	:

iris_casestudy/
│
├─ itis_casestudy.py        # Main Python script with pipeline
├─ iris.csv                 # Dataset file
├─ knn_iris.pkl             # Saved KNN model
├─ dt_iris.pkl              # Saved Decision Tree model
└─ README.md                # This file

##############################################################################################



######	----------->>>>>>>>>>	:: Features

- Preprocessing pipeline handling numeric and categorical features
- Train KNN and Decision Tree classifiers
- Save and load trained models using `joblib`
- Predict new data
- Prints accuracy of trained models

##############################################################################################


##########--------->>>>>> Installation


1. Clone the repository or download files
2. Install dependencies:

```bash
pip install pandas numpy scikit-learn joblib
```

---

## Usage

Run the main script:

```bash
python itis_casestudy.py
```
################################################################################################

This will:

1. Load the dataset
2. Shuffle and preprocess data
3. Train KNN and Decision Tree classifiers
4. Save the trained models
5. Load models and predict new sample

############################################################################################

## Predicted Classes

| Numeric Output | Iris Species   |
|----------------|---------------|
| 0              | setosa        |
| 1              | versicolor    |
| 2              | virginica     |



########	 Author		:	Avadhut Yashwant Mote	#############################
