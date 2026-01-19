#####################		Titanic Survival Prediction		#########################



This project predicts whether a passenger on the Titanic survived or not
based on their details.\
It uses **Logistic Regression** with **StandardScaler** for feature
scaling, and comes with a **GUI** for interactive predictions.

------------------------------------------------------------------------

# Features	:

-   Load and preprocess Titanic dataset
-   Handle missing values & drop unnecessary columns
-   Split data into training/testing sets
-   Standardize features using `StandardScaler`
-   Train a Logistic Regression model
-   Save & load model using `joblib`
-   GUI interface for interactive prediction

------------------------------------------------------------------------

# Installation	:

1.  Clone this repository or download files.
2.  Install required libraries:


	pip install pandas numpy scikit-learn joblib
```

------------------------------------------------------------------------

# Project Files	:

  File Name                                     Description
  --------------------------------------------- -------------------------------------
  `titanic_casestudy.py`                        Main Python script (training + GUI)
  `MarvellousTitanicDataset.csv`                Dataset file
  `titanic.pkl`                                 Saved trained model
  `Titanic_Survival_Model_Documentation.docx`   Documentation file
  `README.md`                                   Project information

------------------------------------------------------------------------

## 🧠 How It Works

1.  Load dataset → Clean & preprocess data\
2.  Split into **X (features)** and **Y (target)**\
3.  Scale features using `StandardScaler`\
4.  Train **Logistic Regression** model\
5.  Calculate and display accuracy\
6.  Save model to `.pkl` file\
7.  Load saved model & predict on sample input\
8.  Launch GUI for live predictions

------------------------------------------------------------------------

# GUI Usage

-   Run the script:

``` bash
python titanic_casestudy.py
```

-   Enter the following details:
    -   **Age**
    -   **Fare**
    -   **Sex** (1 = Male, 0 = Female)
    -   **sibsp** (Siblings/Spouses aboard)
    -   **Parch** (Parents/Children aboard)
    -   **Pclass** (Passenger Class)
    -   **Embarked** (Port of Embarkation)
-   Click **Predict** to get survival result.

------------------------------------------------------------------------

# Model Accuracy

The accuracy is printed in console after training the model.

------------------------------------------------------------------------

# Author

AVADHUT YASHWANT MOTE
Version: 1.0

------------------------------------------------------------------------


