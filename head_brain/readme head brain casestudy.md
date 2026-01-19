######################################			 	Head Brain Linear Regression Project			####################################################


This project implements a **Linear Regression model** using the Head & Brain dataset.
It demonstrates a complete machine learning workflow, including data preprocessing, visualization, model training, evaluation, and prediction.


#############################################################################################################################################################################



####### -------------------- Required Libraries	------->>>>>>>>>>>>>>

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib
```

######################################################################



###### 		 Functions Overview	------>>>>>>>



1. `load_data(csv_filename)`

---------- Loads dataset into a DataFrame.

2. `name_of_headers(df)`
-
---------- Returns dataset column headers.

3. `feature_header_of_dataset(df)`

----------- Returns feature headers (first 3 columns).

4. `target_header_of_dataset(df)`

------------ Returns target column(s).

5. `datatypes_of_headers(df)`

------------- Displays datatypes of each column.

6. `shuffle_the_data(df)`

-------------Shuffles dataset rows randomly.

7. `statistical_info(df)`

---------------Returns statistical summary (mean, std, min, max, quartiles).

8. `checking_null_values(df)`

--------------Checks number of missing values.

9. `boxplot_of_dataset(df)`

--------------Creates `boxplot.png` to check outliers.

10. `split_data_vertically(df, target_header)`

-------------Splits dataset into features (X) and target (Y).

11. `split_data_into_four_parts(X, Y)`

--------------Splits into `X_train, X_test, Y_train, Y_test`.

12. `using_standerd_scaler(X_train, X_test)`

--------------Scales features using `StandardScaler`.

13. `linear_regression(X_train_scaled, X_test_scaled, Y_train, Y_test)`

*____________ Trains Linear Regression model.
*_____________Returns model, R² score, and MSE.

14. `save_model(lrmodel, filename)`

--------------Saves trained model as `.pkl`.

15. `load_model(filename)`

--------------Loads saved model.

16. `predict_new_data(scaler, loaded_model)`

--------------Predicts brain size for new sample data.

###############################################################################################


############		Main Workflow (`main()`)	--------->>>>>>>

1. Load dataset (`MarvellousHeadBrain.csv`)
2. Display headers, datatypes, statistics
3. Shuffle and clean dataset
4. Visualize outliers (boxplot)
5. Split dataset into training/testing sets
6. Apply scaling
7. Train Linear Regression model
8. Evaluate using **R² score** and **MSE**
9. Save and load model (`head_brain.pkl`)
10. Predict new data

###################################################################################################



######			 Example Output		-------------->>>>>>>


* R² Score**: \~0.65 (varies with dataset)
* MSE**: Depends on dataset
* Prediction Example**: Predicted brain size for new input

###################################################################################################


---

################# 	 Project Structure	-------------->>>>>>>>>>>

```
├── MarvellousHeadBrain.csv    # Dataset
├── head_brain.pkl             # Saved Model
├── boxplot.png                # Visualization
├── head_brain.py              # Main Script
└── README.md                  # Documentation
```

####################################################################################################



######		 Installation		___________>>>>>>>>

Create a `requirements.txt` with:


pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

Run the script:

```bash
python head_brain.py
```

---

###############			Notes		############################################################

* Make sure `MarvellousHeadBrain.csv` is in the same directory.
* The model is saved as `head_brain.pkl`.
* Outlier visualization is saved as `boxplot.png`.

#############################################################################################################



###	AUTHOR	:	AVADHUT YASHWANT MOTE	#######		

THANK YOU	!!!!!!!
