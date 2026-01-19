#############			# Sentiment Analysis - Bagging + Boosting + TF-IDF #		#############################################




This project performs **sentiment analysis** using multiple machine learning algorithms such as Logistic Regression, Random Forest (Bagging), Gradient Boosting, and XGBoost (Boosting).  
It uses **TF-IDF vectorization** for text feature extraction and achieves robust performance across models.

###################################################################################################################################################



## 📂 Project Structure

```
sentiment_analysis/
│
├── sentiment_analysis_bagging.py   # Main Python script
├── test.csv                        # Dataset file (text, sentiment)
└── README.md                       # Project documentation
```

######################################################################################################################################################

## ⚙️ Features

- Data cleaning and preprocessing
- TF-IDF text vectorization
- Multi-model training:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - XGBoost
- Model comparison and performance metrics

#######################################################################################################################################################

## 🧰 Requirements

Install dependencies before running the script:

```bash
pip install pandas numpy scikit-learn xgboost
```

########################################################################################################################################################

## 📘 Dataset Format

Your CSV file (`test.csv`) should have at least the following columns:

| text | sentiment |
|------|------------|
| "I love this product!" | positive |
| "This is terrible." | negative |
| "It’s okay, not great." | neutral |

#####################################################################################################################################################

## 🚀 How to Run

1. Place your dataset (`test.csv`) in the same directory.
2. Run the script:

```bash
python sentiment_analysis_bagging.py
```

3. The script will:
   - Clean and preprocess text.
   - Vectorize using TF-IDF.
   - Train and evaluate models.
   - Print accuracy and classification reports.

#######################################################################################################################################################

## 🧾 Example Output

```
🎯 Logistic Regression Results:
Accuracy: 0.6436
🎯 Random Forest Results:
Accuracy: 0.6535
🎯 Gradient Boosting Results:
Accuracy: 0.6351
🎯 XGBoost Results:
Accuracy: 0.6294
```

#########################################################################################################################################################

## 🏆 Model Comparison

| Model | Accuracy |
|--------|-----------|
| Logistic Regression | 0.64 |
| Random Forest | 0.65 |
| Gradient Boosting | 0.63 |
| XGBoost | 0.63 |

#########################################################################################################################################################

## 📈 Future Improvements

- Add deep learning models (LSTM, BERT)
- Use cross-validation
- Hyperparameter tuning
- Handle data imbalance with SMOTE

############################################################################################################################################################

## 👨‍💻 Author

**AVADHUT YASHWANT MOTE**  
*Data Science Enthusiast *

#############################################################################################################################################################
