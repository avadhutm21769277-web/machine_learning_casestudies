# 🩺 Diabetes Prediction System

This project is based on the **PIMA Indians Diabetes Dataset**.  
It uses a **Random Forest Classifier** to build a machine learning model that predicts the likelihood of a patient having diabetes.  
A simple **Tkinter GUI** is also provided for user-friendly interaction.

---

## 📌 Project Structure

```
Diabetes_Prediction_Project/
│
├── data/
│   └── diabetes.csv
│
├── model/
│   └── diabetes.pkl
│
├── src/
│   └── diabetes_prediction.py
│
├── README.md
└── REPORT.pdf
```

---

## 🚀 Features
- Data preprocessing (handling missing and zero values)  
- Model training using Random Forest Classifier  
- Save trained model (`diabetes.pkl`)  
- Load model for prediction without retraining  
- Predict function (Diabetic / Non-Diabetic)  
- User-friendly GUI built with Tkinter  

---

## 🛠️ Libraries Used
- pandas  
- numpy  
- scikit-learn (sklearn)  
- joblib  
- tkinter  

---

## 📂 Dataset
- **File**: `diabetes.csv`  
- **Attributes**:
  - Pregnancies  
  - Glucose  
  - BloodPressure  
  - SkinThickness  
  - Insulin  
  - BMI  
  - DiabetesPedigreeFunction  
  - Age  
  - Outcome (Target: 0 = Non-Diabetic, 1 = Diabetic)

---

## ⚙️ How to Run the Project

1. Clone or download the project folder. 
 
2. Install the required libraries:
   
   pip install pandas numpy scikit-learn joblib
   ```
3. Run the main Python file:
   
   python src/diabetes_prediction.py
   ```
4. After training, the model will be saved as `diabetes.pkl`.  
5. The GUI window will open → Enter patient details and click **Predict Diabetes**.  

---

#Output
- **Terminal**: Training Accuracy (e.g., 78%)  
- **GUI**: Popup message → "Diabetic possibility" or "Non-Diabetic possibility"  



#Author
Prepared by: *AVADHUT YASHWANT MOTE*  
Project: Diabetes Prediction System  
