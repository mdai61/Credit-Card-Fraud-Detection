Credit Card Fraud Detection
This project implements a machine learning model to detect fraudulent credit card transactions using the Credit Card Fraud Detection dataset from Kaggle. The dataset contains anonymized transaction data, including transaction amounts, times, and a binary "Class" label indicating whether a transaction is fraudulent (1) or legitimate (0). The dataset exhibits a significant class imbalance, with a vast majority of transactions being legitimate.
Dataset
The Credit Card Fraud Detection dataset consists of credit card transactions made by European cardholders in September 2013. The data has been transformed using PCA for confidentiality, so the original features are not directly interpretable. Key features include:
•	Time: Number of seconds elapsed between each transaction and the first transaction in the dataset.
•	Amount: Transaction amount.
•	V1 - V28: PCA-transformed features.
•	Class: Target variable (1 = Fraudulent, 0 = Legitimate).
Project Overview
This project addresses class imbalance and utilizes hyperparameter tuning to optimize a Random Forest Classifier for fraud detection. The key steps include:
1. Data Exploration and Visualization
•	Analysis of transaction distributions for fraud and legitimate transactions.
•	Visualization of data patterns to identify anomalies and insights.
2. Data Preprocessing
•	Handling class imbalance using RandomUnderSampler from imblearn.
•	Consideration of SMOTE (Synthetic Minority Over-sampling Technique) for oversampling.
3. Model Selection and Hyperparameter Tuning
•	Random Forest Classifier chosen for its robustness and performance.
•	Hyperparameter tuning using Optuna for optimizing: 
o	n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features.
•	Stratified K-fold cross-validation performed with F1-score as the metric.
4. Threshold Tuning
•	Default classification threshold of 0.5 may not be optimal.
•	Iterative tuning over a range of thresholds to maximize F1-score.
•	Focus on improving recall to reduce false negatives.
5. Model Evaluation
The trained model is evaluated using:
•	Confusion Matrix: Visual representation of model performance.
•	Classification Report: Precision, recall, F1-score, and support for each class.
•	F1-score: Balance between precision and recall.
•	AUC-ROC Curve: Measures the model's ability to distinguish between classes.
•	AUC-PR Curve: More informative for imbalanced datasets.
•	Precision-Recall Curve: Shows trade-off between precision and recall.
Why Threshold Tuning?
Fraud detection prioritizes minimizing false negatives (missed fraudulent transactions) over reducing false positives. Adjusting the classification threshold ensures higher recall, making the model more effective in real-world applications.
Installation
To run this project, ensure you have the necessary dependencies installed:
pip install numpy pandas scikit-learn imbalanced-learn optuna matplotlib seaborn
Usage
1.	Clone the repository:
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
2.	Run the preprocessing and model training script:
python train_model.py
3.	Evaluate the model and visualize results:
python evaluate_model.py
Future Improvements
•	Experiment with other models (XGBoost, LightGBM, Neural Networks).
•	Implement additional feature engineering techniques.
•	Deploy the model as a real-time fraud detection API.
License
This project is licensed under the MIT License. See LICENSE for details.
________________________________________
Feel free to contribute and suggest improvements!

