Credit Card Fraud Detection
This project implements a machine learning model to detect fraudulent credit card transactions using the Credit Card Fraud Detection dataset from Kaggle. The dataset contains anonymized transaction data, including transaction amounts, times, and a binary "Class" label indicating whether a transaction is fraudulent (1) or legitimate (0). The dataset exhibits a significant class imbalance, with a vast majority of transactions being legitimate.



I've used the Modular Design Pattern along with elements of the Factory Pattern and Pipeline Pattern to structure your project efficiently. Here’s how these patterns are applied:

1. Modular Design Pattern
This ensures each functionality is encapsulated in separate modules, making the codebase easier to maintain and scale.

Separation of Concerns:
utils/ → Handles data loading & preprocessing.
models/ → Contains training and evaluation logic.
pipelines/ → Orchestrates training and evaluation workflows.
main.py → Provides a single entry point to run different processes.
2. Pipeline Pattern
The project follows a sequential data-processing pipeline:

train_pipeline.py orchestrates:

Data loading
Preprocessing
Model training with hyperparameter tuning
evaluate_pipeline.py orchestrates:

Loading test data
Loading the trained model
Evaluating performance metrics and visualizing results
This ensures reusability and extensibility while making the workflow clear.

3. Factory Pattern (Limited Usage)
The ModelTrainer and ModelEvaluator classes act like factories, initializing models with optimal hyperparameters.
Instead of directly creating model instances, these classes allow flexible hyperparameter tuning using Optuna, which is an implicit factory approach.
Why This Approach?
✅ Scalability – New models or preprocessing steps can be added without modifying existing code.
✅ Maintainability – Clear separation between data handling, training, and evaluation.
✅ Reusability – Each component (e.g., Preprocessor, ModelTrainer) can be used independently.
