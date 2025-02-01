from utils.data_loader import DataLoader
from utils.preprocessor import Preprocessor
from models.model_evaluator import ModelEvaluator

# Load and preprocess data
data_loader = DataLoader("data/creditcard.csv")
data = data_loader.load_data()
X, y = data_loader.preprocess_data(data)

# Split data
preprocessor = Preprocessor()
_, X_test, _, y_test = preprocessor.split_data(X, y)

# Evaluate model
evaluator = ModelEvaluator("models/fraud_detection_model.pkl", X_test, y_test)
evaluator.evaluate_model()
evaluator.plot_precision_recall_curve()