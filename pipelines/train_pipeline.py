from utils.data_loader import DataLoader
from utils.preprocessor import Preprocessor
from models.model_trainer import ModelTrainer

# Load and preprocess data
data_loader = DataLoader("data/creditcard.csv")
data = data_loader.load_data()
X, y = data_loader.preprocess_data(data)

# Split data
preprocessor = Preprocessor()
X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)

# Train model
trainer = ModelTrainer(X_train, X_test, y_train, y_test)
trainer.train_model()
