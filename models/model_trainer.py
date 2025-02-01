import optuna
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class ModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.best_model = None
    
    def objective(self, trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 50, step=5),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None])
        }
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred)
    
    def train_model(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=50)
        self.best_model = RandomForestClassifier(**study.best_params, random_state=42)
        self.best_model.fit(self.X_train, self.y_train)
        joblib.dump(self.best_model, "models/fraud_detection_model.pkl")
        print("Training complete. Model saved.")
