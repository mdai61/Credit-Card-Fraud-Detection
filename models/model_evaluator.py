import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self, model_path, X_test, y_test):
        self.model = joblib.load(model_path)
        self.X_test = X_test
        self.y_test = y_test
    
    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))
    
    def plot_precision_recall_curve(self):
        y_pred_prob = self.model.predict_proba(self.X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(self.y_test, y_pred_prob)
        plt.plot(recall, precision, marker='.', label="RandomForest")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.show()
