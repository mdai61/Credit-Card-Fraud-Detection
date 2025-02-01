import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load_data(self):
        data = pd.read_csv(self.file_path)
        return data
    
    def preprocess_data(self, data):
        X, y = data.drop(columns=['Class']), data['Class']
        undersampler = RandomUnderSampler()
        X_resampled, y_resampled = undersampler.fit_resample(X, y)
        return X_resampled, y_resampled