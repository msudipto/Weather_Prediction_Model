import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    # Load data
    data = pd.read_csv(file_path)
    
    # Basic preprocessing (handle missing values, normalize data, etc.)
    data = data.dropna()  # Placeholder, consider more nuanced handling
    
    # Feature engineering (add any meaningful features here)
    # Example: data['new_feature'] = data['feature1'] * data['feature2']
    
    return data

def split_data(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test