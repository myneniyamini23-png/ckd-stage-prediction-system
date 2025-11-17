import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data():
    # Load the CSV
    df = pd.read_csv('data/ckd_full_dataset.csv')

    # Separate features and labels
    X = df.drop(['cluster', 'ckd_pred', 'ckd_stage'], axis=1)
    y = df[['ckd_pred', 'ckd_stage']]

    # Encode categorical columns
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val

