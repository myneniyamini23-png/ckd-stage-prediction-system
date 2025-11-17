from base_models import get_base_models
from load_data import load_data  # <-- Import your new load function

# 🚀 Load preprocessed dataset
X_train, X_val, y_train, y_val = load_data()

# 🚀 Get all base models
models = get_base_models()

# 🛠 Train and evaluate models
def train_models():
    print("\nTraining base models...")

    for model_name, model in models.items():
        model.fit(X_train, y_train['ckd_pred'])  # Train on 'ckd_pred' label
        acc = model.score(X_val, y_val['ckd_pred'])
        print(f"{model_name} Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train_models()
