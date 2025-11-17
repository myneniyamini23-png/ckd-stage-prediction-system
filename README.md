# 🩺 CKD Stage Prediction System

A hybrid machine-learning and deep-learning system for predicting **Chronic Kidney Disease (CKD)** and its stages using ensemble base models and a CNN-BiLSTM hybrid architecture.

---

## 🚀 Features

* Hybrid DL model combining **Conv1D + BiLSTM** layers
* Multiple base ML models: **RandomForest, KNN, SVM**
* Automatic data preprocessing
* Multiclass stage prediction + binary CKD detection
* Modular and extendable codebase
* Model training, evaluation & saving

---

## 📂 Project Structure

```
├── base_models.py
├── hybrid_model.py
├── preprocessing.py
├── train_hybrid.py
├── train_base_models.py
├── data/
│   └── ckd_full_dataset.csv
├── models/
│   └── hybrid_ckd_model.h5
```

---

## 🧠 Base Models

The following ML models are included in `get_base_models()`:

* **RandomForestClassifier**
* **KNeighborsClassifier (KNN)**
* **Support Vector Machine (SVM)**

Used for baseline CKD prediction performance.

---

## 🤖 Hybrid Deep Learning Model

`build_hybrid_model()` constructs a hybrid model with:

* **Conv1D layers** for feature extraction
* **BiLSTM layers** for sequential learning
* **Dense layers** for classification
* Supports **binary** and **multiclass** output

---

## 📊 Dataset & Preprocessing

`preprocessing.py` handles:

* Loading the CKD dataset
* Label encoding of categorical features
* Standard scaling
* Train–validation split

Outputs:

```
X_train, X_val, y_train, y_val
```

---

## 🏋️‍♂️ Training Base Models

Run:

```bash
python train_base_models.py
```

Outputs accuracy for:

* RandomForest
* KNN
* SVM

---

## 🧬 Training the Hybrid Model

Run:

```bash
python train_hybrid.py
```

This will:

1. Load preprocessed dataset
2. Build hybrid CNN-BiLSTM model
3. Train for 20 epochs
4. Save final model → `models/hybrid_ckd_model.h5`

---

## 📈 Model Output

The system predicts:

### ✔️ `ckd_pred`

Binary: CKD present or not.

### ✔️ `ckd_stage`

Multiclass: CKD Stage 1–5

---

## 🛠 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Core libraries include:

* TensorFlow
* scikit-learn
* pandas
* numpy

---

## 📝 Usage

Import base models:

```python
from base_models import get_base_models
```

Import and train hybrid:

```python
from hybrid_model import build_hybrid_model
```

---


