import tensorflow as tf
from hybrid_model import build_hybrid_model
from preprocessing import load_data

# 1. Load data
X_train, X_val, y_train, y_val = load_data()

# 2. Build model
input_shape = (X_train.shape[1], 1)
num_classes = 1  # binary classification
model = build_hybrid_model(input_shape, num_classes)

# 3. Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Train model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)

# 5. Save model if you want
model.save('models/hybrid_ckd_model.h5')
