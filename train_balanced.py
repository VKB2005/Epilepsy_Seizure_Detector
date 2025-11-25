import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve, auc
from model import build_model
import matplotlib.pyplot as plt

# --- Configuration ---
PROCESSED_DIR = 'processed_data'
# Ratio: For every 1 seizure, keep 5 normal windows.
NORMAL_TO_SEIZURE_RATIO = 5 

def load_balanced_data():
    x_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, '*_X.npy')))
    y_files = [f.replace('_X.npy', '_y.npy') for f in x_files]

    print("--- Pass 1: Scanning files to count classes ---")
    total_seizures = 0
    total_normals = 0
    
    for y_f in y_files:
        try:
            y_chunk = np.load(y_f, allow_pickle=True)
            total_seizures += np.sum(y_chunk == 1)
            total_normals += np.sum(y_chunk == 0)
        except Exception as e:
            print(f"Skipping check for {y_f}: {e}")

    print(f"  Found {total_seizures} Seizures and {total_normals} Normal windows.")
    
    if total_seizures == 0:
        raise ValueError("No seizures found! Re-check seizure_files.txt and preprocessing.")

    target_normals = int(total_seizures * NORMAL_TO_SEIZURE_RATIO)
    sampling_rate = target_normals / total_normals
    sampling_rate = min(sampling_rate, 1.0)
    
    print(f"  Target Normal Windows: {target_normals}")
    print(f"  Sampling Rate: {sampling_rate:.4f} (Keeping ~{sampling_rate*100:.2f}% of normal data)")

    print("--- Pass 2: Loading and downsampling on-the-fly ---")
    all_X = []
    all_y = []
    
    for x_f, y_f in zip(x_files, y_files):
        try:
            y_chunk = np.load(y_f, allow_pickle=True)
            
            seizure_mask = (y_chunk == 1)
            normal_mask = (y_chunk == 0) & (np.random.rand(len(y_chunk)) < sampling_rate)
            final_mask = seizure_mask | normal_mask
            
            if np.any(final_mask):
                X_chunk = np.load(x_f, allow_pickle=True).astype(np.float32)
                all_X.append(X_chunk[final_mask])
                all_y.append(y_chunk[final_mask])
                
        except Exception as e:
            print(f"Skipping {x_f}: {e}")
            continue

    X_balanced = np.vstack(all_X)
    y_balanced = np.hstack(all_y)
    
    indices = np.arange(len(X_balanced))
    np.random.shuffle(indices)
    
    return X_balanced[indices], y_balanced[indices]

# --- Main Execution ---

# 1. Load Data
X, y = load_balanced_data()
print(f"Final Balanced Dataset Shape: {X.shape}")

# 2. Split Data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Normalization
normalizer = tf.keras.layers.Normalization()
normalizer.adapt(X_train)

# 4. Build Model
INPUT_SHAPE = (256, 23)
model = build_model(INPUT_SHAPE, normalizer)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

# 5. Training
# Note: Monitoring val_loss often yields a more stable F1 than monitoring recall alone
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True, monitor="val_loss", mode='min')
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss", mode='min')

print("\n--- Starting Training ---")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint_cb, early_stopping_cb]
)

# 6. Evaluation & Optimization
print("\n--- Evaluation & Threshold Optimization ---")
best_model = tf.keras.models.load_model("best_model.keras")
y_pred_prob = best_model.predict(X_val)

# --- NEW: Threshold Optimization Logic ---
print("Calculating optimal threshold...")
thresholds = np.arange(0.01, 1.0, 0.01)
f1_scores = []

for thresh in thresholds:
    y_pred_temp = (y_pred_prob > thresh).astype(int)
    f1_scores.append(f1_score(y_val, y_pred_temp))

# Find max F1
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"\n🏆 Best Threshold Found: {best_threshold:.2f}")
print(f"🚀 Optimized F1-Score: {best_f1:.4f}")

# Final Report using the BEST threshold
y_final_pred = (y_pred_prob > best_threshold).astype(int)

print(f"\n--- Classification Report (Threshold = {best_threshold:.2f}) ---")
print(classification_report(y_val, y_final_pred, target_names=['Normal', 'Seizure']))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_final_pred))

# --- Plotting ---
def plot_results(history, y_true, y_pred_prob, best_thresh):
    plt.figure(figsize=(18, 6))

    # Plot 1: Recall & Loss
    plt.subplot(1, 3, 1)
    plt.plot(history.history['recall'], label='Train Recall')
    plt.plot(history.history['val_recall'], label='Val Recall')
    plt.plot(history.history['loss'], label='Train Loss', linestyle='--')
    plt.title('Recall and Loss')
    plt.xlabel('Epoch')
    plt.legend()

    # Plot 2: Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    auc_score = auc(recall, precision)
    plt.subplot(1, 3, 2)
    plt.plot(recall, precision, marker='.', label=f'AUC = {auc_score:.2f}')
    # Mark the optimized threshold on the curve
    plt.scatter(recall[best_idx], precision[best_idx], marker='o', color='red', label='Best Threshold', zorder=5)
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)

    # Plot 3: Confusion Matrix
    cm = confusion_matrix(y_true, (y_pred_prob > best_thresh).astype(int))
    plt.subplot(1, 3, 3)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix (Thresh={best_thresh:.2f})')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Normal', 'Seizure'])
    plt.yticks(tick_marks, ['Normal', 'Seizure'])
    
    thresh_val = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh_val else "black")

    plt.tight_layout()
    plt.savefig('final_optimized_results.png')
    print("\nGraphs saved to 'final_optimized_results.png'")

plot_results(history, y_val, y_pred_prob, best_threshold)