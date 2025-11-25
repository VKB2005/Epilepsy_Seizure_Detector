# 🧠 AI Epilepsy Seizure Detection

## Overview
Automated seizure detection from EEG signals using a hybrid **CNN-LSTM Deep Learning model**. The system processes raw multichannel EEG data, classifies 1-second windows as "Seizure" or "Normal," and visualizes the results in an interactive web dashboard.

**Performance:**
* **F1-Score:** 0.93
* **Recall:** 0.90 (Detects 90% of seizures)
* **Precision:** 0.97

## 🛠 Tech Stack
* **Python** (TensorFlow/Keras, NumPy, Scikit-learn)
* **Data Processing:** MNE (Neurophysiological data analysis)
* **Visualization:** Matplotlib, Streamlit
* **Dataset:** CHB-MIT Scalp EEG Database

## ⚙️ How It Works
1.  **Preprocessing:** Raw `.edf` files are sliced into 1-second windows (256 samples).
2.  **Filtering:** Artifacts and non-seizure noise are removed using a custom duration filter.
3.  **Model:** * **CNN Layers:** Extract spatial features from 23 EEG channels.
    * **LSTM Layers:** Capture temporal dependencies over time.
4.  **UI:** A Streamlit dashboard allows doctors to upload files and view highlighted seizure events.

## 🚀 How to Run
1.  Install dependencies: `pip install -r requirements.txt`
2.  Run the app: `streamlit run app.py`
3.  Upload an `.edf` file to test.
