import streamlit as st
import numpy as np
import tensorflow as tf
import mne
import matplotlib.pyplot as plt
import tempfile
import os

# --- Page Config ---
st.set_page_config(
    page_title="Epilepsy Seizure Detection AI",
    page_icon="🧠",
    layout="wide"
)

# --- CSS for better styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 AI Epilepsy Seizure Detector")
st.markdown("""
**Deep Learning Model:** Hybrid CNN-LSTM  
**Performance:** 93% F1-Score | 90% Recall  
**Status:** Ready for Deployment
""")

# --- Sidebar ---
st.sidebar.header("Configuration")
st.sidebar.info("Upload a .edf file to detect seizures.")

# 1. THRESHOLD SLIDER (Critical Step)
# We set the default to 0.67 because that was your optimized result!
THRESHOLD = st.sidebar.slider(
    "Detection Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.67, 
    help="Confidence level required to classify a window as a seizure."
)

# --- Helper Functions ---

def filter_short_seizures(predictions, min_duration_seconds=10):
    """
    Removes seizure detections that are shorter than 'min_duration_seconds'.
    Real seizures are usually sustained events.
    """
    filtered_preds = predictions.copy()
    
    # We look for continuous runs of '1's
    is_seizure = False
    start_idx = 0
    
    for i in range(len(filtered_preds)):
        if filtered_preds[i] == 1:
            if not is_seizure:
                is_seizure = True
                start_idx = i
        else:
            if is_seizure:
                is_seizure = False
                duration = i - start_idx
                # If the run was too short, delete it (set to 0)
                if duration < min_duration_seconds:
                    filtered_preds[start_idx:i] = 0
                    
    # Handle case where seizure ends at the very last window
    if is_seizure:
        duration = len(filtered_preds) - start_idx
        if duration < min_duration_seconds:
            filtered_preds[start_idx:] = 0
            
    return filtered_preds

@st.cache_resource
def load_seizure_model():
    """Loads the trained model once and caches it for speed."""
    try:
        model = tf.keras.models.load_model("best_model.keras")
        return model
    except Exception as e:
        return None

def preprocess_edf(file_path):
    """Loads EDF, extracts data, and windows it for the model."""
    # 1. Load the file using MNE
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    
    # 2. Extract Data (Transpose to [samples, channels])
    # The model expects 23 channels. 
    data = raw.get_data().T
    sfreq = raw.info['sfreq']
    
    # 3. Create Windows (Same logic as training: 1 second, 50% overlap)
    window_seconds = 1
    overlap = 0.5
    window_samples = int(sfreq * window_seconds)
    step_samples = int(window_samples * (1 - overlap))
    
    windows = []
    timestamps = [] # To keep track of time for plotting
    
    start = 0
    while start + window_samples <= len(data):
        window = data[start : start + window_samples]
        windows.append(window)
        timestamps.append(start / sfreq) # Start time in seconds
        start += step_samples
        
    return np.array(windows), np.array(timestamps), raw

# --- Main App Logic ---

# Load Model
model = load_seizure_model()
if model is None:
    st.error("❌ Could not find 'best_model.keras'. Please train the model first.")
    st.stop()
else:
    st.sidebar.success("✅ Model Loaded")

# File Uploader
uploaded_file = st.file_uploader("Upload EEG Recording", type=["edf"])

if uploaded_file is not None:
    st.divider()
    st.subheader("Analysis Results")
    
    # Save uploaded file to temp so MNE can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        with st.spinner("Processing EEG signals..."):
            # Preprocess
            X_windows, timestamps, raw_obj = preprocess_edf(tmp_path)
        
        # Check shape compatibility
        if X_windows.shape[-1] != 23:
            st.error(f"⚠️ Shape Mismatch: Model expects 23 channels, file has {X_windows.shape[-1]}.")
            st.warning("Prediction might be inaccurate due to channel mismatch.")
        
        # Make Predictions
        predictions_prob = model.predict(X_windows, verbose=0)
        
        # APPLY THE THRESHOLD (The User's Slider)
        raw_predictions = (predictions_prob > THRESHOLD).astype(int).flatten()

        # --- NEW: APPLY DURATION FILTER ---
        # Ignore any seizure shorter than 5 seconds (you can adjust this number)
        predictions = filter_short_seizures(raw_predictions, min_duration_seconds=5)
        
        # Stats (now based on the filtered predictions)
        num_seizures = np.sum(predictions)
        duration = raw_obj.times[-1]
        
        # Metrics Columns
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Duration", f"{duration:.1f} sec")
        col2.metric("Total Windows", f"{len(predictions)}")
        col3.metric("Seizures Detected", f"{num_seizures}", delta_color="inverse")
        
        if num_seizures > 0:
            st.error(f"⚠️ **Seizure Activity Detected!** Found {num_seizures} events.")
        else:
            st.success("✅ **Normal EEG.** No seizure activity detected.")
        
        # Visualization Section
        st.divider()
        st.subheader("Signal Visualization")
        
        # Channel Selection
        channel_names = raw_obj.ch_names
        selected_channel = st.selectbox("Select Electrode to View", channel_names)
        channel_idx = channel_names.index(selected_channel)
        
        # Plotting
        raw_data = raw_obj.get_data()[channel_idx]
        times = raw_obj.times
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(times, raw_data, label=selected_channel, color='#2c3e50', linewidth=0.8)
        
        # Highlight Seizures
        # We overlay red blocks where the model predicted a seizure
        for i, is_seizure in enumerate(predictions):
            if is_seizure:
                start_time = timestamps[i]
                # Highlight the 1-second window
                ax.axvspan(start_time, start_time + 1, color='red', alpha=0.3)
        
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude (µV)")
        ax.set_title(f"Electrode {selected_channel} (Red areas = AI Detected Seizures)")
        ax.legend(loc="upper right")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred processing the file: {e}")
    finally:
        os.remove(tmp_path)