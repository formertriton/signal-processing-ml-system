"""
Signal Processing ML System - Interactive Demo Dashboard
Showcases all capabilities of the system
"""

import streamlit as st
import numpy as np
import pickle
import sys
import os
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal

# Add src to path
sys.path.append('src')

from signal_generation.signal_generator import SignalGenerator
from feature_extraction.feature_extractor import FeatureExtractor
from anomaly_detection.anomaly_detector import AnomalyDetector
import joblib


# Page config
st.set_page_config(
    page_title="Signal Processing ML System",
    page_icon="📡",
    layout="wide"
)

# Title
st.title("🛡️ Signal Processing & Machine Learning System")
st.markdown("### Advanced RF Signal Classification and Anomaly Detection")
st.markdown("---")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a module:", [
    "🏠 Overview",
    "📡 Signal Generation",
    "🔍 Feature Extraction",
    "🤖 ML Classification",
    "⚠️ Anomaly Detection",
    "📊 Full Pipeline Demo"
])

# Load models (cached)
@st.cache_resource
def load_models():
    """Load all trained models"""
    try:
        # Load classifier
        models_dir = 'models/trained'
        model_files = [f for f in os.listdir(models_dir) if f.startswith('random_forest')]
        if model_files:
            rf_model = joblib.load(os.path.join(models_dir, model_files[0]))
        else:
            rf_model = None
        
        # Load label encoder
        encoder_files = [f for f in os.listdir(models_dir) if f.startswith('label_encoder')]
        if encoder_files:
            label_encoder = joblib.load(os.path.join(models_dir, encoder_files[0]))
        else:
            label_encoder = None
        
        # Load scaler
        scaler_files = [f for f in os.listdir(models_dir) if f.startswith('scaler')]
        if scaler_files:
            scaler = joblib.load(os.path.join(models_dir, scaler_files[0]))
        else:
            scaler = None
        
        # Load anomaly detector
        anomaly_files = [f for f in os.listdir(models_dir) if 'anomaly_detector' in f]
        if anomaly_files:
            anomaly_detector = AnomalyDetector.load(os.path.join(models_dir, anomaly_files[0]))
        else:
            anomaly_detector = None
        
        return rf_model, label_encoder, scaler, anomaly_detector
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None


# ==================== OVERVIEW PAGE ====================
if page == "🏠 Overview":
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📋 Project Overview")
        st.markdown("""
        This system demonstrates advanced signal processing and machine learning 
        capabilities for **defense and intelligence applications**.
        
        **Key Features:**
        - 📡 Synthetic RF signal generation
        - 🔍 Advanced feature extraction
        - 🤖 Multi-model classification
        - ⚠️ Anomaly detection
        - 📊 Real-time visualization
        """)
        
        st.header("🎯 Skills Demonstrated")
        st.markdown("""
        - **Signal Processing**: FFT, STFT, Wavelet Analysis
        - **Machine Learning**: Random Forest, SVM, Neural Networks
        - **Anomaly Detection**: Isolation Forest, One-Class SVM
        - **Software Engineering**: Clean architecture, testing
        - **Data Science**: Feature engineering, visualization
        """)
    
    with col2:
        st.header("📊 System Capabilities")
        
        # Display metrics
        st.metric("Signal Types", "6", help="Pulse, Chirp, FSK, PSK, QAM, Noise")
        st.metric("Features Extracted", "60+", help="Time, frequency, and wavelet features")
        st.metric("ML Models", "4", help="RF, SVM, GB, Neural Network")
        st.metric("Detection Methods", "3", help="Isolation Forest, One-Class SVM, Ensemble")
        
        st.header("🚀 Quick Start")
        st.markdown("""
        1. **Signal Generation**: Create synthetic RF signals
        2. **Feature Extraction**: Extract meaningful features
        3. **Classification**: Identify signal types
        4. **Anomaly Detection**: Find unusual patterns
        """)


# ==================== SIGNAL GENERATION PAGE ====================
elif page == "📡 Signal Generation":
    st.header("📡 RF Signal Generation")
    st.markdown("Generate synthetic radio frequency signals for analysis")
    
    # Signal parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        signal_type = st.selectbox("Signal Type", 
                                   ["pulse", "chirp", "fsk", "psk", "qam", "noise"])
    
    with col2:
        sample_rate = st.number_input("Sample Rate (Hz)", value=1000000, step=100000)
    
    with col3:
        duration = st.number_input("Duration (ms)", value=1.0, step=0.1) / 1000
    
    if st.button("Generate Signal", type="primary"):
        # Generate signal
        generator = SignalGenerator(sample_rate=int(sample_rate), duration=duration)
        
        with st.spinner("Generating signal..."):
            if signal_type == "pulse":
                sig = generator.generate_pulse()
            elif signal_type == "chirp":
                sig = generator.generate_chirp()
            elif signal_type == "fsk":
                sig = generator.generate_fsk()
            elif signal_type == "psk":
                sig = generator.generate_psk()
            elif signal_type == "qam":
                sig = generator.generate_qam()
            else:
                sig = generator.generate_noise()
        
        st.success(f"✓ {signal_type.upper()} signal generated!")
        
        # Time domain plot
        time = np.linspace(0, duration, len(sig))
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Time domain
        axes[0].plot(time * 1000, sig, linewidth=0.8)
        axes[0].set_xlabel("Time (ms)")
        axes[0].set_ylabel("Amplitude")
        axes[0].set_title(f"{signal_type.upper()} Signal - Time Domain")
        axes[0].grid(alpha=0.3)
        
        # Frequency domain
        fft = np.fft.fft(sig)
        frequencies = np.fft.fftfreq(len(sig), 1/sample_rate)
        positive_idx = frequencies > 0
        
        axes[1].plot(frequencies[positive_idx] / 1000, np.abs(fft[positive_idx]))
        axes[1].set_xlabel("Frequency (kHz)")
        axes[1].set_ylabel("Magnitude")
        axes[1].set_title(f"{signal_type.upper()} Signal - Frequency Domain")
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)


# ==================== FEATURE EXTRACTION PAGE ====================
elif page == "🔍 Feature Extraction":
    st.header("🔍 Feature Extraction")
    st.markdown("Extract features from signals for machine learning")
    
    signal_type = st.selectbox("Generate and Extract Features", 
                               ["pulse", "chirp", "fsk", "psk", "qam", "noise"])
    
    if st.button("Extract Features", type="primary"):
        # Generate signal
        generator = SignalGenerator()
        extractor = FeatureExtractor()
        
        with st.spinner("Generating signal and extracting features..."):
            if signal_type == "pulse":
                sig = generator.generate_pulse()
            elif signal_type == "chirp":
                sig = generator.generate_chirp()
            elif signal_type == "fsk":
                sig = generator.generate_fsk()
            elif signal_type == "psk":
                sig = generator.generate_psk()
            elif signal_type == "qam":
                sig = generator.generate_qam()
            else:
                sig = generator.generate_noise()
            
            features = extractor.extract_all_features(sig)
        
        st.success(f"✓ Extracted {len(features)} features!")
        
        # Display features in categories
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Time Domain Features")
            time_features = {k: v for k, v in features.items() 
                           if any(x in k for x in ['mean', 'std', 'rms', 'energy', 'zero_crossing'])}
            for name, value in list(time_features.items())[:10]:
                st.metric(name.replace('_', ' ').title(), f"{value:.4f}")
        
        with col2:
            st.subheader("📈 Frequency Domain Features")
            freq_features = {k: v for k, v in features.items() 
                           if 'spectral' in k or 'frequency' in k}
            for name, value in list(freq_features.items())[:10]:
                st.metric(name.replace('_', ' ').title(), f"{value:.4f}")


# ==================== ML CLASSIFICATION PAGE ====================
elif page == "🤖 ML Classification":
    st.header("🤖 Machine Learning Classification")
    st.markdown("Classify signals using trained ML models")
    
    rf_model, label_encoder, scaler, _ = load_models()
    
    if rf_model is None:
        st.error("❌ Models not found! Please train models first by running train_models.py")
    else:
        signal_type = st.selectbox("Select Signal Type to Classify", 
                                   ["pulse", "chirp", "fsk", "psk", "qam", "noise"])
        
        if st.button("Classify Signal", type="primary"):
            # Generate and classify
            generator = SignalGenerator()
            extractor = FeatureExtractor()
            
            with st.spinner("Generating signal, extracting features, and classifying..."):
                # Generate signal
                if signal_type == "pulse":
                    sig = generator.generate_pulse()
                elif signal_type == "chirp":
                    sig = generator.generate_chirp()
                elif signal_type == "fsk":
                    sig = generator.generate_fsk()
                elif signal_type == "psk":
                    sig = generator.generate_psk()
                elif signal_type == "qam":
                    sig = generator.generate_qam()
                else:
                    sig = generator.generate_noise()
                
                # Extract features
                features = extractor.extract_all_features(sig)
                feature_vector = np.array(list(features.values())).reshape(1, -1)
                
                # Handle NaN
                nan_mask = np.isnan(feature_vector)
                if np.any(nan_mask):
                    col_means = np.nanmean(feature_vector, axis=0)
                    col_means[np.isnan(col_means)] = 0
                    feature_vector[nan_mask] = np.take(col_means, nan_mask[1])
                
                # Scale and predict
                feature_scaled = scaler.transform(feature_vector)
                prediction = rf_model.predict(feature_scaled)[0]
                probabilities = rf_model.predict_proba(feature_scaled)[0]
                
                predicted_label = label_encoder.inverse_transform([prediction])[0]
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Classification Result")
                if predicted_label == signal_type:
                    st.success(f"✅ Correctly classified as: **{predicted_label.upper()}**")
                else:
                    st.warning(f"⚠️ Classified as: **{predicted_label.upper()}** (True: {signal_type.upper()})")
            
            with col2:
                st.subheader("Confidence")
                max_prob = np.max(probabilities) * 100
                st.metric("Confidence", f"{max_prob:.1f}%")
            
            # Probability bar chart
            st.subheader("Classification Probabilities")
            fig, ax = plt.subplots(figsize=(10, 4))
            labels = label_encoder.classes_
            ax.barh(labels, probabilities * 100)
            ax.set_xlabel("Probability (%)")
            ax.set_title("Classification Probabilities for All Signal Types")
            ax.grid(axis='x', alpha=0.3)
            st.pyplot(fig)


# ==================== ANOMALY DETECTION PAGE ====================
elif page == "⚠️ Anomaly Detection":
    st.header("⚠️ Anomaly Detection")
    st.markdown("Detect unusual or suspicious signals")
    
    _, _, _, anomaly_detector = load_models()
    
    if anomaly_detector is None:
        st.error("❌ Anomaly detector not found! Please run detect_anomalies.py first")
    else:
        signal_type = st.selectbox("Generate Signal to Check", 
                                   ["pulse", "chirp", "fsk", "psk", "qam", "noise"])
        
        add_anomaly = st.checkbox("Add artificial anomaly (extreme noise)", value=False)
        
        if st.button("Check for Anomalies", type="primary"):
            generator = SignalGenerator()
            extractor = FeatureExtractor()
            
            with st.spinner("Analyzing signal for anomalies..."):
                # Generate signal
                if signal_type == "pulse":
                    sig = generator.generate_pulse()
                elif signal_type == "chirp":
                    sig = generator.generate_chirp()
                elif signal_type == "fsk":
                    sig = generator.generate_fsk()
                elif signal_type == "psk":
                    sig = generator.generate_psk()
                elif signal_type == "qam":
                    sig = generator.generate_qam()
                else:
                    sig = generator.generate_noise()
                
                # Add artificial anomaly if requested
                if add_anomaly:
                    sig = sig + np.random.randn(len(sig)) * 5
                
                # Extract features
                features = extractor.extract_all_features(sig)
                feature_vector = np.array(list(features.values())).reshape(1, -1)
                
                # Predict anomalies
                results = anomaly_detector.predict_all_methods(feature_vector)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pred = results['isolation_forest'][0]
                if pred == -1:
                    st.error("🚨 **Isolation Forest**: ANOMALY")
                else:
                    st.success("✅ **Isolation Forest**: Normal")
            
            with col2:
                pred = results['one_class_svm'][0]
                if pred == -1:
                    st.error("🚨 **One-Class SVM**: ANOMALY")
                else:
                    st.success("✅ **One-Class SVM**: Normal")
            
            with col3:
                pred = results['ensemble'][0]
                if pred == -1:
                    st.error("🚨 **Ensemble**: ANOMALY")
                else:
                    st.success("✅ **Ensemble**: Normal")
            
            if add_anomaly:
                st.info("ℹ️ Artificial anomaly was added to test detection capability")


# ==================== FULL PIPELINE DEMO ====================
elif page == "📊 Full Pipeline Demo":
    st.header("📊 Complete Pipeline Demonstration")
    st.markdown("See the entire system in action: Generation → Features → Classification → Anomaly Detection")
    
    rf_model, label_encoder, scaler, anomaly_detector = load_models()
    
    if rf_model is None or anomaly_detector is None:
        st.error("❌ Models not found! Please train models first.")
    else:
        signal_type = st.selectbox("Select Signal Type", 
                                   ["pulse", "chirp", "fsk", "psk", "qam", "noise"])
        
        if st.button("Run Full Pipeline", type="primary"):
            # Initialize
            generator = SignalGenerator()
            extractor = FeatureExtractor()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Generate
            status_text.text("Step 1/4: Generating signal...")
            progress_bar.progress(25)
            
            if signal_type == "pulse":
                sig = generator.generate_pulse()
            elif signal_type == "chirp":
                sig = generator.generate_chirp()
            elif signal_type == "fsk":
                sig = generator.generate_fsk()
            elif signal_type == "psk":
                sig = generator.generate_psk()
            elif signal_type == "qam":
                sig = generator.generate_qam()
            else:
                sig = generator.generate_noise()
            
            # Step 2: Extract features
            status_text.text("Step 2/4: Extracting features...")
            progress_bar.progress(50)
            
            features = extractor.extract_all_features(sig)
            feature_vector = np.array(list(features.values())).reshape(1, -1)
            
            # Handle NaN
            nan_mask = np.isnan(feature_vector)
            if np.any(nan_mask):
                col_means = np.nanmean(feature_vector, axis=0)
                col_means[np.isnan(col_means)] = 0
                feature_vector[nan_mask] = np.take(col_means, nan_mask[1])
            
            # Step 3: Classify
            status_text.text("Step 3/4: Classifying signal...")
            progress_bar.progress(75)
            
            feature_scaled = scaler.transform(feature_vector)
            prediction = rf_model.predict(feature_scaled)[0]
            predicted_label = label_encoder.inverse_transform([prediction])[0]
            
            # Step 4: Anomaly detection
            status_text.text("Step 4/4: Checking for anomalies...")
            progress_bar.progress(100)
            
            anomaly_results = anomaly_detector.predict_all_methods(feature_vector)
            is_anomaly = anomaly_results['ensemble'][0] == -1
            
            status_text.text("✅ Pipeline complete!")
            
            # Display results
            st.success("### Pipeline Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Generated Signal", signal_type.upper())
                st.metric("Features Extracted", len(features))
            
            with col2:
                st.metric("Classification", predicted_label.upper())
                correct = "✅" if predicted_label == signal_type else "❌"
                st.metric("Correct?", correct)
            
            with col3:
                anomaly_status = "🚨 ANOMALY" if is_anomaly else "✅ Normal"
                st.metric("Anomaly Status", anomaly_status)


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 👤 Author")
st.sidebar.markdown("**Maxim** - @formertriton")
st.sidebar.markdown("[GitHub Repository](https://github.com/formertriton/signal-processing-ml-system)")