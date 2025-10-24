# 🛡️ Signal Processing & Machine Learning System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)

Advanced RF signal processing and machine learning system for signal classification and anomaly detection. Designed for defense and intelligence applications.

![System Overview](results/all_signals_comparison.png)

## 🎯 Project Overview

This system demonstrates comprehensive signal processing and machine learning capabilities including:

- **📡 RF Signal Generation**: Synthetic generation of 6 signal types (Pulse, Chirp, FSK, PSK, QAM, Noise)
- **🔍 Feature Extraction**: 60+ features across time, frequency, and wavelet domains
- **🤖 ML Classification**: 4 trained models achieving 95%+ accuracy
- **⚠️ Anomaly Detection**: Multiple detection algorithms for identifying suspicious signals
- **📊 Interactive Dashboard**: Real-time visualization and analysis

## 🚀 Key Features

### Signal Processing
- Synthetic RF signal generation with realistic parameters
- Advanced preprocessing (filtering, normalization, windowing)
- Multi-domain feature extraction:
  - Time domain: RMS, energy, zero-crossing rate, statistical moments
  - Frequency domain: FFT, spectral centroid, bandwidth, entropy
  - Time-frequency: STFT, wavelet decomposition, spectrogram analysis

### Machine Learning
- **Random Forest**: Ensemble learning with 100 trees
- **Support Vector Machine**: RBF kernel classification
- **Gradient Boosting**: Advanced boosting algorithm
- **Neural Network**: Multi-layer perceptron with early stopping

### Anomaly Detection
- **Isolation Forest**: Unsupervised outlier detection
- **One-Class SVM**: Novelty detection algorithm
- **Ensemble Method**: Combined approach for robust detection

### Visualization
- Real-time signal visualization
- Interactive spectrograms and time-frequency analysis
- Confusion matrices and performance metrics
- PCA-based anomaly visualization

## 📋 Requirements

- Python 3.9+
- Windows 10/11 with PowerShell
- 8GB RAM minimum (16GB recommended)
- Git for version control

## 🔧 Installation
```powershell
# Clone the repository
git clone https://github.com/formertriton/signal-processing-ml-system.git
cd signal-processing-ml-system

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## 📖 Project Structure
```
signal-processing-ml-system/
├── src/
│   ├── signal_generation/       # RF signal generators
│   ├── preprocessing/            # Signal preprocessing
│   ├── feature_extraction/       # Feature extraction engines
│   ├── ml_models/               # ML classifiers
│   ├── anomaly_detection/       # Anomaly detectors
│   └── visualization/           # Plotting and dashboards
├── data/
│   ├── raw/                     # Raw signal datasets
│   └── processed/               # Processed features
├── models/
│   └── trained/                 # Trained ML models
├── results/                     # Output visualizations
├── tests/                       # Unit and integration tests
├── config/                      # Configuration files
├── demo_dashboard.py            # Interactive Streamlit demo
└── README.md
```

## 🎮 Usage

### Quick Start - Interactive Dashboard
```powershell
streamlit run demo_dashboard.py
```

This launches an interactive web interface where you can:
- Generate and visualize signals
- Extract and analyze features
- Classify signals in real-time
- Detect anomalies
- Run the complete pipeline

### Command Line Usage

#### 1. Generate Dataset
```powershell
python src\signal_generation\generate_dataset.py
```

#### 2. Extract Features
```powershell
python src\feature_extraction\extract_features.py
```

#### 3. Train ML Models
```powershell
python src\ml_models\train_models.py
```

#### 4. Run Anomaly Detection
```powershell
python src\anomaly_detection\detect_anomalies.py
```

#### 5. Create Visualizations
```powershell
python src\visualization\plot_signals.py
```

## 📊 Results

### Classification Performance

| Model | Train Accuracy | Test Accuracy |
|-------|---------------|---------------|
| Random Forest | 99.8% | 97.2% |
| Gradient Boosting | 98.5% | 96.8% |
| Neural Network | 97.9% | 95.4% |
| SVM | 96.2% | 94.1% |

### Anomaly Detection

- **Isolation Forest**: 10% anomaly rate
- **One-Class SVM**: 12% anomaly rate
- **Ensemble Method**: 15% anomaly rate

All visualizations and detailed results are saved in the `results/` directory.

## 🎓 Skills Demonstrated

- **Signal Processing**: FFT, STFT, Wavelet analysis, filtering, spectral analysis
- **Machine Learning**: Classification, ensemble methods, neural networks, cross-validation
- **Anomaly Detection**: Unsupervised learning, outlier detection, novelty detection
- **Software Engineering**: Clean architecture, modular design, version control
- **Data Science**: Feature engineering, model evaluation, visualization
- **Defense Applications**: RF analysis, threat detection, signal intelligence

## 🔬 Technical Details

### Signal Types
- **Pulse**: Simple sinusoidal burst (radar pulse)
- **Chirp**: Linear frequency modulation (LFM radar)
- **FSK**: Frequency Shift Keying (digital communications)
- **PSK**: Phase Shift Keying (secure communications)
- **QAM**: Quadrature Amplitude Modulation (high-speed data)
- **Noise**: White Gaussian noise (interference/jamming)

### Feature Engineering
The system extracts 60+ features including:
- Statistical moments (mean, variance, skewness, kurtosis)
- Signal energy and power metrics
- Spectral characteristics (centroid, bandwidth, rolloff)
- Wavelet decomposition coefficients
- Time-frequency representations

## 🤝 Contributing

This is a portfolio project, but suggestions and feedback are welcome! Feel free to:
- Open issues for bugs or feature requests
- Submit pull requests for improvements
- Star the repository if you find it useful

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

## 📧 Contact

**Maxim** - [@formertriton](https://github.com/formertriton)

## 🙏 Acknowledgments

- Built with Python, NumPy, SciPy, Scikit-learn, and TensorFlow
- Signal processing theory from IEEE standards
- ML techniques from academic research

---

*Developed for demonstration of signal processing, machine learning, and software engineering capabilities for defense and intelligence applications.*