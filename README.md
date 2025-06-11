# Voice Gender Recognition using Deep Learning

##  Overview

This project presents a **deep learning-based voice gender classification system** that takes an audio file and predicts the speaker's gender — *male* or *female*. With the increasing integration of voice-enabled systems into modern devices and virtual assistants, voice-based biometric recognition is becoming essential for personalization and accessibility.

We use a combination of **feature extraction**, **classical ML**, **Artificial Neural Networks (ANN)**, and **Convolutional Neural Networks (CNN)**, trained on the **OPENSLR12 (LibriSpeech) dataset**. The system is further enhanced by deployment-ready Flask APIs and model voting logic for robustness.

---

##  Dataset

**OPENSLR12 - LibriSpeech Dataset**

- Audio format: `.flac` files  
- Source: Books read aloud by male and female speakers  
- Data includes:  
  - `train-clean-100`: Folder with audio files by speaker and book ID  
  - `SPEAKERS.TXT`: Contains speaker gender and metadata  
- Size:  
  - Train set: ~6.3 GB (300 hours)  
  - Test set: ~346 MB  

---

##  Features Extracted

| Type                    | Examples                                               |
|-------------------------|--------------------------------------------------------|
| Audio-based             | MFCCs, Chroma, Mel Spectrogram, Tonnetz               |
| Statistical             | Mean Freq, Min/Max Freq, Speech Rate                  |
| Praat-based (Parselmouth) | Articulation Rate, Pause Count, Syllables, Speaking Time |

---

##  Models Implemented

###  Deep Learning
- **Artificial Neural Network (ANN)** – Baseline, improved, and hyperparameter-tuned  
- **Convolutional Neural Network (CNN)** – Spectrogram-based classification with image augmentation

### Traditional ML
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Stochastic Gradient Descent (SGD)  
- Random Forest  
- XGBoost  

---

##  Tools & Libraries

- Python 3.x  
- TensorFlow 2.3, Keras, Keras-Tuner  
- Scikit-learn, XGBoost  
- Librosa, Soundfile, Parselmouth  
- NumPy, Pandas  
- Matplotlib, Seaborn  
- Flask (API)  

---

##  Exploratory Data Analysis (EDA)

- Visualizations of gender-based voice features  
- Distribution plots and correlation heatmaps  
- Feature selection rationale for modeling  
- Insights into effective vs. noisy features (e.g., articulation, pauses, duration)

---

##  Deployment

###  REST API with Flask

- Upload `.flac` or `.wav` files
- Features extracted → Model prediction  
- Two APIs included:
  - **Best Model (CNN-based)**  
  - **Voting System (ANN + CNN + ML)**

###  Docker Support

- Dockerfile included for easy deployment
- Can be tested with Postman or other REST clients

---

##  How to Run

```bash
# Clone the repository
git clone https://github.com/yourusername/voice-gender-recognition.git
cd voice-gender-recognition

# Install dependencies
pip install -r requirements.txt

# Launch Flask API (Local)
python app/api.py
