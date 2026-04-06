# 🧠⚡ SeizureX — EEG Seizure Detection

SeizureX is a machine learning project aimed at detecting epileptic seizures from EEG signals using the CHB-MIT Scalp EEG dataset. The goal is to build a reliable seizure detection system that can generalize across patients and eventually move toward real-time monitoring and assistive healthcare applications.

---

## 🎯 Goal

- Detect seizures from EEG signals  
- Build cross-patient seizure detection model  
- Reduce false positives  
- Move toward real-time detection  
- Explore deep learning approaches (CNN / LSTM)

---

## ⚙️ What SeizureX Does

- Loads EEG EDF files  
- Segments signals into windows  
- Extracts statistical & frequency features  
- Labels seizure vs non-seizure data  
- Trains machine learning model  
- Evaluates on unseen EEG recordings  

---

## 📈 Current Progress

- ✅ Working ML pipeline  
- ✅ Cross-patient testing implemented  
- ✅ **94% accuracy achieved**  
- ⚠️ Few false positives present  
- 🔄 Improving model using CNN / LSTM  

---

## 📊 Dataset Used

This project uses the **CHB-MIT Scalp EEG Database**.

- 🟢 CHB01 → Training (46 files)  
- 🔵 CHB02 → Testing (37 files)  

---

## 📓 Run the Project

`seizureX.ipynb`

Download and run in:
- Google Colab  
- Kaggle  
- Jupyter Notebook  
- Local environment  
