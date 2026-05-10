<h1 align="center">🩻 ChestX-AI Assistant</h1>

<p align="center">
  <b>AI-powered chest X-ray analysis & report generation</b>
</p>

<p align="center">
  An end-to-end AI system that analyzes chest X-rays, predicts possible conditions, and generates a radiology-style report using an LLM, exported as a professional PDF.
</p>

---

## 🚀 Features

- 🧠 Deep Learning Model (ResNet18 / DenseNet121)  
- 📊 Multi-label classification (5 diseases)  
- 🔥 Grad-CAM explainability (optional)  
- 📝 LLM-powered report generation (Groq / OpenAI compatible)  
- 📄 Professional PDF report output  
- ⚡ One-command pipeline  

---

## 🧠 Supported Findings

- Atelectasis  
- Cardiomegaly  
- Consolidation  
- Pleural Effusion  
- Pneumonia  

---

## 📂 Project Structure


ChestX-AI-Assistant/
├── src/
│ ├── models/
│ │ └── predict_sample.py
│ ├── explainability/
│ │ └── gradcam.py
│ ├── reporting/
│ │ └── new_generate_report_groqcloud.py
│
├── data/
│ └── raw/
│ └── patient_xray.png
│
├── outputs/
│ └── reports/
│ ├── report.pdf
│ ├── gradcam_result.png
│ └── sample_evidence.json
│
├── run_pipeline.py
├── requirements.txt
└── README.md


---

## ⚙️ Installation

### 1. Install dependencies
```bash
pip install torch torchvision pandas numpy scikit-learn pillow matplotlib opencv-python tqdm openai reportlab
```

### 2. Run the code
``` bash 
export GROQ_API_KEY="your_api_key_here"

python run_pipeline.py \
  --image data/raw/patient_xray.png \
  --patient "John Doe" \
  --dob "1975-03-12" \
  --mrn "MRN-00421" \
  --out outputs/reports/report.pdf
```
X-ray Image
   ↓
Deep Learning Model (ResNet)
   ↓
Disease Probabilities
   ↓
LLM (Groq)
   ↓
Radiology Report
   ↓
PDF Output

---

## 📄 Example Output

### Findings

> The cardiac silhouette appears enlarged, consistent with cardiomegaly.  
> No focal consolidation is identified. No pleural effusion is seen.  

### Impression


---

## 📊 Outputs Generated

- 📄 `report.pdf` → Final radiology report  
- 🔥 `gradcam_result.png` → Model attention visualization  
- 📁 `sample_evidence.json` → Model prediction output  

---

## ⚠️ Disclaimer

> This project is for **research and educational purposes only**.  
> It does NOT provide medical advice or diagnosis.  
> All outputs must be reviewed by a qualified medical professional.
