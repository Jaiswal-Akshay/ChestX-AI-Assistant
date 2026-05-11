<h1 align="center">🩻 ChestX-AI Assistant</h1>

<p align="center">
  <b>AI-powered chest X-ray analysis & report generation</b>
</p>

<p align="center">
  An end-to-end AI system that analyzes chest X-rays, predicts possible conditions, and generates a radiology-style report using an LLM, exported as a professional PDF.
</p>

---

## 📥 Dataset Setup (CheXpert)

This project uses the **CheXpert v1.0-small dataset**.

---

### 1. Download the Dataset (Kaggle)

Download the dataset from: https://www.kaggle.com/datasets/ashery/chexpert

Unzip the downloaded file and place it in: \ChestX-AI-Assistant\data\raw\CheXpert-v1.0-small

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
### System Pipeline

> ### How it Works
> 1. **Capture:** The system takes a chest `X-ray Image`.
> 2. **Analyze:** A **ResNet** deep learning model calculates `Disease Probabilities`.
> 3. **Interpret:** The **Groq LLM** converts raw data into a structured `Radiology Report`.
> 4. **Deliver:** The final result is exported as a formatted `PDF Output`.

---

##  Example Output

### Findings

> The cardiac silhouette appears enlarged, consistent with cardiomegaly.  
> No focal consolidation is identified. No pleural effusion is seen.  

### Impression


---

## 📊 Outputs Generated

-  `report.pdf` → Final radiology report  
-  `gradcam_result.png` → Model attention visualization  
-  `sample_evidence.json` → Model prediction output  

---

##  Disclaimer

> This project is for **research and educational purposes only**.  
> It does NOT provide medical advice or diagnosis.  
> All outputs must be reviewed by a qualified medical professional.
