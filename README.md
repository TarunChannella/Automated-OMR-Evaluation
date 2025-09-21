# 🤖 Cyber OMR Scorer

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)  
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25-orange)](https://streamlit.io/)  
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8-brightgreen)](https://opencv.org/)  
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-yellowgreen)](https://pandas.pydata.org/)  

Automated **evaluation and scoring of OMR (Optical Mark Recognition) sheets** using **Python**, **OpenCV**, and **Streamlit**.  
The app allows evaluators to quickly detect filled bubbles, assign scores based on customizable rules, and export results as CSV in a **futuristic cyber-themed UI** ✨.

---

## 🔹 Features

- 🖼 **Upload multiple OMR sheets** and process them at once  
- 🎯 **Detects filled bubbles** with OpenCV (robust against noise/blur)  
- 📊 **Per-subject and total scoring** with positive/negative marking  
- 📝 **Answer Key Management**  
  - Create / Edit / Delete answer keys (JSON-based)  
- 💾 **Download consolidated results as CSV**  
- 🖥 **Streamlit-powered UI** with **neon cyberpunk styling**  
- 🚨 **Warnings for missing/incomplete bubble detection**  

---

## 🛠 Tech Stack

- **Python 3.x**  
- **Streamlit** – Web dashboard  
- **OpenCV** – Image preprocessing and bubble detection  
- **NumPy** – Numerical operations  
- **Pandas** – Data analysis and CSV export  
- **Pillow (PIL)** – Image handling  

---

## 📂 Project Structure
Cyber-OMR-Scorer/
│
├─ answer_keys/ # JSON answer key files
│ └─ setA.json
│
├─ results/ # Generated CSV outputs
│ └─ omr_scored_results.csv
│
├─ omr_dashboard.py # Main Streamlit application
├─ requirements.txt # Dependencies list
├─ README.md # Project documentation
└─ .gitignore # Ignored files/folders



---

## ⚡ How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/Cyber-OMR-Scorer.git
   cd Cyber-OMR-Scorer
2. Create a virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

4. Run the Streamlit app

streamlit run omr_dashboard.py


---

⚡ Do you also want me to make a **`.gitignore` file content** for this repo (to ignore things like `venv/`, `__pycache__/`, `results/`, etc.)?


