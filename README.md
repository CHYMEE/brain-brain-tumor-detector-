
# 🧠 Brain Tumor Detection using GRUNet

This Streamlit app uses a Gated Recurrent Unit (GRU) neural network to predict whether a brain MRI scan shows signs of a tumor.

## 📂 Project Files
- `app.py` – The main Streamlit application
- `requirements.txt` – Python dependencies
- `trained_grunet_brain_tumor.pth` – Trained PyTorch model

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 Deploy on Streamlit Cloud
1. Push this repository to GitHub
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account and deploy using:
   - **Main file:** `app.py`
   - **Branch:** `main`

## 🖼 Sample App Screenshot
![App Screenshot](banner.png)

## 🧪 Example Predictions
- BrainTumor 🟥 → Tumor Detected
- Healthy ✅ → No Tumor

---

Built with ❤️ using PyTorch and Streamlit
