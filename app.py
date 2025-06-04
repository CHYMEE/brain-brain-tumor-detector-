
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ----- MODEL DEFINITION -----
class GRUNet(nn.Module):
    def __init__(self, input_size=64, hidden_size=128, num_classes=2):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.squeeze(1)  # (B, H, W)
        h0 = torch.zeros(1, x.size(0), 128)
        out, _ = self.gru(x, h0)
        return self.fc(out[:, -1, :])

# ----- LOAD MODEL -----
@st.cache_resource
def load_model():
    model = GRUNet()
    model.load_state_dict(torch.load("trained_grunet_brain_tumor.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# ----- IMAGE TRANSFORMATION -----
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ----- STREAMLIT UI -----
st.title("🧠 Brain Tumor Detector - GRUNet")
st.write("Upload a grayscale or color brain MRI scan. The model will predict if it shows a tumor.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('L')  # Force grayscale
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0)  # Shape: (1, 1, 64, 64)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        prediction = 'Brain Tumor' if pred.item() == 0 else 'Healthy'

    st.subheader("🩺 Prediction Result")
    st.success(f"This MRI image is predicted as: **{prediction}**")
