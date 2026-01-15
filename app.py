import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# define the model architecture
class ResNetEmotionClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNetEmotionClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

# load model
@st.cache_resource
def load_model():
    device = torch.device('cpu')  # using CPU for deployment
    model = ResNetEmotionClassifier(num_classes=7)
    checkpoint = torch.load('best_resnet18_emotion_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['class_to_idx']

# image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# predict emotion
def predict_emotion(model, image, class_to_idx):
    with torch.no_grad():
        img_tensor = preprocess_image(image)
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_idx = output.argmax(1).item()
        confidence = probabilities[0][predicted_idx].item()
        
        # get emotion name from index
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        emotion = idx_to_class[predicted_idx]
        
        return emotion, confidence, probabilities[0].numpy()

# streamlit UI
st.set_page_config(page_title="Expression Recognition", page_icon="😊", layout="wide")

st.title("🎭 Facial Expression Classification using ResNet-18 Transfer Learning")
st.write("Upload a facial image to detect the emotion using ResNet18")

# load model
model, class_to_idx = load_model()
idx_to_class = {v: k for k, v in class_to_idx.items()}

# file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # display image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("Prediction Results")
        
        # make prediction
        with st.spinner("Analyzing emotion..."):
            emotion, confidence, probabilities = predict_emotion(model, image, class_to_idx)
        
        # display prediction
        st.success(f"**Predicted Emotion:** {emotion.upper()}")
        st.info(f"**Confidence:** {confidence*100:.2f}%")
        
        # show all probabilities
        st.subheader("All Emotion Probabilities")
        prob_dict = {idx_to_class[i]: probabilities[i]*100 for i in range(len(probabilities))}
        prob_dict = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
        
        for emotion_name, prob in prob_dict.items():
            st.progress(prob/100)
            st.write(f"{emotion_name}: {prob:.2f}%")

# Sidebar info
st.sidebar.header("About")
st.sidebar.info(
    """
    This app uses a ResNet18 model trained on the FER2013 dataset 
    to recognize 7 emotions:
    - Angry
    - Disgust
    - Fear
    - Happy
    - Sad
    - Surprise
    - Neutral
    
    **Model Accuracy:** 65.42%
    """
)

st.sidebar.header("How to Use")
st.sidebar.write(
    """
    1. Upload a facial image (JPG, JPEG, or PNG)
    2. Wait for the model to analyze
    3. View the predicted emotion and confidence
    """
)