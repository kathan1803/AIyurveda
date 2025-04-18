import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("weights/AIyurveda.h5")  
    return model

model = load_model()

# Class labels
CLASS_NAMES = [
    "Acne and Rosacea Photos",
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "Atopic Dermatitis Photos",
    "Bullous Disease Photos",
    "Cellulitis Impetigo and other Bacterial Infections",
    "Eczema Photos",
    "Exanthems and Drug Eruptions",
    "Hair Loss Photos Alopecia and other Hair Diseases",
    "Herpes HPV and other STDs Photos",
    "Light Diseases and Disorders of Pigmentation",
    "Lupus and other Connective Tissue diseases",
    "Melanoma Skin Cancer Nevi and Moles",
    "Nail Fungus and other Nail Disease",
    "Poison Ivy Photos and other Contact Dermatitis",
    "Psoriasis pictures Lichen Planus and related diseases",
    "Scabies Lyme Disease and other Infestations and Bites",
    "Seborrheic Keratoses and other Benign Tumors",
    "Systemic Disease",
    "Tinea Ringworm Candidiasis and other Fungal Infections",
    "Urticaria Hives",
    "Vascular Tumors",
    "Vasculitis Photos",
    "Warts Molluscum and other Viral Infections"
]

# Preprocessing
def preprocess_image(image):
    image = image.resize((256, 256))
    img_array = np.array(image)

    if img_array.shape[-1] == 4:  # If image has an alpha channel
        img_array = img_array[..., :3]

    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit App UI
st.title("🌿 AIyurveda - Skin Disease Detection")
st.write("Upload an image and let the model detect the disease.")

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    if st.button("Predict Disease"):
        with st.spinner("Analyzing..."):
            processed_img = preprocess_image(image)
            predictions = model.predict(processed_img)[0]
            top_idx = np.argmax(predictions)
            predicted_label = CLASS_NAMES[top_idx]

            st.success(f"✅ **Predicted Disease:** {predicted_label}")
            st.write("📊 **Confidence Scores:**")
            for i in np.argsort(predictions)[::-1][:5]:  # Top 5
                st.write(f"{CLASS_NAMES[i]}: {predictions[i]*100:.2f}%")