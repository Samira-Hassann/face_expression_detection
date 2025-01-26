import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import helper 

st.title("Face Expression Detection")
model = load_model('artifacts/model_weights.keras')
uploaded_img = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])

if st.button("Predict Emotion"):
    
    if uploaded_img :
        image = Image.open(uploaded_img).convert('RGB')  
        st.image(image, caption='Uploaded Image', use_column_width=True)
        image_np = np.array(image)

        preprocessed_image = helper.image_preprocessing(image_np)
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0) 

        prediction = model.predict(preprocessed_image)
        emotion = helper.getcode(np.argmax(prediction))

        if emotion:
            st.write(f"Detected Emotion: {emotion}")
        else:
            st.write("No clear emotion detected.")
    else:
        st.write("please upload photo .")


