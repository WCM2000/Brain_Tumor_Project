import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('resnet50_model.keras')

# Define the class names 
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary Tumor']

def preprocess_image(image):
    """Preprocess the image to fit the model's input requirements."""
    image = image.resize((224, 224))  # Resize to 224x224 as expected by the model
    image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

def predict(image):
    """Predict the class of the image using the loaded model."""
    image_array = preprocess_image(image)
    
    # Debugging: print the shape of the input image array
    st.write("Preprocessed image shape:", image_array.shape)
    
    predictions = model.predict(image_array)
    
    # Debugging: print the raw prediction values
    st.write("Raw predictions:", predictions)
    
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_percentage = np.max(predictions) * 100
    
    return class_names[predicted_class], predicted_percentage

# Set up Streamlit page configuration
st.set_page_config(page_title="Brain Tumor Classifier", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Brain Tumor Classifier", "How This Works?"])

# Home Page
if page == "Home":
    st.title("Welcome to Brain Tumor Classifier")

    home_image_url = "https://miro.medium.com/v2/resize:fit:493/1*_pCEmeOrVkayCFER33o4Cw.jpeg"
    st.image(home_image_url, caption="Brain Tumor Detection", width=600)  

    st.write("""
    This application is designed to classify MRI scans and provide insights about potential brain tumors.
    
    **How to use the app:**
    1. Navigate to the "Brain Tumor Classifier" page using the sidebar.
    2. Upload an MRI scan image in PNG, JPG, or JPEG format.
    3. Wait for the app to provide predictions and information about potential tumors.
    
    **Disclaimer:**
    This tool is for educational and research purposes only and is not intended for medical diagnostics.
    """)

# Brain Tumor Classifier Page
elif page == "Brain Tumor Classifier":
    st.title("Brain Tumor MRI Image Prediction")

    # Display instructions
    st.write("""
    **Step 1**: Upload an MRI scan using the button below.

    **Step 2**: View predictions and additional information.
    """)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image with a custom size
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=400) 
        
        # Make predictions
        class_name, percentage = predict(image)
        
        # Display the results
        st.write(f"Prediction: **{class_name}**")
        st.write(f"Prediction Confidence: **{percentage:.2f}%**")

# "How This Works?" Page
elif page == "How This Works?":
    st.title("How This Works?")

    # Add an expander to provide detailed information
    with st.expander("Click here to learn more about the model and process"):
        st.write("""
        This application uses a Convolutional Neural Network (CNN) model trained on MRI images to classify brain tumors.
        Here's how the process works:
        
        1. **Model Architecture**: The CNN model consists of several layers including convolutional layers, pooling layers, and dense layers. 
        The model is designed to extract features from MRI images and use these features to predict whether a tumor is present.
        
        2. **Data Preprocessing**: Uploaded MRI images are preprocessed by resizing them to a 224x224 resolution and normalizing the pixel values.
        
        3. **Prediction**: The preprocessed image is fed into the model, which outputs probabilities for each class (e.g., tumor or no tumor).
        
        4. **Interpretation**: The class with the highest probability is chosen as the prediction. The confidence level indicates how certain the model is about its prediction.
        
        This tool is built for educational purposes and is not intended for clinical use.
        """)

        # Adding the image with a caption
        cnn_image_url = "https://saturncloud.io/images/blog/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way.webp"
        st.image(cnn_image_url, caption="Convolutional Neural Network Overview", width=500)
