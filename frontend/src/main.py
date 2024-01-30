"""
Project: Prototyping a Machine Learning Application with Streamlit.
Streamlit app integrated with a pretrained ViT model for image classification.
"""
import os
import requests
import cv2
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

#from model.predict import predict_digit
CANVAS_SIZE = 250

def predict_digit(img):
    """
    Sends a request to the backend with the image data
    to perform a classification on the image.
    """
    # Retrieve the URL of the backend from the environment variables
    url_backend = os.environ["URL_BACKEND"]
    # Send a GET request to the backend with the image data as a JSON payload
    request = requests.get(url_backend, json={"image": img.tolist()})
    # Retrieve the predicted probabilities from the response
    answer = request.json()
    prob = answer["prob"]
    return np.array(prob)

def main():
    """
    Main Streamlit function
    Read an image and show a probability
    """

    # Set title
    st.set_page_config(page_title="Digit Recognizer")
    st.title("Digit Recognizer")
  
    # Description
    st.markdown("A Digit Recognizer built with a pretrained ViT model, FastAPI, \
    StreamLit and HuggingFace Model to predict digits 0-10. The pretrained ViT \
    model is a fine-tuned version of google/vit-base-patch16–224-in21k on the \
    mnist dataset.")
    st.markdown("---")

    # Value Initialization
    prob = None
    canvas_image = None


    st.markdown(
        """
    <style>
    button {
        height: auto;
        padding-top: 15px !important;
        padding-bottom: 15px !important;
    }
    [data-testid="stMetricValue"] {
            font-size: 200px;
        }
        
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Row 1
    with st.container():
        left_col, right_col = st.columns((3.5,5.5), gap='medium')

        with left_col:
            #st.subheader(":1234: Draw a number")
            st.markdown("""
            ### Draw a Number:
            """)
        
        with right_col:
            # Create a canvas component for users to draw an image
            canvas_image = st_canvas(
                fill_color="black",
                stroke_width=20,
                stroke_color="gray",
                width=CANVAS_SIZE,
                height=CANVAS_SIZE,
                drawing_mode="freedraw",
                key="canvas",
                display_toolbar=True,
            )
        
        
            if canvas_image is not None and canvas_image.image_data is not None:
                if st.button("Predict"):
                    # Convert the canvas image to RGB format
                    img = cv2.cvtColor(canvas_image.image_data, cv2.COLOR_RGBA2RGB)

                    with st.spinner("Wait for it..."):
                        # Predict the digit and store the predicted probabilities
                        prob = predict_digit(img) * 100.0

    st.markdown("---")
    # Row 2
    with st.container():
        left_col, right_col = st.columns((3.5,5.5), gap='medium')

        with left_col:
            st.markdown("""
            ### Prediction:
            """)
            
        with right_col:
            if prob is not None:
                st.metric(label="Predicted digit:", value=f"{prob.argmax()}")

                

if __name__ == "__main__":
    main()
