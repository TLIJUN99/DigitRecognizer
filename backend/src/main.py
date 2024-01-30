"""
FastAPI app to receive a HTTP GET request at root endpoint ("/") 
and returns prediction probability of digit image.
"""
import numpy as np
from fastapi import FastAPI, Request
from model.predict import predict_digit

# Create FastAPI instance
app = FastAPI()


# Define GET route to predict the digit
@app.get("/")
async def get_prediction(info: Request):
    # Parse input image from the request
    req_info = await info.json()
    img = np.array(req_info["image"])

    # Predict the digit using the imported function
    prob = predict_digit(img)

    # Return probability of each class in a JSON format
    return {"prob": prob.tolist()}
