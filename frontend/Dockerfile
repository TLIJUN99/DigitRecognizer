FROM python:3.11-slim

# Set working directory
WORKDIR /app/

# Copy the current directory contents into the container at /app
COPY ./src /app

RUN mkdir -p /app/config
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Install the Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create a new user 'appuser' and switch to it
# This user will run the application in the container
RUN useradd appuser && chown -R appuser /app
USER appuser

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable
ENV URL_BACKEND="http://localhost:8000/"

# Run streamlit when the container launches
CMD ["streamlit", "run", "main.py"]