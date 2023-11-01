# Receipts Prediction App

This application predicts the number of scanned receipts for each month in 2022 using a trained LSTM model based on 2021 data.
## Prerequisites:
- Docker installed on your machine. If you don't have Docker, you can download and install it from [Docker's official website](https://www.docker.com/get-started).

## Setup and Run:
### 1. Clone the repository:
```
git clone <repository-url>
cd <repository-directory>
```

### 2. Build the Docker Image:
Inside the project directory, run the following command to build the Docker image:
```
docker build -t receipts-prediction-app .
```

### 3. Run the Docker Container:
After building the image, run the Docker container using:
```
docker run -i -p 8501:8501 -e TRAIN=y receipts-prediction-app
```
if you trained once and only want to deploy the web set env variable **TRAIN=n**

### 4. Access the App:
Once the container is running, you can access the Streamlit app by opening your browser and navigating to:
```
http://localhost:8501
```