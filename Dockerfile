FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
# Train the model first and then run the app
CMD ["sh", "-c", "python train.py && streamlit run --server.port=8501 app.py"]