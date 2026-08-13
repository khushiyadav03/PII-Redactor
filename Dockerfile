FROM python:3.11-slim

# Hinglish: Tesseract and OpenCV runtime dependencies install karte hain
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Hinglish: spaCy model download compile-time par hi run karte hain container start delay reduce karne ke liye
RUN python -m spacy download en_core_web_sm

COPY . .

EXPOSE 8000

# Hinglish: API server start command
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
