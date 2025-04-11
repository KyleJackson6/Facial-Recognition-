FROM python:3.10-slim

WORKDIR /app

# Install dependencies needed by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Disable camera on Cloud Run
ENV DISABLE_CAMERA=true

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
