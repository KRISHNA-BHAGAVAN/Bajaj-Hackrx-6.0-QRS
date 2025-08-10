# Stage 1: Base Image and System Dependencies
# Use the recommended Python 3.12 version on a slim Debian base.
FROM python:3.12-slim

# Set metadata labels
LABEL author="Krishna Bhagavan"
LABEL description="Entity - An Intelligent Query–Retrieval System."

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Create a non-root user for security
# Create a new group 'appgroup' and a new user 'appuser'
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Stage 4: Application Code
# Copy source code and change ownership to the new user
COPY . .
RUN chown -R appuser:appgroup /app

# Switch to the non-root user
USER appuser

# Stage 5: Expose Port and Run Application
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]