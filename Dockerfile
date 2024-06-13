
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    pkg-config \
    libhdf5-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . .
# Expose port 80 to the outside world
EXPOSE 80

# Run the FastAPI server using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]

