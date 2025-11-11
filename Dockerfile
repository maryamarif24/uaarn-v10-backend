# Step 1: Use Python 3.8 (compatible with TensorFlow 1.15)
FROM python:3.8-slim

# Step 2: Set working directory
WORKDIR /app

# Step 3: Copy your project files
COPY . .

# Step 4: Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Step 5: Expose the port your app will run on
EXPOSE 8080

# Step 6: Run your FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
