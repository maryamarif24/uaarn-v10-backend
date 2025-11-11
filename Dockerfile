# Step 4: Install dependencies (build)
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Step 6: Start command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
