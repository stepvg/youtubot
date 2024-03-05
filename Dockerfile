# Install base Python image
FROM python:3.11-slim-bullseye

# Copy files to the container
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Run uvicorn server
CMD ["python", "main.py"]