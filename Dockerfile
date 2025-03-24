# Use a base image with Python and system dependencies
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code to the working directory
COPY . .

# Expose the port for Streamlit
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]
