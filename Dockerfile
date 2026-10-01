# Start from the official Python 3.10 image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy your dependency list and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files into the container
COPY . .

# Expose the ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Create a startup script to run BOTH backend and frontend
RUN echo '#!/bin/bash\n\
uvicorn app.main:app --host 0.0.0.0 --port 8000 &\n\
sleep 3\n\
streamlit run frontend/dashboard.py --server.port 8501 --server.address 0.0.0.0\n\
' > start.sh

# Make the script executable
RUN chmod +x start.sh

# Run the startup script when the container launches
CMD ["./start.sh"]
