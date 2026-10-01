# Start from the official Python 3.10 image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy your dependency list and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files into the container
COPY . .

# 1. Bind FastAPI strictly to internal localhost (127.0.0.1) so the internet cannot see it.
# 2. Bind Streamlit to 0.0.0.0 using Render's official $PORT variable so it becomes the public site.
RUN echo '#!/bin/bash\n\
uvicorn app.main:app --host 127.0.0.1 --port 8000 &\n\
sleep 3\n\
streamlit run frontend/dashboard.py --server.port $PORT --server.address 0.0.0.0\n\
' > start.sh

# Make the script executable
RUN chmod +x start.sh

# Run the startup script when the container launches
CMD ["./start.sh"]