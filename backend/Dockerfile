FROM python:3.9-slim

WORKDIR /app

# Install uv package manager
RUN pip install uv

# Copy only the dependency files to leverage Docker layer caching
COPY pyproject.toml requirements.txt ./

# Install dependencies using uv
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY ./app ./app

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]