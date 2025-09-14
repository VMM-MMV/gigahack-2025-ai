# Use a Python base image
FROM python:3.11-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency definition files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen-lock

# Copy the rest of your application code
COPY . .