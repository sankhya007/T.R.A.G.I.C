# Start from an official Python image.
# "slim" means no bloat — just the OS and Python, nothing extra.
FROM python:3.10-slim

# Set the working directory inside the container.
# Every command below runs from here.
WORKDIR /app

# Install system-level libraries that OpenCV needs.
# These aren't Python packages — they're OS-level and won't be
# picked up by pip, which is why we install them separately here.
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (before the rest of your code).
# Docker caches each step. If requirements.txt hasn't changed,
# it won't re-run pip install on every build — saves a lot of time.
COPY requirements.txt .

# Install Python dependencies.
# --no-cache-dir keeps the image smaller (no pip cache stored inside).
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project into the container.
COPY . .

# This is just documentation — it tells anyone reading the Dockerfile
# that the container expects output to go here.
# The actual folder is created by docker-compose via a volume mount.
VOLUME ["/app/output"]

# Default command: show usage instructions.
# Users override this when running (see docker-compose.yml).
CMD ["python", "-c", "print('TRAGIC container ready. Run a simulation script directly.')"]
