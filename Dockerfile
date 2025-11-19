# Use a base image with Python and minimal footprint
FROM python:3.13-slim

# Copy uv binaries from GHCR image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /code

# Install compiler dependencies for building Python packages
RUN apt-get update && apt-get install -y gcc g++ python3-dev && rm -rf /var/lib/apt/lists/*

# Ensure the virtualenv PATH is first
ENV PATH="/code/.venv/bin:$PATH"

# Copy dependency management files AND README.md
COPY pyproject.toml uv.lock .python-version README.md LICENSE ./
COPY scripts/ ./scripts/

# Install dependencies using uv
RUN uv sync --locked

# Copy application code
COPY predict.py ./
COPY outputs/models/ ./models/
COPY examples/ ./examples/

# Expose the API port
EXPOSE 9696

# Launch the FastAPI application
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]