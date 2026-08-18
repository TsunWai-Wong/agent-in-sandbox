# Dockerfile.sandbox
FROM python:3.12-slim

# Non-root user — never run agents as root
RUN useradd -m -s /bin/bash agent
WORKDIR /workspace

# Install common tools (locked versions, no auto-update)
RUN pip install --no-cache-dir \
    ruff==0.4.4 \
    pytest==8.2.0 \
    httpx==0.27.0

# Drop all capabilities, agent gets only what's listed
USER agent