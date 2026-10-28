FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY configs ./configs

ENTRYPOINT ["kws"]
CMD ["--help"]
