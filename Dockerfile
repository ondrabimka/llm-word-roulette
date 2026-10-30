FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY configs /app/configs
COPY data/stopwords /app/data/stopwords
COPY docs /app/docs
COPY scripts /app/scripts

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

EXPOSE 8000

CMD ["sh", "/app/scripts/serve_dashboard.sh"]
