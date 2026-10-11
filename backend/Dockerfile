FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY products/maf ./products/maf
COPY products/sentinel ./products/sentinel

RUN pip install --no-cache-dir ".[production]" \
    && pip install --no-cache-dir ./products/sentinel ./products/maf

ENV CLM_DATA_DIR=/app/data
ENV CLM_RUNTIME_MODE=production
VOLUME ["/app/data"]

EXPOSE 8800

CMD ["uvicorn", "clm_aws_decision_support.webapp:app", "--host", "0.0.0.0", "--port", "8800"]