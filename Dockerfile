FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY anomsvc ./anomsvc
# Train at build time so the image is self-contained and the artifact always
# matches the scikit-learn version installed in the image.
RUN mkdir -p models && python -m anomsvc.train --out models/model.joblib

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "anomsvc.api:app", "--host", "0.0.0.0", "--port", "8000"]
