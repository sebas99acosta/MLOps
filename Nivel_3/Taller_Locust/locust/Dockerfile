# Load generator. Runs on the PC (never on the API's VM).
FROM python:3.11-slim

RUN pip install --no-cache-dir locust==2.46.7

WORKDIR /locust
COPY locustfile.py payloads.json ./

EXPOSE 8089

ENTRYPOINT ["locust", "-f", "/locust/locustfile.py"]
