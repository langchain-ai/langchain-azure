FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8088
CMD ["python", "main.py"]
