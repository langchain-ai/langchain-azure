FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
# pip 26 fixes parsing of the SDK's version-like v1 extra.
RUN python -m pip install --no-cache-dir --upgrade "pip>=26.0"
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8088
CMD ["python", "main.py"]
