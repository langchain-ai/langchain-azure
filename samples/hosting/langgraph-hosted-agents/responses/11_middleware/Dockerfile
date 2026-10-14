FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8088
CMD ["python", "-m", "langchain_azure_ai.agents.hosting.run", "--protocol", "responses"]
