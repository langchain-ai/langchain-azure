FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
# pip 26 fixes parsing of the SDK's version-like v1 extra.
RUN python -m pip install --no-cache-dir --upgrade "pip>=26.0"
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8088

# Precompile bytecode during the build to reduce cold-start import work.
# Standard library and dependencies are best-effort; application code is strict.
# Clearing PYTHONDONTWRITEBYTECODE permits writes; cached bytecode is still readable.
RUN PYTHONDONTWRITEBYTECODE= python -m compileall -q $(python -c "import sysconfig as s; print(s.get_paths()['stdlib'], s.get_paths()['purelib'])") || true; \
    PYTHONDONTWRITEBYTECODE= python -m compileall -q .

CMD ["python", "main.py"]
