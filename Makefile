# -----------------------------------------------------------
# ONCOGUARD MLOPS AUTOMATION (WSL/LINUX ROBUST)
# -----------------------------------------------------------

# Define explicit paths to the virtual environment
VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

# 1. INSTALLATION (Auto-creates venv if missing)
install:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r api/requirements.txt

# 2. CODE QUALITY
format:
	$(PYTHON) -m black api/ tests/

lint:
	$(PYTHON) -m pylint --disable=R,C api/main.py

# 3. TESTING
test:
	$(PYTHON) -m pytest tests/

# 4. LOCAL EXECUTION
run:
	$(PYTHON) -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 5. DOCKER OPERATIONS (These don't need venv)
docker-build:
	docker build -t oncoguard-api .

docker-run:
	docker run -d -p 8000:8000 --name my-onco-api oncoguard-api

docker-stop:
	docker stop my-onco-api && docker rm my-onco-api

all: install format lint test docker-build