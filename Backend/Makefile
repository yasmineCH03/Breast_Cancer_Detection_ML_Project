# OncoAI Medical API - Simple Makefile
PYTHON = python3
VENV = venv
PORT = 8000

.PHONY: help venv install test api clean

help:
	@echo "🏥 OncoAI Medical API - Available Commands:"
	@echo "  make venv    - Create virtual environment"
	@echo "  make install - Install dependencies"
	@echo "  make test    - Run basic tests"
	@echo "  make api     - Start API server"
	@echo "  make clean   - Clean temporary files"

venv:
	@echo "📦 Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "✅ Virtual environment created"
	@echo "💡 Activate with: source $(VENV)/bin/activate"

install:
	@echo "📚 Installing dependencies..."
	@if [ ! -d "$(VENV)" ]; then echo "❌ Virtual environment not found. Run 'make venv' first"; exit 1; fi
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install fastapi uvicorn scikit-learn joblib numpy pandas
	@echo "✅ Dependencies installed"

test:
	@echo "🧪 Testing setup..."
	@if [ ! -d "$(VENV)" ]; then echo "❌ Virtual environment not found"; exit 1; fi
	$(VENV)/bin/python -c "import fastapi; print('✅ FastAPI:', fastapi.__version__)"
	$(VENV)/bin/python -c "import sklearn; print('✅ Scikit-learn:', sklearn.__version__)"
	$(VENV)/bin/python -c "import joblib; print('✅ Joblib:', joblib.__version__)"
	@echo "✅ All tests passed!"

api:
	@echo "🚀 Starting API server..."
	@echo "📚 Docs: http://localhost:$(PORT)/docs"
	@echo "🔗 Health: http://localhost:$(PORT)/api/health"
	@echo ""
	@if [ ! -f "app/main.py" ]; then echo "❌ app/main.py not found"; exit 1; fi
	$(VENV)/bin/python app/main.py

clean:
	@echo "🧹 Cleaning temporary files..."
	@rm -rf __pycache__
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Clean complete"

.PHONY: install-dev lint typecheck pytest eda-wbcd train-wbcd evaluate-wbcd

install-dev:
	@if [ ! -d "$(VENV)" ]; then echo "❌ Virtual environment not found. Run 'make venv' first"; exit 1; fi
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements-dev.txt

lint:
	@if [ ! -d "$(VENV)" ]; then echo "❌ Virtual environment not found"; exit 1; fi
	$(VENV)/bin/black --check .
	$(VENV)/bin/isort --check-only .
	$(VENV)/bin/ruff Backend

typecheck:
	@if [ ! -d "$(VENV)" ]; then echo "❌ Virtual environment not found"; exit 1; fi
	$(VENV)/bin/mypy Backend/app

pytest:
	@if [ ! -d "$(VENV)" ]; then echo "❌ Virtual environment not found"; exit 1; fi
	$(VENV)/bin/pytest -q

eda-wbcd:
	@if [ ! -d "$(VENV)" ]; then echo "❌ Virtual environment not found"; exit 1; fi
	$(VENV)/bin/python app/scripts/extract_wbcd_eda.py

train-wbcd:
	@if [ ! -d "$(VENV)" ]; then echo "❌ Virtual environment not found"; exit 1; fi
	$(VENV)/bin/python app/scripts/run_train_wbcd.py

evaluate-wbcd:
	@if [ ! -d "$(VENV)" ]; then echo "❌ Virtual environment not found"; exit 1; fi
	$(VENV)/bin/python app/scripts/run_evaluate_wbcd.py
