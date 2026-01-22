.PHONY: install dev test lint format clean run serve docker-build docker-run

# Python
PYTHON := python3
PIP := pip3

# Project
PROJECT_NAME := overnight-predict
SRC_DIR := src
TEST_DIR := tests

# Default target
all: install

# Install production dependencies
install:
	$(PIP) install -e .

# Install development dependencies
dev:
	$(PIP) install -e ".[dev]"

# Run tests
test:
	pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=term-missing

# Run tests with coverage report
test-cov:
	pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=html
	@echo "Coverage report generated at htmlcov/index.html"

# Run linting
lint:
	ruff check $(SRC_DIR) $(TEST_DIR)
	mypy $(SRC_DIR)

# Format code
format:
	ruff format $(SRC_DIR) $(TEST_DIR)
	ruff check --fix $(SRC_DIR) $(TEST_DIR)

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Run overnight coding session
run:
	$(PYTHON) -m src.cli start

# Start API server
serve:
	$(PYTHON) -m src.cli serve

# Start API server with auto-reload
serve-dev:
	$(PYTHON) -m src.cli serve --reload

# Show status
status:
	$(PYTHON) -m src.cli status

# Initialize new project
init:
	$(PYTHON) -m src.cli init

# Build Docker image
docker-build:
	docker build -t $(PROJECT_NAME):latest .

# Run in Docker
docker-run:
	docker run -it --rm \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/logs:/app/logs \
		-p 8000:8000 \
		--env-file .env \
		$(PROJECT_NAME):latest

# Docker compose up
docker-up:
	docker-compose up -d

# Docker compose down
docker-down:
	docker-compose down

# Help
help:
	@echo "OvernightPredict - AI-powered autonomous code generation"
	@echo ""
	@echo "Available targets:"
	@echo "  install     - Install production dependencies"
	@echo "  dev         - Install development dependencies"
	@echo "  test        - Run tests"
	@echo "  test-cov    - Run tests with coverage report"
	@echo "  lint        - Run linting"
	@echo "  format      - Format code"
	@echo "  clean       - Clean build artifacts"
	@echo "  run         - Run overnight coding session"
	@echo "  serve       - Start API server"
	@echo "  serve-dev   - Start API server with auto-reload"
	@echo "  status      - Show system status"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run  - Run in Docker container"
	@echo "  help        - Show this help message"
