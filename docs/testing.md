# Testing Guide

## Test Categories

The project contains several types of tests:

1. **Unit Tests** (tests/core/, tests/client/, tests/service/)
   - Test individual components in isolation
   - Fast execution
   - Examples: test_llm.py, test_client.py

2. **Integration Tests** (tests/integration/)
   - Test interactions between components
   - Slower execution
   - Example: test_docker_e2e.py

3. **Application Tests** (tests/app/)
   - Test the Streamlit web interface
   - Example: test_streamlit_app.py

## Running Tests

### Prerequisites
- Python 3.9+ installed
- Dependencies installed: `pip install -r requirements.txt`

### Running All Tests
```bash
pytest tests/
```

### Running Specific Test Categories
```bash
# Unit tests
pytest tests/core/ tests/client/ tests/service/

# Integration tests
pytest tests/integration/

# Application tests
pytest tests/app/
```

### Running Individual Test Files
```bash
# Example: Run LLM tests
pytest tests/core/test_llm.py
```

### Running Specific Test Functions
```bash
# Example: Run specific test function
pytest tests/core/test_llm.py::test_llm_initialization
```

## Test Coverage

To generate coverage reports:
```bash
pytest --cov=src --cov-report=html tests/
```

This will:
1. Run all tests
2. Generate coverage report in HTML format
3. Create coverage report in `htmlcov/` directory

View coverage by opening `htmlcov/index.html` in your browser.

## Continuous Integration

Tests are automatically run on GitHub Actions:
- Configuration: `.github/workflows/test.yml`
- Runs on every push and pull request
- Includes:
  - Unit tests
  - Integration tests
  - Code coverage reporting
  - Code quality checks

## Debugging Tests

To debug failing tests:
1. Run with `-v` for verbose output
2. Use `pytest --pdb` to drop into debugger on failure
3. Check logs in `tests/logs/` directory
