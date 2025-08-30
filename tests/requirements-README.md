# Testing Dependencies Guide

This directory contains different requirement files for various testing scenarios.

## Files Overview

### `requirements-minimal.txt`
**Purpose**: Essential dependencies for basic testing
**Usage**: Minimal test jobs in CI/CD, local development
**Contains**:
- Core pytest framework
- Google Gemini embeddings
- Qdrant client
- Basic utilities

### `requirements-test.txt`
**Purpose**: Comprehensive testing dependencies
**Usage**: Full integration and end-to-end testing
**Contains**:
- All minimal dependencies
- Advanced testing frameworks (coverage, mocking, async)
- Performance testing tools
- HTTP mocking utilities
- Data generation tools

## Installation

### For Local Development
```bash
# Basic testing
pip install -r tests/requirements-minimal.txt

# Full testing suite
pip install -r tests/requirements-test.txt
```

### For CI/CD
- **Minimal tests**: Use `requirements-minimal.txt` for fast feedback
- **Integration tests**: Use `requirements-test.txt` for comprehensive testing
- **End-to-end tests**: Use `requirements-test.txt` + API secrets

## Dependency Categories

### Core Testing Framework
- `pytest` - Main testing framework
- `pytest-cov` - Coverage reporting
- `pytest-mock` - Mocking support

### Pipeline Testing
- `langchain-google-genai` - Google Gemini embeddings
- `qdrant-client` - Vector database client
- `requests` - HTTP client for API testing

### Advanced Testing (Full only)
- `hypothesis` - Property-based testing
- `faker` - Test data generation
- `locust` - Load testing
- `responses` - HTTP response mocking

### Optional Dependencies
Some dependencies in `requirements-test.txt` are marked as optional:
- Performance testing tools
- Database testing utilities
- Documentation testing

## Usage in GitHub Actions

The workflow uses different requirement files based on test level:

```yaml
# Minimal tests
pip install -r requirements.txt
pip install -r tests/requirements-minimal.txt

# Integration/E2E tests  
pip install -r requirements.txt
pip install -r tests/requirements-test.txt
```

## Adding New Dependencies

When adding new test dependencies:

1. **Core dependencies** → Add to `requirements-minimal.txt`
2. **Advanced/optional** → Add to `requirements-test.txt`
3. **Update this documentation**
4. **Test locally** before committing

## Troubleshooting

### Common Issues

1. **Import errors**: Check if dependency is in correct requirements file
2. **Version conflicts**: Update version constraints
3. **Missing optional deps**: Install full test requirements

### Debugging Dependencies
```bash
# Check what's installed
pip list

# Install specific requirements
pip install -r tests/requirements-test.txt --dry-run

# Check for conflicts
pip check
```
