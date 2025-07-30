# GitHub Workflows Documentation

This directory contains documentation for required GitHub Actions workflows. Since Terry cannot create GitHub workflow files directly, this documentation provides templates and requirements for manual setup.

## Required Workflows

### 1. CI/CD Pipeline (`.github/workflows/ci.yml`)

```yaml
name: CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, "3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,test]"
    
    - name: Run pre-commit
      run: pre-commit run --all-files
    
    - name: Run tests
      run: pytest --cov=pg_neo_graph_rl --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      if: matrix.python-version == '3.11'
```

### 2. Security Scanning (`.github/workflows/security.yml`)

```yaml
name: Security

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * 0'  # Weekly

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Run Bandit security scan
      uses: PyCQA/bandit-action@v1
      with:
        path: "src/"
        level: high
        confidence: high
    
    - name: Run Safety check
      run: |
        pip install safety
        safety check
    
    - name: Run Semgrep
      uses: returntocorp/semgrep-action@v1
      with:
        config: auto
```

### 3. Documentation Build (`.github/workflows/docs.yml`)

```yaml
name: Documentation

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  docs:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"
    
    - name: Install dependencies
      run: |
        pip install -e ".[docs]"
    
    - name: Build documentation
      run: |
        cd docs
        make html
    
    - name: Deploy to GitHub Pages
      if: github.ref == 'refs/heads/main'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs/_build/html
```

### 4. Release Automation (`.github/workflows/release.yml`)

```yaml
name: Release

on:
  release:
    types: [published]

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"
    
    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    
    - name: Build package
      run: python -m build
    
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

## Setup Instructions

1. **Create workflow files**: Copy the YAML content above into corresponding files in `.github/workflows/`

2. **Configure secrets**: Add these secrets to your repository:
   - `PYPI_API_TOKEN`: For PyPI publishing
   - `CODECOV_TOKEN`: For coverage reporting (optional)

3. **Enable GitHub Pages**: 
   - Go to repository Settings > Pages
   - Select "GitHub Actions" as source

4. **Branch protection**: Configure branch protection rules for `main`:
   - Require pull request reviews
   - Require status checks (CI tests)
   - Require up-to-date branches

## Workflow Features

- **Multi-Python testing**: Tests across Python 3.9-3.12
- **Security scanning**: Automated vulnerability detection
- **Code quality**: Pre-commit hooks and linting
- **Documentation**: Auto-build and deploy docs
- **Release automation**: Automated PyPI publishing
- **Coverage reporting**: Integration with Codecov

## Customization

Modify workflows based on your needs:
- Add GPU testing for JAX workflows
- Include integration tests with external services
- Add performance benchmarking
- Configure notifications (Slack, email)
- Add deployment to cloud platforms