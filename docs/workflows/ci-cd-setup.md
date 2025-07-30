# CI/CD Setup Guide

This document provides guidance for setting up Continuous Integration and Continuous Deployment (CI/CD) workflows for pg-neo-graph-rl.

## Overview

The recommended CI/CD setup includes:
- **Testing**: Automated test execution across multiple Python versions
- **Quality Assurance**: Code formatting, linting, and type checking
- **Security Scanning**: Dependency vulnerability scanning and static analysis
- **Documentation**: Automated documentation building and deployment
- **Packaging**: Automated package building and publishing

## GitHub Actions Workflows

### 1. Main CI Workflow

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11"]
        jax-version: ["cpu", "gpu"]
        exclude:
          # Exclude GPU tests on some Python versions to reduce CI time
          - python-version: "3.10"
            jax-version: "gpu"
          - python-version: "3.11"
            jax-version: "gpu"

    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/pyproject.toml') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,${{ matrix.jax-version }}]"
    
    - name: Run pre-commit hooks
      run: |
        pre-commit install
        pre-commit run --all-files
    
    - name: Run tests
      run: |
        pytest --cov=pg_neo_graph_rl --cov-report=xml --cov-report=html
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
        fail_ci_if_error: true

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python 3.9
      uses: actions/setup-python@v4
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install bandit safety
    
    - name: Run Bandit security check
      run: bandit -r pg_neo_graph_rl/ -f json -o bandit-report.json
    
    - name: Run Safety check
      run: safety check --json --output safety-report.json
    
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          bandit-report.json
          safety-report.json

  docs:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python 3.9
      uses: actions/setup-python@v4
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
        pip install sphinx sphinx-rtd-theme
    
    - name: Build documentation
      run: |
        cd docs/
        make html
    
    - name: Upload documentation
      uses: actions/upload-artifact@v3
      with:
        name: documentation
        path: docs/_build/html/
```

### 2. Release Workflow

Create `.github/workflows/release.yml`:

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0  # Full history for versioning
    
    - name: Set up Python 3.9
      uses: actions/setup-python@v4
      with:
        python-version: 3.9
    
    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    
    - name: Build package
      run: python -m build
    
    - name: Check package
      run: twine check dist/*
    
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
      run: twine upload dist/*
    
    - name: Create GitHub Release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: ${{ github.ref }}
        release_name: Release ${{ github.ref }}
        body: |
          See [CHANGELOG.md](CHANGELOG.md) for details.
        draft: false
        prerelease: false
```

### 3. Documentation Deployment

Create `.github/workflows/docs.yml`:

```yaml
name: Documentation

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  docs:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python 3.9
      uses: actions/setup-python@v4
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
        pip install sphinx sphinx-rtd-theme
    
    - name: Build documentation
      run: |
        cd docs/
        make html
    
    - name: Deploy to GitHub Pages
      if: github.ref == 'refs/heads/main'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs/_build/html
```

## Required Secrets

Configure these secrets in your GitHub repository:

1. **PYPI_TOKEN**: PyPI API token for package publishing
2. **CODECOV_TOKEN**: Codecov token for coverage reporting (optional)

## Branch Protection Rules

Configure branch protection for `main` branch:

1. **Require pull request reviews**: 1 reviewer minimum
2. **Require status checks**: All CI jobs must pass
3. **Require branches to be up to date**: Ensure latest changes
4. **Include administrators**: Apply rules to admins too

## Alternative CI/CD Platforms

### GitLab CI

Create `.gitlab-ci.yml`:

```yaml
stages:
  - test
  - security
  - build
  - deploy

variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"

cache:
  paths:
    - .cache/pip
    - venv/

test:
  stage: test
  image: python:3.9
  parallel:
    matrix:
      - PYTHON_VERSION: ["3.9", "3.10", "3.11"]
        JAX_VERSION: ["cpu", "gpu"]
  script:
    - pip install -e ".[dev,$JAX_VERSION]"
    - pytest --cov=pg_neo_graph_rl --cov-report=xml
  coverage: '/TOTAL.*\s+(\d+%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

security:
  stage: security
  image: python:3.9
  script:
    - pip install bandit safety
    - bandit -r pg_neo_graph_rl/
    - safety check
  allow_failure: true

build:
  stage: build
  image: python:3.9
  script:
    - pip install build
    - python -m build
  artifacts:
    paths:
      - dist/

deploy:
  stage: deploy
  image: python:3.9
  script:
    - pip install twine
    - twine upload dist/*
  only:
    - tags
  when: manual
```

### Jenkins Pipeline

Create `Jenkinsfile`:

```groovy
pipeline {
    agent any
    
    environment {
        PYTHON_VERSION = '3.9'
    }
    
    stages {
        stage('Setup') {
            steps {
                sh '''
                    python -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -e ".[dev]"
                '''
            }
        }
        
        stage('Lint') {
            steps {
                sh '''
                    . venv/bin/activate
                    pre-commit run --all-files
                '''
            }
        }
        
        stage('Test') {
            parallel {
                stage('Unit Tests') {
                    steps {
                        sh '''
                            . venv/bin/activate
                            pytest tests/unit/ --cov=pg_neo_graph_rl
                        '''
                    }
                }
                stage('Integration Tests') {
                    steps {
                        sh '''
                            . venv/bin/activate
                            pytest tests/integration/
                        '''
                    }
                }
            }
        }
        
        stage('Security') {
            steps {
                sh '''
                    . venv/bin/activate
                    bandit -r pg_neo_graph_rl/
                    safety check
                '''
            }
        }
        
        stage('Build') {
            when {
                anyOf {
                    branch 'main'
                    tag pattern: 'v\\d+\\.\\d+\\.\\d+', comparator: 'REGEXP'
                }
            }
            steps {
                sh '''
                    . venv/bin/activate
                    pip install build
                    python -m build
                '''
                archiveArtifacts artifacts: 'dist/*', fingerprint: true
            }
        }
        
        stage('Deploy') {
            when {
                tag pattern: 'v\\d+\\.\\d+\\.\\d+', comparator: 'REGEXP'
            }
            steps {
                sh '''
                    . venv/bin/activate
                    pip install twine
                    twine upload dist/*
                '''
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
        success {
            emailext (
                subject: "BUILD SUCCESS: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                body: "Build succeeded. Check console output at ${env.BUILD_URL}",
                to: "${env.CHANGE_AUTHOR_EMAIL}"
            )
        }
        failure {
            emailext (
                subject: "BUILD FAILURE: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                body: "Build failed. Check console output at ${env.BUILD_URL}",
                to: "${env.CHANGE_AUTHOR_EMAIL}"
            )
        }
    }
}
```

## Local Development Workflow

### Pre-commit Setup

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

### Testing Workflow

```bash
# Run specific test types
pytest -m unit                    # Unit tests only
pytest -m integration            # Integration tests only
pytest -m "not slow"             # Skip slow tests

# Run with coverage
pytest --cov=pg_neo_graph_rl --cov-report=html

# Run parallel tests
pytest -n auto
```

### Release Workflow

```bash
# Update version and create tag
git tag v0.1.0
git push origin v0.1.0

# Manual build and check
python -m build
twine check dist/*

# Upload to test PyPI first
twine upload --repository testpypi dist/*

# Upload to production PyPI
twine upload dist/*
```

## Monitoring and Metrics

### Coverage Tracking
- Use Codecov or Coveralls for coverage tracking
- Set coverage thresholds in `pyproject.toml`
- Monitor coverage trends over time

### Performance Monitoring
- Run benchmark tests in CI
- Track performance regression
- Monitor resource usage

### Security Monitoring
- Regular dependency updates with Dependabot
- Automated security scanning
- SBOM generation for compliance

## Best Practices

1. **Fast Feedback**: Keep CI jobs under 10 minutes
2. **Parallel Execution**: Run independent jobs in parallel
3. **Caching**: Cache dependencies and build artifacts
4. **Matrix Testing**: Test across multiple Python/JAX versions
5. **Artifact Management**: Store build artifacts and test reports
6. **Notifications**: Configure appropriate notifications
7. **Security**: Never store secrets in code, use secret management
8. **Documentation**: Keep CI configuration documented and updated

For specific implementation details, refer to the template files in this directory.