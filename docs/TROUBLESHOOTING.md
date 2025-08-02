# Troubleshooting Guide

This guide helps resolve common issues with the PG-Neo-Graph-RL development and deployment environment.

## Development Environment Issues

### Python Environment Problems

#### Issue: Import errors after installation
```bash
ModuleNotFoundError: No module named 'pg_neo_graph_rl'
```

**Solutions:**
1. **Editable Installation:**
   ```bash
   pip install -e .
   ```

2. **Check PYTHONPATH:**
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

3. **Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   pip install -e ".[dev]"
   ```

#### Issue: JAX installation problems
```bash
No module named 'jax' or CUDA version mismatch
```

**Solutions:**
1. **CPU-only installation:**
   ```bash
   pip install -e ".[cpu]"
   ```

2. **GPU installation:**
   ```bash
   pip install -e ".[gpu]"
   # Verify CUDA version compatibility
   nvidia-smi
   python -c "import jax; print(jax.devices())"
   ```

3. **Version conflicts:**
   ```bash
   pip uninstall jax jaxlib
   pip install -e ".[cpu]" --force-reinstall
   ```

### Pre-commit Hook Issues

#### Issue: Pre-commit hooks failing
```bash
An error has occurred: FatalError: git failed
```

**Solutions:**
1. **Reinstall hooks:**
   ```bash
   pre-commit uninstall
   pre-commit clean
   pre-commit install
   ```

2. **Update hooks:**
   ```bash
   pre-commit autoupdate
   pre-commit run --all-files
   ```

3. **Skip hooks temporarily:**
   ```bash
   git commit --no-verify -m "commit message"
   ```

#### Issue: Black formatting conflicts
```bash
would reformat <file>
```

**Solutions:**
1. **Auto-fix formatting:**
   ```bash
   black pg_neo_graph_rl/ tests/ scripts/
   isort pg_neo_graph_rl/ tests/ scripts/
   ```

2. **Line length conflicts:**
   ```bash
   # Check .editorconfig and pyproject.toml settings
   black --line-length 88 pg_neo_graph_rl/
   ```

## Testing Issues

### pytest Problems

#### Issue: Tests not discovered
```bash
no tests ran in 0.00s
```

**Solutions:**
1. **Check test discovery:**
   ```bash
   pytest --collect-only
   ```

2. **Explicit test paths:**
   ```bash
   pytest tests/ -v
   ```

3. **Python path issues:**
   ```bash
   PYTHONPATH=. pytest tests/
   ```

#### Issue: JAX tests failing
```bash
XLA compilation failed
```

**Solutions:**
1. **Force CPU mode:**
   ```bash
   JAX_PLATFORM_NAME=cpu pytest tests/
   ```

2. **Disable JIT compilation:**
   ```bash
   JAX_DISABLE_JIT=1 pytest tests/
   ```

3. **Memory issues:**
   ```bash
   XLA_PYTHON_CLIENT_PREALLOCATE=false pytest tests/
   ```

### Coverage Issues

#### Issue: Coverage not working
```bash
Coverage.py warning: No data to report
```

**Solutions:**
1. **Reinstall coverage:**
   ```bash
   pip install coverage[toml]
   pytest --cov=pg_neo_graph_rl --cov-report=html
   ```

2. **Coverage configuration:**
   ```bash
   # Check pyproject.toml [tool.coverage] section
   coverage run -m pytest tests/
   coverage report
   ```

## Docker Issues

### Build Problems

#### Issue: Docker build failing
```bash
ERROR: failed to solve: process "/bin/sh -c pip install -e ." did not complete successfully
```

**Solutions:**
1. **Check Dockerfile syntax:**
   ```bash
   docker build --no-cache --target development .
   ```

2. **Build with verbose output:**
   ```bash
   docker build --progress=plain --no-cache .
   ```

3. **Platform-specific builds:**
   ```bash
   docker build --platform linux/amd64 .
   ```

#### Issue: Permission denied in container
```bash
Permission denied: '/app'
```

**Solutions:**
1. **Check user permissions:**
   ```bash
   docker run --rm -it pg-neo-graph-rl:dev whoami
   docker run --rm -it pg-neo-graph-rl:dev ls -la /app
   ```

2. **Fix ownership:**
   ```bash
   # In Dockerfile
   COPY --chown=app:app . .
   ```

### Docker Compose Issues

#### Issue: Services not starting
```bash
exited with code 1
```

**Solutions:**
1. **Check service logs:**
   ```bash
   docker-compose logs service-name
   docker-compose up --no-deps service-name
   ```

2. **Rebuild services:**
   ```bash
   docker-compose down
   docker-compose build --no-cache
   docker-compose up
   ```

3. **Port conflicts:**
   ```bash
   # Check port usage
   netstat -tulpn | grep :8080
   # or
   docker-compose ps
   ```

#### Issue: Volume mount problems
```bash
volume mount failed
```

**Solutions:**
1. **Check volume paths:**
   ```bash
   docker-compose config
   ```

2. **Reset volumes:**
   ```bash
   docker-compose down -v
   docker volume prune
   docker-compose up
   ```

## CI/CD Issues

### GitHub Actions Problems

#### Issue: Workflow not triggering
```yaml
Workflow is not running
```

**Solutions:**
1. **Check workflow syntax:**
   ```bash
   # Use GitHub CLI or online YAML validator
   yamllint .github/workflows/ci.yml
   ```

2. **Verify triggers:**
   ```yaml
   on:
     push:
       branches: [main, develop]  # Check branch name
   ```

3. **File permissions:**
   ```bash
   # Ensure workflow files are committed
   git add .github/workflows/
   git commit -m "Add workflows"
   ```

#### Issue: Secret access denied
```bash
Error: Input required and not supplied: password
```

**Solutions:**
1. **Check secret names:**
   ```yaml
   # Secrets are case-sensitive
   password: ${{ secrets.DOCKER_PASSWORD }}
   ```

2. **Verify secret scope:**
   - Repository secrets vs. environment secrets
   - Check secret permissions in organization settings

3. **Token permissions:**
   ```yaml
   permissions:
     contents: read
     packages: write
   ```

### Build Failures in CI

#### Issue: Tests failing in CI but passing locally
```bash
Test failed in CI environment
```

**Solutions:**
1. **Environment differences:**
   ```yaml
   env:
     PYTHONPATH: ${{ github.workspace }}
     JAX_PLATFORM_NAME: cpu
   ```

2. **Dependency caching issues:**
   ```yaml
   - name: Clear pip cache
     run: pip cache purge
   ```

3. **Resource constraints:**
   ```yaml
   # Add timeout and resource limits
   timeout-minutes: 30
   ```

## Performance Issues

### Slow Training/Testing

#### Issue: JAX compilation taking too long
```python
# Long compilation times
```

**Solutions:**
1. **Disable JIT for debugging:**
   ```python
   import jax
   jax.config.update('jax_disable_jit', True)
   ```

2. **Pre-compile functions:**
   ```python
   @jax.jit
   def compiled_function(x):
       return x * 2
   
   # Warm up compilation
   compiled_function(jax.numpy.array(1.0))
   ```

3. **Memory optimization:**
   ```python
   jax.config.update('jax_enable_x64', False)  # Use 32-bit
   ```

### Memory Issues

#### Issue: Out of memory errors
```bash
CUDA out of memory / OOM
```

**Solutions:**
1. **Reduce batch size:**
   ```python
   # In configuration
   batch_size = 32  # Instead of 256
   ```

2. **Memory management:**
   ```python
   import jax
   jax.config.update('jax_memory_preallocation', False)
   ```

3. **Clear caches:**
   ```python
   jax.clear_caches()
   ```

## Monitoring Issues

### Grafana Problems

#### Issue: Grafana not accessible
```bash
Connection refused to localhost:3000
```

**Solutions:**
1. **Check service status:**
   ```bash
   docker-compose ps grafana
   docker-compose logs grafana
   ```

2. **Port conflicts:**
   ```bash
   netstat -tulpn | grep :3000
   # Change port in docker-compose.yml if needed
   ```

3. **Configuration issues:**
   ```bash
   # Check Grafana configuration
   docker-compose exec grafana cat /etc/grafana/grafana.ini
   ```

### Prometheus Issues

#### Issue: Metrics not collected
```bash
No data points
```

**Solutions:**
1. **Check Prometheus targets:**
   - Visit `http://localhost:9090/targets`
   - Verify all targets are "UP"

2. **Configuration validation:**
   ```bash
   # Validate prometheus.yml
   docker-compose exec prometheus promtool check config /etc/prometheus/prometheus.yml
   ```

3. **Network connectivity:**
   ```bash
   docker-compose exec prometheus wget -qO- http://app:8080/metrics
   ```

## Security Issues

### Secret Management

#### Issue: Secrets exposed in logs
```bash
Secret values visible in output
```

**Solutions:**
1. **Mask secrets:**
   ```bash
   echo "::add-mask::$SECRET_VALUE"
   ```

2. **Use proper secret handling:**
   ```yaml
   env:
     SECRET: ${{ secrets.SECRET_NAME }}
   ```

3. **Audit logs:**
   ```bash
   # Check for exposed secrets
   git log --all --full-history -- "**/*secret*"
   ```

### Dependency Vulnerabilities

#### Issue: Security vulnerabilities detected
```bash
Safety check failed
```

**Solutions:**
1. **Update dependencies:**
   ```bash
   pip install --upgrade package-name
   safety check
   ```

2. **Pin secure versions:**
   ```toml
   # In pyproject.toml
   dependencies = [
       "package-name>=1.2.3",  # Known secure version
   ]
   ```

3. **Ignore false positives:**
   ```bash
   # Only if confirmed safe
   safety check --ignore 12345
   ```

## Getting Help

### Debug Information Collection

When reporting issues, include:

```bash
# System information
python --version
pip --version
docker --version
docker-compose --version

# Package versions
pip list | grep -E "(jax|flax|optax)"

# Environment variables
echo $PYTHONPATH
echo $JAX_PLATFORM_NAME

# Git status
git status
git log --oneline -5
```

### Log Collection

```bash
# Collect relevant logs
docker-compose logs > docker-logs.txt
pytest --tb=long > test-output.txt 2>&1
pip check > dependency-check.txt 2>&1
```

### Support Channels

1. **GitHub Issues:** For bugs and feature requests
2. **GitHub Discussions:** For questions and community help
3. **Documentation:** Check docs/ directory first
4. **Stack Overflow:** Tag with `pg-neo-graph-rl`

### Creating Bug Reports

Include:
1. **Environment details** (OS, Python version, package versions)
2. **Steps to reproduce** the issue
3. **Expected vs. actual behavior**
4. **Error messages and stack traces**
5. **Configuration files** (sanitized)
6. **Minimal reproducible example**

---

**Last Updated:** January 2025  
**Maintainer:** Development Team  
**Contributions:** Community contributions welcome via PR
