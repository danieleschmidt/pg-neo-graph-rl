#!/bin/bash
set -e

echo "Setting up pg-neo-graph-rl development environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    print_error "Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

print_status "Python version check passed: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install development dependencies
print_status "Installing development dependencies..."
pip install -e ".[dev,monitoring,benchmarks]"

# Setup pre-commit hooks
print_status "Setting up pre-commit hooks..."
pre-commit install

# Run initial checks
print_status "Running initial code quality checks..."
pre-commit run --all-files || print_warning "Some pre-commit checks failed. Please review and fix."

# Create necessary directories
print_status "Creating project directories..."
mkdir -p logs
mkdir -p data
mkdir -p results
mkdir -p monitoring/data

# Set up Git hooks (if in git repo)
if [ -d ".git" ]; then
    print_status "Setting up additional Git hooks..."
    
    # Pre-push hook to run tests
    cat > .git/hooks/pre-push << 'EOF'
#!/bin/bash
echo "Running tests before push..."
source venv/bin/activate
pytest tests/unit/ -x --tb=short
if [ $? -ne 0 ]; then
    echo "Tests failed. Push aborted."
    exit 1
fi
EOF
    chmod +x .git/hooks/pre-push
fi

# Check if Docker is available
if command -v docker &> /dev/null; then
    print_status "Docker detected. Building development image..."
    docker build -t pg-neo-graph-rl:dev --target development .
else
    print_warning "Docker not found. Skipping Docker setup."
fi

# Check JAX installation
print_status "Verifying JAX installation..."
python3 -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'JAX devices: {jax.devices()}')" || {
    print_error "JAX verification failed"
    exit 1
}

# Generate requirements.txt
print_status "Generating requirements.txt..."
pip freeze > requirements-dev.txt

print_status "Development environment setup complete!"
print_status "Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Run tests: make test"
echo "3. Start development: make install-dev"
echo "4. View documentation: make docs && make serve-docs"

print_status "Available make targets:"
make help