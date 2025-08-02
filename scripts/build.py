#!/usr/bin/env python3
"""Build automation script for PG-Neo-Graph-RL."""

import argparse
import subprocess
import sys
import os
import shutil
import json
from pathlib import Path
from typing import List, Dict, Any
import time


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_step(message: str):
    """Print a build step message."""
    print(f"{Colors.OKBLUE}[BUILD]{Colors.ENDC} {message}")


def print_success(message: str):
    """Print a success message."""
    print(f"{Colors.OKGREEN}[SUCCESS]{Colors.ENDC} {message}")


def print_warning(message: str):
    """Print a warning message."""
    print(f"{Colors.WARNING}[WARNING]{Colors.ENDC} {message}")


def print_error(message: str):
    """Print an error message."""
    print(f"{Colors.FAIL}[ERROR]{Colors.ENDC} {message}")


def run_command(cmd: List[str], cwd: str = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print_step(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd, 
            check=check, 
            capture_output=True, 
            text=True
        )
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {' '.join(cmd)}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        if check:
            sys.exit(1)
        return e


def clean_build_artifacts():
    """Clean build artifacts and cache directories."""
    print_step("Cleaning build artifacts...")
    
    directories_to_clean = [
        "build",
        "dist",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "htmlcov",
        "__pycache__"
    ]
    
    files_to_clean = [
        ".coverage",
        "coverage.xml",
        "test-results.xml",
    ]
    
    # Remove directories
    for dir_name in directories_to_clean:
        for path in Path(".").rglob(dir_name):
            if path.is_dir():
                print(f"Removing {path}")
                shutil.rmtree(path, ignore_errors=True)
    
    # Remove files
    for file_name in files_to_clean:
        for path in Path(".").rglob(file_name):
            if path.is_file():
                print(f"Removing {path}")
                path.unlink()
    
    # Remove egg-info directories
    for path in Path(".").glob("*.egg-info"):
        if path.is_dir():
            print(f"Removing {path}")
            shutil.rmtree(path, ignore_errors=True)
    
    print_success("Cleanup completed")


def run_linting():
    """Run code linting and formatting checks."""
    print_step("Running linting and formatting checks...")
    
    # Run pre-commit hooks
    run_command(["pre-commit", "run", "--all-files"])
    
    print_success("Linting completed")


def run_type_checking():
    """Run static type checking."""
    print_step("Running type checking...")
    
    run_command(["mypy", "pg_neo_graph_rl/"])
    
    print_success("Type checking completed")


def run_security_checks():
    """Run security checks."""
    print_step("Running security checks...")
    
    # Run bandit for security issues
    run_command([
        "bandit", "-r", "pg_neo_graph_rl/", 
        "-f", "json", "-o", "bandit-report.json"
    ], check=False)
    
    # Run safety for known vulnerabilities
    run_command([
        "safety", "check", "--json", "--output", "safety-report.json"
    ], check=False)
    
    print_success("Security checks completed")


def run_tests(test_type: str = "all"):
    """Run tests based on type."""
    print_step(f"Running {test_type} tests...")
    
    cmd = ["python", "-m", "pytest"]
    
    if test_type == "unit":
        cmd.extend(["tests/unit/"])
    elif test_type == "integration":
        cmd.extend(["tests/integration/"])
    elif test_type == "fast":
        cmd.extend(["-m", "not slow"])
    elif test_type == "benchmark":
        cmd.extend(["tests/benchmarks/", "--benchmark-only"])
    
    # Add coverage and reporting
    if test_type != "benchmark":
        cmd.extend([
            "--cov=pg_neo_graph_rl",
            "--cov-report=html",
            "--cov-report=xml",
            "--cov-report=term",
            "--junit-xml=test-results.xml"
        ])
    
    run_command(cmd)
    
    print_success(f"{test_type.capitalize()} tests completed")


def build_package():
    """Build the Python package."""
    print_step("Building Python package...")
    
    # Ensure build tool is available
    run_command(["python", "-m", "pip", "install", "build", "twine"])
    
    # Build package
    run_command(["python", "-m", "build"])
    
    # Check package
    run_command(["twine", "check", "dist/*"])
    
    print_success("Package build completed")


def build_documentation():
    """Build documentation."""
    print_step("Building documentation...")
    
    # Check if sphinx is available
    try:
        run_command(["sphinx-build", "--version"])
    except:
        print_warning("Sphinx not available, installing...")
        run_command(["pip", "install", "sphinx", "sphinx-rtd-theme"])
    
    # Build docs
    docs_dir = Path("docs")
    if docs_dir.exists():
        run_command([
            "sphinx-build", "-W", "-b", "html", 
            "docs/", "docs/_build/html"
        ])
        print_success("Documentation build completed")
    else:
        print_warning("Documentation directory not found, skipping...")


def build_docker_images(targets: List[str] = None):
    """Build Docker images."""
    if targets is None:
        targets = ["development", "production", "gpu"]
    
    for target in targets:
        print_step(f"Building Docker image: {target}")
        
        tag = f"pg-neo-graph-rl:{target}"
        cmd = [
            "docker", "build",
            "--target", target,
            "--tag", tag,
            "."
        ]
        
        run_command(cmd)
        
        print_success(f"Docker image {tag} built successfully")


def generate_sbom():
    """Generate Software Bill of Materials."""
    print_step("Generating SBOM...")
    
    try:
        # Try using cyclonedx-bom if available
        run_command([
            "cyclonedx-bom", "-o", "sbom.json", "-f", "json"
        ], check=False)
    except:
        # Fallback to pip freeze
        result = run_command(["pip", "freeze"], check=False)
        
        sbom_data = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.4",
            "metadata": {
                "component": {
                    "type": "application",
                    "name": "pg-neo-graph-rl",
                    "version": "unknown"
                }
            },
            "components": []
        }
        
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                if '==' in line:
                    name, version = line.split('==')
                    sbom_data["components"].append({
                        "type": "library",
                        "name": name,
                        "version": version
                    })
        
        with open("sbom.json", "w") as f:
            json.dump(sbom_data, f, indent=2)
    
    print_success("SBOM generated")


def create_release_artifacts():
    """Create release artifacts."""
    print_step("Creating release artifacts...")
    
    # Create release directory
    release_dir = Path("release")
    release_dir.mkdir(exist_ok=True)
    
    # Copy built packages
    dist_dir = Path("dist")
    if dist_dir.exists():
        for file in dist_dir.glob("*"):
            shutil.copy2(file, release_dir / file.name)
    
    # Copy documentation
    docs_build = Path("docs/_build/html")
    if docs_build.exists():
        shutil.copytree(
            docs_build, 
            release_dir / "docs", 
            dirs_exist_ok=True
        )
    
    # Copy SBOM
    sbom_file = Path("sbom.json")
    if sbom_file.exists():
        shutil.copy2(sbom_file, release_dir / "sbom.json")
    
    # Copy security reports
    for report in ["bandit-report.json", "safety-report.json"]:
        report_file = Path(report)
        if report_file.exists():
            shutil.copy2(report_file, release_dir / report)
    
    # Create checksums
    import hashlib
    checksums = {}
    for file in release_dir.rglob("*"):
        if file.is_file() and file.name not in ["checksums.json"]:
            with open(file, "rb") as f:
                content = f.read()
                checksums[str(file.relative_to(release_dir))] = {
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "size": len(content)
                }
    
    with open(release_dir / "checksums.json", "w") as f:
        json.dump(checksums, f, indent=2)
    
    print_success("Release artifacts created")


def main():
    """Main build script."""
    parser = argparse.ArgumentParser(description="Build automation for PG-Neo-Graph-RL")
    
    parser.add_argument(
        "command",
        choices=[
            "clean", "lint", "type-check", "security", "test", "build", 
            "docs", "docker", "sbom", "release", "all"
        ],
        help="Build command to run"
    )
    
    parser.add_argument(
        "--test-type",
        choices=["all", "unit", "integration", "fast", "benchmark"],
        default="all",
        help="Type of tests to run"
    )
    
    parser.add_argument(
        "--docker-targets",
        nargs="*",
        default=["development", "production"],
        help="Docker targets to build"
    )
    
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running tests in 'all' command"
    )
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    try:
        if args.command == "clean":
            clean_build_artifacts()
        
        elif args.command == "lint":
            run_linting()
        
        elif args.command == "type-check":
            run_type_checking()
        
        elif args.command == "security":
            run_security_checks()
        
        elif args.command == "test":
            run_tests(args.test_type)
        
        elif args.command == "build":
            build_package()
        
        elif args.command == "docs":
            build_documentation()
        
        elif args.command == "docker":
            build_docker_images(args.docker_targets)
        
        elif args.command == "sbom":
            generate_sbom()
        
        elif args.command == "release":
            create_release_artifacts()
        
        elif args.command == "all":
            clean_build_artifacts()
            run_linting()
            run_type_checking()
            run_security_checks()
            
            if not args.skip_tests:
                run_tests("fast")  # Run fast tests for full pipeline
            
            build_package()
            build_documentation()
            generate_sbom()
            create_release_artifacts()
        
        elapsed_time = time.time() - start_time
        print_success(f"Build completed in {elapsed_time:.2f} seconds")
    
    except KeyboardInterrupt:
        print_error("Build interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Build failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()