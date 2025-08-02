#!/usr/bin/env python3
"""Release automation script for PG-Neo-Graph-RL."""

import argparse
import subprocess
import sys
import os
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import tempfile
import shutil


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


def print_step(message: str):
    """Print a release step message."""
    print(f"{Colors.OKBLUE}[RELEASE]{Colors.ENDC} {message}")


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


def get_current_version() -> str:
    """Get current version from git tags."""
    try:
        result = run_command(["git", "describe", "--tags", "--abbrev=0"])
        return result.stdout.strip()
    except:
        return "0.0.0"


def parse_version(version: str) -> tuple:
    """Parse semantic version string."""
    # Remove 'v' prefix if present
    version = version.lstrip('v')
    
    # Match semantic version pattern
    match = re.match(r'^(\d+)\.(\d+)\.(\d+)(?:-([^+]+))?(?:\+(.+))?$', version)
    if not match:
        raise ValueError(f"Invalid version format: {version}")
    
    major, minor, patch = map(int, match.groups()[:3])
    prerelease = match.group(4)
    build = match.group(5)
    
    return major, minor, patch, prerelease, build


def increment_version(current_version: str, bump_type: str) -> str:
    """Increment version based on bump type."""
    major, minor, patch, prerelease, build = parse_version(current_version)
    
    if bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif bump_type == "minor":
        minor += 1
        patch = 0
    elif bump_type == "patch":
        patch += 1
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")
    
    return f"{major}.{minor}.{patch}"


def check_working_directory():
    """Check if working directory is clean."""
    print_step("Checking working directory status...")
    
    result = run_command(["git", "status", "--porcelain"])
    if result.stdout.strip():
        print_error("Working directory is not clean. Please commit or stash changes.")
        print(result.stdout)
        sys.exit(1)
    
    print_success("Working directory is clean")


def check_branch(allowed_branches: List[str] = None):
    """Check if on allowed release branch."""
    if allowed_branches is None:
        allowed_branches = ["main", "master", "release"]
    
    print_step("Checking current branch...")
    
    result = run_command(["git", "branch", "--show-current"])
    current_branch = result.stdout.strip()
    
    if current_branch not in allowed_branches:
        print_error(f"Not on allowed release branch. Current: {current_branch}, Allowed: {allowed_branches}")
        sys.exit(1)
    
    print_success(f"On allowed release branch: {current_branch}")


def run_pre_release_checks():
    """Run comprehensive pre-release checks."""
    print_step("Running pre-release checks...")
    
    # Run tests
    run_command(["python", "-m", "pytest", "tests/", "-x", "--tb=short"])
    
    # Run linting
    run_command(["pre-commit", "run", "--all-files"])
    
    # Run type checking
    run_command(["mypy", "pg_neo_graph_rl/"])
    
    # Run security checks
    run_command(["bandit", "-r", "pg_neo_graph_rl/"], check=False)
    
    print_success("Pre-release checks completed")


def update_changelog(version: str):
    """Update CHANGELOG.md with new version."""
    print_step("Updating CHANGELOG.md...")
    
    changelog_path = Path("CHANGELOG.md")
    if not changelog_path.exists():
        print_warning("CHANGELOG.md not found, creating basic version...")
        with open(changelog_path, "w") as f:
            f.write("# Changelog\n\n")
            f.write("All notable changes to this project will be documented in this file.\n\n")
            f.write("The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),\n")
            f.write("and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).\n\n")
    
    # Read current changelog
    with open(changelog_path, "r") as f:
        content = f.read()
    
    # Check if version already exists
    if f"## [{version}]" in content:
        print_warning(f"Version {version} already exists in CHANGELOG.md")
        return
    
    # Find the position to insert new version
    lines = content.split('\n')
    insert_pos = None
    
    for i, line in enumerate(lines):
        if line.startswith("## [") or line.startswith("## Unreleased"):
            insert_pos = i
            break
    
    if insert_pos is None:
        # Add after header
        for i, line in enumerate(lines):
            if line.startswith("# "):
                insert_pos = i + 1
                while insert_pos < len(lines) and not lines[insert_pos].strip():
                    insert_pos += 1
                break
    
    # Create new version entry
    today = datetime.now().strftime("%Y-%m-%d")
    new_entry = [
        f"## [{version}] - {today}",
        "",
        "### Added",
        "- New features and enhancements",
        "",
        "### Changed", 
        "- Modifications to existing functionality",
        "",
        "### Fixed",
        "- Bug fixes and corrections",
        "",
        "### Security",
        "- Security improvements and patches",
        ""
    ]
    
    # Insert new entry
    if insert_pos is not None:
        lines[insert_pos:insert_pos] = new_entry
    else:
        lines.extend(new_entry)
    
    # Write updated changelog
    with open(changelog_path, "w") as f:
        f.write('\n'.join(lines))
    
    print_success("CHANGELOG.md updated")


def create_git_tag(version: str):
    """Create and push git tag."""
    print_step(f"Creating git tag v{version}...")
    
    # Create annotated tag
    tag_message = f"Release version {version}"
    run_command(["git", "tag", "-a", f"v{version}", "-m", tag_message])
    
    print_success(f"Git tag v{version} created")


def build_and_upload_package(test_pypi: bool = False):
    """Build and upload package to PyPI."""
    print_step("Building package...")
    
    # Clean previous builds
    shutil.rmtree("dist", ignore_errors=True)
    shutil.rmtree("build", ignore_errors=True)
    
    # Build package
    run_command(["python", "-m", "build"])
    
    # Check package
    run_command(["twine", "check", "dist/*"])
    
    print_success("Package built successfully")
    
    # Upload to PyPI
    if test_pypi:
        print_step("Uploading to Test PyPI...")
        run_command(["twine", "upload", "--repository", "testpypi", "dist/*"])
        print_success("Package uploaded to Test PyPI")
    else:
        print_step("Uploading to PyPI...")
        run_command(["twine", "upload", "dist/*"])
        print_success("Package uploaded to PyPI")


def build_and_push_docker_images(version: str):
    """Build and push Docker images."""
    targets = ["production", "gpu"]
    
    for target in targets:
        print_step(f"Building Docker image: {target}")
        
        # Build image with version tag
        run_command([
            "docker", "build",
            "--target", target,
            "--tag", f"pg-neo-graph-rl:{target}-{version}",
            "--tag", f"pg-neo-graph-rl:{target}-latest",
            "."
        ])
        
        # Push images (commented out - requires Docker Hub setup)
        # run_command(["docker", "push", f"pg-neo-graph-rl:{target}-{version}"])
        # run_command(["docker", "push", f"pg-neo-graph-rl:{target}-latest"])
        
        print_success(f"Docker image {target} built")


def create_github_release(version: str, draft: bool = True):
    """Create GitHub release."""
    print_step("Creating GitHub release...")
    
    # Generate release notes from changelog
    changelog_path = Path("CHANGELOG.md")
    release_notes = ""
    
    if changelog_path.exists():
        with open(changelog_path, "r") as f:
            content = f.read()
            
        # Extract notes for this version
        lines = content.split('\n')
        capturing = False
        
        for line in lines:
            if f"## [{version}]" in line:
                capturing = True
                continue
            elif line.startswith("## [") and capturing:
                break
            elif capturing and line.strip():
                release_notes += line + '\n'
    
    if not release_notes:
        release_notes = f"Release version {version}"
    
    # Create release using GitHub CLI
    cmd = [
        "gh", "release", "create", f"v{version}",
        "--title", f"Release {version}",
        "--notes", release_notes
    ]
    
    if draft:
        cmd.append("--draft")
    
    # Add release assets
    dist_files = list(Path("dist").glob("*"))
    for file in dist_files:
        cmd.append(str(file))
    
    try:
        run_command(cmd)
        print_success("GitHub release created")
    except:
        print_warning("GitHub CLI not available or not authenticated")


def post_release_tasks(version: str):
    """Perform post-release tasks."""
    print_step("Performing post-release tasks...")
    
    # Push changes and tags
    run_command(["git", "push"])
    run_command(["git", "push", "--tags"])
    
    print_success("Post-release tasks completed")


def main():
    """Main release script."""
    parser = argparse.ArgumentParser(description="Release automation for PG-Neo-Graph-RL")
    
    parser.add_argument(
        "bump_type",
        choices=["major", "minor", "patch"],
        help="Version bump type"
    )
    
    parser.add_argument(
        "--version",
        help="Specific version to release (overrides bump_type)"
    )
    
    parser.add_argument(
        "--test-pypi",
        action="store_true",
        help="Upload to Test PyPI instead of PyPI"
    )
    
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip pre-release checks"
    )
    
    parser.add_argument(
        "--skip-docker",
        action="store_true",
        help="Skip Docker image build"
    )
    
    parser.add_argument(
        "--draft-release",
        action="store_true",
        help="Create draft GitHub release"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform dry run without making changes"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print_warning("DRY RUN MODE - No changes will be made")
    
    try:
        # Determine version
        if args.version:
            new_version = args.version
        else:
            current_version = get_current_version()
            new_version = increment_version(current_version, args.bump_type)
        
        print_step(f"Preparing release: {new_version}")
        
        if not args.dry_run:
            # Pre-release checks
            check_working_directory()
            check_branch()
            
            if not args.skip_checks:
                run_pre_release_checks()
            
            # Update changelog
            update_changelog(new_version)
            
            # Commit changelog changes
            run_command(["git", "add", "CHANGELOG.md"])
            run_command(["git", "commit", "-m", f"chore: update changelog for v{new_version}"])
            
            # Create git tag
            create_git_tag(new_version)
            
            # Build and upload package
            build_and_upload_package(args.test_pypi)
            
            # Build Docker images
            if not args.skip_docker:
                build_and_push_docker_images(new_version)
            
            # Create GitHub release
            create_github_release(new_version, args.draft_release)
            
            # Post-release tasks
            post_release_tasks(new_version)
        
        print_success(f"Release {new_version} completed successfully!")
        
        if args.test_pypi:
            print_step("Test the release with:")
            print(f"pip install --index-url https://test.pypi.org/simple/ pg-neo-graph-rl=={new_version}")
        
    except KeyboardInterrupt:
        print_error("Release interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Release failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()