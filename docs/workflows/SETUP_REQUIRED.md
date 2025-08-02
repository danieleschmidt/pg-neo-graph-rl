# Manual Setup Requirements

This document outlines the manual setup steps required after implementing the SDLC, due to GitHub App permission limitations.

## GitHub Actions Workflows

**⚠️ Manual Action Required:** The CI/CD workflows could not be created automatically due to insufficient GitHub App permissions.

### Required Workflows

Copy the following workflow files from `docs/workflows/examples/` to `.github/workflows/`:

1. **Main CI Workflow** (`ci.yml`)
   - Automated testing across Python versions
   - Code quality checks
   - Security scanning
   - Documentation building
   - Docker image builds

2. **Release/CD Workflow** (`cd.yml`)
   - Automated releases on tag push
   - PyPI package publishing
   - Docker image publishing
   - GitHub release creation
   - Documentation deployment

3. **Security Scanning** (`security-scan.yml`)
   - SAST (Static Application Security Testing)
   - Dependency vulnerability scanning
   - Container security scanning
   - Secrets detection
   - License compliance checking

4. **Dependency Updates** (`dependency-update.yml`)
   - Automated dependency updates
   - Security vulnerability patching
   - Docker base image updates
   - GitHub Actions version updates

### Setup Commands

```bash
# Create workflows directory
mkdir -p .github/workflows

# Copy workflow templates
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/cd.yml .github/workflows/
cp docs/workflows/examples/security-scan.yml .github/workflows/
cp docs/workflows/examples/dependency-update.yml .github/workflows/

# Commit the workflows
git add .github/workflows/
git commit -m "feat: add GitHub Actions workflows"
git push
```

## Repository Secrets Configuration

### Required Secrets

Configure these secrets in **Settings → Secrets and variables → Actions**:

#### Essential Secrets
- `PYPI_API_TOKEN` - PyPI API token for package publishing
- `DOCKER_USERNAME` - Docker Hub username
- `DOCKER_PASSWORD` - Docker Hub password or access token

#### Optional Secrets
- `CODECOV_TOKEN` - Codecov token for coverage reporting
- `DEPENDENCY_UPDATE_TOKEN` - GitHub token with write permissions for dependency PRs
- `TEST_PYPI_API_TOKEN` - Test PyPI token for pre-release testing
- `GITLEAKS_LICENSE` - Gitleaks Pro license (if available)

### Secret Setup Commands

```bash
# Using GitHub CLI (recommended)
gh secret set PYPI_API_TOKEN
gh secret set DOCKER_USERNAME
gh secret set DOCKER_PASSWORD
gh secret set CODECOV_TOKEN

# Or via GitHub web interface:
# https://github.com/YOUR_USERNAME/YOUR_REPO/settings/secrets/actions
```

## Branch Protection Rules

**⚠️ Manual Action Required:** Configure branch protection for the `main` branch.

### Required Settings

1. Go to **Settings → Branches**
2. Add rule for `main` branch with:
   - ☑️ Require a pull request before merging (1 reviewer)
   - ☑️ Require status checks to pass before merging
   - ☑️ Require branches to be up to date before merging
   - ☑️ Include administrators
   - ☑️ Allow force pushes (disabled)
   - ☑️ Allow deletions (disabled)

### Required Status Checks

Enable these status checks (they'll appear after first workflow runs):
- `quality`
- `test (ubuntu-latest, 3.9, cpu)`
- `test (ubuntu-latest, 3.10, cpu)`
- `test (ubuntu-latest, 3.11, cpu)`
- `security`
- `docs`

## Repository Settings

### General Settings

1. **Features to Enable:**
   - ☑️ Issues
   - ☑️ Projects
   - ☑️ Wiki (optional)
   - ☑️ Discussions (recommended)

2. **Pull Requests:**
   - ☑️ Allow merge commits
   - ☑️ Allow squash merging (recommended)
   - ☑️ Allow rebase merging
   - ☑️ Automatically delete head branches

3. **Security:**
   - ☑️ Enable vulnerability alerts
   - ☑️ Enable automated security fixes
   - ☑️ Enable private vulnerability reporting

### Code Security and Analysis

1. **Dependabot:**
   - ☑️ Enable Dependabot alerts
   - ☑️ Enable Dependabot security updates
   - ☑️ Enable Dependabot version updates

2. **Code Scanning:**
   - ☑️ Enable CodeQL analysis
   - ☑️ Enable secret scanning
   - ☑️ Enable push protection for secrets

## GitHub Pages (Documentation)

### Setup Instructions

1. Go to **Settings → Pages**
2. Source: **Deploy from a branch**
3. Branch: **gh-pages** (will be created by workflow)
4. Folder: **/ (root)**
5. Custom domain (optional): `pg-neo-graph-rl.readthedocs.io`

## Issue and PR Templates

**✓ Already Created:** Issue and PR templates are already in place:
- `.github/ISSUE_TEMPLATE/bug_report.md`
- `.github/ISSUE_TEMPLATE/feature_request.md`
- `.github/PULL_REQUEST_TEMPLATE.md`

## Repository Topics and Description

### Recommended Topics

Add these topics to your repository (**Settings → General → Topics**):

```
federated-learning, graph-neural-networks, reinforcement-learning, 
jax, python, machine-learning, distributed-systems, traffic-control, 
power-grid, swarm-robotics, multi-agent-systems
```

### Repository Description

```
Federated Graph-Neural Reinforcement Learning toolkit for city-scale traffic, power-grid, or swarm control
```

### Website URL

```
https://pg-neo-graph-rl.readthedocs.io
```

## Third-Party Integrations

### Codecov Integration

1. Visit [codecov.io](https://codecov.io)
2. Sign up with GitHub account
3. Add your repository
4. Copy the token to `CODECOV_TOKEN` secret

### Docker Hub Integration

1. Create Docker Hub account
2. Create repository: `your-username/pg-neo-graph-rl`
3. Generate access token
4. Add credentials to secrets

### PyPI Integration

1. Create PyPI account
2. Generate API token
3. Add token to `PYPI_API_TOKEN` secret
4. Consider creating Test PyPI account for testing

## Monitoring Setup

### GitHub Insights

1. Enable **Insights** tab
2. Configure **Community Standards**
3. Set up **Traffic** monitoring
4. Enable **Dependency Graph**

### External Monitoring

Consider integrating:
- **UptimeRobot** for uptime monitoring
- **Better Uptime** for status pages
- **Sentry** for error tracking
- **DataDog** for comprehensive monitoring

## Verification Checklist

After completing the setup, verify:

- [ ] All workflow files are in `.github/workflows/`
- [ ] Required secrets are configured
- [ ] Branch protection rules are active
- [ ] First CI run completes successfully
- [ ] Security scanning is enabled
- [ ] Documentation builds and deploys
- [ ] Docker images build successfully
- [ ] Dependabot is configured
- [ ] Issue templates work correctly

## Troubleshooting

### Common Issues

1. **Workflows not triggering:**
   - Check workflow file syntax with `yamllint`
   - Verify branch names match your default branch
   - Ensure proper indentation in YAML files

2. **Secret access issues:**
   - Verify secret names match exactly (case-sensitive)
   - Check repository permissions for secrets
   - Confirm tokens haven't expired

3. **Build failures:**
   - Review workflow logs in **Actions** tab
   - Check dependency conflicts
   - Verify Docker builds locally first

### Getting Help

- **GitHub Actions:** [GitHub Actions Documentation](https://docs.github.com/en/actions)
- **Repository Settings:** [GitHub Repository Settings](https://docs.github.com/en/repositories)
- **Security Features:** [GitHub Security Features](https://docs.github.com/en/code-security)

---

**📝 Note:** This setup needs to be completed by a repository maintainer with appropriate permissions. The automated SDLC implementation has prepared all necessary templates and configurations, but manual activation is required due to security restrictions.
