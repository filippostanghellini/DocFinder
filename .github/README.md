# GitHub Actions CI/CD Documentation

This directory contains GitHub Actions workflows for continuous integration and deployment.

## Workflows

### 🔄 CI (`ci.yml`)
**Triggers:** Push to main, Pull Requests, Manual dispatch

**Jobs:**
- **Lint**: Runs Ruff linter and formatter checks
- **Test**: Runs tests on Python 3.10, 3.11, 3.12 across Ubuntu, macOS, and Windows
- **Test Install**: Verifies package installation and CLI availability
- **Security**: Runs Safety and Bandit security checks
- **Build**: Creates distribution packages and validates them
- **All Checks Passed**: Meta-job that ensures all required checks succeeded

**Caching:**
- Pip packages cached per OS and Python version
- HuggingFace models cached to speed up embedding tests

**Coverage:**
- Uploads coverage report to Codecov (requires `CODECOV_TOKEN` secret)
- Reports coverage in PR comments

### 📦 Release (`release.yml`)
**Triggers:** GitHub Release published, Manual dispatch

**Jobs:**
- **Build**: Creates source and wheel distributions
- **Publish to TestPyPI**: For manual testing (requires `TESTPYPI_API_TOKEN`)
- **Publish to PyPI**: Automatic on release (requires `PYPI_API_TOKEN`)
- **Upload Release Assets**: Attaches built packages to GitHub release
- **Docker Build**: Builds and pushes multi-arch Docker images (requires Docker Hub credentials)

**Trusted Publishing (PyPI):**
- Uses PyPI's trusted publishers (no API tokens in secrets if configured)
- Requires setting up trusted publisher on PyPI.org

### 🔒 CodeQL (`codeql.yml`)
**Triggers:** Push to main, Pull Requests, Weekly schedule (Monday 2 AM)

**Purpose:**
- Advanced security analysis using GitHub's CodeQL engine
- Detects security vulnerabilities and code quality issues
- Results appear in Security tab

### 🔍 Dependency Review (`dependency-review.yml`)
**Triggers:** Pull Requests only

**Purpose:**
- Reviews dependency changes in PRs
- Blocks PRs introducing vulnerabilities rated "moderate" or higher
- Comments summary directly in PR

### 🤖 Dependabot (`dependabot.yml`)
**Schedule:** Weekly on Mondays at 3 AM

**Updates:**
- Python dependencies (production and development grouped separately)
- GitHub Actions versions
- Auto-labels PRs and assigns to you
- Limits concurrent PRs to avoid spam

### 🧹 Stale (`stale.yml`)
**Schedule:** Daily at 1 AM

**Behavior:**
- Issues: Marked stale after 60 days, closed after 7 more days
- PRs: Marked stale after 45 days, closed after 14 more days
- Exempts: Issues/PRs labeled "pinned", "security", or "bug"

## Required Secrets

### For PyPI Publishing
**Option 1: API Tokens (easier)**
- `PYPI_API_TOKEN`: Get from https://pypi.org/manage/account/token/
- `TESTPYPI_API_TOKEN`: Get from https://test.pypi.org/manage/account/token/

**Option 2: Trusted Publishing (recommended)**
1. Go to PyPI project settings
2. Add trusted publisher: `github.com/filippostanghellini/DocFinder`
3. Workflow: `release.yml`
4. No secrets needed!

### For Coverage
- `CODECOV_TOKEN`: Get from https://codecov.io after linking repo

### For Docker (optional)
- `DOCKER_USERNAME`: Your Docker Hub username
- `DOCKER_PASSWORD`: Docker Hub access token (not your password!)

## Setting Up Secrets

```bash
# Go to: https://github.com/filippostanghellini/DocFinder/settings/secrets/actions
# Click "New repository secret" for each secret above
```

## Badge URLs

Add these to your README.md:

```markdown
![CI](https://github.com/filippostanghellini/DocFinder/workflows/CI/badge.svg)
![Release](https://github.com/filippostanghellini/DocFinder/workflows/Release/badge.svg)
![CodeQL](https://github.com/filippostanghellini/DocFinder/workflows/CodeQL/badge.svg)
[![codecov](https://codecov.io/gh/filippostanghellini/DocFinder/branch/main/graph/badge.svg)](https://codecov.io/gh/filippostanghellini/DocFinder)
[![PyPI](https://img.shields.io/pypi/v/docfinder)](https://pypi.org/project/docfinder/)
```

## Manual Workflow Triggers

### Test Release Process
```bash
# Go to Actions tab → Release → Run workflow
# Enter version number (e.g., 0.1.0)
# This will publish to TestPyPI for testing
```

### Create a Production Release
```bash
# Method 1: Via GitHub UI
# Go to Releases → Draft a new release → Create new tag → Publish

# Method 2: Via CLI
git tag v0.1.0
git push origin v0.1.0
gh release create v0.1.0 --generate-notes
```

## Local Testing

Test workflows locally with [act](https://github.com/nektos/act):

```bash
# Install act
brew install act  # macOS
# or download from GitHub releases

# Run CI workflow
act pull_request

# Run specific job
act -j lint

# With secrets
act -s CODECOV_TOKEN=xxx
```

## Troubleshooting

### Tests fail on Windows
- Check file path separators (use `pathlib.Path`)
- Check line endings (configure `.gitattributes`)

### Coverage upload fails
- Verify `CODECOV_TOKEN` is set
- Check Codecov.io repo is linked

### PyPI publish fails
- Verify version in `pyproject.toml` is incremented
- Check package name isn't taken
- Ensure trusted publisher is configured OR token is valid

### Docker build fails
- Verify Dockerfile exists
- Check secrets are set
- Ensure base image is available
