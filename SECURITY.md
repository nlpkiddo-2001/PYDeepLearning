# 🔒 Security Best Practices

This document outlines security guidelines and best practices for this project.

---

## 🚨 Never Commit Secrets

**CRITICAL:** Never commit API keys, tokens, passwords, or other secrets to Git.

### What NOT to Commit:
- `.env` files with real credentials
- API keys (OpenAI, Anthropic, Google, etc.)
- Database passwords
- Authentication tokens
- SSH private keys
- Database connection strings with passwords
- Any hardcoded credentials

### What TO Commit:
- `.env.example` - Template with placeholder values
- `SECURITY.md` - This file
- `.gitignore` - Rules to prevent accidental commits

---

## ✅ How to Handle Credentials

### 1. **Use Environment Variables**
```python
import os

# ✅ GOOD
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

# ❌ BAD
api_key = "sk-abc123xyz..."
```

### 2. **Use .env Files (Local Development)**
```bash
# .env (NEVER commit this)
OPENAI_API_KEY=sk-abc123xyz...
ZOHO_AUTH_TOKEN=token123...

# .env.example (Always safe to commit)
OPENAI_API_KEY=your-openai-api-key
ZOHO_AUTH_TOKEN=your-zoho-token
```

### 3. **Load Environment Variables**
```python
from dotenv import load_dotenv
import os

# Load from .env file
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
```

### 4. **Use Secrets Management (Production)**

For production environments, use:
- **GitHub Secrets** (for Actions)
- **AWS Secrets Manager**
- **HashiCorp Vault**
- **Docker secrets** (for containerized apps)
- **Environment variable services**

---

## 🔐 Setup Instructions

### Development Setup

1. **Copy the template:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your actual credentials:**
   ```bash
   nano .env
   # or
   vim .env
   ```

3. **Verify `.env` is in `.gitignore`:**
   ```bash
   cat .gitignore | grep "^.env$"
   ```

4. **Install dependencies:**
   ```bash
   pip install python-dotenv
   ```

---

## 📋 Pre-Commit Hooks

Prevent secrets from being committed using pre-commit hooks:

### Install detect-secrets

```bash
pip install detect-secrets
```

### Setup pre-commit hook

Create `.git/hooks/pre-commit`:
```bash
#!/bin/bash
detect-secrets scan --baseline .secrets.baseline
if [ $? -eq 1 ]; then
    echo "❌ Secrets detected. Commit blocked."
    exit 1
fi
```

Make it executable:
```bash
chmod +x .git/hooks/pre-commit
```

---

## 🔍 Check for Secrets Before Committing

### Using git-secrets
```bash
brew install git-secrets  # macOS
# or
apt-get install git-secrets  # Ubuntu

# Setup for this repo
git secrets --install
git secrets --register-aws

# Scan commits
git secrets --scan
```

### Using detect-secrets
```bash
# Scan current directory
detect-secrets scan

# Baseline for comparison
detect-secrets scan > .secrets.baseline
```

---

## 🚨 If You Accidentally Commit a Secret

### Immediate Steps:
1. **DO NOT PUSH** if it's only in local commits
2. **Rotate the credential** immediately (revoke/regenerate)
3. **Remove from git history** (see below)
4. **Notify security team** (internal)

### Remove from Git History

**Using git-filter-repo (recommended):**
```bash
# Install
pip install git-filter-repo

# Remove file from history
git filter-repo --path <filename> --invert-paths

# Force push
git push origin --force-with-lease --all
```

**Using git filter-branch (alternative):**
```bash
git filter-branch --force --index-filter \
  'git rm -r --cached --ignore-unmatch <filename>' \
  --prune-empty -- --all

git push origin --force --all
```

---

## 🛡️ GitHub Security Features

### Enable Secret Scanning
1. Go to **Settings → Code security & analysis**
2. Enable **Secret scanning**
3. Enable **Push protection** (premium)

### Set Up Branch Protection
1. Go to **Settings → Branches**
2. Add branch protection rule for `main`
3. Require status checks to pass
4. Require secret scanning to pass

### Review Security Advisories
1. Go to **Security → Advisories**
2. Check for vulnerable dependencies

---

## 🔒 Third-Party Credentials

### For Each Service:

**OpenAI:**
```
OPENAI_API_KEY=sk-...
OPENAI_ORG_ID=org-...
```

**Anthropic:**
```
ANTHROPIC_API_KEY=sk-ant-...
```

**Google:**
```
GOOGLE_API_KEY=AIzaSy...
GOOGLE_PROJECT_ID=my-project
```

**Zoho:**
```
ZOHO_LLM_URL=https://...
ZOHO_AUTH_TOKEN=...
```

**Kaggle:**
```
KAGGLE_USERNAME=your-username
KAGGLE_KEY=your-key
```

---

## 📝 Code Review Checklist

Before submitting a PR, check:

- [ ] No `.env` files included
- [ ] No hardcoded API keys
- [ ] No hardcoded passwords
- [ ] No hardcoded tokens
- [ ] All credentials use environment variables
- [ ] `.env.example` updated if new env vars added
- [ ] Sensitive files are in `.gitignore`
- [ ] No secrets in commit messages
- [ ] No secrets in comments

---

## 🚨 Common Mistakes

### ❌ Mistake 1: Hardcoded Credentials
```python
# BAD
OPENAI_API_KEY = "sk-abc123xyz..."
```

### ✅ Solution
```python
# GOOD
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

---

### ❌ Mistake 2: Printing Secrets in Logs
```python
# BAD
print(f"Token: {token}")
logger.info(f"Using key: {api_key}")
```

### ✅ Solution
```python
# GOOD
logger.debug("Attempting API call")
# Don't log the actual token
```

---

### ❌ Mistake 3: Secrets in Comments
```python
# BAD
# Use this token: zalb_791c15dd5f=f0b83a68b20b8687864dc215f54d0554
```

### ✅ Solution
```python
# GOOD
# Use the ZOHO_AUTH_TOKEN environment variable
```

---

## 📞 Incident Response

If you discover an exposed secret:

1. **Report immediately** to the security team
2. **Rotate the credential** (revoke/regenerate)
3. **Check access logs** for unauthorized usage
4. **Remove from git history** (see instructions above)
5. **Document the incident** with timeline
6. **Update security procedures** to prevent recurrence

---

## 📚 Additional Resources

- [OWASP Secrets Management](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
- [GitHub Secret Scanning](https://docs.github.com/en/code-security/secret-scanning)
- [git-secrets Documentation](https://github.com/awslabs/git-secrets)
- [detect-secrets Documentation](https://github.com/Yelp/detect-secrets)
- [CWE-798: Use of Hard-Coded Credentials](https://cwe.mitre.org/data/definitions/798.html)

---

## Questions?

If you have questions about security practices, please reach out to the security team or create an issue with the `security` label.

**Last Updated:** 2026-09-16  
**Status:** Active
