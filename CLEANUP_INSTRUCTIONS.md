# 🔧 Git History Cleanup Instructions

> **Exposed Token Removal from Git History**

**Status:** This guide walks you through removing the exposed Zoho token from all git commits.

---

## ⚠️ Important Warnings

- **This REWRITES git history** - Cannot be undone without backup
- **Force push is required** - Will overwrite remote history
- **Other developers affected** - They must re-clone the repository
- **Backup is created automatically** - But keep your own too
- **CI/CD builds may break** - References to old commits will fail

---

## Quick Start (Automated)

If you have Bash available, use the automated script:

```bash
# Make script executable
chmod +x git-history-cleanup.sh

# Run the cleanup
./git-history-cleanup.sh check_ttft.py

# Follow the prompts
```

The script will:
1. Check for `git-filter-repo`
2. Create a backup
3. Rewrite history
4. Verify cleanup
5. Offer to force-push

---

## Manual Method (Step-by-Step)

If you prefer to do it manually or the script doesn't work:

### Step 1: Install git-filter-repo

```bash
pip install git-filter-repo
```

Verify installation:
```bash
git filter-repo --version
```

### Step 2: Backup Your Repository

```bash
# Create a bare clone as backup
cd ~
git clone --bare /path/to/PYDeepLearning PYDeepLearning.backup

# Or zip it
zip -r PYDeepLearning_backup.zip /path/to/PYDeepLearning/.git
```

### Step 3: Navigate to Your Repository

```bash
cd /path/to/PYDeepLearning
```

### Step 4: Review the File to Remove

```bash
# See the file in recent commits
git log --oneline check_ttft.py | head -5

# Check the content
git show HEAD:check_ttft.py | grep -i "zalb\|crm"
```

### Step 5: Run git-filter-repo

**Option A: Remove entire file**
```bash
git filter-repo --path check_ttft.py --invert-paths --force
```

**Option B: Remove multiple files**
```bash
git filter-repo \
  --path check_ttft.py \
  --path .env \
  --path credentials.json \
  --invert-paths --force
```

**Option C: Remove by pattern**
```bash
git filter-repo --path-glob '*.env*' --invert-paths --force
```

### Step 6: Verify Cleanup

```bash
# Check file is gone from history
git log --all -- check_ttft.py
# Should output: fatal: your current branch 'main' does not have any commits yet

# Verify recent commits
git log --oneline -10

# Check for the exposed token
git log -S "zalb_791c15dd5f" --all
# Should output nothing
```

### Step 7: Force Push to GitHub

⚠️ **This is the point of no return!**

```bash
# Push all branches with force-with-lease (safer than --force)
git push origin --force-with-lease --all

# Push all tags
git push origin --force-with-lease --tags
```

### Step 8: Verify on GitHub

1. Visit: https://github.com/nlpkiddo-2001/PYDeepLearning/commits
2. Search for the commit with the token
3. Verify it's no longer in history

```bash
# Also check locally
git log --all --grep="985cfc79" --oneline
# Should show nothing
```

---

## Troubleshooting

### Problem: "git filter-repo not found"

**Solution:**
```bash
pip install git-filter-repo
# Add to PATH if needed
export PATH="$PATH:$HOME/.local/bin"
```

---

### Problem: "fatal: You are in the middle of a filter-repo operation"

**Solution:**
```bash
# Clean up from failed attempt
git filter-repo --help  # This completes the operation
# Then try again
```

---

### Problem: Force push is rejected

**Solution 1: Check branch protection**
```
GitHub Settings → Branches → Branch protection rules
Temporarily disable for main branch, push, then re-enable
```

**Solution 2: Use force-with-lease (safer)**
```bash
git push origin --force-with-lease --all
```

---

### Problem: "remote: error: denying non-fast-forward refs"

**Solution:**
```bash
# Temporarily disable branch protection on GitHub
# Settings → Branches → Branch protection rules → Edit → Uncheck all

# Then push
git push origin --force-with-lease --all

# Re-enable branch protection
```

---

## Recovery: If Something Goes Wrong

### Restore from Backup

```bash
# Stop everything
cd ~

# Remove corrupted repo
rm -rf PYDeepLearning

# Restore from backup
git clone --bare PYDeepLearning.backup PYDeepLearning/.git
cd PYDeepLearning
git config --bool core.bare false
```

### Reset GitHub (Nuclear Option)

```bash
# Delete remote repository on GitHub (Settings → Danger Zone)
# Create new empty repository
# Push clean version
git push -u origin main
```

---

## For Team Members (After Cleanup)

If you're working with a team, they need to:

```bash
# Backup their work
git stash

# Remove old clone
cd ~
rm -rf PYDeepLearning

# Re-clone
git clone https://github.com/nlpkiddo-2001/PYDeepLearning.git

# Verify history is clean
git log --all --grep="zalb" --oneline
# Should show nothing
```

---

## Verification Checklist

After cleanup, verify everything:

- [ ] Exposed token is gone from git history
- [ ] Old commit is no longer accessible
- [ ] GitHub shows cleaned history
- [ ] Local repository is synced
- [ ] Tags are pushed correctly
- [ ] CI/CD pipelines are updated
- [ ] Team members re-cloned repository
- [ ] `.env.example` is committed (not `.env`)
- [ ] SECURITY.md and SECURITY_INCIDENT_REPORT.md are in repo
- [ ] Branch protection is re-enabled

---

## What Happens Next?

### ✅ Completed
- Exposed token removed from current code
- Environment variables implemented
- Security documentation created
- Backup created

### 🔄 In Progress
- You are here: running git history cleanup

### 📋 Still To Do
1. Force push to GitHub
2. Notify Zoho security team
3. Rotate the exposed token (Zoho side)
4. Monitor access logs (Zoho side)
5. Enable GitHub secret scanning
6. Setup pre-commit hooks

---

## Additional Resources

- [git-filter-repo Official Guide](https://github.com/newren/git-filter-repo/blob/master/README.md)
- [GitHub: Removing Sensitive Data](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
- [OWASP: Secrets Management](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
- [CWE-798: Hard-Coded Credentials](https://cwe.mitre.org/data/definitions/798.html)

---

## Questions?

If you encounter issues:

1. Check the troubleshooting section above
2. Consult the git-filter-repo documentation
3. Verify you have a backup
4. Contact your security team

**Remember:** You can always restore from backup if something goes wrong.
