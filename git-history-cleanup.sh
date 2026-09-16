#!/bin/bash

################################################################################
# Git History Cleanup Script - Remove Exposed Secrets
# 
# This script removes sensitive files from git history using git-filter-repo
# and force-pushes the cleaned history back to GitHub.
#
# ⚠️  IMPORTANT: This script REWRITES GIT HISTORY. Make sure you:
#   1. Have a backup of your repository
#   2. Are the only developer (or notify your team)
#   3. Update any build systems or CI/CD that reference old commits
#
# Usage:
#   bash git-history-cleanup.sh [--file-to-remove] [--remove-patterns]
#
# Examples:
#   bash git-history-cleanup.sh check_ttft.py
#   bash git-history-cleanup.sh .env
#   bash git-history-cleanup.sh --patterns "*.key" "*.pem" "*.credentials"
#
################################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ─────────────────────────────────────────────────────────────────────────────

print_header() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# ─────────────────────────────────────────────────────────────────────────────

print_header "Git History Cleanup - Exposed Secrets Removal"

# Default: remove check_ttft.py (contains exposed Zoho token)
FILES_TO_REMOVE=("check_ttft.py")

# Parse command line arguments
if [ $# -gt 0 ]; then
    FILES_TO_REMOVE=("$@")
fi

echo -e "\n${BLUE}Files to be removed from history:${NC}"
for file in "${FILES_TO_REMOVE[@]}"; do
    echo "  • $file"
done

# ─────────────────────────────────────────────────────────────────────────────

print_warning "This script will REWRITE git history!"
print_warning "Make sure you have a backup before proceeding."

read -p "Do you want to continue? (yes/no): " confirm
if [[ "$confirm" != "yes" ]]; then
    echo "Cancelled."
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────

# Check if git-filter-repo is installed
print_header "Checking Prerequisites"

if ! command -v git-filter-repo &> /dev/null; then
    print_error "git-filter-repo is not installed"
    echo ""
    echo "Install it with:"
    echo "  pip install git-filter-repo"
    exit 1
fi

print_success "git-filter-repo is installed"

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    print_error "Not in a git repository"
    exit 1
fi

print_success "Current directory is a git repository"

# ─────────────────────────────────────────────────────────────────────────────

print_header "Repository Information"

REPO_URL=$(git config --get remote.origin.url)
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
COMMIT_COUNT=$(git rev-list --count HEAD)

echo "Repository URL: $REPO_URL"
echo "Current branch: $CURRENT_BRANCH"
echo "Total commits: $COMMIT_COUNT"

# ─────────────────────────────────────────────────────────────────────────────

print_header "Checking for Sensitive Content"

echo "Scanning for files to remove..."

# Build the filter-repo arguments
FILTER_ARGS=()
for file in "${FILES_TO_REMOVE[@]}"; do
    FILTER_ARGS+=("--path" "$file")
    
    # Check if file exists in history
    if git log --all --full-history --diff-filter=D -- "$file" | grep -q commit; then
        print_success "Found '$file' in git history (will be removed)"
    elif git log --all -- "$file" 2>/dev/null | grep -q commit; then
        print_success "Found '$file' in git history (will be removed)"
    else
        print_warning "'$file' not found in history (may have already been removed)"
    fi
done

# ─────────────────────────────────────────────────────────────────────────────

print_header "Creating Backup"

BACKUP_DIR="${HOME}/git_backup_$(date +%Y%m%d_%H%M%S)"
print_warning "Creating backup at: $BACKUP_DIR"

# Clone the current repo as backup
git clone --bare . "$BACKUP_DIR"
if [ -d "$BACKUP_DIR" ]; then
    print_success "Backup created successfully"
    echo "Backup location: $BACKUP_DIR"
else
    print_error "Failed to create backup"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────

print_header "Running git-filter-repo"

echo "This may take a while depending on repository size..."
echo ""

# Execute git-filter-repo with invert-paths (remove the specified files)
if git filter-repo "${FILTER_ARGS[@]}" --invert-paths --force; then
    print_success "git-filter-repo completed successfully"
else
    print_error "git-filter-repo failed"
    echo ""
    echo "Your repository has been backed up at: $BACKUP_DIR"
    echo "You can restore from backup if needed."
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────

print_header "Verifying Cleanup"

echo "Checking if sensitive files still exist..."

FILES_FOUND=0
for file in "${FILES_TO_REMOVE[@]}"; do
    if git log --all -- "$file" 2>/dev/null | grep -q commit; then
        print_error "File still exists in history: $file"
        FILES_FOUND=$((FILES_FOUND + 1))
    else
        print_success "File successfully removed from history: $file"
    fi
done

if [ $FILES_FOUND -gt 0 ]; then
    print_error "Some files are still in history. Cleanup may have failed."
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────

print_header "Ready to Push"

echo ""
print_warning "The history has been rewritten locally."
echo ""
echo "Next steps:"
echo ""
echo "  1. Review the changes:"
echo "     git log --oneline -n 10"
echo ""
echo "  2. Force push to GitHub (THIS CANNOT BE UNDONE):"
echo "     git push origin --force-with-lease --all"
echo "     git push origin --force-with-lease --tags"
echo ""
echo "  3. Other developers need to:"
echo "     - Backup their work"
echo "     - Delete their local clone"
echo "     - Re-clone the repository"
echo ""

read -p "Do you want to force push now? (yes/no): " confirm_push
if [[ "$confirm_push" != "yes" ]]; then
    echo ""
    print_warning "Push cancelled. You can push manually later with:"
    echo "  git push origin --force-with-lease --all"
    echo "  git push origin --force-with-lease --tags"
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────

print_header "Force Pushing to GitHub"

if git push origin --force-with-lease --all; then
    print_success "All branches pushed successfully"
else
    print_error "Failed to push branches"
    echo "Try pushing manually: git push origin --force-with-lease --all"
    exit 1
fi

if git push origin --force-with-lease --tags; then
    print_success "All tags pushed successfully"
else
    print_warning "Failed to push tags (this is usually not critical)"
fi

# ─────────────────────────────────────────────────────────────────────────────

print_header "Cleanup Complete! ✅"

echo ""
echo "Summary:"
echo "  • Git history has been rewritten"
echo "  • Sensitive files have been removed"
echo "  • Changes have been force-pushed to GitHub"
echo ""
echo "Backup saved at: $BACKUP_DIR"
echo ""
print_warning "Team members need to:"
echo "  • Backup their work"
echo "  • Re-clone the repository"
echo "  • Update any local branches"
echo ""
echo "Verify on GitHub:"
echo "  • Check commit history: $REPO_URL/commits"
echo "  • Confirm sensitive files are gone"
echo ""

print_success "Incident remediation complete!"
