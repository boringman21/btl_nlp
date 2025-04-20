# Fixing Git Large File Issues

## The Problem

You're encountering an error when trying to push to GitHub because some files in your Git history exceed GitHub's file size limits:

```
remote: error: File LeakDataset/Logger_Data_2024_Bau_Bang-2/find_query_3.csv is 114.05 MB; this exceeds GitHub's file size limit of 100.00 MB
remote: error: File LeakDataset/Logger_Data_2024_Bau_Bang-2/find_query_1.csv is 103.66 MB; this exceeds GitHub's file size limit of 100.00 MB
remote: error: GH001: Large files detected. You may want to try Git Large File Storage - https://git-lfs.github.com.
```

Even though you've already deleted these files from your working directory, they still exist in your Git history, which is why GitHub rejects your push.

## Solution - Using the Provided PowerShell Script

### 1. Run the PowerShell Script

The `fix_git_history.ps1` script will remove the large files from your Git history.

```powershell
# Open PowerShell and run:
.\fix_git_history.ps1
```

The script performs the following steps:
1. Creates a backup branch to preserve your current state
2. Removes the LeakDataset directory from all commits in your history
3. Cleans up and optimizes the repository
4. Updates .gitignore to prevent these files from being committed again
5. Offers to push the changes to GitHub

### 2. Manual Approach (if the script doesn't work)

If you prefer to run the commands manually:

```powershell
# Create a backup branch
git branch backup-before-cleanup

# Remove LeakDataset from all commits
git filter-branch --force --index-filter "git rm -r --cached --ignore-unmatch LeakDataset/" --prune-empty --tag-name-filter cat -- --all

# Clean up
git for-each-ref --format="delete %(refname)" refs/original/ | git update-ref --stdin
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Force push to GitHub
git push --force origin main
```

## Preventing Future Issues

### Option 1: Use .gitignore

We've added a comprehensive `.gitignore` file that excludes:
- The entire LeakDataset directory
- Common large file types (CSV, Excel, databases)
- Temporary Python files and directories

### Option 2: Use Git LFS (Large File Storage)

For handling large files properly:

1. Install Git LFS: [https://git-lfs.github.com](https://git-lfs.github.com)
2. Set up Git LFS:
   ```bash
   git lfs install
   git lfs track "*.csv" "*.docx"  # Track file patterns
   git add .gitattributes
   git commit -m "Configure Git LFS"
   ```

### Option 3: Store Data Externally

Consider storing large datasets on:
- Google Drive or Dropbox
- A specific data hosting service
- MLflow, DVC, or other ML-specific tools

## Notes

- The `--force` push will rewrite history on GitHub
- If others are working on this repository, they'll need to re-clone or carefully rebase
- The original large files will still be in your backup branch if needed 