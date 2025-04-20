# PowerShell script to fix Git history by removing large files
Write-Host "=======================================================" -ForegroundColor Green
Write-Host "  Fixing Git history - Removing large files from history" -ForegroundColor Green
Write-Host "=======================================================" -ForegroundColor Green
Write-Host ""

# Check if current directory is a git repository
if (-not (Test-Path .git)) {
    Write-Host "Error: Current directory is not a Git repository." -ForegroundColor Red
    exit 1
}

# Step 1: Create a backup branch
Write-Host "Step 1: Creating backup branch..." -ForegroundColor Cyan
git branch backup-before-cleanup

# Step 2: Identify what's causing the issue
Write-Host "Step 2: These are the large files that need to be removed from Git history:" -ForegroundColor Cyan
git filter-branch --tree-filter "ls -la LeakDataset/Logger_Data_2024_Bau_Bang-2/ 2>/dev/null || echo 'Files already removed'" HEAD --max-count=1

# Step 3: Use git filter-branch to remove the large files from history
Write-Host "Step 3: Removing large files from Git history..." -ForegroundColor Cyan
Write-Host "This may take a while..." -ForegroundColor Yellow

# Remove LeakDataset folder from all commits
git filter-branch --force --index-filter "git rm -r --cached --ignore-unmatch LeakDataset/" --prune-empty --tag-name-filter cat -- --all

# Step 4: Clean up the repository
Write-Host "Step 4: Cleaning up repository..." -ForegroundColor Cyan
git for-each-ref --format="delete %(refname)" refs/original/ | git update-ref --stdin
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Step 5: Verify the fix
Write-Host "Step 5: Verifying files are no longer tracked..." -ForegroundColor Cyan
git count-objects -v

# Step 6: Instructions for pushing
Write-Host ""
Write-Host "=======================================================" -ForegroundColor Green
Write-Host "  Repository history has been cleaned!" -ForegroundColor Green
Write-Host "=======================================================" -ForegroundColor Green
Write-Host ""
Write-Host "To push these changes, use the following command:" -ForegroundColor Yellow
Write-Host "git push --force origin main" -ForegroundColor Cyan
Write-Host ""
Write-Host "WARNING: This will rewrite your remote history." -ForegroundColor Red
Write-Host "If other people are working on this repository, they may need to re-clone or use git pull --rebase." -ForegroundColor Yellow
Write-Host ""
Write-Host "Would you like to push now? (y/n)" -ForegroundColor Cyan
$response = Read-Host

if ($response -eq "y" -or $response -eq "Y") {
    Write-Host "Pushing changes to remote repository..." -ForegroundColor Cyan
    git push --force origin main
    Write-Host "Done!" -ForegroundColor Green
} else {
    Write-Host "Push canceled. You can manually push later using: git push --force origin main" -ForegroundColor Yellow
}

# Step 7: Add LeakDataset to .gitignore if needed
if (-not (Test-Path .gitignore)) {
    "LeakDataset/" | Out-File -FilePath .gitignore -Append
    Write-Host "Added LeakDataset/ to .gitignore" -ForegroundColor Green
} elseif (-not (Select-String -Path .gitignore -Pattern "LeakDataset/" -Quiet)) {
    "LeakDataset/" | Out-File -FilePath .gitignore -Append
    Write-Host "Added LeakDataset/ to .gitignore" -ForegroundColor Green
} else {
    Write-Host "LeakDataset/ is already in .gitignore" -ForegroundColor Green
}

# Final instructions
Write-Host ""
Write-Host "To prevent future issues with large files, consider using Git LFS:" -ForegroundColor Cyan
Write-Host "1. Install Git LFS: https://git-lfs.github.com" -ForegroundColor Yellow
Write-Host "2. Set up Git LFS for your repository: git lfs install" -ForegroundColor Yellow
Write-Host "3. Track large file types: git lfs track '*.csv' '*.docx'" -ForegroundColor Yellow 