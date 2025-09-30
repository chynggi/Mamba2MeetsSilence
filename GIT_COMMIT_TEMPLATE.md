# Git Commit Template for data folder fix

## Option 1: Simple commit message
```bash
git add data/ .gitignore data/README.md
git commit -m "Fix: Add data package to Git repository

- Removed data/ from .gitignore (was incorrectly excluding source code)
- data/ folder contains Python modules, not dataset files
- Added data/README.md to clarify purpose
- Fixes ModuleNotFoundError on fresh clones"
```

## Option 2: Detailed commit message
```bash
git commit -m "Fix: data package was excluded from Git due to .gitignore

Problem:
- .gitignore contained 'data/' which excluded the entire data/ source folder
- This caused ModuleNotFoundError: No module named 'data' on Linux servers
- Windows local development worked because data/ folder existed locally

Solution:
- Removed 'data/' from .gitignore
- Added data/README.md to clarify it's source code, not datasets
- Updated README.md with troubleshooting note

Files changed:
- .gitignore: Removed data/ exclusion
- data/: Added entire package (dataset.py, transforms.py, __init__.py)
- data/README.md: New file explaining folder purpose
- README.md: Added import troubleshooting note
- BUGFIX_DATA_GITIGNORE.md: Detailed bug fix documentation

Testing:
- test_imports.py now passes all import tests
- Verified on both Windows and Linux environments"
```

## Option 3: Conventional Commits format
```bash
git commit -m "fix(git): include data package in repository

The data/ folder was incorrectly excluded by .gitignore, causing import
failures on fresh clones. This folder contains source code, not dataset files.

BREAKING CHANGE: Projects cloned before this fix will need to pull latest
changes to get the data/ package.

Closes #<issue-number-if-exists>"
```

## After committing, push to remote:
```bash
git push origin main
```

## Verification after push:
```bash
# On a different machine or clean clone:
git clone https://github.com/chynggi/Mamba2MeetsSilence.git
cd Mamba2MeetsSilence
python test_imports.py
# All imports should succeed ✓
```
