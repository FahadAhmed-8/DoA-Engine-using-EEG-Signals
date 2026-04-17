# GitHub Migration — v2 flat → v2-under-prefix + v3

One-time reorganization so the remote repo mirrors the local layout:

```
Mini Project 2/
├── 01_docs/
├── 02_literature/
├── 03_data/                    (catalogue tracked; raw .mat files ignored)
├── 04_pipeline_v2/             (was root of remote)
├── 05_pipeline_v3/             (new)
├── .github/                    (new — CI, issue templates)
├── .gitignore                  (new — comprehensive)
├── CITATION.cff                (new)
├── CONTRIBUTING.md             (new)
├── LICENSE                     (new — MIT)
├── MIGRATION.md                (this file — delete after migration)
└── README.md                   (already exists locally)
```

Current state: remote `main` has the flat v2 layout, 4 commits, no README,
no LICENSE, no CI. Local `.git` lives inside `04_pipeline_v2/` — its HEAD
matches remote `main`.

The steps below **move `.git` up one level and rewrite-at-commit-time so
`04_pipeline_v2/<path>` is the new tracked location**. History is preserved
via git's rename detection when you run `git log --follow`.

> Back up first. Before anything, copy the whole `Mini Project 2` folder to
> a second location. If the migration goes sideways you can restore.

---

## Step 0 — sanity check (10 seconds)

```powershell
cd "C:\Users\ROG STRIX\OneDrive\Desktop\6TH Sem\Mini Project 2\04_pipeline_v2"
git status                  # must be clean
git log --oneline -5        # you should see 4 commits, tip = 07e1bb5
git remote -v               # must point to FahadAhmed-8/DoA-Engine-using-EEG-Signals
```

If `git status` is **not** clean: commit or stash everything first. The
migration moves the git index, so uncommitted changes will be confusing.

---

## Step 1 — move `.git` up to the project root

```powershell
cd "C:\Users\ROG STRIX\OneDrive\Desktop\6TH Sem\Mini Project 2"

# Move the git directory one level up.
# On PowerShell:
Move-Item 04_pipeline_v2\.git .git

# (On Git Bash or WSL: mv 04_pipeline_v2/.git .git)
```

Now run:

```powershell
git status
```

You should see a **large list of "deleted"** (the files git thinks vanished
from their old paths at root) and a **large list of "untracked"** (the same
files at their new `04_pipeline_v2/…` paths, plus all the new v3 / docs /
literature / data / scaffold files).

That is expected.

---

## Step 2 — stage everything and verify rename detection

```powershell
git add -A
git status --short | Select-String -Pattern "^R " | Measure-Object
```

The `R` (rename) count should be roughly the number of files that existed
at remote root before (around 30-40 Python files). Git's default threshold
detects moves at commit time; `git log --follow -- 04_pipeline_v2/config/config.py`
will later trace through the rename back to the original history.

If you want to see the rename detections explicitly:

```powershell
git diff --cached --stat --find-renames
```

---

## Step 3 — commit the reorganization

```powershell
git commit -m "chore: reorganize to v2/v3 layout

Move frozen v2 pipeline into 04_pipeline_v2/ prefix.
Add 05_pipeline_v3/ scaffold, root README, .github/ CI + templates,
LICENSE (MIT), CITATION.cff, CONTRIBUTING, .gitignore,
v2/v3 requirements.txt.

v2 history preserved via git rename detection:
  git log --follow -- 04_pipeline_v2/<path>"
```

---

## Step 4 — push

Because this commit adds many files and moves paths, but is a clean
fast-forward on `main`, a normal push works:

```powershell
git push origin main
```

No `--force` needed — we're adding a commit on top of the existing tip, not
rewriting history.

---

## Step 5 — verify on GitHub

Open https://github.com/FahadAhmed-8/DoA-Engine-using-EEG-Signals and confirm:

- [ ] Root shows README.md, LICENSE, CITATION.cff, .github/ folder
- [ ] Root shows `01_docs/`, `02_literature/`, `03_data/` (with
      `case_catalogue.csv` inside), `04_pipeline_v2/`, `05_pipeline_v3/`
- [ ] The old root-level `analysis/`, `config/`, `data/`, etc. are gone
      (they moved into `04_pipeline_v2/`)
- [ ] Code tab shows MIT License badge
- [ ] "About" side panel auto-updates with the CITATION snippet once GitHub
      picks it up (may take a few minutes)
- [ ] CI workflow kicks off under the Actions tab

---

## Step 6 — update the remote description

Your current description is out of date ("24 patients, 28 experiments").
Edit on the GitHub web UI → About (pencil icon) and paste:

> Subject-independent Depth of Anaesthesia estimation from single-channel
> EEG (VitalDB cohort). v2 (frozen): multi-entropy fusion + classical ML,
> best LOPO RMSE 11.5. v3 (in progress): HEED-Net dual-stream architecture
> with domain-adversarial learning for an IEEE Transactions submission.

Keep the topics (`anesthesia`, `biomedical`, `eeg`, `entropy`, etc.); add
`deep-learning` and `vitaldb` to them.

---

## Step 7 — once confirmed, delete this file

```powershell
git rm MIGRATION.md
git commit -m "docs: remove completed migration guide"
git push origin main
```

---

## If Step 2's status output looks wrong

Safety net: revert the `.git` move and try again:

```powershell
cd "C:\Users\ROG STRIX\OneDrive\Desktop\6TH Sem\Mini Project 2"
Move-Item .git 04_pipeline_v2\.git
```

You are now back to the original state. Re-read Step 0 and try again.

---

## Alternative: clean-slate start

If Step 2 shows too much divergence to trust (for example, remote has
commits you didn't know about), the safest path is a clean reset:

```powershell
# Archive the old history on a branch first
cd "C:\Users\ROG STRIX\OneDrive\Desktop\6TH Sem\Mini Project 2"
git checkout -b v2-archive-original
git push origin v2-archive-original

# Then rewind main to a single clean reorg commit
git checkout main
git reset --soft 4b825dc642cb6eb9a060e54bf8d69288fbee4904  # empty tree
git add -A
git commit -m "v3 reorg: merge v2 under 04_pipeline_v2/, add v3 scaffold"
git push origin main --force-with-lease
```

This loses v2's 4 commits from `main` but they remain on the
`v2-archive-original` branch for reference.
