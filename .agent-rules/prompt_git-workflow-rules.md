# Git Workflow Rules

## Issues

### Every change starts with an issue
- No code without an issue. No PR without an issue.
- Issue number links to CHANGELOG entry, LEARNING_LOG entry, branch name, and PR title.
- If a change surfaces mid-work (bug found, refactor needed), create the issue first, then fix it.

### Title format
- Prefix with label in brackets: `[Enhancement]`, `[Bug]`, `[Fix]`, `[Docs]`, `[Refactor]`, `[Chore]`
- Followed by a clear action: what is being added, fixed, or changed
- Examples:
  - `[Enhancement] Add retry logic with exponential backoff`
  - `[Bug] Agent creates duplicate comments on re-run`
  - `[Fix] Handle missing config.json gracefully`
  - `[Docs] Update README with webhook setup instructions`
  - `[Refactor] Extract shared logic into core.ts`
  - `[Chore] Update dependencies`

### Apply matching GitHub labels
- `enhancement`, `bug`, `fix`, `documentation`, `refactor`, `chore`
- Add phase/milestone labels if the project uses phased roadmaps

### Issue body — keep it light
- **What** — one or two sentences on the change
- **Why** — why this matters or what breaks without it
- **Expectations** — what "done" looks like (observable behavior, not implementation details)
- No essays. If it needs more context, link to a doc or discussion.

### Filing issues
- When asked to file an issue (or when a bug/enhancement/task is identified during conversation), delegate to a background agent
- Never block the conversation to file — hand it off and keep talking
- Confirm with issue number and link once filed
- If multiple issues surface at once, file them all separately — don't bundle unrelated changes
- Issue filing is fire-and-forget from the conversation's perspective

## Milestones

### Milestones mirror the ROADMAP
- Each roadmap phase = one GitHub Milestone
- Milestone title matches the phase: `Phase 4 — Intelligence` or `v0.4.0 — Intelligence`
- Milestone description = the phase's one-line vision from ROADMAP.md
- Every issue belongs to a milestone (except bugs — see below)

### Milestone branches
- Feature development happens on a milestone branch, not directly on main
- Branch name: `milestone/{phase-name}` (e.g., `milestone/intelligence`, `milestone/resilience`)
- Individual issue branches branch FROM the milestone branch, not from main
- PRs for issues merge INTO the milestone branch
- When all issues in a milestone are complete, the milestone branch merges into main — this is the minor version bump

### Branch structure
```
main
 └── milestone/resilience          ← phase branch
      ├── issue-17-retry-backoff   ← issue branches off milestone
      ├── issue-22-graceful-shutdown
      └── issue-33-structured-logging
```

### Bugs and hotfixes bypass milestones
- `[Bug]` and `[Fix]` issues branch directly from main
- PR merges directly into main
- No milestone branch needed — bugs are patch bumps on the current version
- If a bug is discovered during milestone work, fix it on main first, then rebase the milestone branch

## Workflow

### 1. Issue
- Create the issue with title prefix, label, and light description
- Assign to a GitHub Milestone (= roadmap phase)

### 2. Study
- Read relevant code, docs, and related issues before touching anything
- If the issue involves a new pattern, document the "why this design" in LEARNING_LOG.md before coding

### 3. Code
- Branch from the milestone branch: `issue-{N}-{short-description}`
- If bug/fix: branch from main instead
- Commit messages reference the issue: `Fix #N: description` or `Closes #N: description`
- Follow existing project conventions — don't introduce new patterns without documenting why

### 4. PR
- PR title matches the issue: `Fix #N: {issue title without prefix}`
- PR body links the issue: `Closes #N`
- PR targets the milestone branch (or main for bugs)
- Draft PR if work is incomplete or awaiting review
- Update CHANGELOG.md, LEARNING_LOG.md, README.md in the same PR — not in a follow-up

### 5. Merge issue PR
- Squash merge into the milestone branch
- Verify docs are updated before merging
- Delete the issue branch after merge

### 6. Merge milestone into main
- When all issues in the milestone are done and reviewed
- Merge milestone branch into main (merge commit, not squash — preserve the issue history)
- Tag the merge with the minor version
- Delete the milestone branch after merge
- Close the GitHub Milestone

### Versioning
- Version bump rules: see **docs-versioning-rules.md**
- Short version: each issue = patch bump, milestone merge = minor bump, bugs on main = patch bump

## Parallel milestones
- Multiple milestone branches can exist simultaneously when phases are independent
- The Architect (or you) decides which milestones can overlap
- If two milestones touch shared files, coordinate merge order — earlier milestone merges first, later one rebases
