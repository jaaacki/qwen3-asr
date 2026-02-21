# Documentation & Versioning Rules

## Living Documents

Four documents are maintained. Updated after EVERY completed issue — never deferred.

### ROADMAP.md — Where we're going
- Phased plan with issues grouped by milestone
- Each phase has a one-line vision ("the system can now X")
- Issues ordered by dependency within each phase
- Unplaced issues go in a Backlog section until assigned to a phase
- Updated when: phases complete, new issues appear, priorities shift

### LEARNING_LOG.md — Why we did what we did
- Running narrative of decisions, patterns, and lessons
- Each entry references its issue number
- Each entry connects to previous entries — readable as a continuous story
- Three entry types:
  - **Why this design** — the pattern, trade-off, or architecture choice
  - **What just happened** — post-implementation: pattern used, why this over alternatives, one "aha moment" that's easy to miss
  - **What could go wrong** — edge cases, review findings, things that break at scale
- Written for someone learning — plain terms, no unexplained jargon

### CHANGELOG.md — What we have done
- Reverse chronological, newest at top
- Format:
  ```
  ## [X.Y.Z] — YYYY-MM-DD
  ### Added / Changed / Fixed / Removed
  - Description referencing issue (#N)
  ```
- Every completed issue gets an entry — no silent changes

### README.md — How to use it
- Updated incrementally as features land
- Fresh clone test: could someone set up and use everything right now?
- New commands, config, tools, setup steps — documented when they work, not later

## Versioning

### Patch (0.x.Y)
- Default. One issue = one patch bump.
- The system works the same way, just better.

### Minor (0.X.0)
- The system can do something it fundamentally couldn't before.
- Typically aligns with a milestone branch merging into main.
- CHANGELOG entry includes a narrative: why this is a meaningful boundary.
- Use judgment, not rigid rules.

### Major (X.0.0)
- Project reaches its stated vision or a major architectural reset.
- Rare and deliberate.

### Conventions
- Map phases to target minor versions in CHANGELOG header.
- Plan can be overridden — document the reasoning when it happens.
- Package version file stays in sync.
- Git tags on minor+ bumps.
- CHANGELOG is the source of truth.
