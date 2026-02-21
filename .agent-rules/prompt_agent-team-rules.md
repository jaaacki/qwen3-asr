# Agent Team Rules

## Perspectives

Three types of perspective. How many of each — that's your call.

### Architect (one)
- Big picture: dependencies, sequencing, design patterns
- Asks: "why this order?", "what does this enable next?", "where does this fit?"
- Writes "why this design" entries in LEARNING_LOG.md
- Final say on minor version bumps

### Builder (as many as needed)
- Spawn one Builder per parallel issue, or one Builder handling several — match the workload
- Does the work: code, tests, config, docs
- Asks: "what pattern am I using?", "what's the alternative?", "what's the aha moment?"
- Writes "what just happened" entries in LEARNING_LOG.md
- Updates CHANGELOG.md and README.md as work lands
- Multiple Builders coordinate through the Architect — if two Builders' work starts overlapping, either one stops or the Architect re-assigns

### Critic (one or more)
- Stress-tests plans, code, and assumptions
- Asks: "what breaks?", "is this safe to run twice?", "what's the hidden coupling?"
- Writes "what could go wrong" entries in LEARNING_LOG.md
- Can challenge any decision
- When multiple Builders are active, Critic checks for cross-Builder conflicts: shared files, state collisions, import chains

## Scaling

- Start small. One Builder is fine for serial work.
- When the Architect identifies N independent issues, spin up N Builders to work them simultaneously.
- There is no upper limit — if 5 issues are independent, run 5 Builders.
- Scale back down when work converges on shared files or gets into integration territory.
- The Architect decides when to scale up or down. The Critic can flag when parallel work is getting risky.

## Picking Work

1. Look at open issues and the ROADMAP
2. Identify what's ready — no unmet dependencies, nothing blocked
3. Pick issues to work on. Use judgment:
   - Can these run in parallel? (different files, no shared state → yes, spin up multiple Builders)
   - Does one unlock others? (prioritize it)
   - Is something quick and independent? (hand it to a Builder alongside bigger work)
   - Already in progress on a branch? (leave it alone)
4. State what you're picking, who's working what, and what you're deferring
5. Do the work
6. Reassess — what's now unblocked? Pick again. Adjust Builder count.

No fixed batch size. No mandatory phase order. No rigid cycle. Match the situation.

## Rules

- Always explain why you're picking something and why you're skipping something
- If two issues touch the same files, one Builder — serial, not parallel
- If a Builder discovers a conflict mid-work, stop and escalate to Architect
- Cross-phase work is fine when dependencies are met
- Quick wins alongside big work is encouraged
- When multiple Builders finish at the same time, Architect sequences the merges to avoid conflicts
