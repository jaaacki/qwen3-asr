# Testing Rules

## Test files live with code
- Every source file gets a test file next to it
- Naming: `xxx.test.ts`, `xxx_test.py`, `xxx.test.js` — match the project's convention, pick one and stick with it
- Tests move when code moves. Tests rename when code renames. Tests delete when code deletes.
- If a source file has no test file, that's a gap — flag it

### Structure
```
src/
  config.ts
  config.test.ts        ← lives next to the code it tests
  agent.ts
  agent.test.ts
  github-tools.ts
  github-tools.test.ts

tests/                   ← test infrastructure only
  setup.ts               ← global setup, teardown
  mocks/                 ← shared mock factories
  fixtures/              ← test data, sample payloads
  helpers/               ← test utilities, custom matchers
```

- `tests/` directory is for infrastructure — shared mocks, fixtures, helpers, global config
- No actual test cases in `tests/` — those live next to their source files
- If a mock is used by only one test file, keep it in that test file. Move to `tests/mocks/` only when shared across multiple test files.

## Tests adapt with code

### Code changes = test changes
- Every PR that changes code must update its tests in the same PR
- New function → new tests
- Changed behavior → updated tests
- Removed code → removed tests
- No "I'll add tests later" — tests and code ship together

### Migrations and schema changes
- When state formats, config schemas, or data structures change, tests for the old format are updated — not just deleted
- Migration tests verify: old format → new format works
- Keep one test for the migration path until the old format is fully deprecated
- Document in the test why the migration test exists and when it can be removed

### Refactors
- If a refactor doesn't change behavior, tests should still pass without modification — that's how you know the refactor is safe
- If tests break on a pure refactor, either the refactor changed behavior (fix it) or the tests were testing implementation details (fix the tests)

## Test reporting

### Always report results after running tests
- Format:
  ```
  Tests: X passed, Y failed, Z skipped
  ```
- If all pass, keep it brief
- If any fail, list each failure: test name, file, and a one-line reason

### Failed tests that are intentionally ignored
- If a failing test is skipped or ignored, state why:
  - Known issue with a linked issue number
  - Dependency not yet implemented
  - Flaky test under investigation
- Never silently skip — the user should always know what's being ignored and why
- Use the framework's skip mechanism (`.skip`, `@pytest.mark.skip`, `xit`) with a reason string

### What to report
```
Tests: 67 passed, 2 failed, 1 skipped

Failed:
  ✗ agent.test.ts > should retry on transient error
    — retry util (#17) not yet implemented
  ✗ config.test.ts > should validate webhook port
    — webhook config (#12) not yet merged

Skipped:
  ○ github-tools.test.ts > should handle rate limit
    — flaky, investigating (#45)
```

## Principles
- Tests prove the code works — not that it was written
- Test behavior, not implementation — don't assert internal state unless there's no observable output
- One assertion per test when practical — makes failures specific
- Test names describe what should happen: "should retry on 503", not "test retry"
- Fast tests run first, slow tests (API mocks, integration) run separately if needed
