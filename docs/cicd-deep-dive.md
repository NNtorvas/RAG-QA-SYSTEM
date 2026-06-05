# CI/CD Deep Dive — RAG QA System

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Layer 1 — Local Gates (pre-commit)](#2-layer-1--local-gates-pre-commit)
3. [Layer 2 — Pre-Push Gate (version check)](#3-layer-2--pre-push-gate-version-check)
4. [Layer 3 — Remote Pipeline (GitHub Actions)](#4-layer-3--remote-pipeline-github-actions)
   - [cd.yml — The Orchestrator](#cdyml--the-orchestrator)
   - [_prep.yml — Version Gate + Tagging](#_prepyml--version-gate--tagging)
   - [_build-push.yml — Docker Build & Push](#_build-pushyml--docker-build--push)
5. [Why This Is a Gold Standard](#5-why-this-is-a-gold-standard)
6. [Full Worked Example: Shipping Version 0.6.0](#6-full-worked-example-shipping-version-060)
7. [What Each Piece Protects Against](#7-what-each-piece-protects-against)

---

## 1. The Big Picture

The pipeline has **three independent layers of defence**, each catching problems at the earliest possible point:

```
Developer machine                    GitHub
─────────────────────────────────    ──────────────────────────────────────────
[1] git commit  →  pre-commit hooks  │
                   • formatting      │
                   • linting         │
                   • file hygiene    │
                                     │
[2] git push    →  pre-push hook     │
                   • version bump    │  [3] GitHub Actions
                                    ├──→  cd.yml (orchestrator)
                                     │       └─ _prep.yml
                                     │            • extract version
                                     │            • validate semver
                                     │            • create git tag
                                     │       └─ _build-push.yml
                                     │            • build Docker images
                                     │            • push to GHCR (3 tags)
```

**Why three layers?** Because fixing a problem costs more the later it is found:
- A formatting error caught at commit takes 2 seconds to fix.
- A version bump missed at push takes 10 seconds.
- A broken image discovered after it is already tagged "latest" in production is a rollback incident.

---

## 2. Layer 1 — Local Gates (pre-commit)

### What runs it

`pre-commit` hooks are configured in `.pre-commit-config.yaml` and installed once via:

```bash
make hooks
# which runs: pre-commit install --hook-type pre-commit --hook-type pre-push
```

After that, every `git commit` and `git push` automatically runs the relevant hooks —
the developer does not have to remember to run anything.

### What runs on `git commit`

#### File hygiene (pre-commit-hooks)

```yaml
- id: trailing-whitespace
- id: end-of-file-fixer
- id: check-yaml
- id: check-added-large-files
  args: ["--maxkb=500"]
```

These are cosmetic but important. Trailing whitespace creates noisy diffs.
A YAML file with a syntax error silently breaking a workflow is a real failure mode.
The 500 KB limit prevents accidentally committing model weights, datasets, or generated
embeddings into the repository.

#### Black (auto-formatter)

```yaml
- repo: https://github.com/psf/black
  rev: 24.4.2
  hooks:
    - id: black
      language_version: python3
```

Black **rewrites** the file in place before the commit is recorded. The developer does
not get a diff to review and reject — the file is simply corrected. This is intentional:
style debates become impossible when the formatter is non-negotiable.

Configuration lives in `pyproject.toml`:

```toml
[tool.black]
line-length = 110
skip_string_normalization = true
```

`skip_string_normalization = true` means Black will not change `'single'` to `"double"`
quotes. This preserves the author's string style while still enforcing everything else.

#### Flake8 (linter)

```yaml
- id: flake8
  additional_dependencies: [flake8-pyproject]
```

The `flake8-pyproject` plugin is the critical detail: without it, Flake8 reads its config
from `setup.cfg` or `.flake8`. With it, the config comes from `pyproject.toml`,
keeping all tooling config in one file:

```toml
[tool.flake8]
max-line-length = 110
extend-ignore = ["E203", "W503", "W504"]
```

`E203/W503/W504` are ignored because Black's line-wrapping style conflicts with them —
this is the standard set of ignores for Black + Flake8 co-existence.

### What runs on `git push`

Two hooks are pinned to `stages: [pre-push]`, meaning they only fire when pushing, not
on every commit (they are too slow or too semantic to run on each save).

#### Version bump check

```yaml
- repo: local
  hooks:
    - id: version-bump-check
      language: system
      entry: python scripts/check_version_bump.py
      stages: [pre-push]
      pass_filenames: false
      always_run: true
```

`always_run: true` bypasses the usual "only run if staged files match" filter — this hook
runs on every push regardless of what changed. See Layer 2 below for the full logic.

---

## 3. Layer 2 — Pre-Push Gate (version check)

`scripts/check_version_bump.py` enforces that `backend/__version__.py` is always bumped
before anything reaches `main`.

### How it works

```python
def get_local_version() -> str:
    return _read_version(Path(VERSION_FILE).read_text())

def get_remote_version() -> str | None:
    result = subprocess.run(
        ["git", "show", f"origin/main:{VERSION_FILE}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None               # first push — no remote yet
    return _read_version(result.stdout)
```

`git show origin/main:backend/__version__.py` reads the file **as it exists on the remote
right now**, without fetching the whole repo. This is efficient and always compares
against the live state of `main`, not a stale local copy.

`_read_version` uses `exec()` to evaluate the Python file rather than regex-parsing it.
This means it correctly handles any legal Python that assigns `__version__`, not just a
bare string literal.

```python
def main() -> int:
    local = get_local_version()
    remote = get_remote_version()

    if remote is None:
        print(f"[version-gate] No remote version found — first push. Local: {local}")
        return 0                  # safe: can't compare, allow through

    if parse_semver(local) <= parse_semver(remote):
        print(
            f"[version-gate] FAIL: bump __version__ before pushing.\n"
            f"  remote/main : {remote}\n"
            f"  local       : {local}"
        )
        return 1                  # exit 1 aborts the push

    print(f"[version-gate] OK: {remote} -> {local}")
    return 0
```

`parse_semver` converts `"0.5.0"` into `(0, 5, 0)`. Tuple comparison in Python is
lexicographic, so `(0, 5, 0) < (0, 6, 0)` is `True` — a correct semver ordering.

**Example failure:**
```
remote/main : 0.5.0
local       : 0.5.0   ← same, not bumped
→ exit 1, push aborted
```

**Example success:**
```
remote/main : 0.5.0
local       : 0.6.0   ← bumped
→ exit 0, push proceeds
```

---

## 4. Layer 3 — Remote Pipeline (GitHub Actions)

### cd.yml — The Orchestrator

```yaml
on:
  push:
    branches: [main]
  workflow_dispatch:
```

Two triggers:
- `push` to `main` — normal delivery path.
- `workflow_dispatch` — allows manually triggering the pipeline from the GitHub UI without
  committing. Useful for re-running a delivery after a transient failure.

```yaml
jobs:
  prepare:
    uses: ./.github/workflows/_prep.yml
    permissions:
      contents: write

  build-push:
    needs: prepare
    uses: ./.github/workflows/_build-push.yml
    with:
      version: ${{ needs.prepare.outputs.version }}
    secrets: inherit
    permissions:
      contents: read
      packages: write
```

**Key architectural decision: reusable workflows.**
`_prep.yml` and `_build-push.yml` use `on: workflow_call` — they are callable sub-workflows,
not independent pipelines. `cd.yml` is a pure orchestrator: it calls them in order and
wires the output of one (`version`) into the input of the other.

**Why split them?**
- Each workflow has exactly the permissions it needs and nothing more (see "Least Privilege" below).
- `_prep.yml` can be tested independently or reused by a future CI workflow.
- If the build fails, you can re-run only the build job without re-running the tag step.
- The split makes the dependency graph explicit: build cannot start before tagging succeeds.

`secrets: inherit` passes all repository secrets to `_build-push.yml` so it can access
`GITHUB_TOKEN` for GHCR authentication without having to thread secrets through explicitly.

---

### _prep.yml — Version Gate + Tagging

#### Step 1: Checkout with full history

```yaml
- uses: actions/checkout@v4
  with:
    fetch-depth: 0
```

By default, `actions/checkout` does a **shallow clone** (depth=1) for speed. That is fine
for most steps, but `git describe --tags --abbrev=0` needs to walk back through history to
find the most recent tag. `fetch-depth: 0` fetches the entire history, enabling this.

#### Step 2: Extract version

```yaml
- name: Extract version from __version__.py
  id: extract
  run: |
    VERSION=$(python -c "exec(open('backend/__version__.py').read()); print(__version__)")
    echo "version=$VERSION" >> $GITHUB_OUTPUT
```

`exec(open(...).read())` evaluates the file as Python code. `print(__version__)` captures
the result to stdout, which is assigned to the `VERSION` shell variable.

`echo "version=$VERSION" >> $GITHUB_OUTPUT` is the GitHub Actions way to expose a step
output. After this, other steps and other jobs can reference it as
`${{ steps.extract.outputs.version }}` or `${{ needs.prepare.outputs.version }}`.

#### Step 3: Validate semver bump against latest git tag

```yaml
- name: Validate semver bump against latest tag
  run: |
    CURR=${{ steps.extract.outputs.version }}
    PREV_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0")
    PREV="${PREV_TAG#v}"
    python -c "
    import sys
    curr = tuple(int(x) for x in '${CURR}'.split('.'))
    prev = tuple(int(x) for x in '${PREV}'.split('.'))
    if curr <= prev:
        print(f'[version-gate] FAIL: ...')
        sys.exit(1)
    print(f'[version-gate] OK: ...')
    "
```

`git describe --tags --abbrev=0` finds the most recent annotated tag reachable from HEAD.
The `2>/dev/null` suppresses the error message when no tags exist (first deploy).
`|| echo "v0.0.0"` is the fallback — if no tags exist, treat the previous version as
`0.0.0`, which any real version will be greater than.

`"${PREV_TAG#v}"` is bash parameter expansion that strips a leading `v` prefix.
So `v0.5.0` becomes `0.5.0`. This normalises tags whether they were created with or
without the prefix.

**Why validate here when the pre-push hook already checked?**
The pre-push hook can be skipped with `git push --no-verify`. This server-side check is
the authoritative gate. The hook is developer convenience; the CI check is enforcement.

#### Step 4: Create annotated git tag

```yaml
- name: Create annotated git tag on last non-merge commit
  run: |
    VERSION=${{ steps.extract.outputs.version }}
    COMMIT=$(git rev-list --max-count=1 --no-merges HEAD)
    git config user.name "GitHub Actions"
    git config user.email "github-actions@users.noreply.github.com"
    git tag -a "v${VERSION}" "${COMMIT}" -m "Release v${VERSION}"
    git push origin "v${VERSION}"
```

`git rev-list --max-count=1 --no-merges HEAD` finds the most recent non-merge commit.
When a PR is merged via a merge commit, the HEAD is that merge commit. Tagging the
non-merge commit ensures the tag points to the actual feature work, not the merge commit
itself.

`git tag -a` creates an **annotated** tag, not a lightweight tag. Annotated tags store the
tagger name, email, date, and message — they are proper objects in the Git object database.
`git describe` only finds annotated tags (unless `--tags` is passed), which is why this
distinction matters for the validation step.

---

### _build-push.yml — Docker Build & Push

#### Least-privilege permissions

```yaml
permissions:
  contents: read
  packages: write
```

- `contents: read` — only needs to read the code for the build context.
- `packages: write` — needed to push images to GHCR.

The job has no `contents: write` — it cannot create tags, push commits, or modify the repo.
This is the principle of least privilege: each job gets exactly the permissions its tasks require.

#### Registry login without a stored secret

```yaml
- name: Log in to GHCR
  uses: docker/login-action@v3
  with:
    registry: ghcr.io
    username: ${{ github.actor }}
    password: ${{ secrets.GITHUB_TOKEN }}
```

`GITHUB_TOKEN` is a short-lived token created automatically by GitHub for each workflow run.
It requires no configuration, rotates automatically, and is scoped to this repository only.
There is no long-lived credential stored anywhere.

#### Normalising the image owner to lowercase

```yaml
- name: Set versioned image references
  id: refs
  run: |
    OWNER=$(echo "${{ github.repository_owner }}" | tr '[:upper:]' '[:lower:]')
    BASE="ghcr.io/${OWNER}/rag-qa-system"
    echo "backend=${BASE}-backend:${{ inputs.version }}" >> $GITHUB_OUTPUT
    echo "frontend=${BASE}-frontend:${{ inputs.version }}" >> $GITHUB_OUTPUT
    echo "backend_cache=${BASE}-backend:buildcache" >> $GITHUB_OUTPUT
    echo "frontend_cache=${BASE}-frontend:buildcache" >> $GITHUB_OUTPUT
```

`github.repository_owner` preserves the case from the GitHub account name (e.g., `NNtorvas`).
Docker registry paths must be lowercase. `tr '[:upper:]' '[:lower:]'` normalises it.

This step also centralises all image name construction. Every subsequent step that needs
a registry path reads it from `steps.refs.outputs.*` rather than constructing its own string.
If the naming convention changes, one step changes — not five.

#### Three image tags per push

The `docker/metadata-action` step produces three tags for each image:

```yaml
tags: |
  type=raw,value=${{ inputs.version }}    # e.g. 0.6.0
  type=sha,format=short                   # e.g. sha-b38fb38
  type=raw,value=latest,enable={{is_default_branch}}
```

| Tag | Example | Use case |
|-----|---------|----------|
| Version | `0.6.0` | Pin a specific release. Immutable — never overwritten. |
| SHA | `sha-b38fb38` | Trace exactly which commit produced this image. |
| `latest` | `latest` | Always points to the most recent push to `main`. Only set on the default branch. |

The SHA tag is the most operationally important for debugging. When something is wrong in
production, you can read the image tag, look up the SHA in git, and find the exact code.

#### Registry-based layer cache

```yaml
cache-from: type=registry,ref=${{ steps.refs.outputs.backend_cache }}
cache-to:   type=registry,ref=${{ steps.refs.outputs.backend_cache }},mode=max,image-manifest=true,oci-mediatypes=true
```

Docker build layers are cached in GHCR as a separate manifest
(`ghcr.io/nntorvas/rag-qa-system-backend:buildcache`).

**Why registry cache instead of GitHub Actions cache?**
GitHub Actions cache has a 10 GB limit shared across all workflows. Docker layer caches
for Python images can be large. Registry cache lives in GHCR (separate quota) and is
shared between any runner that can pull from that registry — including local `docker build`
invocations if you authenticate.

`mode=max` caches every intermediate layer, not just the final image layers. This
maximises cache hits when only late-stage layers (like `COPY . .`) change.

`image-manifest=true,oci-mediatypes=true` makes the cache manifest OCI-compliant,
enabling broader compatibility.

`provenance: false` suppresses the automatic SBOM/provenance attestation that
`docker/build-push-action@v6` adds by default. This avoids a known issue where
multi-platform attestations can confuse some older registry clients.

#### CVE scanning (not currently active)

Docker image CVE scanning is not enabled in this pipeline. When re-adding it, the standard
approach is [Trivy](https://github.com/aquasecurity/trivy-action):

```yaml
- name: Backend — Trivy CVE scan
  id: trivy-backend
  continue-on-error: true
  uses: aquasecurity/trivy-action@v0.36.0
  with:
    image-ref: ${{ steps.refs.outputs.backend }}
    format: sarif
    output: trivy-backend.sarif
    severity: CRITICAL,HIGH
    ignore-unfixed: true
    exit-code: 1

- name: Backend — upload SARIF to GitHub Security
  if: always()
  uses: github/codeql-action/upload-sarif@v4
  with:
    sarif_file: trivy-backend.sarif
    category: trivy-backend
```

The `security-events: write` permission on the job is also required for the SARIF upload.
Repeat both steps for the frontend image, then add a gate step at the end:

```yaml
- name: Fail if CVEs found
  if: always()
  run: |
    if [[ "${{ steps.trivy-backend.outcome }}" == "failure" || \
          "${{ steps.trivy-frontend.outcome }}" == "failure" ]]; then
      echo "Unfixed CRITICAL/HIGH CVEs found — see the Security tab for details."
      exit 1
    fi
```

Key design points when adding this back:
- `continue-on-error: true` on each scan step keeps the job alive so both images are always
  fully scanned and their SARIF files uploaded before the gate delivers a verdict.
- `ignore-unfixed: true` avoids blocking on CVEs that have no available fix — only report
  what the team can actually act on.
- The gate step reads `steps.<id>.outcome`, which reflects what Trivy found regardless of
  `continue-on-error`.

---

## 5. Why This Is a Gold Standard

### Defence in depth — the same gate at two levels

The version bump check runs both locally (pre-push hook, `scripts/check_version_bump.py`)
and remotely (`_prep.yml` validation step). The local check is fast feedback for the
developer. The remote check cannot be bypassed with `--no-verify`. Neither alone is
sufficient.

### Single source of truth for the version

The version lives in exactly one place: `backend/__version__.py`. Every other system reads
it from there:

- `make version` → `exec(open('backend/__version__.py').read()); print(__version__)`
- `_prep.yml` extract step → same exec pattern
- `_prep.yml` tag step → `${{ steps.extract.outputs.version }}`
- Docker image tags → `${{ inputs.version }}` (passed from prep output)

There is no chance of a `pyproject.toml` version getting out of sync with the image tag.

### Reusable workflows enforce separation of concerns

`_prep.yml` does only two things: validate the version and create a tag.
`_build-push.yml` does only two things: build Docker images and scan them.
`cd.yml` does only one thing: call them in order and wire outputs to inputs.

This mirrors the single-responsibility principle. You can read any one file and completely
understand what it does without reading the others.

### Least privilege throughout

| Workflow | Permissions | Why minimal |
|----------|-------------|-------------|
| `cd.yml` calling `_prep.yml` | `contents: write` | Needs to push the git tag |
| `cd.yml` calling `_build-push.yml` | `contents: read`, `packages: write` | Can't touch the repo, can't create tags |

If the build workflow was compromised (e.g., a malicious action in the supply chain),
it could not write back to the repository. The blast radius is limited.

### No long-lived credentials

`GITHUB_TOKEN` is scoped to the run, auto-rotates, and requires no management.
GHCR authentication uses this token exclusively. There are no service accounts, deploy
keys, or stored PATs.

### Build reproducibility

Every image push generates a version tag and a SHA tag. The SHA tag is permanent —
it is never overwritten. Six months from now, you can run:

```bash
docker pull ghcr.io/nntorvas/rag-qa-system-backend:sha-b38fb38
```

and get exactly the image that was built from commit `b38fb38`. This is essential for
incident investigation, rollback, and audit.

---

## 6. Full Worked Example: Shipping Version 0.6.0

**Before pushing:**

```
backend/__version__.py: __version__ = "0.5.0"   ← currently on main
```

**Step 1 — Developer bumps the version locally:**

```python
# backend/__version__.py
__version__ = "0.6.0"
```

**Step 2 — Developer commits:**

```
git commit -m "add streaming response support"
```

Pre-commit hooks fire:
```
Trim Trailing Whitespace...............................Passed
Fix End of Files.......................................Passed
Check Yaml.............................................Passed
Check for added large files............................Passed
black..................................................Passed
flake8.................................................Passed
```

The commit is recorded.

**Step 3 — Developer pushes:**

```
git push origin main
```

Pre-push hooks fire:

```
Version bump check.....................................Passed
  [version-gate] OK: 0.5.0 -> 0.6.0
```

Push proceeds to GitHub.

**Step 4 — GitHub Actions: cd.yml triggers**

Job `prepare` starts, calling `_prep.yml`.

**Step 5 — _prep.yml: Extract version**

```bash
VERSION=$(python -c "exec(open('backend/__version__.py').read()); print(__version__)")
# VERSION = "0.6.0"
echo "version=0.6.0" >> $GITHUB_OUTPUT
```

**Step 6 — _prep.yml: Validate semver**

```bash
CURR=0.6.0
PREV_TAG=$(git describe --tags --abbrev=0)   # returns "v0.5.0"
PREV="0.5.0"                                 # strip "v" prefix

# Python comparison:
# curr = (0, 6, 0)
# prev = (0, 5, 0)
# (0, 6, 0) > (0, 5, 0) → OK
```

Output: `[version-gate] OK: 0.5.0 -> 0.6.0`

**Step 7 — _prep.yml: Create git tag**

```bash
COMMIT=$(git rev-list --max-count=1 --no-merges HEAD)
# COMMIT = "b38fb38b058c6999a1cbfc22ae957c0bafe9693e"

git tag -a "v0.6.0" "b38fb38..." -m "Release v0.6.0"
git push origin "v0.6.0"
```

A new annotated tag `v0.6.0` now exists in the repository, pointing at the feature commit
(not any merge commit).

**Step 8 — cd.yml: `prepare` outputs `version=0.6.0`, job `build-push` starts**

**Step 9 — _build-push.yml: Set image references**

```bash
OWNER="nntorvas"   # lowercased from "NNtorvas"
BASE="ghcr.io/nntorvas/rag-qa-system"

backend=ghcr.io/nntorvas/rag-qa-system-backend:0.6.0
frontend=ghcr.io/nntorvas/rag-qa-system-frontend:0.6.0
backend_cache=ghcr.io/nntorvas/rag-qa-system-backend:buildcache
frontend_cache=ghcr.io/nntorvas/rag-qa-system-frontend:buildcache
```

**Step 10 — _build-push.yml: docker/metadata-action produces tags**

For the backend image:
```
ghcr.io/nntorvas/rag-qa-system-backend:0.6.0
ghcr.io/nntorvas/rag-qa-system-backend:sha-b38fb38
ghcr.io/nntorvas/rag-qa-system-backend:latest
```

**Step 11 — _build-push.yml: Build and push**

Docker BuildKit reads from the registry cache, rebuilds only changed layers,
and pushes all three tags to GHCR.

**Step 12 — _build-push.yml: Build and push frontend image**

Same pattern as the backend: metadata action produces three tags, build-push-action builds
and pushes using the registry cache.

**Final state:**

| Artifact | Value |
|----------|-------|
| Git tag | `v0.6.0` on commit `b38fb38` |
| Backend images | `:0.6.0`, `:sha-b38fb38`, `:latest` |
| Frontend images | `:0.6.0`, `:sha-b38fb38`, `:latest` |
| Job result | Success |

---

## 7. What Each Piece Protects Against

| Failure scenario | Caught by |
|-----------------|-----------|
| Committing unformatted Python | Black (pre-commit) |
| Committing code with lint errors | Flake8 (pre-commit) |
| Committing trailing whitespace | pre-commit-hooks (pre-commit) |
| Accidentally committing a 200 MB model file | check-added-large-files (pre-commit) |
| Pushing to main without bumping version | check_version_bump.py (pre-push) + _prep.yml (remote) |
| Bypassing local hooks with `--no-verify` | _prep.yml semver check (remote, cannot be bypassed) |
| Losing track of which commit maps to which image | SHA tag on every image |
| Version tag going to a merge commit instead of feature commit | `--no-merges` in `git rev-list` |
| Registry path failing because GitHub username has capitals | `tr '[:upper:]' '[:lower:]'` in refs step |
| Build cache filling GitHub Actions cache quota | Registry-based cache in GHCR |
| Over-privileged CI job being exploited | Per-job least-privilege permissions |
