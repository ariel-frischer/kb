---
name: release
description: Private autospec release workflow for syncing dev/prod/main across GitLab and GitHub, gating on GitHub Actions, fixing failed CI through dev, and preparing GoReleaser releases.
---

# autospec Release Workflow

Use this private skill when asked to release autospec, prepare a release, sync `dev` into `main`, publish to GitLab and GitHub, or recover from failed release CI.

User instructions for the current release take precedence over this workflow. Preserve the safety rules: do not use `git stash`, do not use `git add -A` or `git add .`, and do not rewrite shared branches unless the user explicitly asks.

## Current repo shape

- GitLab remote: `origin` (`git@gitlab.com:ariel-frischer/autospec.git`)
- GitHub remote: `gh` (`git@github.com:ariel-frischer/autospec.git`)
- Main development branch: `dev`
- Release branch: `main`
- Optional staging branch: `prod` only if it exists locally/remotely or the user explicitly asks for it.
- GitHub `CI` and docs workflows run on pushes to `main`.
- GitHub `Release` workflow runs on `v*.*.*` tags and uses GoReleaser.
- GoReleaser publishes GitHub releases from `.goreleaser.yml`.
- GitLab CI runs on `main` and `dev`; `dev` jobs are manual.
- `dev` is GitLab-only. Never push `dev` to the GitHub `gh` remote.

## Preflight

1. Read the user's release instructions and extract any required version, branch, changelog, or hotfix constraints.
2. Check repo state:
   ```bash
   git status --short --branch
   git remote -v
   git fetch origin --prune --tags
   git fetch gh --prune --tags
   gh auth status
   glab auth status
   ```
3. If there are unrelated local changes, do not overwrite them. Either work around them or ask before proceeding.
4. Validate locally before merging:
   ```bash
   make fmt
   make lint
   make test
   make build
   ```

## Sync Branches

Use `origin` as the GitLab remote and `gh` as the GitHub remote. Keep release branches and tags aligned across both remotes, but keep `dev` on GitLab only.

1. Update `dev`:
   ```bash
   git switch dev
   git pull --ff-only origin dev
   ```
2. If `prod` exists or the user requested it, merge `dev` into `prod` first:
   ```bash
   git switch prod || git switch -c prod origin/prod
   git pull --ff-only origin prod
   git merge --no-ff dev
   make fmt
   make lint
   make test
   make build
   git push origin prod
   git push gh prod
   ```
   If `prod` does not exist and was not requested, use `dev` as the source branch for `main`.
3. Merge the selected source branch into `main`:
   ```bash
   git switch main
   git pull --ff-only origin main
   git merge --no-ff <source-branch>
   make fmt
   make lint
   make test
   make build
   git push origin main
   git push gh main
   ```

## GitHub CI Gate

After pushing `main`, wait for every GitHub Actions run for the pushed SHA to finish.

```bash
sha=$(git rev-parse HEAD)
gh run list --branch main --commit "$sha" --limit 20 --json databaseId,workflowName,status,conclusion,headSha
gh run watch <run-id> --exit-status
```

Watch all relevant runs for the SHA, including `CI` and docs if docs changed. If any required run fails:

1. Inspect the failing logs:
   ```bash
   gh run view <run-id> --log-failed
   ```
2. Fix on `dev`, not directly on `main`, unless the user explicitly asked for a direct hotfix.
3. Commit only the files changed for the fix:
   ```bash
   git status --short
   git add <specific-files>
   oco -c "fix release CI failure"
   ```
4. Repeat the branch sync and GitHub CI gate until `main` is green on GitHub.
5. Merge the CI fix back through any branch used in the flow so `dev`, optional `prod`, and `main` all contain the fix. Push `dev` only to `origin`.

## Prepare Release

1. Select the release version from user instructions, or inspect:
   ```bash
   make version
   git tag --sort=-version:refname | head -10
   ```
2. Ensure `internal/changelog/changelog.yaml` has a release entry for the selected version and date. Move relevant `unreleased` bullets into that version when preparing a real release.
3. Regenerate and verify changelog:
   ```bash
   make changelog-sync
   make changelog-check
   ```
4. Run the release build gate:
   ```bash
   make release-build VERSION=vX.Y.Z
   ```
5. Commit changelog/version-prep files if changed, merge them through the branch sync workflow, and wait for GitHub CI on `main` again. Push `dev` only to GitLab.

## GitLab Comparison MR

When the user wants to review all changes since the last GitHub release, create a GitLab MR for comparison only:

1. Find the latest GitHub release tag:
   ```bash
   gh release view --json tagName,publishedAt,url
   ```
2. Create and push two comparison branches without switching away from the current worktree:
   ```bash
   current_sha=$(git rev-parse HEAD)
   release_tag=vX.Y.Z
   safe_tag=${release_tag//./-}
   base_branch="compare/${safe_tag}-base"
   current_branch="compare/current-${current_sha:0:12}"
   git branch -f "$base_branch" "$release_tag"
   git branch -f "$current_branch" "$current_sha"
   git push -f origin "$base_branch:$base_branch"
   git push -f origin "$current_branch:$current_branch"
   ```
3. Create the MR from current to base:
   ```bash
   glab mr create \
     --source-branch "$current_branch" \
     --target-branch "$base_branch" \
     --title "Compare ${release_tag} to ${current_sha:0:12}" \
     --description "Comparison-only MR for reviewing all changes from the latest GitHub release (${release_tag}) to current commit ${current_sha}."
   ```
4. Do not merge this MR. Close/delete the comparison branches after the user is done reviewing.

## Publish Release

Prefer the GitHub tag workflow because `.github/workflows/release.yml` runs tests, extracts changelog notes, and invokes GoReleaser.

```bash
git switch main
git pull --ff-only origin main
git pull --ff-only gh main
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
git push gh vX.Y.Z
```

Then watch the GitHub release workflow:

```bash
gh run list --workflow release.yml --branch main --limit 10 --json databaseId,status,conclusion,headSha,event
gh run watch <run-id> --exit-status
gh release view vX.Y.Z --json tagName,url,body,assets
```

Verify the release body contains the curated changelog entries for the version, not only the generic installation/checksum template. The release workflow installs `chlog` in CI and generates the full release notes file before invoking GoReleaser.

If the release workflow fails, inspect logs, fix on `dev`, merge forward to `main`, delete/recreate the tag only with explicit user approval, and rerun the release.

## Final Checks

- `git status --short --branch` is clean or only has unrelated user changes.
- `origin/main` and `gh/main` point at the same commit.
- `dev` contains any release CI fixes and is pushed only to `origin/dev`.
- GitHub Actions for `main` are green.
- GitHub release `vX.Y.Z` exists with assets and changelog release notes.
- Add a bead comment with the release version, main SHA, CI run URL, and release URL when a bead is active.
