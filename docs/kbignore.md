# .kbignore Patterns

Exclude files from `kb index` by placing a `.kbignore` in your source directory.

## How it works

- Patterns use **fnmatch** glob syntax (like shell globs, not full gitignore)
- Lines starting with `#` are comments, blank lines are ignored
- Each pattern matches against the **relative path** from the source directory and the **filename**
- Directory patterns end with `/` — matches any file under that directory
- Lookup: checks `<source-dir>/.kbignore`, then `<source-dir>/../.kbignore` (first match wins)

## Common patterns

### Obsidian / note-taking

```
.obsidian/
.trash/
templates/
*.excalidraw.md
```

### Development docs

```
node_modules/
vendor/
dist/
build/
_site/
.venv/
CHANGELOG.md
CHANGELOG-*.md
```

### Work-in-progress

```
drafts/
WIP-*
*.draft.md
*.tmp.md
TODO.md
```

### Generated / auto-created

```
*.gen.md
*.auto.md
api-reference/
_generated/
```

### Large files that chunk poorly

```
*.min.js
*.bundle.js
package-lock.json
yarn.lock
```

### Private / sensitive

```
internal/
private/
*-secret*
.env*
```

## Example `.kbignore`

A typical setup for a repo with docs, notes, and generated content:

```
# Tooling
.obsidian/
node_modules/
.venv/

# Not useful to index
CHANGELOG.md
LICENSE

# Work in progress
drafts/
*.draft.md
WIP-*

# Generated
_generated/
api-reference/
```

## Pattern matching details

| Pattern | Matches | Doesn't match |
|---|---|---|
| `drafts/` | `drafts/foo.md`, `drafts/sub/bar.md` | `my-drafts/foo.md` |
| `*.draft.md` | `notes.draft.md`, `sub/notes.draft.md` | `notes.md` |
| `WIP-*` | `WIP-feature.md` | `my-WIP-feature.md` (path) |
| `CHANGELOG.md` | `CHANGELOG.md` | `docs/CHANGELOG.md` (path only matches filename) |

Note: filename patterns (no `/`) match against both the full relative path and the bare filename. Directory patterns (trailing `/`) only match against the relative path prefix.
