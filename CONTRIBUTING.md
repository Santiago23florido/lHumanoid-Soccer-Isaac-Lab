# Contributing

## Nothing lands on `main` directly

`main` is the branch that is supposed to be reproducible. Every change reaches
it through a pull request, so that the diff is readable once before it becomes
the state of the project rather than after.

```bash
git switch -c <type>/<short-description>   # fix/, feat/, docs/, chore/, test/
# ... work, commit ...
git push -u origin HEAD
gh pr create --fill
gh pr merge --squash --delete-branch       # or merge on GitHub
```

This is enforced in two places, deliberately.

| Layer | Mechanism | Can be bypassed? |
| --- | --- | --- |
| Local | [`tools/githooks/pre-push`](tools/githooks/pre-push) | Yes — `--no-verify`, or `ALLOW_PUSH_TO_MAIN=yes` |
| Remote | GitHub branch ruleset, `main` | No, including for the repository owner |

The local hook exists for speed: it fails in a millisecond and prints the
commands to recover, before a mistake becomes a commit that has been pulled.
It is not the guarantee. A hook that lives in the working tree can always be
skipped, so the ruleset is what actually holds, and the hook is a courtesy to
whoever typed the wrong thing.

### Enabling the local hook

Hooks are not part of a checkout, so a fresh clone starts unguarded. One
command fixes that:

```bash
git config core.hooksPath tools/githooks
```

This points git at the versioned hook directory instead of `.git/hooks`, which
means the hook survives cloning and is reviewed like any other file. Verify it:

```bash
git push --dry-run origin HEAD:main    # must fail, with instructions
```

If that command succeeds, the hook is not installed. Note that git reports
`Everything up-to-date` without running any hook when there is nothing to
push, so test it with a branch that is genuinely ahead.

On Windows the hook must keep LF line endings or `sh` rejects it with
`bad interpreter`. [`.gitattributes`](.gitattributes) pins this; do not
override it.

### The remote ruleset

```bash
gh api /repos/:owner/:repo/rulesets --jq '.[] | "\(.name): \(.enforcement)"'
```

It requires a pull request for `main` and refuses deletion and force-push. It
has no bypass actors, so it applies to the owner too — which is the point. To
change it, edit the ruleset in the repository settings rather than working
around it.

## Checks before opening a pull request

```powershell
& "C:\Users\USER\miniconda3\shell\condabin\conda-hook.ps1"; conda activate env_isaaclab
python -m pytest -q          # no simulator required
ruff check .
ruff format --check .
```

Tests that need Isaac Sim are marked and skipped by default; the per-track
READMEs give the commands that run them.

## Conventions

- **Comments and documentation in English.** Code, docstrings, commit messages,
  markdown. The LaTeX book under `docs/` is the exception and is in Spanish.
- **Commit messages** are `<type>(<scope>): <imperative summary>`, with a body
  that says *why*. The reasoning is the part that is expensive to recover later.
- **Robot assets are not committed.** The NAO meshes are CC BY-NC-ND; see
  [`nao/README.md`](nao/README.md) for fetching them.
- **Numbers in documentation must be reproducible** from a script in the
  repository. A quoted figure with no derivation is a figure nobody can check.
