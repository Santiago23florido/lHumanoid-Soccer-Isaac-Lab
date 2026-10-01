# Working in this repository

## Never push to `main`

Not with a merge commit, not with a fast-forward, not "just this once because
the pull request was already reviewed". Work goes on a branch and reaches
`main` through a pull request, always.

```bash
git switch -c <type>/<short-description>
git push -u origin HEAD
gh pr create --fill
```

Merging the pull request is the user's call. Do not merge it, and do not ask
to merge it as a formality — opening it is where the hand-off is.

Specifically forbidden:

- `git push origin main`, in any form
- `git push --no-verify`, which exists only to skip the guard
- setting `ALLOW_PUSH_TO_MAIN`, which is the repository owner's escape hatch
  and not an agent's
- merging a branch into local `main` and pushing the result

[`tools/githooks/pre-push`](tools/githooks/pre-push) will refuse, and the
GitHub ruleset on `main` will refuse after it. Both refusals are correct. If a
push to `main` fails, the fix is to open a pull request, never to find a way
around the check.

**Why this is written down:** it had already happened. A branch was merged
into local `main` and pushed, which put five commits on the shared branch
without anything to read them against. No hook existed to stop it.

## Constraints that are not negotiable

- **A condition may define a task. It may not prescribe a strategy.**
  `nao_stand` terminates on a step, and that is correct: "recover without
  stepping" *is* the task it poses, and the zero-step envelope is its
  definition. Keep it. The walking task learned from the teacher poses an open
  objective instead — track the command, do not fall — so it must carry no
  condition on *how*. In particular it must not inherit the standing task's
  zero-step criterion: for a stepping skill that criterion measures how close
  to standing still a state is, not whether the student can reach it.
  Capturability there is **instrumentation** — compute it, log it, report the
  margin the gait chose — never a limit. See
  [`transfer/docs/plan.md`](transfer/docs/plan.md).
- **Robot assets are never committed.** The NAO meshes and USD are CC BY-NC-ND.
  Check `.gitignore` before adding anything under `nao/assets/`.
- **The PDFs under `docs/` are not committed.** Both are gitignored on purpose.
- **Comments and documentation in English**, including commit messages. The
  LaTeX book under `docs/` is in Spanish.

## Claims must be checkable

Every number in the documentation comes from a script in the repository. When
reporting a result, give the command that produces it. When a number turns out
to be wrong, correct it *and* say what the wrong one affected — this project
has already shipped a sampled extreme that was 6 % low and a joint count that
described a robot the asset is not.

Do not report work as verified unless the verification was run. `pytest` output
pasted from memory is not a test run.

## Environment

```powershell
& "C:\Users\USER\miniconda3\shell\condabin\conda-hook.ps1"; conda activate env_isaaclab
```

Isaac Lab 2.3.2 / Isaac Sim 5.1.0 / rsl-rl-lib 5.0.1 / torch 2.7.0+cu128.

- Tasks are registered by **this repository's** `scripts/train.py`, not by
  Isaac Lab's. Isaac Lab's own `train.py` does not import the extension and
  will report the task as unknown.
- `scripts/list_tasks.py` queries `gym.envs.registry`. Do not reintroduce a
  hardcoded list; the previous one drifted and omitted a whole track.
- Renderers run **windowed**. The headless RTX path segfaults in
  `rtx.scenedb` on this machine.
