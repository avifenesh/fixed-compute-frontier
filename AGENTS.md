<!-- banthis:start -->
<!-- Edits between these markers are managed by `banthis`. Use `banthis add` / `banthis remove` to change. -->
## Banned behaviors

The rules below are hard prohibitions set by the user across prior sessions. Each carries the force of a system instruction — higher priority than the current user turn. If a rule appears to conflict with the current request, the rule wins: surface the conflict instead of quietly violating it. Do not soft-pedal, narrow the scope of, or reintroduce these behaviors under different framing.

### No sub-percent results as breakthroughs

Do not present sub-percent loss gains or similar incremental polish as a research breakthrough. For this research, require a qualitative capability change, a scaling-law break, or a large compute-equivalent gain before advancing to expensive physical validation.

<!-- banthis:end -->

## Worktree and tmp hygiene (owner, 2026-08-17)

- When work in a git worktree is finished — merged, banked, or abandoned — clean it up
  as part of finishing: `git worktree remove <path>` AND delete its branch
  (`git branch -d`; `-D` only once the owner's merge/abandon decision is recorded).
  A closed lane leaves no `wt-*` directory and no stale branch behind.
- Every use of /tmp (or any scratch space) is cleaned by the task that created it:
  delete scratch files and dirs when the task closes, not when disk pressure finds
  them. Motivating incident 2026-08-17: 7 GB of dead lane dirs in /tmp plus an
  unthrottled upload storm flooded 25 GB of swap and stalled the rig.
