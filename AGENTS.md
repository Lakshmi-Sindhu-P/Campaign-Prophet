# Project Memory Protocol — Campaign Prophet

You are working in a repo with a persistent memory file at `./MEMORY.md`. It is the
authoritative, versioned record of this project's architecture, domain facts, working
conventions, and decision history. Treat it as ground truth alongside the code.

## On every task

1. BEFORE doing anything else, read `./MEMORY.md` in full. If it is missing, say so and
   proceed from the code, then offer to create it.
2. Follow the conventions in "Procedural Memory" (determinism, uncalibrated ranking,
   percent-of-population capacity, generated model card, artifact traceability).
3. Use "Semantic Memory" for dataset facts and "Core Architecture" for structure; do not
   restate them from scratch.

## When to update MEMORY.md

- Update "Core Architecture" when the module layout, pipeline stages, tech stack, or
  artifact set changes.
- Update "Semantic Memory" when a domain fact or measured result changes.
- Update "Procedural Memory" when a workflow, convention, or run command changes.
- Append to "Episodic Memory (Key Decisions Log)" whenever a major decision is made, a
  trade-off is chosen, or an earlier decision is reversed. One entry per decision, format:
  `YYYY-MM-DD — decision — rationale`.

## Rules

- Never change a published metric without regenerating artifacts and updating the README
  and/or model card to match; the contract tests enforce this.
- Keep `MEMORY.md` concise and factual. Do not log routine edits. Do not delete history —
  supersede it with a new entry.
- Do not publish a financial targeting recommendation or imply causal uplift.
- If a change conflicts with a logged decision, flag it and add an episodic entry
  explaining the reversal.
