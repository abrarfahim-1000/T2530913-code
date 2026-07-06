# What We're Doing — Plain English Overview

## The Problem

Power grids can develop faults — overloads, line trips, cascading failures. A GNN (a neural network built to understand networks/graphs) can be trained to detect and locate these faults automatically.

But an AI trained on one grid layout doesn't reliably generalize to a different layout it's never seen. On an unfamiliar grid it can give a confident but wrong, or physically impossible, answer — dangerous in a safety-critical system.

## Our Core Idea

We built a closed-loop system: safety rules get extracted from official standards documents, turned into runtime checks, and used to hard-gate every prediction a fault-detecting AI makes — blocking it and citing the exact rule violated whenever a prediction doesn't check out. No prior published system closes this loop end-to-end (extraction → compiled runtime logic → gating the AI at inference → citations). That closed loop is the contribution.

The AI's unreliability on unfamiliar grids is the *reason* this safety layer is needed — not the contribution itself.

## Step by Step: How We Build This

**Step 1 — Teach the AI to read fault patterns.**
We train the GNN on a simulated 36-node power grid, feeding it thousands of examples of normal operation, overloads, line trips, and cascading failures. It learns to classify what kind of fault is happening and roughly where.

**Step 2 — Build the rulebook, automatically.**
Instead of manually reading through hundreds of pages of official grid safety standards (IEEE documents, grid operation manuals) and hand-coding rules — which is slow and error-prone — we use an AI language model to read those documents and extract the rules itself, in a structured format. A second AI model then double-checks each extracted rule against the original text to make sure nothing was misread or invented.

**Step 3 — Store the rulebook as a connected map.**
The extracted rules, along with the grid's own layout (which line connects to which node, etc.), are stored in a "knowledge graph" — essentially a structured, queryable map of how everything relates to everything else.

**Step 4 — Build the safety checker (the "shield").**
This is the core piece. Every time the trained AI (Step 1) makes a prediction, the shield independently checks that prediction against the rulebook (Step 3) and against basic physical laws (like: current in must equal current out). If the prediction passes, it's allowed through. If it fails, it's blocked — and the system explains exactly which rule it violated and where that rule came from.

**Step 5 — Provide the evidence that the shield is actually necessary.**
The system (Steps 1–4) is the contribution. This step is the evidence for why it matters. We take the AI trained only on the 36-node grid and test it — frozen, no retraining — on completely different, larger grid layouts (14, 57, and 118-node grids) it's never seen. We expect its accuracy to degrade there. Then we measure how much of that degradation the shield catches and blocks before it would reach a real decision. That measurement backs up the claim that a hard safety gate — not just trusting the AI's confidence — is required.

## What We're NOT Doing (on purpose)

- We're not trying to build something that fixes or repairs grid problems automatically — only *detects and flags* them. Repair is a separate, explicitly excluded problem.
- We're not deploying on a real grid — this is simulation-based research.
- We're not retraining the AI on the new grid layouts — the whole point is testing how it holds up *without* retraining.

## The One-Sentence Version

**"We built a closed-loop system that turns official safety standards into runtime checks that hard-block a fault-detecting AI whenever its prediction violates a rule or the physics of the grid — and we prove this gate is necessary, not optional, by showing how badly the AI fails on unfamiliar grid layouts without it."**
