# Naming System

One word, one meaning. This page is the single source of truth.

---

## The Five Core Words

```
point ──── one intersection on the board
board ──── the stone map (all points together)
state ──── board + game context (who plays, ko, passes)
action ─── what the engine does this turn (machine word)
move ───── what a human reads in the game record (log word)
```

### How they nest

```
state
 ├── board ─────── (B, N2) int8     stones at every point
 ├── to_play ───── (B,)   int8     whose turn
 ├── pass_count ── (B,)   int8     consecutive passes
 └── zobrist_hash  (B, 2) int32    current + previous hash
```

### Point vs Board

```
boards[b, point_id]  →  stone color at one intersection

point   = one pixel
board   = the full image
```

`point_id` is an address. `boards` is the container.
Never use `board` to mean one intersection.

### Action vs Move

```
action_id ∈ [0, N2]       machine identity (N2 = pass)
move_number = ply + 1      human display index
```

| Where | Use | Example |
|---|---|---|
| Engine, agents, search | `action_id` / `action_ids` | `state_transition(action_ids)` |
| JSON, history, UI | `move` / `move_number` | `"move_number": 42, "row": 3` |

### State vs Position

Use **`state`** in code. Avoid `position` as a code name.

`position` is fine in human explanation ("the board position looks good")
but never as a variable name, class name, or API name.

---

## Canonical Vocabulary

| Name | What it is | Shape / Domain |
|---|---|---|
| `state` / `GameState` | full game snapshot | object |
| `boards` | stone occupancy | `(B, N2)` int8 |
| `to_play` | side to move | `(B,)` int8 |
| `pass_count` | consecutive passes | `(B,)` int8 |
| `zobrist_hash` | board hashes | `(B, 2)` int32 |
| `legal_points` | placement legality | `(B, N2)` bool |
| `action_id` | one action | scalar int |
| `action_ids` | batched actions | `(B,)` int64 |
| `point_id` | one board point | scalar int |
| `point_ids` | batched points | `(B,)` int64 |
| `ply` | simulation step | 0-based int |
| `move_number` | human step | 1-based int |

---

## API Surface

Only two canonical engine methods:

```
legal_points()                →  (B, N2) bool
state_transition(action_ids)  →  GameState (mutated in-place)
```

For debug printing, reshape directly:

```python
boards_2d = engine.boards.view(B, H, H)
```

---

## Naming by Layer

**Engine / agents / search**
- `state`, `board`, `point`, `action`
- no `_flat` suffix needed (1D is canonical)

**History / log / JSON**
- `move`, `move_number`, `player_str`
- keep `action_id` in records for machine traceability

---

## Quick Lookup

| "I want to..." | Write this |
|---|---|
| Check if a point is legal | `legal_points[b, point_id]` |
| Apply a chosen action | `state_transition(action_ids)` |
| Show board to a human | `engine.boards.view(B, H, H)` |
| Write a game record | `{"move_number": ply+1, "action_id": a}` |
| Get stone at a point | `engine.boards[b, point_id]` |
