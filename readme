# Mastering Go — Learning AlphaGo Through Coding

A from-scratch implementation of the key building blocks behind AlphaGo / AlphaZero,
built on **PyTorch + CUDA** with fully batched tensor operations.

The goal is *not* to produce a competitive engine, but to deeply understand every
component — board representation, legal-move computation, MCTS, and eventually
neural-network-guided search — by writing it all by hand.

## Current progress

| Component | Status | Description |
|-----------|--------|-------------|
| Batched Go engine | Done | Full rules: captures, suicide, simple ko, Chinese scoring |
| Random agent | Done | Uniform random over legal moves + pass |
| MCTS agent | Done | PUCT selection, leaf reconstruction, terminal evaluation, sign-flip backup |
| Neural network policy/value | Not started | Next milestone — replace uniform prior and terminal eval with a trained net |
| Self-play training loop | Not started | Generate games via MCTS+NN, train the network, iterate |

## Repository layout

```
mastering_go_cat_RL_cuda/
│
├── engine/                        # Core Go rules engine (batched, GPU)
│   ├── stones.py                  #   Stone enum: EMPTY=0, BLACK=1, WHITE=2
│   ├── game_state.py              #   GameState dataclass (boards, to_play, pass_count, zobrist)
│   ├── game_state_machine.py      #   GameStateMachine — legal moves, state transitions, scoring
│   └── board_rules_checker.py     #   BoardRulesChecker — union-find groups, CSR captures, territory
│
├── agents/                        # Bot implementations
│   ├── random_bot.py              #   RandomBot — uniform sampling over legal actions
│   ├── mcts_bot.py                #   MCTSBot — clones state, runs MCTS, returns best move
│   ├── mcts_tree.py               #   MCTSTree — batched tensor tree (N, W, Q, P, child table)
│   └── mcts_ops.py                #   MCTS phases: selection, reconstruction, expansion, eval, backup
│
├── utils/                         # Shared helpers
│   ├── shared.py                  #   Device selection, @timed_method decorator, timing reports
│   ├── game_history.py            #   GameHistory — pre-allocated trajectory buffers
│   ├── board_history_tracker.py   #   Per-move snapshot + JSON export (older pipeline)
│   └── save_debug_history_minimal.py  # Minimal JSON export from GameHistory
│
├── duel_random_vs_mcts.py         # Driver: Random vs MCTS batched games
├── duel_random_vs_mcts_last_step.py
├── duel_random_vs_random.py       # Driver: Random vs Random batched games
├── duel_random_random_stable.py
├── test.py                        # Smoke test: manual stone placement + scoring
├── bench_workspace_alloc..py      # Micro-benchmark: alloc-per-call vs workspace reuse
│
├── frontend/
│   └── index.html                 # Browser-based Go board viewer (loads game JSON)
│
├── debug_games/                   # Recorded game JSONs (output of duel scripts)
│
├── NN/                            # [stale] Earlier MCTS experiment on old engine API
└── archieved/                     # [stale] Iterative snapshots of prior engine versions
```

## How it works

### 1. Engine — batched Go on the GPU

Everything operates on **B games in parallel** as `(B, H, W)` tensors.

`GameStateMachine` is the central object. It wraps a `GameState` and mutates it
in place through two main methods:

- **`legal_moves()`** — calls `BoardRulesChecker` to compute union-find groups,
  CSR-packed capture metadata, and per-move Zobrist candidate hashes, then
  applies a simple-ko filter. Returns `(B, H, W) bool`.

- **`state_transition(actions)`** — takes `(B, 2)` row/col moves (negative = pass),
  applies captures via the CSR data, places stones, updates Zobrist hashes,
  flips the player, and invalidates caches.

Scoring uses Chinese-style area counting (stones + surrounded territory).

### 2. Agents

**RandomBot** asks the engine for legal moves, builds a uniform distribution over
legal board points plus pass, and samples one action per game via `torch.multinomial`.

**MCTSBot** implements a standard PUCT search in six phases:

1. **Selection** — walk the tree from root using PUCT: `Q + c * P * sqrt(N_parent) / (1 + N_child)`
2. **Leaf reconstruction** — trace parent pointers to recover the action sequence,
   clone the root state for that single game, replay moves through a fresh engine
3. **Expansion** — compute legal moves at the leaf, assign a uniform prior
   *(this is where a neural network policy will plug in)*
4. **Evaluation** — score the leaf position via `game_outcomes(komi)`
   *(this is where a neural network value head will plug in)*
5. **Backup** — propagate the value up the path, flipping sign at each level
6. **Root action selection** — pick the action with the most visits

### 3. Duel scripts

Each `duel_*.py` script creates an empty batch of games, wires up two bots, and
runs a ply loop:

```
state = create_empty_game_state(B, H, device)
engine = GameStateMachine(state)

for ply in range(max_plies):
    moves = bot.select_moves(engine)
    engine.state_transition(moves)

scores = engine.compute_scores(komi)
```

Game trajectories are optionally saved as JSON for inspection in the browser viewer.

## Running

```bash
# Run a Random-vs-Random duel (16 games, 9x9 board, 10 plies)
python duel_random_random_stable.py

# Run a Random-vs-MCTS duel (16 games, 5x5 board, 50 plies)
python duel_random_vs_mcts.py

# Smoke test: manual scoring check
python test.py

# Serve the game viewer (then open http://localhost:8001/frontend/)
python3 -m http.server 8001
```

## Dependencies

- Python 3.10+
- PyTorch (with CUDA for GPU acceleration; CPU and MPS also work)

## Roadmap

The long-term arc follows the AlphaGo → AlphaZero progression:

1. ~~Batched Go engine with full rules~~
2. ~~Random agent baseline~~
3. ~~MCTS with terminal evaluation~~
4. **Neural network (policy + value head)** — replace uniform prior and terminal eval
5. **Self-play data generation** — MCTS + NN plays against itself, stores trajectories
6. **Training loop** — supervised learning on self-play data, iterating policy/value
7. **Evaluation** — pit new network against previous version
