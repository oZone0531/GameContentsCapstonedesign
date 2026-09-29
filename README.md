# Bang RL Training System

This repository contains the `bang_rl` Python package for a five-player Bang RL environment, configurable DummyAgent behavior, MainAgent inference modules, dataset preparation, PPO training seams, evaluation protocols, and GUI-independent application services.

## Installation

Run these commands from the repository root (`GameContentsCapstonedesign`), so the relative paths in the example configuration resolve predictably.

```powershell
# Headless learning, data, evaluation, and test environment
python -m pip install -r requirements.lock

# Optional editable package install with test tooling
python -m pip install -e ".[test]"

# Optional PyQt6/pyqtgraph GUI environment
python -m pip install -r requirements-gui.lock
# or: python -m pip install -e ".[gui]"
```

`requirements.lock` intentionally excludes PyQt6 and pyqtgraph. `requirements-gui.lock` includes it with the non-GUI lock. The GUI is optional: `import bang_rl.gui` remains safe in a headless installation, while constructing a concrete GUI widget requires the `[gui]` extra.

### Launching the GUI (Windows)

Two helper batch files are provided in the repository root:

```powershell
# 1. Install base + GUI dependencies (run once)
install_gui.bat

# 2. Launch the main menu (game / training / settings)
run_gui.bat
```

`run_gui.bat` starts `python -m bang_rl.gui`. Each screen (game, training, settings)
has a "← Back to menu" button, human games show a readable game log, and roles are
displayed by name (Sheriff, Deputy, Outlaw A/B, Traitor) instead of numeric codes.

## Validation

```powershell
python -m compileall -q bang_rl
python -m pytest -q
```

There is no command-line training or evaluation entry point in this implementation. Use the Python service APIs below (or the GUI when the optional dependencies are installed).

## Configuration

`config.example.yaml` is a validated starting point. Load it through the public config boundary:

```python
from bang_rl.config import load_config

config = load_config("config.example.yaml")
```

Important defaults are schema-owned: `max_episode_steps=2000`, long-game thresholds `200/500`, unlimited public event history (`event_history.max_events: null`), PPO rollout/gamma/GAE/epoch/minibatch values `256/0.99/0.95/4/64`, PPO `clip_epsilon=0.2`, and EmotionInference `initial_baseline=0.0`. Reward numbers are example configuration values, not implementation constants.

The schema accepts YAML or JSON. Environment and dataset paths are interpreted relative to the process working directory. The example uses the repository’s checked-in runtime-data locations:

- Canonical Parquet shards: `bang_rl/datasets/canonical/`
- Derived NPZ tensor caches: `bang_rl/datasets/cache/`
- Final inference artifacts: `bang_rl/models/`
- JSON Lines logs and reports: `bang_rl/logs/`

The schema defaults remain `datasets/canonical`, `datasets/cache`, and `models`; set explicit paths as in the example when running from this repository root. Generated data and artifacts are ignored by Git; only `.gitkeep` placeholders are retained.

## Public services and entry points

The GUI-independent application boundary is exported from `bang_rl.app`:

```python
from bang_rl.app import (
    ConfigService,
    DatasetService,
    EvaluationService,
    GameService,
    TrainingService,
)
```

- `GameService` creates sessions, exposes read-only `PublicGameState`, and submits actions.
- `TrainingService` starts a job, reports progress, pauses/resumes in the current process, and cancels a job. Pause/resume is memory-only.
- `EvaluationService` runs the primary or custom protocol and reports normal/truncated denominators separately.
- `DatasetService` exposes canonical and cache paths and Parquet collection boundaries.
- `ConfigService` loads, validates, saves, and snapshots `SystemConfig`.

Environment and agent code communicate through canonical Observation/Action data. The Environment owns terminal-condition evaluation; agents and GUI code do not read its private role array. Public observations do not contain other living players’ hidden roles, DummyAgent private emotion labels, reward breakdowns, or privileged diagnostics.

## Game and learning contracts

- The action space is `Discrete(6)`: `0=PASS`, `1..5=PLAYER_0..PLAYER_4`. The six-entry `uint8` Action Mask is separate and must be applied by every agent and by PPO probability calculations. The environment action-space helper identifies this contract as `discrete-6-v1`; persisted snapshot/preprocessor/dataset metadata uses its existing `action_space_version: "1.0"` field.
- Invalid actions are not converted to PASS: the Environment logs them on the `invalid_action` channel, leaves state/selection/step/emotion unchanged, and raises an explicit exception. AEC cleanup calls are separate and are not game steps.
- A valid BANG or PASS is one Environment Step. Normal wins set `terminated=True`; reaching `max_episode_steps` without a normal result sets `truncated=True`. A truncated episode has no winner/loser, terminal reward, bootstrap, or PPO training transition, but its event log and statistics are retained.
- MainAgent learning counts Environment Steps and MainAgent transitions separately. GUI progress is MainAgent transitions; PPO updates are a third unit.
- Canonical Phase 1 data is Parquet. Phase 2/3 normal-only tensor caches are derived NPZ shards. Truncated episodes remain in the canonical dataset and logs but are excluded from ML caches and PPO updates. Splits are episode-level 80/10/10 by default.
- EmotionInference predicts continuous `[0,1]` values with Sigmoid and MSE against the DummyAgent’s behavior-immediately-before-action state. Outputs are displayed as “estimated emotion”; they are not claims about human psychology.
- Turn order always starts with the Sheriff and proceeds clockwise across the seats regardless of seat index.

### GUI game modes and replay formats

The optional GUI offers human-play modes (Human vs Random/Dummy/Main AI) plus three spectator modes where all five seats are AI and the human only watches: Random AI x5, Dummy AI x5, and Main AI 1 vs Dummy AI 4. In a spectator mode the human cannot BANG or PASS; only the AI-turn-speed slider and the log view (including hover-to-inspect) are available.

Replays are stored in two formats. The public format keeps observation-only records for AI-training datasets and never exposes living non-sheriff, non-self roles. The full format is written for human-played and spectator games and additionally stores the authoritative reproduction state (hidden roles, deck and seat order, seed, and whose turn it is) so a moment can be resumed with a "Continue from here" action. Both formats render the same board and log; only full-format replays can be resumed. The replay browser can also rename and delete saved replays.

## Artifacts and compatibility

Phase 2 supervised modules and the final Phase 3 game model are different artifact types:

- Phase 2 module artifact: `.phase2model`, `phase2_inference_module`, format `1.0`; used only to initialize Phase 3 inference modules.
- Final game artifact: `.bangmodel`, `inference`, format `1.0`; a ZIP archive whose root contains exactly `manifest.json`, `metadata.json`, and `model.safetensors`. The weights file uses `intent`, `emotion`, and `ppo` namespaces, and the manifest records SHA-256 values.

Final artifacts are inference-only. Compatibility checks cover model architecture, Environment/Observation/Action/Reward/Emotion schema versions, preprocessing, and artifact format. Optimizer, rollout, worker, Environment-in-progress, and resume state are never stored. A saved artifact can be loaded for game execution, not used to resume a terminated process.

## Logs and evaluation

Structured JSON Lines channels are `episode_events`, `termination`, `reward_breakdown`, `invalid_action`, `training`, `evaluation`, and `privileged_diagnostics`. Privileged role/label diagnostics are physically separated from agent-visible/public channels.

The primary evaluation protocol is `deterministic_v1`: 500 games, 100 games per MainAgent seat, cyclic balanced seats, and `base_seed + evaluation_index`. Win Rate is normal wins divided by normal games; truncated games are excluded from that denominator. If there are no normal games, Win Rate is `N/A`. Custom and stochastic runs have separate protocol/run identities and are not merged into primary results.

## Deliberate limitations

This implementation does not provide Hardware Benchmark runners, automatic Worker/Batch/Buffer tuning, OOM-based batch search, Role Count Projection/Bayesian assignment, intermediate training checkpoints, or training resume after process termination. GUI Pause/Resume only coordinates the in-memory current job; it does not create a disk checkpoint.
