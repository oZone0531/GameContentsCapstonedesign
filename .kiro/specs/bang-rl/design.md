# Design — Bang RL Training System

> 버전: 0.7  
> 작성일: 2026-09-21  
> 기준 문서: `.kiro/specs/bang-rl/requirements.md` v0.7  
> 상태: Requirements-to-Design 확정 구현 기준  
> 범위: 설계 문서만 작성하며 구현 코드는 포함하지 않는다.

> 변경 이력: v0.7 — RoleFusion Hard Override만 구현하고 Role Count Projection은 확장 지점으로만 유지. EmotionInference initial baseline `0.0` 및 PPO `clip_epsilon=0.2`를 Config/GUI 설정으로 추가. Hardware Benchmark·자동 Worker/Batch 최적화를 제거하고 사용자 직접 설정·입력 검증·GUI 도움말로 변경.

## 목차

1. [설계 목표와 추적성](#1-설계-목표와-추적성)
2. [전체 아키텍처와 모듈 경계](#2-전체-아키텍처와-모듈-경계)
3. [Environment 도메인 모델](#3-environment-도메인-모델)
4. [AEC 실행 및 종료 계약](#4-aec-실행-및-종료-계약)
5. [Action Space와 Action Mask](#5-action-space와-action-mask)
6. [Observation과 정보 공개 모델](#6-observation과-정보-공개-모델)
7. [Agent 인터페이스와 DummyAgent](#7-agent-인터페이스와-dummyagent)
8. [MainAgent 3단계 구조](#8-mainagent-3단계-구조)
9. [감정 실행 계층과 EmotionInference](#9-감정-실행-계층과-emotioninference)
10. [Reward 및 학습 신호 경계](#10-reward-및-학습-신호-경계)
11. [PPO Transition과 Rollout](#11-ppo-transition과-rollout)
12. [Dataset 및 3-Phase Pipeline](#12-dataset-및-3-phase-pipeline)
13. [Evaluation Protocol](#13-evaluation-protocol)
14. [Application 및 GUI](#14-application-및-gui)
15. [Config, Version, Artifact Compatibility](#15-config-version-artifact-compatibility)
16. [Pause와 Resource 관리](#16-pause와-resource-관리)
17. [Logging과 오류 처리](#17-logging과-오류-처리)
18. [검증 전략과 인수 기준 추적](#18-검증-전략과-인수-기준-추적)
19. [디렉터리 레이아웃](#19-디렉터리-레이아웃)

---

## 1. 설계 목표와 추적성

### 1.1 설계 목표

이 설계는 다음 원칙을 만족한다.

- Environment는 Agent 구현 타입을 모른다.
- 모든 Agent는 동일한 Observation/Action 계약을 사용한다.
- GUI는 Environment 내부 상태가 아닌 Application Service와 DTO만 사용한다.
- 정상 종료와 제한 종료를 별도 상태로 다룬다.
- `max_episode_steps`에 도달한 Episode는 기록·통계에는 남기지만 모든 모델 학습에서 제외한다.
- MainAgent의 환경 Step과 PPO transition을 별도 카운터로 관리한다.
- 비공개 역할·실제 Emotion Label·Reward 내역은 행동 입력 경로에 들어가지 않는다.
- 학습 완료 결과는 추론용 Artifact로 저장하며, 중간 Checkpoint와 종료 후 Resume은 제공하지 않는다.

### 1.2 Requirements 추적 매트릭스

| 설계 영역 | 주요 Requirements |
|---|---|
| 모듈 독립성 | REQ-ARCH-001~006 |
| Environment/AEC | REQ-ENV-001~009, REQ-STEP-001~020 |
| Action | REQ-ACT-001~010 |
| Observation/역할 공개 | REQ-OBS-001~003, REQ-OBS-006~018 |
| 승패 | REQ-WIN-001~008 |
| DummyAgent | REQ-DUMMY-001~010 |
| MainAgent | REQ-MAIN-001~009 |
| PPO/학습 신호 | REQ-PPO-001~018 |
| Reward | REQ-RWD-001~009 |
| Dataset/분할 | REQ-DATA-001~010, REQ-SPLIT-001~005 |
| 평가 | REQ-EVAL-001~008 |
| Config/Artifact/Pause | REQ-CFG-001~008, REQ-INFER-001~003, REQ-COMPAT-001~002, REQ-PAUSE-001~007 |
| 검증 | AC-01~AC-25 |

---

## 2. 전체 아키텍처와 모듈 경계

### 2.1 계층 구조

```text
┌──────────────────────────────────────────────────────────┐
│ GUI: PyQt6 + pyqtgraph                                  │
│ Main Menu │ Game Screen │ Training Monitor │ Settings    │
└──────────────────────┬───────────────────────────────────┘
                       │ DTO/Service API only
┌──────────────────────▼───────────────────────────────────┐
│ Application Layer                                        │
│ GameService │ TrainingService │ EvaluationService        │
│ ConfigService │ PresetService │ DatasetService            │
│ CompatibilityService                                   │
└───────┬─────────────────────┬──────────────────────┬─────┘
        │                     │                      │
┌───────▼────────┐   ┌────────▼─────────┐   ┌────────▼──────┐
│ BangEnvironment │   │ Training Runtime │   │ Evaluation    │
│ PettingZoo AEC │   │ SL / PPO         │   │ Runner        │
└───────┬────────┘   └────────┬─────────┘   └───────────────┘
        │                      │
        │ canonical Obs/Action │
┌───────▼──────────────────────▼───────────────────────────┐
│ Agent Layer                                               │
│ HumanAgent │ RandomAgent │ DummyAgent │ MainAgent         │
└──────────────────────────────────────────────────────────┘
```

### 2.2 의존성 규칙

```text
GUI → Application Layer
Application Layer → Environment / Agents / Training / Data
Training → Environment / Agent interface / Reward service
Agents → canonical Observation/Action types only
Environment → domain types / Config snapshot / logging interface only
Environment ↛ agents
Environment ↛ gui
Agents ↛ gui
```

Environment는 `DummyAgent`, `MainAgent`, `HumanAgent`의 클래스를 import하지 않는다. 감정 상태도 Environment가 직접 제어하지 않으며, 실행 계층의 EmotionRuntime이 공개 이벤트를 Agent에 전달한다(REQ-ARCH-002, REQ-EMO-007).

### 2.3 작업 단위와 실행 세션

`TrainingJob`은 하나의 수집·학습 실행을 나타낸다. Job 시작 시 Config, Preset, Reward Config, Emotion Schema, Model Architecture, Optimizer, Seed, 데이터 분할 정보를 불변 Snapshot으로 만든다. 일반 Interactive Game의 설정 변경과 TrainingJob의 설정 고정은 서로 다른 정책으로 관리한다.

`SystemConfig.dataset`은 `DatasetConfig`를 사용하며 Canonical Shard 기준을 다음과 같이 관리한다.

```yaml
dataset:
  canonical_shard_max_episodes: 10000
  canonical_shard_max_bytes: 256000000  # 256 MB
```

두 값은 Config에서 조정할 수 있고, Shard rollover는 Episode 수 또는 byte 수 중 먼저 도달한 기준을 사용한다. `pyproject.toml`에는 `safetensors==0.5.3`을 직접 고정하고, 전체 Python 의존성은 생성된 `requirements.lock`에 transitive dependency와 함께 고정·커밋한다. 개발/CI 설치는 lock 파일을 사용한다.

```text
TrainingJob
├── job_id
├── mode: collection | sl | rl | evaluation
├── config_snapshot
├── seed_snapshot
├── preset_snapshot
├── reward_snapshot
├── emotion_schema_snapshot
└── runtime_status: running | pausing | paused | pause_failed | stopping | finished
```

---

## 3. Environment 도메인 모델

### 3.1 고정 역할과 팀

```python
class Role(IntEnum):
    SHERIFF = 0
    DEPUTY = 1
    OUTLAW_A = 2
    OUTLAW_B = 3
    TRAITOR = 4

class Team(IntEnum):
    SHERIFF_TEAM = 0
    OUTLAW_TEAM = 1
    TRAITOR_TEAM = 2
```

역할 배정은 5개 슬롯에 대해 정확히 한 번 수행한다. 역할 배정 시 canonical role list `[SHERIFF, DEPUTY, OUTLAW_A, OUTLAW_B, TRAITOR]`를 만들고, `np.random.Generator(np.random.PCG64(episode_seed))`를 생성한 뒤 `rng.permutation(canonical_role_list)` 결과를 slot `0..4`에 순서대로 배치한다. 동일한 `episode_seed`는 동일한 permutation을 생성하며, 다른 RNG·shuffle·추가 random draw를 역할 배정에 사용하지 않는다. Environment 내부에는 실제 역할 배열이 존재하지만, 공개 Observation에는 자기 역할·공개 보안관 역할·탈락 공개 역할·논리적 확정 역할만 노출한다.

### 3.2 내부 게임 상태

```text
EnvironmentState
├── episode_id
├── seed
├── roles[5]                 # privileged, Environment only
├── health[5]                # uint8, sheriff=5, others=4 initially
├── alive[5]                 # bool
├── agent_selection         # current slot
├── fixed_turn_order[5]
├── environment_step        # valid living-player actions only
├── round_index              # optional presentation counter
├── event_log[]              # canonical public/private-separated events
├── terminated[5]
├── truncated[5]
├── terminal_result          # normal termination only
└── termination_status       # running | normal | truncated
```

`environment_step`은 BANG 또는 PASS 같은 유효 생존자 행동마다 1 증가한다. Invalid Action과 AEC 정리 호출은 증가시키지 않는다.

### 3.3 GameEvent

```python
@dataclass(frozen=True)
class GameEvent:
    event_id: int             # uint32-capable monotonic ID
    environment_step: int     # uint32-capable
    event_type: EventType
    actor_idx: int | None
    target_idx: int | None
    hp_before: int | None
    hp_after: int | None
    eliminated_idx: int | None
    revealed_role: int | None # absent/hidden or Role value after elimination
    public: bool
```

PASS의 대상은 `target_idx = None`을 canonical 표현으로 사용하고, 바이너리 저장 시 nullable을 지원하지 않는 포맷에는 `255` sentinel을 사용한다. 역할 공개 이벤트는 탈락 시점에 생성되며 이후 공개 Event Sequence에 포함된다.

### 3.4 TerminalResult

```python
@dataclass(frozen=True)
class TerminalResult:
    terminated: bool
    winners: tuple[int, ...]
    losers: tuple[int, ...]
    winning_team: Team | None
    reason: str
```

`truncated` Episode에는 TerminalResult의 승자·패자를 생성하지 않는다. `truncated`는 별도의 `termination_status`와 종료 원인으로 기록한다.

---

## 4. AEC 실행 및 종료 계약

### 4.1 유효 Step 처리 순서

```text
1. 현재 Agent가 활성 생존자인지 확인
2. Observation의 Action Mask를 생성
3. 전달 Action의 타입·범위·Mask 허용 여부 검증
4. Invalid Action이면 상태 불변 + 로그 + 예외
5. 유효 Action이면 BANG/PASS 적용
6. environment_step += 1
7. 공개 Event 생성 및 Event ID 할당
8. 탈락 시 역할 공개 Event 생성
9. 정상 Terminal Condition 평가
10. 정상 종료가 아니면 max_episode_steps 평가
11. terminated/truncated 또는 다음 생존 Agent 선택
12. 다음 행동 경계에서 EmotionRuntime 처리
```

정상 승패가 같은 행동에서 충족되면 `terminated=True`가 `truncated=True`보다 우선한다(REQ-STEP-007~009).

### 4.2 승패 판정

```text
if sheriff_alive and no living outlaw and no living traitor:
    Sheriff Team wins
elif not sheriff_alive:
    if exactly one living player and that player is traitor:
        Traitor wins
    elif any living non-traitor player:
        Outlaw Team wins
else:
    continue
```

보안관과 배신자만 남은 상태는 별도 Duel State가 아니다. 두 슬롯은 정상 턴 순서로 계속 행동한다. 보안관이 배신자를 제거하면 보안관팀 승리, 배신자가 보안관을 제거하면 배신자 승리다.

보안관팀·무법자팀 결과는 개인 생존 여부와 무관하게 팀 소속 Agent에 귀속한다. 정상 종료 시에만 승자·패자를 만든다.

### 4.3 Truncated 계약

`environment_step == max_episode_steps`이고 정상 승패가 결정되지 않은 경우:

```text
termination_status = truncated
terminated = False
truncated = True
winner/loser = none
terminal_reward = none
bootstrap = none
PPO training inclusion = false
```

Truncated Episode는 discarded game이다. 원본 Event Log, 종료 상태, 통계는 저장하지만 Phase 2 지도학습과 Phase 3 PPO Dataset에서는 제외한다.

### 4.4 Long-game diagnostics

`0 < warn_turn_threshold < critical_turn_threshold < max_episode_steps`를 Config 검증 규칙으로 사용한다. 환경 Step이 각 threshold를 초과하는 시점에 Episode별·threshold별 1회만 WARNING/CRITICAL 로그를 남긴다. 로그는 Environment Event와 상태를 제공하고 실행/로깅 계층이 Preset·Experiment·Version metadata를 결합한다.

---

## 5. Action Space와 Action Mask

### 5.1 Action Space

모든 Agent는 `Discrete(6)`을 사용한다.

```text
0 = PASS
1 = PLAYER_0 공격
2 = PLAYER_1 공격
3 = PLAYER_2 공격
4 = PLAYER_3 공격
5 = PLAYER_4 공격
```

Action Space는 가능한 행동의 전체 집합이고, 현재 선택 가능 여부는 별도 Action Mask로 결정한다.

### 5.2 Mask 생성

```python
mask = np.zeros(6, dtype=np.uint8)
mask[0] = 1
for target_idx in living_players:
    if target_idx != acting_idx:
        mask[target_idx + 1] = 1
```

탈락 대상과 자기 자신 공격은 0이다. 사람 UI, RandomAgent, DummyAgent, MainAgent, PPO 모두 이 Mask를 적용한다. PPO Rollout은 action mask를 보존하며 log-prob 재계산에도 같은 mask를 사용한다.

### 5.3 Invalid Action

Environment는 타입, 범위, 현재 Mask를 최종 검증한다. 실패 시 다음 상태를 보장한다.

- 게임 상태 불변
- agent selection 불변
- environment step 불변
- 공개 행동 Event 미생성
- Emotion 갱신·감쇠 미실행
- 별도 Invalid Action 로그 기록
- 명시적 예외 발생
- PASS 변환 및 자동 재선택 없음

탈락·종료 Agent의 AEC `step(None)` 정리는 이 검증 경로와 분리한다.

---

## 6. Observation과 정보 공개 모델

### 6.1 Canonical Observation

```python
ObsDict = {
    "current_player_idx": np.uint8,      # shape (1,)
    "environment_step": np.uint32,       # shape (1,)
    "round_index": np.uint32,             # optional presentation value
    "healths": np.uint8,                  # shape (5,)
    "alive": np.bool_,                    # shape (5,)
    "turn_order": np.uint8,               # shape (5,)
    "sheriff_idx": np.uint8,              # public
    "self_role": np.uint8,                # self only, 0..4
    "role_visibility": np.uint8,          # shape (5,)
    "revealed_roles": np.uint8,           # shape (5,), 255=not revealed
    "confirmed_role_distribution": np.float32, # shape (5, 5)
    "event_sequence": EventSequence,      # raw public sequence, K configurable
    "action_mask": np.uint8,              # shape (6,), 0/1
}
```

`role_visibility`는 hidden/revealed/self/confirmed 같은 공개 상태를 표현한다. `revealed_roles`는 공개되지 않은 슬롯에 `255` sentinel을 사용한다. 확정된 역할은 `confirmed_role_distribution`에서 one-hot으로 표현하고, 미공개 역할만 추론 모듈의 출력을 사용한다.

### 6.2 정보 차단

다음은 Agent Observation, 공개 Event Sequence, MainAgent 추론 입력, PPO policy/value 입력에 포함하지 않는다.

- 다른 생존자의 비공개 실제 Role
- DummyAgent의 실제 EmotionModule 내부 상태
- Role Label
- Emotion Label
- Reward 값·Reward Breakdown
- 특권 진단 로그

허용되는 것은 자기 역할, 공개 보안관 역할, 탈락 공개 역할, 공개 정보로 논리적으로 확정한 역할과 공개 정보에서 계산한 추정값이다.

### 6.3 Event Sequence 전처리

Environment는 원본 이벤트를 제공한다. `event_history.max_events`의 기본값은 `null`이며, 이는 Episode 시작부터 현재까지 전체 이력을 뜻한다. Model 계층의 `model_history_length`는 **64**로 설정한다. 최근 64개 이벤트를 선택하고, 실제 이력이 64개보다 짧으면 **왼쪽 padding**을 적용하여 최신 이벤트가 sequence의 뒤쪽에 위치하도록 한다. 64개보다 길면 최신 64개를 유지하고 오래된 이벤트를 truncation한다. Environment의 원본 Event History 자체에는 이 padding/truncation을 적용하지 않는다.

일반 결측값은 `null`을 사용하고, 정수형 저장 포맷에서 nullable을 지원하지 않는 경우에만 `255` sentinel을 사용한다. Event ID와 `environment_step`으로 시간 순서를 보존한다.

---

## 7. Agent 인터페이스와 DummyAgent

### 7.1 공통 Agent 계약

```python
class BaseAgent(Protocol):
    def act(self, observation: ObsDict, action_mask: np.ndarray) -> int: ...
    def on_episode_start(self, context: EpisodeContext) -> None: ...
    def on_episode_end(self, result: EpisodeResult) -> None: ...
```

`act()`는 Observation과 Action Mask만 입력으로 받는다. 학습기의 Reward는 Agent 행동 입력 API와 별도의 Training Runtime 경로로 전달한다.

### 7.2 DummyAgent 구성

```text
DummyAgent
├── RoleBeliefEstimator
├── IntentModule
├── EmotionModule
└── ActionSelector
```

`RoleBeliefEstimator`는 학습된 MainAgent 모듈에 의존하지 않는다.

1. 자기 역할·보안관·탈락 공개 역할을 반영한다.
2. 남은 역할 수를 계산한다.
3. 미확정 생존자에 남은 역할 비율 확률을 부여한다.
4. 공개 정보만으로 확정되면 one-hot 확정값으로 바꾼다.

배신자의 진영 판단은 실제 비공개 역할이 아니라 이 추정값을 사용한다.

```text
strong_faction:
  1) living survivor count가 큰 진영
  2) 동률이면 Σ(public_hp × faction_probability)가 큰 진영
  3) 동률이면 Config의 tie policy
```

배신자는 역할에 따른 추가 행동 제한을 받지 않지만, 공통 Action Mask는 준수한다. 1대1이 아니면 보안관 직접 공격 가중치를 낮추고, 보안관·배신자만 생존하면 보안관 공격 가중치를 높인다.

### 7.3 Dummy EmotionModule

EmotionModule은 Config에서 정의된 N개 상태를 관리한다. 모든 값은 `[0, 1]`로 정규화한다. 감정 이름·범위·초기값·변화량·감쇠·행동 영향은 Config/Preset에 둔다. Environment는 내부 EmotionState를 직접 변경하지 않는다.

---

## 8. MainAgent 3단계 구조

```text
MainAgent
├── ObsPreprocessor
├── IntentInferenceModule
│   └── SequenceEncoder: GRU 기본, 교체 가능
├── RoleFusion
│   └── ConfirmedRoleHardOverride (현재 구현)
├── EmotionInferenceModule
└── PPOModule
    ├── Policy Head
    └── Value Head
```

### 8.1 IntentInferenceModule

- 입력: 타 플레이어 4명의 공개 행동 이력과 공개 상태
- 기본 Encoder: GRU
- 추상 인터페이스: `SequenceEncoder`
- 고정 구현값: `hidden_size=128`, `num_layers=1`, `batch_first=True`, dropout 없음
- SequenceEncoder는 Observation Preprocessor가 제공하는 입력 차원을 사용하고, valid mask를 반영하여 마지막 유효 시점의 hidden state를 출력한다.
- 각 target player에 동일하게 적용하는 공유 Role Head: `Linear(128, 64) → ReLU → Linear(64, 5)`
- 학습 시 Role Head의 최종 출력은 logits이며 Cross-Entropy에 사용하고, 추론 시 logits에 softmax를 적용한다.
- 출력: 타 플레이어 4명 × 역할 5개의 role probability, shape `(4, 5)`
- 출력은 `RoleFusion`으로 전달되며 PPOModule이 raw output을 직접 사용하지 않는다.

### 8.2 RoleFusion

`RoleFusion`은 `IntentInferenceModule`의 raw role distribution과 공개·확정 역할 정보를 연결하는 중간 계층이다. PPOModule은 IntentInference의 raw output을 직접 사용하지 않고, RoleFusion이 반환한 fused distribution을 사용한다.

```python
class RoleFusion(Protocol):
    def fuse(
        self,
        raw_role_distribution: np.ndarray,  # shape (4, 5)
        role_visibility: np.ndarray,
        revealed_roles: np.ndarray,
        confirmed_role_distribution: np.ndarray,
    ) -> np.ndarray:                         # shape (4, 5)
        ...
```

현재 버전의 동작은 다음과 같다.

```text
for each target player:
    if confirmed role exists:
        fused_distribution = one_hot(confirmed role)
    else:
        fused_distribution = raw IntentInference distribution
```

- 공개된 Sheriff, 탈락 시 공개된 Role, 논리적으로 확정된 Role은 Hard Override한다.
- 모델의 기존 확률은 확정 Role의 one-hot 결과에 영향을 주지 않는다.
- 미확정 Role의 raw 확률은 그대로 전달한다.
- Remaining Role Count 계산, Role Count Constraint, Role Assignment 조합 생성, Assignment Weight, Marginalization, Bayesian Projection, 확률 재분배는 현재 구현 범위에 포함하지 않는다.
- Role Count Projection을 위한 계산이나 호출 경로도 현재 버전에서는 만들지 않는다.

RoleFusion의 interface 경계는 향후 다음 확장을 허용한다.

```text
raw_role_distribution
        ↓
ConfirmedRoleHardOverride
        ↓
[future extension point: RoleCountProjection]
        ↓
fused_role_distribution
        ↓
PPOModule
```

향후 Projection을 추가하더라도 입력은 공개·확정된 Role, 고정된 게임 Role 구성, 공개 게임 상태, IntentInference 결과로 제한한다. 실제 hidden true role, Role Label, 특권 진단 로그는 RoleFusion에 전달하지 않는다.

### 8.3 EmotionInferenceModule

- 입력: 타 플레이어의 공개 행동 이력, 공격·피격, 체력 변화, 생존 상태
- 입력 금지: 실제 Emotion Label, 비공개 Role, Reward
- 고정 구현값: GRU `hidden_size=128`, `num_layers=1`, `batch_first=True`, dropout 없음
- SequenceEncoder는 valid mask를 반영하여 마지막 유효 시점의 hidden state를 출력한다.
- 각 target player에 동일하게 적용하는 공유 Emotion Head: `Linear(128, 64) → ReLU → Linear(64, E) → Sigmoid`
- 출력 활성화 함수: `Sigmoid`
- 출력: 감정 변수 수 E에 따른 연속 벡터 shape `(4, E)`
- 출력 범위: `[0.0, 1.0]`
- 빈 History: valid mask로 구분하고 Config의 `initial_baseline`을 기준값으로 사용한다. 초기값은 `0.0`이며 Emotion Schema에 별도 baseline 값을 저장하지 않는다.
- Target: 행동 직전 DummyAgent EmotionModule 상태
- Loss: MSE
- Emotion Schema Version이 출력 차원과 일치해야 한다.
- `initial_baseline`은 Config Schema와 Job Config Snapshot으로 관리하고, GUI에서 변경할 수 있다. 변경값은 다음 Training Job부터 적용한다.

### 8.4 PPOModule

PPOModule 입력은 다음 Concatenate 결과다.

```text
concat(
    self_state,
    board_state,
    confirmed_role_features,
    intent_features,
    emotion_features
)
```

Role Label, Emotion Label, Reward Breakdown, 특권 로그는 입력에 포함하지 않는다. Policy Head에는 현재 Action Mask를 적용하고, Value Head에는 존재하지 않는 다음 상태를 입력으로 만들지 않는다.

PPOModule의 고정 구현값은 다음과 같다.

- Shared trunk: `Linear(input_dim, 256) → ReLU → Linear(256, 256) → ReLU`
- Policy Head: `Linear(256, 6)`
- Value Head: `Linear(256, 1)`
- `input_dim`은 위 Concatenate된 실제 Feature 차원에서 계산하며 별도 임의 padding을 추가하지 않는다.

---

## 9. 감정 실행 계층과 EmotionInference

### 9.1 EmotionRuntime 책임

Environment와 DummyAgent 내부 상태 사이에는 `EmotionRuntime` 또는 실행 조정 계층을 둔다. 이 계층은 공개 이벤트를 Agent에 전달하지만 Environment에 DummyAgent 구현 의존성을 추가하지 않는다.

### 9.2 행동 전 시간 순서

각 유효 Environment Step에서 다음 순서를 따른다.

```text
1. 아직 반영되지 않은 public event 확인
2. Event ID별 1회 반영
3. 살아 있는 관련 DummyAgent 감정 갱신
4. 현재 Environment Step에 대한 감쇠 1회 적용
5. 행동 직전 감정 Label 기록
6. Agent 행동 선택
7. 행동 실행
8. 새 public event 생성
```

Invalid Action은 Step이 아니므로 이 순서를 다시 실행하지 않는다. 행동 선택 전에 정상 반영된 감정 상태를 되돌리지 않는다. 현재 행동과 이후 Event는 해당 Label의 입력에 포함하지 않는다. 비행동 플레이어의 Label도 동일한 관측 기준 시점으로 기록한다.

### 9.3 Schema 변경

감정 이름·개수·순서·값 범위·정규화가 변경되면 새 Emotion Schema Version, 새 Dataset, 새 추론 모델이 필요하다. 변화량·감쇠 같은 동작 파라미터 변경은 Preset Version으로 추적하며 Schema 호환성 변경과 구분한다.

---

## 10. Reward 및 학습 신호 경계

### 10.1 Reward 구성

```text
RewardCalculator
├── TerminalReward
└── RewardShaping
    ├── bang_hit
    ├── kill_enemy
    ├── attack_sheriff_outlaw
    ├── attack_sheriff_deputy
    ├── role_or_faction_attack_terms
    └── pass_penalty
```

실제 Component ID, 조건 문법, 숫자값은 Reward Config에 둔다. Requirements는 항목과 변경 가능성만 고정하고 기본 숫자를 고정하지 않는다. GUI는 항목을 조회·편집하고 Config Version을 갱신한다.

Reward Config Schema는 다음 구조를 고정한다.

```yaml
reward_config:
  version: <string>
  terminal:
    components:
      - id: terminal_win | terminal_loss
        enabled: <bool>
        condition:
          type: terminal_result
          result: win | loss
        value: <float>
  shaping:
    components:
      - id: bang_hit | kill_enemy | attack_sheriff_outlaw | attack_sheriff_deputy | role_or_faction_attack_terms | pass_penalty
        enabled: <bool>
        condition:
          type: event | role_attack | faction_attack | pass
          event: bang_hit | kill_enemy | pass
          actor_role: <Role>          # role_attack only
          target_role: <Role>         # role_attack only
          actor_faction: <Team>       # faction_attack only
          target_faction: <Team>      # faction_attack only
        value: <float>
```

`version`, `id`, `enabled`, `condition`, `value`는 모든 Component에 필수다. `terminal.components`와 `shaping.components`는 서로 다른 namespace로 관리한다. `terminal_result` 조건은 `result`만, `event` 조건은 `event`만, `role_attack` 조건은 `actor_role`·`target_role`만, `faction_attack` 조건은 `actor_faction`·`target_faction`만 사용하며 `pass` 조건은 PASS Event에 적용한다. 조건에 일치하고 `enabled=true`인 Component는 Design §10.1의 해당 Reward 계층 안에서 합산한다. Terminal Reward는 Normal termination에서만 평가하고, Truncated에는 생성하지 않는다. 모든 `value`는 Config에서 읽으며 코드에 숫자를 하드코딩하지 않는다.

### 10.2 정상/제한 종료 Reward

- Normal termination: 팀 결과에 따른 Terminal Reward를 기록하고 PPO Dataset에 포함한다. 초기 Design 기본값은 팀 승리 `+1`, 팀 패배 `-1`이며, 실제 값은 Reward Config에서 변경할 수 있다.
- Truncated: Terminal Reward 없음, Bootstrap 없음, PPO Dataset 제외, 기존 Shaping 계산/로그는 보존 가능
- Reward Config는 Terminal Condition을 변경하지 않는다.

### 10.3 Reward 전달 경계

```text
Environment/Reward Runtime → Training Runtime → PPO buffer
Environment Observation → Agent.act()
```

Reward와 Reward 구성 요소는 PPO 계산에 사용하지만 다음 행동 Observation이나 Policy 입력으로 전달하지 않는다. Reward Breakdown에는 `agent_id`, `environment_step`, `main_agent_transition_id`, `terminal_reward`, shaping component별 값, total, Reward Config Version을 기록한다.

---

## 11. PPO Transition과 Rollout

### 11.1 단위

- Environment Step: 살아 있는 플레이어의 유효 행동 1회
- MainAgent Transition: MainAgent 행동부터 다음 MainAgent 행동 기회 또는 Episode 종료까지
- PPO Update: 수집 transition으로 수행하는 파라미터 갱신

Transition 구간 동안 MainAgent에게 발생한 Reward를 합산하고 gamma를 transition 단위로 적용한다. GUI 목표 학습 Step은 MainAgent transition 수다. Environment Step과 transition 수는 별도 기록하고 병렬 누적값은 Worker 합산이다.

### 11.2 Transition 상태

```python
@dataclass
class MainAgentTransition:
    episode_id: str
    transition_id: int
    observation: ObsDict
    action: int
    action_mask: np.ndarray
    log_prob: float
    value: float
    accumulated_reward: float
    next_observation: ObsDict | None
    terminated: bool
    truncated: bool
    main_agent_eliminated: bool
    rollout_id: str
    environment_steps_spanned: int
    trainable: bool
```

`PendingTransitionLedger`는 MainAgent의 행동 직후 생성된 미완료 transition을 PPO Buffer와 분리하여 보관한다.

```text
MainAgent action
    ↓
Pending Ledger에 transition 생성
    ↓
다음 MainAgent 행동 기회 또는 Episode 종료
    ↓
MainAgent 탈락 시 추가 행동·가짜 PASS 없음
    ↓
Episode 결과 확정
    ├── Normal termination → 최종 팀 보상 반영 후 PPO Buffer에 확정
    └── Truncated          → trainable=False로 폐기, PPO Buffer에 넣지 않음
```

정상 종료 transition은 `trainable=True`, final reward가 기록되고 종료 이후 Bootstrap은 없다. Truncated 기록은 Log/통계용으로만 남긴다.

### 11.3 Rollout 경계

Rollout은 다음 경계를 명시한다.

- `rollout_boundary`: 고정 rollout 길이 또는 update 수집 경계
- `terminated`: 정상 승패 종료
- `truncated`: 운영상 discarded 종료
- `agent_eliminated`: MainAgent 행동 자격 상실

Rollout 경계가 Episode 종료와 다를 수 있으므로 transition에는 두 상태를 별도 기록한다. Truncated transition은 rollout 경계와 무관하게 PPO Update에서 제거한다.

### 11.4 Mask와 PPO

Rollout 수집 시 저장한 Action Mask를 PPO의 log-prob 재계산에 동일하게 사용한다. Mask = 0 행동은 logits를 제거한 뒤 policy distribution을 재정규화한다.

### 11.5 PPO Update

PPO Update는 다음 조건을 만족하는 transition을 입력으로 받는다.

```text
trainable == True
truncated == False
```

정상 Episode 내부의 비-terminal transition은 PPO 학습 대상이며 GAE 및 후속 정상 상태를 사용한 Bootstrap 계산에 사용할 수 있다. 정상 Episode의 마지막 transition은 `terminated=True`로 표시하고 종료 이후 Bootstrap을 적용하지 않는다. Truncated transition은 `terminated` 값과 무관하게 항상 PPO Update에서 제외한다. 초기 Design Config는 다음과 같다.

```yaml
training:
  ppo:
    rollout_transitions_per_worker: 256
    gamma: 0.99
    gae_lambda: 0.95
    ppo_epochs: 4
    minibatch_size: 64
    clip_epsilon: 0.2
  emotion_inference:
    initial_baseline: 0.0
```

- 위 값들은 초기값이며 코드에 하드코딩하지 않는다.
- 모든 값은 Config Schema와 Job Config Snapshot으로 관리한다.
- GUI에서 수정할 수 있다.
- 수정값은 다음 Training Job부터 적용하며 진행 중인 Job에는 적용하지 않는다.
- PPO는 `clip_epsilon`를 Config Snapshot에서 읽는다.
- EmotionInference 빈 History는 `initial_baseline`과 valid mask를 사용한다.
- IntentInference와 EmotionInference의 지도학습 optimizer는 Adam이며 `learning_rate=1e-3`, `betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=0`으로 고정한다.
- PPOModule PPO optimizer는 Adam이며 `learning_rate=3e-4`, `betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=0`으로 고정한다.
- 위 모델 구조와 optimizer 값은 현재 구현 계약이며 GUI에서 변경하지 않는다. 실제 사용값은 Job Snapshot과 최종 Artifact Metadata에 기록한다.

---

## 12. Dataset 및 3-Phase Pipeline

### 12.1 Canonical Parquet

Parquet Shard는 기준 원본이며, `datasets/canonical/`에 저장한다. Zstandard compression을 사용한다. `SystemConfig.dataset.canonical_shard_max_episodes`(초기값 10,000) 또는 `SystemConfig.dataset.canonical_shard_max_bytes`(초기값 256,000,000 bytes = 256 MB) 중 먼저 도달하는 조건으로 분할한다. 두 값은 Config에서 조정할 수 있다.

```text
DatasetRecord
├── episode_id
├── seed
├── termination_status: normal | truncated
├── environment_version
├── observation_schema_version
├── action_space_version
├── emotion_schema_version
├── reward_config_version
├── agent/preset identifiers
├── initial state
├── public event sequence
├── valid actions and hp changes
├── role labels for supervised targets
├── emotion labels for supervised targets
└── final state / terminal or truncation reason
```

원본 Event를 매 Step 전체 누적 복사하지 않는다. 완료 Episode는 Shard에 기록한 뒤 불필요한 메모리를 해제한다.

### 12.2 NPZ Tensor Cache

NPZ는 Parquet에서 생성되는 재생성 가능한 파생 Cache이며 `datasets/cache/`에 저장한다. Cache content hash는 source shard content, preprocessing version, Observation Schema Version, Emotion Schema Version, split name 및 split seed를 조합하여 생성한다.

Cache metadata에는 다음을 넣는다.

- source episode/shard identifiers
- source content hash
- preprocessing version
- Observation Schema Version
- Emotion Schema Version
- split name and split seed
- cache generation timestamp
- feature/label schema

위 입력 중 하나라도 변경되면 content hash가 달라지고 NPZ Cache를 재생성한다. `termination_status=truncated` Episode는 Phase 2와 Phase 3 Cache에서 제외한다. 원본 Parquet과 파생 NPZ의 표현 중복은 허용하지만, 원본 권위는 Parquet에만 둔다.

### 12.3 dtype

- HP, Role, Action, Player Index: `uint8`
- Action Mask: `uint8`, shape `(6,)`
- Alive: `bool` 또는 `uint8`
- Environment Step/Event ID: `uint32` 이상
- 연속 Feature/Emotion: 정밀도 검증 후 `float16` 또는 `float32`
- PASS target: nullable 또는 `255` sentinel

### 12.4 Episode split

Episode ID 단위로 80/10/10(train/validation/test)을 분할한다. 한 Episode의 Observation, Sequence, Role Label, Emotion Label은 같은 split에 있어야 한다. Split Seed와 결과를 기록하고 전처리 통계는 train split으로만 계산한다.

### 12.5 Pipeline

```text
Phase 1: Data Collection
  DummyAgent × 5 → headless Environment → Parquet canonical shards

Phase 1.5: Preprocessing
  Parquet → episode split → normal-only NPZ tensor caches

Phase 2: Supervised Learning
  normal-only cache → IntentInference + EmotionInference
  Emotion: continuous [0,1], MSE
  → Phase 2 inference module artifact

Phase 3: PPO Training
  load compatible Phase 2 inference module artifact
  freeze IntentInference and EmotionInference
  trainable=True && truncated=False transitions → PPOModule PPO update
  normal Episode non-terminal transitions → GAE/Bootstrap eligible
  normal Episode final transition (`terminated=True`) → terminal reward, no post-terminal Bootstrap
  truncated transitions → always excluded
  → final inference artifact for game execution
```

Phase 2 Artifact는 Phase 3에서 IntentInference와 EmotionInference를 초기화하기 위한 inference module artifact다. Phase 2 Artifact가 최종 게임 실행용 `.bangmodel`과 동일한 구성일 필요는 없다. Phase 3 완료 결과는 실제 게임 추론에 사용하는 최종 `.bangmodel` artifact로 저장한다.

중간 학습 Checkpoint나 optimizer/rollout/environment 복원 상태는 저장하지 않는다. 프로그램 종료 후 학습 Resume도 제공하지 않는다.

---

## 13. Evaluation Protocol

### 13.1 Primary Evaluation Protocol

Primary Evaluation은 다음 조건으로 고정한다.

```yaml
evaluation_protocol:
  id: deterministic_v1
  games: 500
  seats: balanced
  games_per_seat: 100
  action_mode: deterministic
  seed_formula: base_seed + evaluation_index
```

- `seats: balanced`는 MainAgent 좌석을 `seat 0 → seat 1 → seat 2 → seat 3 → seat 4` 순서로 cyclic하게 배정하는 deterministic 정책이다.
- 각 좌석에서 정확히 100게임을 수행하여 총 500게임을 구성한다.
- Episode Seed는 `base_seed + evaluation_index`로 생성한다.
- 역할 배정은 Episode Seed에 의해 재현 가능하게 결정한다.
- Primary 결과는 모델 비교의 기준으로 사용한다.

### 13.2 Custom Evaluation

GUI의 평가 게임 수 입력은 Primary Protocol과 구분되는 **Custom Evaluation 설정**으로 취급한다. 사용자가 평가 게임 수, Seed, 좌석 배정 또는 행동 선택 방식을 변경하면 Primary 결과와 별도의 Evaluation Run으로 기록한다. Custom 결과를 Primary 결과와 동일한 조건의 연속 성능으로 표시해서는 안 된다.

Stochastic 행동 평가는 별도 Protocol로 지원하며 Primary deterministic 결과와 합산하지 않는다.

### 13.3 결과 Schema

```text
evaluation_run_id
model_artifact_id
opponent/preset/version
seed_set
seat_assignment_policy
action_selection_mode
total_games
normal_games
truncated_games
normal_wins
normal_losses
win_rate = normal_wins / normal_games
truncated_rate = truncated_games / total_games
role_specific_results
role_classification_accuracy
hidden_role_accuracy
estimated_emotion_mse
```

Normal games가 0이면 Win Rate는 `N/A`다. Truncated는 Win Rate 분모에 넣지 않는다. 서로 다른 Protocol 결과는 같은 조건의 연속 성능 변화로 표시하지 않는다. 확정 역할과 미공개 역할 분류 성능은 분리한다. `role_classification_accuracy`는 평가 시점의 대상별 예측 분포에서 `argmax`한 Role이 실제 Role과 일치한 대상 수를 평가 대상 수로 나눈 값이다. `hidden_role_accuracy`는 같은 공식을 사용하되 해당 평가 시점에 MainAgent에게 공개·확정되지 않은 Role 대상만 분모에 포함한다. `estimated_emotion_mse`는 행동 직전 추정 Emotion과 privileged evaluation target인 DummyAgent 실제 Emotion 상태의 각 차이에 대한 제곱 오차 평균으로 계산하며, 실제 Emotion 상태는 MainAgent 입력에 전달하지 않는다. 실제 Emotion target이 존재하는 평가 샘플만 분모에 포함하고 eligible sample이 0이면 `N/A`로 기록한다. 두 지표 모두 평가 결과에 별도 기록하고 Emotion 출력은 “추정 감정”으로 표기한다.

---

## 14. Application 및 GUI

### 14.1 Service API 경계

```python
class GameService:
    def create_session(self, config: SessionConfig) -> SessionHandle: ...
    def step(self, handle: SessionHandle, action: int) -> StepResult: ...
    def get_state(self, handle: SessionHandle) -> PublicGameState: ...

class TrainingService:
    def start(self, config: TrainingConfig) -> TrainingHandle: ...
    def pause(self, handle: TrainingHandle) -> None: ...
    def resume(self, handle: TrainingHandle) -> None: ...
    def stop(self, handle: TrainingHandle) -> None: ...
    def progress(self, handle: TrainingHandle) -> TrainingProgress: ...

class EvaluationService:
    def run(self, config: EvaluationConfig) -> EvaluationReport: ...

class ConfigService:
    def load(self, path: str, schema: type) -> dict: ...
    def validate(self, value: dict, schema: type) -> list[ValidationError]: ...
    def save(self, path: str, value: dict) -> None: ...
```

`TrainingService.resume()`는 현재 실행 중인 프로세스에서 `paused` 상태를 해제하는 기능이다. 프로그램 종료 후 이전 TrainingJob 상태를 복원하는 Resume은 지원하지 않으며, 이를 위한 Checkpoint 파일도 저장하지 않는다.

GUI는 위 Service와 read-only DTO를 사용하며 Environment 내부 배열이나 Agent private state를 직접 읽지 않는다.

### 14.2 화면

```text
MainWindow (PyQt6)
├── MainMenu
├── GameModeSelect
│   ├── 사람 1명 vs 랜덤 AI 4명
│   ├── 사람 1명 vs DummyAgent 4명
│   └── 사람 1명 vs MainAgent 1명 + DummyAgent 3명
│       └── InferenceModelSelectModal
│           ├── 저장된 .bangmodel 목록
│           ├── 생성 일시
│           ├── 학습 상대/Preset 정보
│           ├── 총 환경 Step / MainAgent transition
│           ├── 최종 Win Rate / Truncated Rate
│           └── Compatibility 검사 결과
├── GameScreen
│   ├── GameBoard
│   ├── ActionPanel
│   └── DebugHUD (타 플레이어 4명, 추정 감정 표기)
├── TrainingSetup
├── TrainingMonitor
│   ├── ProgressBar: MainAgent transition 수
│   ├── environment step counter
│   ├── truncated count/rate
│   └── Win-rate graph (pyqtgraph)
├── EvaluationScreen
└── Settings
    ├── Dummy AI/Preset
    ├── Reward Config
    ├── Training/Model Settings
    │   ├── PPO clip epsilon (initial: 0.2)
    │   └── Emotion initial baseline (initial: 0.0)
    └── Hardware
```

- `GameModeSelect`는 Requirements의 3가지 Game Mode를 제공한다.
- MainAgent가 포함된 모드를 선택하면 `InferenceModelSelectModal`에서 저장된 추론 모델을 선택한다.
- Model Modal은 모델 Metadata와 Compatibility 검사 결과를 표시한다.
- Headless는 학습 Environment의 게임 렌더링을 끄는 의미이며, TrainingMonitor의 ProgressBar·pyqtgraph 그래프는 유지한다.
- Training/Model Settings 변경은 다음 Training Job부터 적용하고, 진행 중인 Job에는 적용하지 않는다.

---

## 15. Config, Version, Artifact Compatibility

### 15.1 Config Snapshot

- 일반 게임의 구조 비변경 Config는 다음 행동 경계부터 적용한다.
- Collection/SL/RL Job Config는 Job 시작 시 고정한다.
- Job 중 변경은 다음 Job부터 적용한다.
- Emotion 구조 변경은 새 Emotion Schema Version과 새 Dataset/모델을 요구한다.
- 변화량·감쇠 같은 Behavior Parameter는 Preset Version으로 추적한다.
- Reward Config 변경은 승패 규칙을 변경하지 않는다.

### 15.2 Version set

```text
environment_version
observation_schema_version
action_space_version
reward_config_version
emotion_schema_version
inference_model_artifact_format_version
dummy_preset_version
preprocessing_version
dataset_split_seed
```

### 15.3 Inference Artifact

SL 또는 RL TrainingJob이 정상 완료되거나 사용자·운영에 의해 조기 종료될 때, 모델 가중치와 필요한 Artifact Metadata가 생성된 상태라면 현재 학습 결과를 `.bangmodel`로 저장한다. `pause_failed`로 안전한 학습 중단으로 전환되는 경우에도 모델 결과가 저장 가능한 상태라면 Artifact를 저장한다.

모델 또는 가중치가 아직 생성되지 않은 매우 초기 단계에서 Job이 종료된 경우에는 유효한 Inference Artifact를 생성하지 않는다. 이 경우 종료·실패 상태와 원인을 로그에 남긴다. Artifact 저장 성공·실패 여부도 로그에 기록한다. 저장된 Artifact는 실제 존재하는 모델 가중치와 Metadata가 일치해야 한다.

단순 Data Collection 또는 Evaluation 실행은 Inference Artifact 자동 저장 대상이 아니다. Evaluation은 기존 Artifact를 읽어 사용한다.

저장 Artifact에는 다음을 포함한다.

- weights and architecture
- preprocessing/schema metadata
- IntentInference and EmotionInference identifiers/weights
- environment, observation, action, and artifact format versions
- reward Config Version
- training configuration, including `clip_epsilon` and `initial_baseline`
- emotion Schema Version
- total environment steps
- total MainAgent transitions
- active training time excluding pause
- normal win rate and truncated rate
- random seed, split seed, hardware settings, and execution statistics

`.bangmodel`은 ZIP archive 형식의 최종 추론용 Bundle이며, Bundle root에는 다음 세 파일만 둔다.

```text
manifest.json
metadata.json
model.safetensors
```

`manifest.json`은 다음 필드를 필수로 가진다.

```json
{
  "artifact_type": "inference",
  "inference_model_artifact_format_version": "1.0",
  "weights_file": "model.safetensors",
  "metadata_file": "metadata.json",
  "weights_sha256": "<sha256>",
  "metadata_sha256": "<sha256>"
}
```

`metadata.json`은 위에서 정의한 Design §15.3의 모델 구조·전처리/Schema·모듈 식별자/버전·Reward Config·학습 설정·카운터·평가·Seed·하드웨어·실행 통계를 포함한다. `model.safetensors`에는 IntentInference, EmotionInference, PPOModule의 실제 추론 가중치를 모듈 namespace로 저장한다. Manifest의 파일명, format version, SHA-256과 실제 Bundle 내용이 일치하지 않으면 Artifact를 호환되지 않는 것으로 처리한다. Phase 2 Inference Module Artifact는 최종 `.bangmodel`과 별도 역할이며 이 최종 Bundle 형식을 학습 Resume에 사용하지 않는다.

Optimizer, rollout buffer, Environment in-progress state, Worker state, 또는 그 밖의 프로그램 재시작 Resume state는 Bundle에 포함하지 않는다. 중간 학습 Checkpoint 저장과 프로그램 종료 후 학습 Resume는 제공하지 않는다.

### 15.4 Compatibility check

Artifact를 Game에 로드하기 전에 Model Architecture, Environment Version, Observation Schema Version, Action Space Version, Reward Config Version, Emotion Schema Version, Inference Model Artifact Format Version을 검사한다. 불일치 시 구체적인 field-level 오류를 반환하고 Game 실행을 거부한다. 이 검사는 학습 Resume가 아니라 inference execution compatibility만 검증한다.

### 15.5 초기 학습 설정값과 GUI 변경

초기 Design Config는 다음 값을 사용한다.

```yaml
training:
  ppo:
    rollout_transitions_per_worker: 256
    gamma: 0.99
    gae_lambda: 0.95
    ppo_epochs: 4
    minibatch_size: 64
    clip_epsilon: 0.2
  emotion_inference:
    initial_baseline: 0.0
```

- 모든 값은 코드에 하드코딩하지 않고 Config Schema와 Job Config Snapshot으로 관리한다.
- GUI의 Training/Model Settings에서 사용자가 수정할 수 있다.
- 수정 내용은 다음 Training Job부터 적용하며 진행 중인 Job에는 적용하지 않는다.
- 저장된 Inference Artifact의 Metadata에는 실제 사용한 값과 Config Version을 기록한다.

---

## 16. Pause와 Resource 관리

### 16.1 사용자 직접 자원 설정

사용자는 GUI에서 다음 값을 직접 설정한다.

- 사용할 CPU 코어 수
- 사용할 RAM 예산
- GPU 사용 여부
- Worker 수
- Batch Size
- 학습 Buffer 크기

프로그램은 입력값의 형식과 명백한 범위 오류만 검증한다. CPU를 몇 개 사용할지, Worker·Batch·Buffer를 어떻게 조합할지는 사용자 설정을 적용하며 자동 최적화하지 않는다.

### 16.2 GUI 권장 도움말

GUI는 다음과 같은 일반 권장사항과 주의사항을 표시할 수 있다.

> 운영체제와 GUI의 원활한 동작을 위해 사용 가능한 CPU 코어 전체보다 1~2개 적게 설정하는 것을 권장합니다.

권장 도움말은 사용자의 설정을 자동으로 변경하거나 특정 값을 강제하지 않는다. RAM과 GPU에 대해서도 입력값의 명백한 오류만 표시하고, 자동 성능 측정이나 OOM 탐색을 수행하지 않는다.

### 16.3 Pause/Resume

`TrainingService.resume()`는 현재 실행 중인 프로세스에서만 `paused` 상태를 해제한다. Pause 요청은 안전한 Environment/optimizer 경계에서 처리한다. 모든 Worker에 pause command를 보내고 각 Worker의 ACK를 기다린다. ACK timeout은 **30초**로 설정한다.

- 30초 안에 모든 Worker ACK가 도착하면 `paused` 상태로 표시한다.
- Timeout이 발생하면 `pause_failed` 상태를 기록한다.
- `pause_failed` 후 자동 재시도하지 않는다.
- 시스템은 안전한 학습 중단 절차로 전환하고 미완료 상태와 오류를 로그에 보존한다.
- Paused 동안 environment_step, MainAgent transition count, PPO update는 증가하지 않는다.
- Model/Environment/optimizer 상태는 메모리에만 유지하며 pause 시간은 training time에서 제외한다.
- 프로그램 종료 시 메모리 상태를 폐기하고, 이전 TrainingJob의 학습 상태를 복원하지 않는다.
- 프로그램 재시작 후 학습 Resume를 제공하지 않으며, 이를 위한 Checkpoint도 저장하지 않는다.

SL 또는 RL TrainingJob이 정상 완료되거나 조기 종료되면 §15.3의 Artifact 저장 조건에 따라 추론용 Artifact를 처리한다.

---

## 17. Logging과 오류 처리

### 17.1 오류 계층

```text
BangRLError
├── EnvironmentError
│   ├── InvalidActionError
│   ├── TerminalStateActionError
│   └── AgentOrderError
├── ConfigError
│   ├── InvalidConfigError
│   ├── MissingConfigKeyError
│   └── SchemaChangedError
├── CompatibilityError
├── DataError
└── TrainingError
    └── ResourceLimitError
```

`InvalidActionError`는 상태 변경·Step 증가·감정 갱신 전에 발생한다. AEC cleanup은 이 예외 경로와 분리한다.

### 17.2 로그 채널

- `episode_events`: 공개 이벤트 및 상태
- `reward_breakdown`: Agent별 구성요소 보상
- `invalid_action`: 전달 Action, Mask, Agent, Step, 상태 요약
- `termination`: terminated/truncated 및 원인
- `evaluation`: Protocol·집계·분모·역할별 결과
- `privileged_diagnostics`: 실제 역할 포함 가능, Agent 채널과 분리
- `training`: phase, step counters, transitions, update metrics

모든 로그에는 `run_id` 또는 `episode_id`를 포함한다. Truncated Episode 전체 진행은 `episode_events`와 `termination` 로그로 재구성할 수 있어야 한다.

---

## 18. 검증 전략과 인수 기준 추적

### 18.1 Environment tests

- PettingZoo AEC API 준수
- PASS/BANG 각각 1 Environment Step
- Invalid Action의 상태·차례·Step·감정 불변 및 예외
- AEC cleanup의 Step/transition 미생성
- 정상 승패 우선, max Step truncated
- truncated의 winner/loser·Terminal Reward·Bootstrap·학습 제외
- role reveal과 confirmed-role visibility

### 18.2 Agent/Model tests

- 모든 Agent가 Mask를 적용
- DummyAgent가 실제 비공개 역할 없이 faction strength 계산
- MainAgent MAIN-005 concatenate 구조
- EmotionInference output 범위/차원/MSE
- Schema 변경 시 Artifact compatibility 거부

### 18.3 Data/Training/Evaluation tests

- Parquet canonical과 NPZ regeneration
- termination_status 기록 및 truncated filter
- episode-level 80/10/10 split
- normal/truncated 평가 통계와 N/A Win Rate
- Reward Config 항목 변경·Version 기록
- pause barrier와 counter freeze
- inference Artifact 저장/로드 및 no-resume 보장

### 18.4 Requirements Acceptance 추적

| Acceptance 범위 | Design 위치 |
|---|---|
| AC-01~AC-05 | §4, §13, §18 |
| AC-06~AC-09 | §6, §7, §8 |
| AC-10~AC-12 | §5, §4, §11 |
| AC-13~AC-15 | §6, §9, §12 |
| AC-16~AC-17 | §14.2, §16.3 |
| AC-18~AC-19 | §15 |
| AC-20~AC-22 | §4, §10, §11, §12, §13 |
| AC-23~AC-25 | §8, §9, §10, §15 |

---

## 19. 디렉터리 레이아웃

```text
bang_rl/
├── env/
│   ├── bang_env.py
│   ├── domain.py
│   ├── events.py
│   ├── terminal.py
│   └── schemas.py
├── agents/
│   ├── base.py
│   ├── random_agent.py
│   ├── human_agent.py
│   ├── dummy_agent/
│   │   ├── dummy_agent.py
│   │   ├── role_belief.py
│   │   ├── intent_module.py
│   │   ├── emotion_module.py
│   │   └── action_selector.py
│   └── main_agent/
│       ├── main_agent.py
│       ├── preprocessor.py
│       ├── encoders.py
│       ├── intent_inference.py
│       ├── emotion_inference.py
│       └── ppo_module.py
├── runtime/
│   ├── emotion_runtime.py
│   ├── reward_runtime.py
│   ├── transition_collector.py
│   └── pause_coordinator.py
├── training/
│   ├── sl_trainer.py
│   ├── ppo_trainer.py
│   ├── rollout_buffer.py
│   ├── vec_env.py
│   └── resource_guard.py
├── data/
│   ├── collector.py
│   ├── parquet_writer.py
│   ├── tensor_cache_writer.py
│   ├── preprocessor.py
│   ├── splitter.py
│   └── loader.py
├── evaluation/
│   ├── evaluator.py
│   ├── protocol.py
│   └── metrics.py
├── app/
│   ├── game_service.py
│   ├── training_service.py
│   ├── evaluation_service.py
│   ├── config_service.py
│   ├── preset_service.py
│   └── compatibility_service.py
├── gui/
│   ├── main_window.py
│   ├── screens/
│   ├── widgets/
│   └── dialogs/
├── config/
├── models/                  # inference artifacts only
├── datasets/                # canonical and derived datasets
├── logs/
└── tests/
```

