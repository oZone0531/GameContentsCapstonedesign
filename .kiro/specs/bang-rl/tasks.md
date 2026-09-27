# Tasks — Bang RL Training System

> 버전: 0.7  
> 작성일: 2026-09-21  
> 기준 문서: `requirements.md` v0.7, `design.md` v0.7  
> 범위: 구현 및 검증 작업만 정의한다. 새로운 요구사항이나 설계 선택을 추가하지 않는다.

---

## 구현 원칙

- Environment는 Agent 구현 타입을 import하거나 구분하지 않는다.
- Invalid Action은 PASS로 변환하지 않고 상태 불변·로그·예외 처리한다.
- `Discrete(6)`와 Action Mask를 분리한다.
- 환경 Step, MainAgent transition, PPO update를 별도 기록한다.
- Normal termination은 ML 학습 대상이며, truncated Episode는 모든 ML Phase 학습에서 제외한다.
- RoleFusion은 현재 ConfirmedRoleHardOverride만 구현한다. Role Count Projection은 구현하지 않는다.
- EmotionInference는 Sigmoid, `[0,1]` 출력, MSE, `initial_baseline=0.0`을 사용한다.
- Reward 숫자값은 코드에 고정하지 않고 Config에서 읽는다.
- Hardware Benchmark, 자동 Worker/Batch 최적화, OOM 탐색은 구현하지 않는다.
- 중간 Checkpoint와 프로그램 종료 후 학습 Resume는 구현하지 않는다.
- Pause/Resume은 현재 실행 중인 프로세스의 메모리 상태에서만 지원한다.
- Task 완료 조건은 자동화 테스트 또는 검증 가능한 결과로 작성한다.

## Phase 의존성

```text
Implementation Phase 0 기반
    ↓
Implementation Phase 1 Environment
    ↓
Implementation Phase 2 Agent 기본 구현
    ├── Implementation Phase 3 MainAgent/Emotion
    ├── Implementation Phase 4 Emotion/Reward Runtime
    └── Implementation Phase 5 Dataset
            ↓
        ML Phase 2 (Implementation Phase 6): SL
            ↓
        ML Phase 3 (Implementation Phase 7): PPO
            ├── Implementation Phase 8 Evaluation
            ├── Implementation Phase 9 GUI/Application
            └── Implementation Phase 10 Compatibility/Logging/통합 검증
```

---

## Phase 0 — 프로젝트 기반 및 Config

- [x] 1. TASK-0-1. Python 패키지 및 디렉터리 초기화 (no dependencies)

**참조:** Requirements REQ-ARCH-001~006, Design §2, §19

**작업 내용:**
- `bang_rl/` 패키지 생성
- Design §19 디렉터리 구조에 따라 `env/`, `agents/`, `runtime/`, `training/`, `data/`, `evaluation/`, `app/`, `gui/`, `config/`, `tests/` 생성
- 실제 Dataset 저장 디렉터리 `datasets/canonical/`, `datasets/cache/` 생성
- `models/`, `logs/` 생성
- 각 Python 패키지에 `__init__.py` 추가
- `pyproject.toml` 작성
- `.gitignore` 작성

**완료 조건:**
- `python -c "import bang_rl"` 성공
- 각 독립 패키지 import 성공
- `env/`가 `agents/`와 `gui/`를 import하지 않는 구조 확인

- [x] 2. TASK-0-2. 의존성 및 개발 환경 정의 (depends on TASK-0-1)

**참조:** Requirements REQ-ENV-001, REQ-GUI-013, REQ-MAIN-003, Design §2, §14

**작업 내용:**
- PettingZoo AEC, Gymnasium, NumPy, Pydantic, PyYAML, PyTorch, pytest 의존성 정의
- PyQt6, pyqtgraph를 GUI용 optional dependency로 정의
- Parquet 처리를 위한 pyarrow 또는 동일 역할의 라이브러리 정의
- `.bangmodel`의 `model.safetensors` 처리를 위한 `safetensors==0.5.3` 의존성 정의
- `pyproject.toml`에 직접 의존성을 고정하고 `requirements.lock`에 전체 transitive dependency를 고정·커밋하는 lock 정책 정의
- 학습용 dependency와 GUI용 optional dependency의 경계 유지

**완료 조건:**
- GUI dependency 없이 학습/테스트 dependency 설치 명령이 성공
- GUI dependency 없이 학습/테스트 환경에서 `pytest --collect-only` 성공
- GUI dependency가 설치된 개발/GUI 환경에서 PyQt6 및 pyqtgraph import 검증
- `safetensors==0.5.3` 직접 의존성과 `requirements.lock` 기반 설치 검증

- [x] 3. TASK-0-3. Config Schema 구현 (depends on TASK-0-2)

**참조:** Requirements REQ-CFG-001~008, REQ-HW-001~007, REQ-RWD-004~008, Design §15

**작업 내용:**
- `SystemConfig`
- `EnvConfig`
- `RewardConfig`
- `IntentPolicyConfig`
- `EmotionConfig`
- `PresetConfig`
- `SLTrainingConfig`
- `RLTrainingConfig`
- `EvaluationConfig`
- `PauseConfig`
- `HardwareConfig`
- `ArtifactConfig`
- `PPOConfig` 구현
- `DatasetConfig` 구현: `canonical_shard_max_episodes=10000`, `canonical_shard_max_bytes=256000000`
- Design §10.1의 Reward Config Schema 구현: `reward_config.version`, terminal/shaping namespace, Component `id`, `enabled`, `condition`, `value`
- Design §8 및 §11.5의 고정 Intent/Emotion/PPO 모델 구조와 optimizer/learning rate를 Job Snapshot 및 Artifact Metadata에 기록

PPO 초기값:

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
dataset:
  canonical_shard_max_episodes: 10000
  canonical_shard_max_bytes: 256000000  # 256 MB
```

Hardware 입력:

```yaml
hardware:
  cpu_cores: <user value>
  ram_gb: <user value>
  use_gpu: <user value>
  workers: <user value>
  batch_size: <user value>
  buffer_size: <user value>
```

**완료 조건:**
- Config 파일이 Pydantic 모델로 검증됨
- `max_episode_steps` 기본값이 2,000
- `warn_turn_threshold < critical_turn_threshold < max_episode_steps` 검증
- `event_history.max_events: null` 또는 양의 정수 검증
- `DatasetConfig`의 `canonical_shard_max_episodes=10000`, `canonical_shard_max_bytes=256000000` 및 Config override 검증
- PPO 초기값 검증
- Reward Config의 `version`, terminal/shaping 분리, Component `id`/`enabled`/`condition`/`value` 필수 필드 검증
- Terminal Reward와 Shaping Reward가 별도 namespace로 저장·계산됨
- Reward 숫자값이 코드에 하드코딩되지 않음
- Design §8 및 §11.5의 모델 구조·optimizer·learning rate가 검증 가능한 고정값으로 Snapshot에 기록됨
- `initial_baseline`이 `[0,1]` 범위인지 검증
- Hardware 입력의 형식 및 명백한 범위 오류 검증
- Benchmark 관련 Config가 없음

- [x] 4. TASK-0-4. Config Loader 및 Job Snapshot 구현 (depends on TASK-0-3)

**참조:** Requirements REQ-CFG-005~008, REQ-REPR-001~004, Design §2.3, §15

**작업 내용:**
- YAML/JSON 로더 구현
- Config Version 관리
- TrainingJob 시작 시 Config/Preset/Reward/Emotion Schema/Model Architecture/Optimizer/Seed/Split 정보를 Snapshot으로 고정
- 일반 게임의 비구조 Config는 다음 행동 경계부터 적용
- 학습 중 Config 변경은 다음 Job부터 적용

**완료 조건:**
- 진행 중 TrainingJob의 Snapshot이 외부 Config 변경으로 변하지 않음
- 변경 Config가 다음 Job에서만 적용됨
- Snapshot에 실제 사용값이 저장됨

- [x] 5. TASK-0-5. Seed 및 Logging 기반 구현 (depends on TASK-0-4)

**참조:** Requirements REQ-STEP-015~019, REQ-REPR-002~003, Design §17

**작업 내용:**
- Episode/Worker/Agent용 Seed 파생 구현
- JSON Lines 로그 포맷 구현
- `episode_events`, `termination`, `reward_breakdown`, `invalid_action`, `training`, `evaluation`, `privileged_diagnostics` 채널 정의
- Episode/Run ID 자동 부여

**완료 조건:**
- 동일 Seed와 동일 Snapshot에서 재현 가능한 RNG 생성
- 로그에 `episode_id` 또는 `run_id` 포함
- privileged log가 Agent 공개 채널과 분리됨

- [x] 6. TASK-0-6. 예외 계층 구현 (depends on TASK-0-5)

**참조:** Requirements REQ-ACT-006~010, REQ-COMPAT-001~002, Design §17

**작업 내용:**
- `InvalidActionError`
- `TerminalStateActionError`
- `AgentOrderError`
- `ConfigError`
- `SchemaChangedError`
- `CompatibilityError`
- `DataError`
- `TrainingError`
- `ResourceLimitError`

**완료 조건:**
- 예외 계층 단위 테스트 통과
- Invalid Action과 AEC 정리 호출 예외 경로가 분리됨

---

Implementation Phase 1 — BangEnvironment

- [x] 7. TASK-1-1. Environment 도메인 타입 구현 (depends on TASK-0-6)

**참조:** Requirements REQ-ENV-002~007, REQ-WIN-001~008, Design §3

**작업 내용:**
- `Role`, `Team`, `ActionType`, `EventType` 구현
- `EnvironmentState` 구현
- 체력 초기값 구현: Sheriff=5, 나머지=4
- 역할 배정 구현: canonical role list와 Episode Seed 기반 NumPy `Generator(PCG64).permutation`을 사용하여 slot 0~4에 배치
- 고정 턴 순서 구현
- `GameEvent`, `TerminalResult` 구현

**완료 조건:**
- 5개 슬롯에 Sheriff 1, Deputy 1, Outlaw 2, Traitor 1 배정
- 역할별 초기 체력 검증
- 동일 Episode Seed에서 canonical role list의 동일 permutation이 생성되고 다른 역할 RNG/shuffle이 사용되지 않음
- Event ID와 Environment Step 필드 존재
- PASS target은 Canonical/public 의미에서 `null`로 표현하고, Dataset/Tensor 저장 또는 변환 과정에서만 `255` sentinel representation을 사용하며 실제 유효 Action이나 실제 Role Label로 해석하지 않음 (`Discrete(6)` Action Space 의미는 변경하지 않음)

- [x] 8. TASK-1-2. Action Space와 Action Mask 구현 (depends on TASK-1-1)

**참조:** Requirements REQ-ACT-001~005, REQ-ACT-008, Design §5

**작업 내용:**
- `Discrete(6)` Action Space 구현
- PASS mask 항상 1
- 자기 자신 공격 mask=0
- 탈락 플레이어 공격 mask=0
- 생존한 타 플레이어 공격 mask=1
- Agent별 공통 Mask API 구현

**완료 조건:**
- 모든 Agent가 동일한 Action Space를 받음
- 모든 상태 조합에서 Mask가 규칙과 일치
- Mask shape `(6,)`, dtype `uint8`, 값 0/1

- [x] 9. TASK-1-3. Invalid Action 최종 검증 구현 (depends on TASK-1-2)

**참조:** Requirements REQ-ACT-005~010, REQ-STEP-002~003, Design §4.1, §5.3, §17

**작업 내용:**
- 타입 오류 검증
- 범위 밖 Action 검증
- Mask 미허용 Action 검증
- Invalid Action log 작성
- `InvalidActionError` 발생
- 상태·차례·Step·Event·감정 상태 rollback/불변 보장
- AEC `step(None)` cleanup 경로 분리

**완료 조건:**
- Invalid Action 후 HP/Alive/Selection/Event/Step이 모두 동일
- 감정 갱신·감쇠가 실행되지 않음
- PASS 자동 변환 없음
- 예외와 별도 로그 생성
- cleanup 호출은 정상적으로 처리되며 일반 Invalid Action으로 기록되지 않음

- [x] 10. TASK-1-4. BANG/PASS 실행 및 Environment Step 구현 (depends on TASK-1-3)

**참조:** Requirements REQ-ENV-004~006, REQ-STEP-001~005, Design §4.1

**작업 내용:**
- 유효 BANG 처리
- 대상 HP 1 감소
- 유효 PASS 처리
- 유효 행동마다 `environment_step += 1`
- Invalid Action 및 cleanup은 Step 미증가
- 탈락 Agent 행동 제외
- 공개 Event 생성

**완료 조건:**
- BANG 1회와 PASS 1회가 각각 Environment Step 1회
- Invalid Action은 Step을 증가시키지 않음
- 탈락 플레이어에게 행동 기회가 주어지지 않음
- 행동별 HP 변화가 Event에 기록됨

- [x] 11. TASK-1-5. 역할 공개 및 Observation 구현 (depends on TASK-1-4)

**참조:** Requirements REQ-OBS-001~003, REQ-OBS-006~018, Design §6

**작업 내용:**
- Sheriff 공개
- 탈락 시 Role Reveal Event 생성
- 자기 역할 공개
- 다른 생존자 hidden role 차단
- `role_visibility`, `revealed_roles`, `confirmed_role_distribution` 구현
- 공개 정보 기반 확정 Role 표현
- Canonical Observation 생성

**완료 조건:**
- 탈락 즉시 역할 공개
- 공개 역할이 이후 Observation/Event에 포함됨
- hidden true role이 생존자 Observation에 없음
- 확정 역할은 one-hot으로 표현
- Observation Space와 실제 dtype/shape 일치

- [x] 12. TASK-1-6. Terminal Condition과 종료 유형 구현 (depends on TASK-1-5)

**참조:** Requirements REQ-WIN-001~008, REQ-STEP-006~014, Design §4.2~4.3

**작업 내용:**
- Sheriff Team 승리 조건
- Traitor 단독 생존 승리 조건
- Sheriff 사망 후 비-Traitor 생존자 존재 시 Outlaw Team 승리
- Duel 상태 미생성
- `terminated`/`truncated` 분리
- `max_episode_steps=2000` 기본값
- 정상 승패 평가 후 truncated 평가
- truncated 시 winner/loser 없음
- truncated 시 Terminal Reward/Bootstrap/PPO 학습 제외

**완료 조건:**
- 정상 승패가 마지막 Step에서 truncated보다 우선
- 모든 정상 생존자 조합에 올바른 승자 정의
- Truncated Episode에 승자·패자가 없음
- `termination_status=normal|truncated` 저장
- `terminated`와 `truncated`가 동시에 true가 되지 않음

- [x] 13. TASK-1-7. Long-game Diagnostic 로그 구현 (depends on TASK-1-6)

**참조:** Requirements REQ-STEP-015~020, Design §4.4, §17

**작업 내용:**
- `warn_turn_threshold` 초과 로그
- `critical_turn_threshold` 초과 로그
- Episode별·threshold별 1회 제한
- Environment Event와 실행 계층 Metadata 결합
- privileged diagnostics 분리

**완료 조건:**
- 임계값 순서 Config 검증
- 200/500 threshold 초과 로그 생성
- 같은 Episode에서 같은 threshold 로그가 반복되지 않음
- Truncated 전체 진행 재구성 가능

- [x] 14. TASK-1-8. PettingZoo AEC 검증 (depends on TASK-1-7)

**참조:** Requirements REQ-ENV-001, REQ-ACT-001~010, REQ-STEP-001~020, Design §4

**완료 조건:**
- PettingZoo AEC API test 통과
- reset/step/observe/agent selection 수명주기 검증
- terminated/truncated와 cleanup 동작 검증
- Environment가 agents/gui를 import하지 않음

---

Implementation Phase 2 — Agent 기본 구현

- [x] 15. TASK-2-1. BaseAgent·HumanAgent·RandomAgent 구현 (depends on TASK-1-8)

**참조:** Requirements REQ-ARCH-003~004, REQ-ACT-004, Design §7.1

**작업 내용:**
- `BaseAgent` Protocol/ABC 구현
- `HumanAgent` 입력 큐 구현
- `RandomAgent` Mask 기반 샘플링 구현

**완료 조건:**
- RandomAgent가 mask=1 Action만 선택
- HumanAgent가 GUI 없이 Queue 테스트 가능
- Agent가 Environment 내부 구조를 import하지 않음

- [x] 16. TASK-2-2. DummyAgent RoleBeliefEstimator 구현 (depends on TASK-2-1)

**참조:** Requirements REQ-DUMMY-003-A~B, REQ-OBS-012~018, Design §7.2

**작업 내용:**
- 자기 역할·Sheriff·탈락 공개 역할 반영
- 남은 역할 기반 기본 확률 계산
- 공개 정보로 논리적 확정 역할 처리
- MainAgent 추론 모듈 의존 금지

**완료 조건:**
- hidden true role 없이 Role distribution 생성
- 확정 역할은 one-hot
- 미확정 역할은 잔여 역할 비율 기반 확률

- [x] 17. TASK-2-3. DummyAgent faction strength와 Intent 정책 구현 (depends on TASK-2-2)

**참조:** Requirements REQ-DUMMY-002~005, REQ-ACT-003~004, Design §7.2

**작업 내용:**
- 생존자 수 우선 진영 비교
- 동률 시 `Σ(public_hp × faction_probability)` 비교
- 동률 Config 정책
- 배신자 공격 우선순위
- 1대1 상황 Sheriff 공격 우선
- 공통 Action Mask 적용

**완료 조건:**
- 실제 hidden role 없이 faction strength 계산
- 자기 자신/탈락자 공격이 차단됨
- 비1대1과 Sheriff-Traitor 1대1 정책 차이 검증

- [x] 18. TASK-2-4. Dummy EmotionModule 구현 (depends on TASK-2-3)

**참조:** Requirements REQ-DUMMY-006~010, REQ-EMO-001~008, Design §7.3, §9

**작업 내용:**
- Config 기반 N개 감정 상태
- 값 범위 `[0,1]`
- initial value, 변화량, 감쇠, 행동 영향 로딩
- EmotionRuntime 호출 API 구현
- event ID 중복 반영 방지
- Invalid Action 시 갱신 방지

**완료 조건:**
- 감정 항목 추가/삭제가 Config만으로 가능
- 환경 Step당 감쇠 최대 1회
- 동일 Event ID 중복 반영 없음
- Invalid Action/재시도 시 감정 변화 없음

- [x] 19. TASK-2-5. Dummy ActionSelector와 통합 (depends on TASK-2-4)

**참조:** Requirements REQ-DUMMY-010, REQ-ACT-008, Design §7

**작업 내용:**
- Intent distribution 생성
- Emotion effect 적용
- Action Mask 적용
- Mask 후 확률 재정규화
- 최종 Action sampling

**완료 조건:**
- DummyAgent 전체 게임 실행
- 항상 mask=1 Action 반환
- Environment 구현과 독립 실행

---

Implementation Phase 3 — MainAgent 추론 구조

- [x] 20. TASK-3-1. Observation Preprocessor 구현 (depends on TASK-2-5)

**참조:** Requirements REQ-OBS-006~008, REQ-MAIN-002~004, Design §6, §8.1, §6.3

**작업 내용:**
- Fixed State/Player Matrix 처리
- Event actor별 분리
- 타 플레이어 4명 분리
- `model_history_length=64`
- 최근 Event를 뒤쪽에 두는 왼쪽 Padding
- 오래된 Event Truncation
- valid mask 생성
- null/255 sentinel 처리

**완료 조건:**
- 0/1/64/초과 길이 Event Sequence 처리
- 출력 길이 항상 64
- 자기 자신 Event 제외
- Event 시간 순서 보존

- [x] 21. TASK-3-2. SequenceEncoder와 GRUEncoder 구현 (depends on TASK-3-1)

**참조:** Requirements REQ-MAIN-003, Design §8.1

**작업 내용:**
- `SequenceEncoder` Interface
- GRU 기본 Encoder: `hidden_size=128`, `num_layers=1`, `batch_first=True`, dropout 없음
- Observation Preprocessor 입력 차원 사용 및 valid mask 기반 마지막 유효 hidden state 처리
- 빈 Sequence와 valid length 처리
- 향후 Encoder 교체 가능한 Factory

**완료 조건:**
- GRU 출력 shape 검증
- 빈 History에서 valid mask 처리
- 동일 입력 결정성 테스트
- IntentInference가 Interface에만 의존

- [x] 22. TASK-3-3. IntentInferenceModule 구현 (depends on TASK-3-2)

**참조:** Requirements REQ-MAIN-002~003, REQ-OBS-012~015, Design §8.1

**작업 내용:**
- 타 플레이어 4명 Role distribution 출력
- shape `(4,5)`
- softmax probability
- 확정 역할과 미확정 역할 분리
- hidden true role 입력 차단

**완료 조건:**
- 각 플레이어 분포 합이 1.0
- hidden true role이 forward 입력에 없음
- 미공개 역할 성능 측정용 출력 제공

- [x] 23. TASK-3-4. RoleFusion Hard Override 구현 (depends on TASK-3-3)

**참조:** Requirements REQ-OBS-012~018, Design §8.2

**작업 내용:**
- `RoleFusion` Interface 구현
- confirmed role 존재 시 one-hot override
- 미확정 Role은 raw IntentInference output pass-through
- PPOModule이 raw output을 직접 사용하지 않도록 연결
- Role Count Projection 호출 경로 미생성

**완료 조건:**
- Sheriff/탈락 공개 Role이 one-hot으로 변환됨
- 기존 model distribution과 관계없이 confirmed role 우선
- 미확정 distribution이 변경 없이 전달됨
- Remaining Role Count/Assignment 계산 코드가 없음

- [x] 24. TASK-3-5. EmotionInferenceModule 구현 (depends on TASK-3-4)

**참조:** Requirements REQ-MAIN-004~004-C, REQ-EMO-001~008, Design §8.3, §9

**작업 내용:**
- 공개 정보만 입력
- GRU `hidden_size=128`, `num_layers=1`, `batch_first=True`, dropout 없음
- valid mask 기반 마지막 유효 hidden state 처리
- 공유 Emotion Head `Linear(128,64) → ReLU → Linear(64,E) → Sigmoid`
- Sigmoid 출력
- 출력 shape `(4,E)`
- 출력 범위 `[0,1]`
- `training.emotion_inference.initial_baseline=0.0` 빈 History 처리
- valid mask 기반 빈 History 처리
- 행동 직전 Emotion Label 정렬
- MSE Loss

**완료 조건:**
- 출력값이 모두 `[0,1]`
- 빈 History에서 baseline 0.0과 valid mask 사용
- 실제 Emotion Label이 입력 Feature에 없음
- 현재 행동·미래 Event 누출 없음
- MSE 계산 검증

- [x] 25. TASK-3-6. PPOModule 구현 (depends on TASK-3-5)

**참조:** Requirements REQ-MAIN-005~008, REQ-PPO-015~018, Design §8.4

**작업 내용:**
- Self State + Board State + Confirmed Role + fused Intent + Emotion concatenate
- Shared trunk `Linear(input_dim,256) → ReLU → Linear(256,256) → ReLU`
- Policy Head `Linear(256,6)`
- Value Head `Linear(256,1)`
- Action Mask 적용
- Role/Emotion Label, Reward, privileged log 입력 차단

**완료 조건:**
- Policy output shape `(6,)`
- Mask=0 Action probability 0
- Value input에 hidden true role 없음
- PPOModule이 RoleFusion output을 사용

---

Implementation Phase 4 — Emotion Runtime 및 Reward Runtime

- [x] 26. TASK-4-1. EmotionRuntime 구현 (depends on TASK-3-6)

**참조:** Requirements REQ-EMO-001~008, Design §9

**작업 내용:**
- 공개 Event 수신
- Event ID별 1회 반영
- 환경 Step당 감쇠 1회
- 행동 직전 Label snapshot
- Label 입력 boundary를 행동 전 Observation까지로 고정
- 현재 행동과 그 행동으로 생성된 새 Event를 해당 Label 입력에서 제외
- 미래 Event를 해당 Label 입력에서 제외
- 비행동 플레이어 상태 정렬
- Invalid Action 시 sequence/update/label snapshot을 진행하지 않음

**완료 조건:**
- 공개 Event 반영 → 감정 갱신/감쇠 → Label 기록 → 행동 선택 → 실행 → 새 Event 생성 순서 검증
- 행동 직전 DummyAgent EmotionModule 상태가 Target인지 검증
- 현재 행동·새 Event·미래 Event가 Label 입력에 포함되지 않음
- 동일 Event ID가 한 번만 반영됨
- 과거/미래 Event 누출 테스트
- Invalid Action/retry 시 Emotion sequence/update/label 변화 없음
- Environment가 DummyAgent 내부 상태를 직접 조작하지 않음

- [x] 27. TASK-4-2. Reward DSL 및 RewardCalculator 구현 (depends on TASK-4-1)

**참조:** Requirements REQ-RWD-001~009, REQ-PPO-002, Design §10

**작업 내용:**
- YAML Declarative Reward DSL
- `reward_config.version` 필드 구현
- Terminal Reward와 Shaping Reward를 별도 namespace로 구현
- 각 Component에 `id`, `enabled`, `condition`, `value` 필드 구현
- `condition.type`별 필드 검증: `terminal_result(result)`, `event(event)`, `role_attack(actor_role/target_role)`, `faction_attack(actor_faction/target_faction)`, `pass`
- Terminal Component ID: `terminal_win`, `terminal_loss`
- Shaping Component ID: `bang_hit`, `kill_enemy`, `attack_sheriff_outlaw`, `attack_sheriff_deputy`, `role_or_faction_attack_terms`, `pass_penalty`
- 조건 만족 Component 모두 합산
- 기본 clipping 비활성화
- Config 기반 clipping 옵션
- Terminal Reward와 Shaping 분리
- Reward 숫자값 코드 하드코딩 금지
- Reward Breakdown Logging
- Reward Config Version 기록

**완료 조건:**
- Component 활성화/비활성화 동작
- 복수 Component 동시 합산
- 기본값 숫자가 코드에 하드코딩되지 않음
- Reward Config 변경이 Terminal Condition을 변경하지 않음
- Normal termination에만 Terminal Reward 기록
- Truncated에는 Terminal Reward 없음

---

Implementation Phase 5 — Canonical Dataset 및 Tensor Cache

- [x] 28. TASK-5-1. EpisodeCollector 구현 (depends on TASK-4-2)

**참조:** Requirements REQ-DATA-001~010, REQ-OBS-009, REQ-EMO-001~008, Design §12

**작업 내용:**
- Episode Event/State 수집
- `termination_status=normal|truncated`에 따라 Normal/Truncated Episode 모두 Canonical Record로 수집하고, Truncated Episode를 버리지 않음
- Role Reveal Event 수집
- Role Label 수집
- Emotion Label 수집
- Reward Breakdown 수집
- Environment/Agent/Reward/Schema metadata 수집
- Environment 종료 후 메모리 정리

**완료 조건:**
- Normal Episode와 Truncated Episode 모두 Canonical Record 생성
- Truncated Episode를 Canonical Dataset 보존 대상에서 버리지 않음
- 원본 Event가 매 Step 전체 복제되지 않음
- Event ID와 environment_step 보존
- PASS target nullable/255 처리

- [x] 29. TASK-5-2. Parquet Canonical Shard Writer 구현 (depends on TASK-5-1)

**참조:** Requirements REQ-PIPE-002~002-B, REQ-DATA-004, REQ-DATA-008, Design §12.1

**작업 내용:**
- `datasets/canonical/` 저장
- `DatasetConfig.canonical_shard_max_episodes`와 `DatasetConfig.canonical_shard_max_bytes`를 사용하여 Episode 수 또는 byte 수 중 먼저 도달하면 rollover
- Zstandard compression
- 10,000 Episode 또는 256MB 중 먼저 도달 시 shard rollover
- Episode 단위 원본 보존
- 정상/Truncated 모두 저장

**완료 조건:**
- shard rollover가 `DatasetConfig`의 Episode/byte 기준을 사용하고 Config override가 적용됨
- Parquet round-trip 값 동일
- shard rollover 조건 테스트
- termination status, version, seed 저장
- 완료 Episode 저장 후 불필요한 메모리 해제

- [x] 30. TASK-5-3. Episode Splitter 구현 (depends on TASK-5-2)

**참조:** Requirements REQ-SPLIT-001~005, Design §12.4

**작업 내용:**
- Episode ID 단위 80/10/10 split
- split seed 기록
- split 결과 manifest 생성
- 한 Episode의 모든 Observation/Sequence/Label을 동일 split에 배치
- train split 기반 전처리 통계 계산

**완료 조건:**
- Episode overlap 0
- split 비율 검증
- 동일 split seed 재실행 결과 동일
- validation/test 정보가 통계 계산에 미사용

- [x] 31. TASK-5-4. NPZ Tensor Cache Writer/Loader 구현 (depends on TASK-5-3)

**참조:** Requirements REQ-DATA-004-A~C, REQ-DATA-010, Design §12.2~12.3

**작업 내용:**
- `datasets/cache/` 저장
- source content hash 생성
- preprocessing/Observation/Emotion Schema/split seed를 hash에 포함
- hash 변경 시 재생성
- `termination_status=normal` Episode만 ML Phase 2/ML Phase 3 학습용 Tensor Cache에 포함하고, Truncated Episode는 Canonical Dataset 원본에 보존하되 Cache에서는 제외
- Truncated Episode를 ML Phase 2/Phase 3에 포함시키는 별도 경로를 만들지 않음
- dtype 적용
- model_history_length=64 padding/mask 저장

**완료 조건:**
- 동일 source/config에서 cache hash 동일
- source/config 변경 시 cache 재생성
- Truncated Episode는 Canonical Dataset 원본에는 보존되지만 ML Phase 2/ML Phase 3 학습용 Cache에는 포함되지 않음
- NPZ round-trip 및 batch shape 검증
- HP/Role/Action/Player Index/Step/Event dtype 검증

---

Implementation Phase 6 — ML Phase 2: Supervised Learning

- [x] 32. TASK-6-1. IntentInference SL Trainer 구현 (depends on TASK-5-4)

**참조:** Requirements REQ-MAIN-002~003, REQ-PIPE-003~004, Design §8.1, §12.5

**작업 내용:**
- normal-only Tensor Cache 로딩
- Role Label Cross-Entropy 학습
- 학습 시 softmax 이전 logits를 Cross-Entropy Loss에 사용
- inference/output 단계에서 logits에 softmax를 적용하여 `(4,5)` role probability 생성
- 최종 공개 IntentInference output은 Design §8.1의 probability 형태 유지
- train/validation/test 분리 사용
- GRU Encoder `hidden_size=128`, `num_layers=1`, `batch_first=True`, dropout 없음
- 공유 Role Head `Linear(128,64) → ReLU → Linear(64,5)`
- Optimizer: Adam, `learning_rate=1e-3`, `betas=(0.9,0.999)`, `eps=1e-8`, `weight_decay=0`
- 학습 loss/accuracy Logging

**완료 조건:**
- validation/test data가 optimizer에 사용되지 않음
- Cross-Entropy 입력이 logits이며, public output은 softmax probability임을 검증
- Role 분류 metric Logging
- 확정/미공개 역할 성능 분리 가능
- IntentInference module weights와 학습 metadata 생성
- 결과가 최종 게임용 `.bangmodel`과 구분됨

- [x] 33. TASK-6-2. EmotionInference SL Trainer 구현 (depends on TASK-6-1)

**참조:** Requirements REQ-MAIN-004-A~C, REQ-DATA-002, REQ-DATA-006~007, Design §8.3, §9

**작업 내용:**
- 공개 정보 Feature만 로딩
- GRU `hidden_size=128`, `num_layers=1`, `batch_first=True`, dropout 없음
- 공유 Emotion Head `Linear(128,64) → ReLU → Linear(64,E) → Sigmoid`
- `[0,1]` 연속 출력
- `initial_baseline=0.0` 빈 History 처리
- MSE Loss
- Optimizer: Adam, `learning_rate=1e-3`, `betas=(0.9,0.999)`, `eps=1e-8`, `weight_decay=0`
- 행동 직전 Label 정렬
- train/validation/test metric Logging

**완료 조건:**
- MSE가 실제 행동 직전 Emotion Label과 계산됨
- 현재 행동과 그 행동으로 생성된 새 Event가 입력에 없음
- 미래 Event가 입력에 없음
- Invalid Action에서 sequence/update/label snapshot이 진행되지 않음
- output 범위 검증
- Emotion Schema Version 불일치 거부
- EmotionInference module weights와 학습 metadata 생성
- 결과가 최종 게임용 `.bangmodel`과 구분됨

- [x] 34. TASK-6-3. ML Phase 2 Inference Module Artifact 저장 (depends on TASK-6-2)

**참조:** Requirements REQ-INFER-001~003, Design §12.5, §15.3

**작업 내용:**
- IntentInference/EmotionInference 가중치 저장
- preprocessing/schema metadata 저장
- Config/Schema Version 저장
- ML Phase 3 초기화에 사용할 Inference Module Artifact 생성
- 중간 Checkpoint·Optimizer state 저장 금지

**완료 조건:**
- ML Phase 2 Artifact를 ML Phase 3에서 로드 가능
- 최종 게임용 `.bangmodel`과 ML Phase 2 Module Artifact를 구분
- Artifact와 실제 가중치/Metadata 일치

---

Implementation Phase 7 — ML Phase 3: PPO Training

- [x] 35. TASK-7-1. Vectorized Environment 및 Worker Runtime 구현 (depends on TASK-6-3)

**참조:** Requirements REQ-HW-001~007, REQ-PIPE-006, Design §2, §16

**작업 내용:**
- 사용자가 지정한 Worker 수 적용
- Worker별 독립 Seed
- DummyVecEnv/SubprocVecEnv 구현
- 사용자가 지정한 Batch/Buffer 적용
- 입력값 명백한 범위 오류 차단
- 자동 Benchmark·자동 최적화·OOM 탐색 없음

**완료 조건:**
- 지정 Worker 수와 실제 Worker 수 일치
- 지정 Batch/Buffer 적용
- CPU/RAM/GPU 설정 Snapshot 기록
- Benchmark 실행 경로가 없음

- [x] 36. TASK-7-2. Pending Transition Ledger 구현 (depends on TASK-7-1)

**참조:** Requirements REQ-PPO-007~014, Design §11.2

**작업 내용:**
- MainAgent 행동 직후 Pending Ledger에 미확정 transition 생성
- MainAgent의 한 행동 직후 생성된 Pending transition은 다음 MainAgent 행동 기회 또는 Episode 종료까지 유지
- Pending transition에 기록하는 reward는 Design §11.1~§11.2의 MainAgent transition reward semantics를 그대로 따르며, Tasks에서 별도의 reward aggregation 규칙을 정의하거나 재해석하지 않음
- 하나의 MainAgent transition에 여러 Environment Step이 포함될 때, 해당 구간의 Environment Step별 reward와 MainAgent transition reward를 별도로 기록하며 Environment Step counter와 transition boundary를 혼동하지 않음
- Reward DSL/RewardCalculator의 shaping reward와 terminal reward를 구분하여 기록
- 다음 MainAgent 행동 기회에서 이전 transition을 확정하되, 이미 확정된 이전 transition은 다시 수정하지 않음
- MainAgent 탈락 시 추가 행동·가짜 PASS·존재하지 않는 next state 생성 금지
- MainAgent 탈락 시점에 확정할 실제 Pending transition이 없으면 terminal reward 전달을 위한 가짜 transition을 생성하지 않음
- MainAgent 탈락 때문에 존재하지 않는 다음 MainAgent 행동 기회를 생성하지 않음
- `terminated`, `truncated`, `main_agent_eliminated` 상태를 Design §11.2에 따라 기록
- Normal Episode 종료 시 Design §11.2에 따라 실제 존재하는 해당 Episode의 마지막 미확정 Pending transition(미완료 transition)에 Episode 결과에 따른 terminal reward를 반영하고 `terminated=True`로 확정
- Normal Episode 결과 확정 후 `trainable=True` transition만 PPO Buffer로 이동
- Truncated Episode에서는 terminal reward를 생성하거나 PPO Buffer에 반영하지 않고 `trainable=False`로 폐기

**완료 조건:**
- MainAgent 행동부터 다음 MainAgent 행동 기회 또는 Episode 종료까지 Pending transition이 유지됨
- 여러 Environment Step이 포함된 MainAgent transition에서 reward 처리 범위가 Design §11.1~§11.2의 transition reward semantics와 일치하고, 별도의 reward aggregation 규칙을 추가하지 않음
- Environment Step counter, Environment Step별 reward, MainAgent transition reward 및 transition boundary가 서로 혼동되지 않음
- MainAgent 탈락 후 transition 추가 생성 없음
- MainAgent 탈락 + 실제 Pending transition 존재 시 해당 실제 transition만 유지·확정 대상으로 처리하고 가짜 transition/PASS/next state를 생성하지 않음
- MainAgent 탈락 + Pending transition 부재 시 terminal reward 전달을 위한 가짜 transition/PASS/next state를 생성하지 않음
- 이미 확정된 이전 transition이 Normal Episode 종료 시 다시 수정되지 않음
- Normal Episode 종료 시 terminal reward가 실제 존재하는 마지막 미확정 Pending transition에만 반영되고, 마지막 `terminated=True` transition이 `trainable=True && truncated=False` 조건으로 PPO Buffer에 반영됨
- 마지막 transition 이후 Bootstrap 없음
- Truncated Episode에서는 terminal reward를 생성하지 않고 transition이 PPO Buffer에 들어가지 않음
- 정상 Episode 비-terminal transition은 GAE/Bootstrap 대상
- `terminated`, `truncated`, `main_agent_eliminated` 상태가 기록됨

- [x] 37. TASK-7-3. PPO Rollout/Update 구현 (depends on TASK-7-2)

**참조:** Requirements REQ-PPO-001~018, Design §11

**작업 내용:**
- Environment Step/transition/PPO update counter 분리
- Worker당 rollout transition 256
- gamma 0.99
- GAE lambda 0.95
- PPO epochs 4
- minibatch 64
- clip epsilon 0.2
- PPO optimizer: Adam, `learning_rate=3e-4`, `betas=(0.9,0.999)`, `eps=1e-8`, `weight_decay=0`
- Rollout 시 실제 선택 Action, Action Mask, 해당 정책의 transition `log_prob`를 PPO의 `old_log_prob`로 저장
- PPO Update 시 동일한 Action Mask를 사용하여 같은 Action에 대한 현재 정책의 `current_log_prob`를 재계산
- PPO ratio는 저장된 `old_log_prob`와 PPO Update에서 계산한 `current_log_prob`로 계산
- Rollout 당시 저장한 `old_log_prob`를 PPO Update 시 현재 정책으로 다시 계산하여 대체하지 않음
- Rollout과 PPO Update에서 동일한 Action Mask semantics를 사용하여 old/current policy 비교 조건 유지
- `trainable=True && truncated=False` transition만 PPO Update 입력
- 정상 Episode 비-terminal transition GAE/Bootstrap
- 마지막 `terminated=True` transition은 Bootstrap 없음

**완료 조건:**
- 모든 초기값이 Config에서 로드됨
- GUI 변경값이 다음 Job부터 적용
- Design §11.5의 PPO Adam optimizer와 `learning_rate=3e-4`, `betas`, `eps`, `weight_decay` 값이 적용됨
- Rollout transition에 실제 선택 Action, Rollout 시점 Action Mask 및 `old_log_prob`가 저장됨
- PPO Update에서 동일한 Action Mask로 `current_log_prob`를 계산하고, 저장된 `old_log_prob`를 대체하지 않음
- PPO ratio가 저장된 `old_log_prob`와 Update 시 계산한 `current_log_prob`로 계산됨
- Truncated transition이 PPO Update에 들어가지 않음
- Rollout boundary와 Episode termination이 별도 기록됨
- Environment Step과 MainAgent transition Metadata 기록

- [x] 38. TASK-7-4. PPO Module Freeze 및 Artifact 저장 (depends on TASK-7-3)

**참조:** Requirements REQ-MAIN-008, REQ-INFER-001~003, Design §8.4, §15.3

**작업 내용:**
- ML Phase 2 Inference Module Artifact 로드
- IntentInference/EmotionInference freeze
- PPOModule만 학습
- `.bangmodel`은 중간 Checkpoint가 아닌, Design §15.3을 source of truth로 하는 최종 추론용 Artifact로 저장
- `.bangmodel`은 ZIP archive이며 root에 `manifest.json`, `metadata.json`, `model.safetensors`만 포함
- `manifest.json`의 `artifact_type=inference`, `inference_model_artifact_format_version=1.0`, weights/metadata 파일명 및 SHA-256을 검증
- `model.safetensors`에 IntentInference/EmotionInference/PPOModule 가중치를 저장하고 `metadata.json`에 Design §15.3 필수 Metadata를 저장
- `TrainingJob` 정상 완료 또는 사용자·운영 조기 종료 시 최종 `.bangmodel` 저장 여부는 Design §15.3의 Inference Artifact 저장 조건을 따름
- `pause_failed` 시 최종 `.bangmodel` 저장 여부도 Design §15.3의 조건을 따르며, Checkpoint 또는 Resume state 저장을 의미하지 않음
- 저장 가능한 PPOModule 가중치가 실제로 생성되지 않은 경우 Artifact를 생성하지 않고 종료/실패 원인을 로그 기록
- 어떤 종료 상황에서도 Optimizer state, Rollout state, Environment state, Worker state 등 프로그램 재시작 Resume에 필요한 상태를 저장하지 않음
- 불완전한 학습 상태를 Checkpoint 형태로 별도 저장하지 않음

**완료 조건:**
- Freeze된 모듈의 `requires_grad=False`
- PPO update가 PPOModule에만 적용됨
- Normal Episode의 `trainable=True && truncated=False` transition만 Update 입력
- 정상 Episode 비-terminal transition에 GAE/Bootstrap 적용
- 마지막 `terminated=True` transition에 Terminal Reward 반영 및 종료 후 Bootstrap 없음
- Truncated transition은 Terminal Reward/Bootstrap/PPO Update에서 제외
- `.bangmodel` 저장 성공/실패 로그 생성
- 실제로 생성된 PPOModule 가중치와 필수 Metadata가 없거나 Design §15.3의 Artifact 저장 조건을 만족하지 않으면 `.bangmodel`을 생성하지 않음
- `.bangmodel` Bundle root에 `manifest.json`, `metadata.json`, `model.safetensors`만 존재하고 manifest의 format version·파일명·SHA-256이 실제 Bundle과 일치함
- `metadata.json`에 Design §15.3의 필수 구성 요소와 Metadata가 존재하고, `model.safetensors`의 실제 가중치와 서로 일치함
- Phase 2 Inference Module Artifact와 최종 게임 실행용 `.bangmodel`의 역할이 구분됨
- Design §15.3에 정의되지 않은 새로운 구체 파일 구성이나 저장 형식을 추가하지 않음
- `.bangmodel`이 중간 Checkpoint가 아니며, Optimizer/Rollout/Environment/Worker state 등 Resume state를 포함하지 않음
- `.bangmodel`을 프로그램 재시작 후 학습 Resume state로 사용할 수 없음
- 불완전한 학습 상태가 별도 Checkpoint 형태로 저장되지 않음

---

Implementation Phase 8 — Evaluation

- [x] 39. TASK-8-1. Primary Evaluation Protocol 구현 (depends on TASK-7-4)

**참조:** Requirements REQ-EVAL-001~008, Design §13

**작업 내용:**
- Primary deterministic Protocol
- 총 500게임
- seat 0→1→2→3→4 cyclic
- 좌석별 100게임
- Seed: `base_seed + evaluation_index`
- Episode Seed 기반 역할 배정: canonical role list에 NumPy `Generator(PCG64(episode_seed)).permutation` 적용
- Normal/Truncated 분리 집계
- Win Rate/N/A 처리
- Role별 성능 집계 및 분석 Metadata 작성에 필요한 평가용 실제 Role 정보 사용
- 평가용 실제 Role 정보는 MainAgent의 Observation, IntentInference, EmotionInference, PPOModule 입력에 전달하지 않음
- 평가용 실제 Role은 성능 집계 및 분석 Metadata 용도로만 사용
- 미공개 Role 성능은 해당 Role이 MainAgent에게 공개되지 않은 상태에서의 평가 결과로 별도 집계
- `role_classification_accuracy`: 5-role argmax 예측이 실제 Role과 일치한 대상 수 / 평가 대상 수
- `hidden_role_accuracy`: 평가 시점에 MainAgent에게 공개·확정되지 않은 Role 대상만 분모로 하는 동일한 정확도
- `estimated_emotion_mse`: 행동 직전 추정 Emotion과 privileged evaluation target인 DummyAgent 실제 Emotion 상태의 모든 차원 제곱 오차 평균
- 실제 Role/Emotion 상태는 통계·metric 계산에만 사용하고 MainAgent 입력에는 전달하지 않음

**완료 조건:**
- 좌석별 정확히 100게임
- 총 500게임
- 동일 Seed/Protocol에서 결과 재현
- Win Rate = normal wins / normal games
- Normal games=0이면 N/A
- Truncated Rate = truncated / total
- 평가용 실제 Role 정보가 MainAgent의 Observation, IntentInference, EmotionInference, PPOModule 입력에 포함되지 않음
- 평가용 실제 Role이 성능 집계 및 분석 Metadata에만 사용됨
- 미공개 Role이 MainAgent에게 공개되지 않은 상태의 성능이 별도로 집계됨
- `role_classification_accuracy`, `hidden_role_accuracy`, `estimated_emotion_mse` 공식이 Design §13.3과 일치함
- 실제 Emotion target이 없는 경우 `estimated_emotion_mse=N/A`로 기록됨
- 실제 Role/Emotion 상태가 MainAgent 입력에 포함되지 않고 metric 계산에만 사용됨

- [x] 40. TASK-8-2. Custom/Stochastic Evaluation 구현 (depends on TASK-8-1)

**참조:** Requirements REQ-EVAL-002, REQ-EVAL-007~008, Design §13.2

**작업 내용:**
- GUI Custom Evaluation 설정
- Primary와 별도 Run ID
- Stochastic Protocol 분리
- Protocol/Seed/Preset/Seat/Action Mode Metadata 저장

**완료 조건:**
- Custom 결과가 Primary 결과와 섞이지 않음
- Stochastic 결과가 Primary Win Rate에 합산되지 않음
- 서로 다른 조건 결과를 연속 성능으로 표시하지 않음

---

Implementation Phase 9 — Application 및 GUI

- [x] 41. TASK-9-1. Application Service 구현 (depends on TASK-7-4)

**참조:** Requirements REQ-ARCH-005, REQ-GUI-001~014, Design §14

**작업 내용:**
- GameService
- TrainingService
- EvaluationService
- ConfigService
- PresetService
- DatasetService
- CompatibilityService
- read-only DTO 정의

**완료 조건:**
- GUI가 Environment 내부 배열에 직접 접근하지 않음
- Service API를 통해 Game/Training/Evaluation 제어
- Training/Model Config 변경은 다음 Job부터 반영

- [x] 42. TASK-9-2. Game Mode GUI 구현 (depends on TASK-9-1)

**참조:** Requirements REQ-GUI-001~004, Design §14.2

**작업 내용:**
- Main Menu
- Game Mode Select
- 사람 vs RandomAgent 4명
- 사람 vs DummyAgent 4명
- 사람 vs MainAgent + DummyAgent 3명
- `.bangmodel` 선택 Modal
- Model Metadata/Compatibility 표시
- GameBoard/ActionPanel/DebugHUD

**완료 조건:**
- 3가지 Game Mode 진입 가능
- MainAgent Mode에서 저장된 Artifact 선택 가능
- Compatibility 실패 모델은 실행되지 않음
- DebugHUD는 타 플레이어 4명만 표시
- Action Mask=0 버튼 비활성화

- [x] 43. TASK-9-3. Training Monitor 및 Settings GUI 구현 (depends on TASK-9-2)

**참조:** Requirements REQ-GUI-005~014, REQ-HW-001~007, Design §14, §16

**작업 내용:**
- Training Setup
- MainAgent transition ProgressBar
- Environment Step counter
- Truncated count/rate
- pyqtgraph Win Rate graph
- PPO clip epsilon 설정
- Emotion initial baseline 설정
- Reward Config Editor
- CPU/RAM/GPU/Worker/Batch/Buffer 입력
- 권장 도움말 및 입력 검증
- Pause/Resume/Stop UI
- Stop은 현재 TrainingJob을 사용자 요청으로 종료하는 동작
- Stop 이후 프로그램을 재시작해 학습을 Resume하는 기능 없음
- Stop 시 Resume state 또는 Checkpoint를 생성하지 않음
- Stop 시 Artifact 저장 여부는 TASK-7-4의 기존 최종 Artifact 저장 조건을 그대로 따름
- Pause는 현재 실행 중인 프로세스의 메모리 상태에서 일시정지·재개하고, Stop은 TrainingJob을 종료하며 둘을 동일한 상태 복원 기능으로 취급하지 않음

**완료 조건:**
- Pause 요청 시 모든 Worker ACK 후 paused 표시
- 30초 timeout 시 `pause_failed` 및 자동 재시도 없는 안전 중단
- Pause 중 Step/transition/update 증가 없음
- Stop은 현재 TrainingJob을 사용자 요청으로 종료함
- Stop 이후 프로그램 재시작 Resume state 또는 Checkpoint가 생성되지 않음
- Stop 시 Artifact 저장 여부가 TASK-7-4의 기존 최종 Artifact 저장 조건과 일치함
- Pause와 Stop의 상태 처리 및 현재 프로세스 메모리 상태 경계가 구분됨
- 프로그램 종료 후 학습 상태 복원 UI 없음
- Benchmark/OOM 탐색 UI 없음

---

Implementation Phase 10 — Compatibility, Logging 및 통합 검증

- [x] 44. TASK-10-1. Inference Artifact Compatibility 구현 (depends on TASK-7-4)

**참조:** Requirements REQ-INFER-001~003, REQ-COMPAT-001~002, REQ-REPR-004, Design §15

**작업 내용:**
- `.bangmodel` ZIP Bundle 로더
- root의 `manifest.json`, `metadata.json`, `model.safetensors` 구조 및 Manifest SHA-256 검증
- `inference_model_artifact_format_version=1.0` 및 `artifact_type=inference` 검증
- Model Architecture 검사
- Environment/Observation/Action/Reward/Emotion/Artifact Version 검사
- preprocessing/config/schema 검사
- optimizer/rollout/environment/worker/resume state 포함 여부 거부
- field-level 오류 메시지

**완료 조건:**
- 호환 Artifact는 Game 실행 가능
- 불일치 Artifact는 실행 거부
- Compatibility 오류에 field명과 현재/요구값 포함
- 학습 Resume state 로드 시도 없음

- [x] 45. TASK-10-2. Logging 및 Metadata 통합 (depends on TASK-10-1)

**참조:** Requirements REQ-STEP-015-A, REQ-RWD-006~009, REQ-REPR-003~004, Design §17

**작업 내용:**
- Episode/Event/Termination 로그
- Reward Breakdown
- Invalid Action 로그
- Training counters
- Evaluation Protocol 결과
- Config/Schema/Artifact Version
- Hardware 수동 설정 및 실행 통계
- Artifact 저장 성공/실패

**완료 조건:**
- Truncated Episode 전체 진행 재구성 가능
- Reward Component별 값 확인 가능
- 실제 사용 Config Version 확인 가능
- Environment Step과 MainAgent transition 분리 기록

- [x] 46. TASK-10-3. 전체 파이프라인 통합 테스트 (depends on TASK-8-2, TASK-9-3, TASK-10-2)

**참조:** Requirements AC-01~AC-25, Design §18

**작업 내용:**
- Environment 단독 실행
- Agent 혼합 실행
- Normal/Truncated 종료 시나리오
- Invalid Action 강제 주입
- Role Reveal/RoleFusion
- Emotion Label 시간 정렬
- Parquet → split → NPZ cache
- Truncated 학습 제외
- ML Phase 2 SL
- ML Phase 3 PPO
- Primary/Custom Evaluation
- Pause/`pause_failed`
- `.bangmodel` 저장/불러오기

**완료 조건:**
- AC-01~AC-25 각각에 대해 자동화 테스트로 검증 가능한 구현 경로와 추적 가능한 테스트 결과가 존재함
- 자동화 테스트를 AC 검증의 기본 방식으로 사용함
- 수동 검토·실행 확인은 자동화 테스트를 대체하지 않으며, 자동화 결과를 보완하는 검증 기록으로만 남김
- Truncated Episode가 ML Phase 2/ML Phase 3 학습에 포함되지 않음
- Role Count Projection 구현·호출 없음
- Benchmark/자동 Worker 최적화 구현·호출 없음
- Checkpoint/프로그램 재시작 Resume 구현·호출 없음

- [x] 47. TASK-10-4. 문서·설정·README 정합성 검증 (depends on TASK-10-3)

**참조:** Requirements v0.7, Design v0.7

**작업 내용:**
- 문서 버전 확인
- Requirements ID 참조 확인
- Design section 참조 확인
- Config 예시와 Schema 일치 확인
- `datasets/canonical`, `datasets/cache`, `models`, `logs` 경로 확인
- 오래된 OQ/Benchmark/Checkpoint/Resume 태스크 검색

**완료 조건:**
- Task가 요구사항 또는 설계 내용에 의존하는 경우 적절한 Requirements/Design 참조가 존재함
- 단순 구현 순서·테스트 실행·문서 정합성 항목에는 불필요한 참조 번호를 강제하지 않음
- Requirements v0.7과 Design v0.7의 실제 section 및 ID 참조가 일치함
- Benchmark 구현 태스크 없음
- 중간 Checkpoint/학습 Resume 태스크 없음
- Phase 의존성 순서와 태스크 순서 일치

---

## 구현 금지 목록

다음은 현재 Tasks에서 구현 대상으로 만들지 않는다.

- Hardware Benchmark
- Games/sec 또는 Steps/sec 자동 측정용 Benchmark Runner
- 자동 Worker/Batch/Buffer 최적화
- OOM 기반 Batch 탐색
- Role Count Projection
- Bayesian Role Assignment
- 중간 학습 Checkpoint
- 프로그램 종료 후 학습 Resume
- Optimizer/Rollout/Environment 상태 복원
- Truncated Episode를 ML Phase 2/ML Phase 3 학습에 포함하는 것

## 최종 구현 순서

```text
Implementation Phase 0 기반
→ Implementation Phase 1 Environment
→ Implementation Phase 2 기본 Agent
→ Implementation Phase 3 MainAgent
→ Implementation Phase 4 Emotion/Reward Runtime
→ Implementation Phase 5 Dataset
→ ML Phase 2 (Implementation Phase 6) SL
→ ML Phase 3 (Implementation Phase 7) PPO
→ Implementation Phase 8 Evaluation
→ Implementation Phase 9 Application/GUI
→ Implementation Phase 10 Compatibility/통합 검증
```

`tasks.md`의 각 작업은 구현 가능한 코드 작업과 검증 기준으로만 구성한다. 새로운 요구사항이나 설계 결정을 이 문서에서 추가하지 않는다.


## Notes — Reconciliation

This section records the final evidence-based task-state reconciliation against requirements.md, design.md, the current implementation, and the supplied automated validation. Requirements and design decisions were not changed.

- **All 47 tasks are complete:** TASK-1-7, TASK-4-1, TASK-5-1, TASK-5-3, TASK-6-2, TASK-7-2, and the dependent TASK-10-3 blockers are resolved and their status markers are now `[x]`.
- **TASK-1-7:** focused long-game diagnostics validation passed (2 tests), including threshold logging and truncated-run reconstruction data.
- **TASK-4-1:** EmotionRuntime integration validation passed (21 tests), including event/decay/label ordering, terminal behavior, and invalid-action/retry boundaries.
- **TASK-5-1:** collector/Parquet validation passed (11 tests), including pre-action alignment, event preservation, finalization, and canonical record handling.
- **TASK-5-3:** Parquet read normalization for empty episodes was updated; the empty-episode split/round-trip blocker is resolved.
- **TASK-6-2:** EmotionInference trainer validation passed (24 tests), including split-scoped metrics, early stopping, schema checks, and artifact metadata.
- **TASK-7-2:** Pending Transition Ledger validation passed (29 tests), including retry and duplicate-result idempotency without duplicate transitions.
- **TASK-10-3:** end-to-end integration validation passed (39 tests). The full repository validation passed all 291 tests with 5 non-failing warnings; the warnings are not blockers.
- **Three-document consistency:** requirements.md and design.md remain unchanged at v0.7; tasks.md preserves the existing task text and dependency order, and its references remain consistent with both documents. The documentation consistency validation passed (3 tests).

**Remaining blockers:** none. No requirements.md or design.md changes were needed, and no unrelated code changes were made as part of this reconciliation.
