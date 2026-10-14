# 메인 AI 총괄 문서

이 문서는 메인 AI(MainAgent)가 어떻게 구성되고 작동하는지, 어떤 방식으로 학습되는지, 학습 결과물(아티팩트)이 어떤 형식으로 저장되는지, 아티팩트별 학습 상태를 어떻게 읽고 이어 학습하는지, 그리고 학습된 아티팩트로 어떻게 플레이하는지를 한곳에 모아 설명합니다.

필드 이름, 클래스 이름, 모드 이름 등 코드가 인식하는 리터럴은 영어 그대로 적습니다. 설명 문장은 한국어로 작성합니다. 모든 서술은 실제 코드와 `README.md`의 "Artifacts and compatibility", "Logs and evaluation" 절에 근거합니다. 이 문서는 추가 문서이며 `README.md`의 내용과 모순되지 않습니다.

관련 소스 위치는 아래와 같습니다.

- 메인 AI 모듈: `bang_rl/agents/main_agent/`의 `preprocessor.py`, `encoder.py`, `intent_inference.py`, `emotion_inference.py`, `role_fusion.py`, `ppo_module.py`
- 고정 아키텍처 스냅샷: `bang_rl/config/schema.py`의 `ModelArchitectureSnapshot`, `OptimizerSnapshot`, `FixedOptimizerConfig`
- 학습 총괄(러너): `bang_rl/training/pipeline_runner.py`
- 아티팩트 형식: `bang_rl/training/final_inference_artifact.py`, `bang_rl/training/inference_artifact.py`
- 저장 경로: `bang_rl/paths.py`
- 서비스/플레이 연결: `bang_rl/app/services.py`

---

## 1. 개요: 메인 AI 구성 요소와 작동 방식

메인 AI는 공개 관측(public observation)만을 입력으로 받아 하나의 유효한 행동을 선택하는 추론 파이프라인입니다. 숨겨진 진짜 역할, 감정 라벨, 보상, 미래 이벤트 같은 특권 정보(privileged state)는 어느 모듈에서도 입력으로 받지 않습니다. 각 모듈은 입력 매핑에서 금지 키(`roles`, `true_role`, `emotion_label`, `reward` 등)를 발견하면 텐서 변환 이전에 거부합니다.

전체 추론 흐름은 다음 순서로 이어집니다.

1. `ObservationPreprocessor`가 공개 관측 매핑을 고정 형상 텐서로 변환합니다.
2. `GRUEncoder`(SequenceEncoder)가 플레이어별 공개 이벤트 스트림을 인코딩합니다.
3. `IntentInferenceModule`이 인코딩 결과로 상대 네 명의 역할 분포(raw role distribution)를 추론합니다.
4. `RoleFusion`이 공개로 확정된 역할을 하드 오버라이드하여 역할 분포를 보정합니다.
5. `EmotionInferenceModule`이 공개 이벤트 스트림으로 상대 네 명의 연속 감정 값을 추론합니다.
6. `PPOModule`이 위 결과들을 특징 블록으로 받아 마스킹된 정책에서 행동을 선택하고 가치(value)를 추정합니다.

### 1.1 ObservationPreprocessor (`preprocessor.py`)

모델 경계입니다. 공개 관측 매핑을 검증하고 고정 형상 텐서로 변환합니다. 주요 상수와 출력 형상은 다음과 같습니다.

- `PLAYER_COUNT`는 5, `OTHER_PLAYER_COUNT`는 4(관측자 본인을 제외한 상대 수).
- `EVENT_FEATURE_SIZE`는 17, `MODEL_HISTORY_LENGTH`는 64(이벤트 스트림 길이).
- `STATE_VECTOR_SIZE`는 9, `PLAYER_FEATURE_SIZE`는 8.
- `ACTION_SPACE_SIZE`만큼의 `action_mask`.

출력은 `ProcessedObservation`이며 다음 고정 형상 필드를 담습니다.

- `state_vector`: `(9,)` float32
- `player_state_matrix`: `(5, 8)` float32
- `role_beliefs`: `(5, 5)` float32 (공개 확정 역할 분포)
- `event_history`: `(4, 64, 17)` float32 (상대 네 명의 좌측 패딩된 공개 이벤트 스트림)
- `event_valid_mask`: `(4, 64)` bool (유효 이벤트 위치)
- `action_mask`: `(ACTION_SPACE_SIZE,)` uint8

전처리기는 공개 필드만 다루며, `confirmed_role_distribution`은 각 행이 0 또는 정규화된 분포여야 하는 등 엄격한 검증을 수행합니다.

### 1.2 GRUEncoder (`encoder.py`)

패딩/마스크를 처리하는 시퀀스 인코더입니다. 좌측 패딩된 고정 길이 스트림에서 유효하지 않은 위치를 제거한 뒤 GRU로 처리하고 마지막 유효 은닉 상태(final valid hidden state)를 반환합니다. 모든 위치가 무효인 스트림은 0 벡터를 반환합니다.

고정 기본값(Design 8.1)은 다음과 같습니다.

- `encoder_type`: `GRU`
- `DEFAULT_GRU_HIDDEN_SIZE`: 128
- `DEFAULT_GRU_NUM_LAYERS`: 1
- `DEFAULT_GRU_BATCH_FIRST`: True
- `DEFAULT_GRU_DROPOUT`: 0.0

`IntentInferenceModule`과 `EmotionInferenceModule`은 모두 인코더의 `output_dim`이 128이어야 함을 생성자에서 검증합니다. 인코더는 교체 가능한 인터페이스(`SequenceEncoderFactory`)로 등록되지만 기본 구현은 `GRUEncoder` 하나입니다.

### 1.3 IntentInferenceModule (`intent_inference.py`)

공개 이벤트 스트림만으로 역할 분포를 추론합니다. 인코더가 스트림당 하나의 표현을 만들고, 공유 역할 헤드(role head)가 이를 5개 역할 로짓으로 매핑합니다. `ROLE_COUNT`는 5, `TARGET_PLAYER_COUNT`는 4로 고정됩니다.

- 입력 형상: `(T, F)` -> `(5,)`, `(4, T, F)` -> `(4, 5)`, `(B, 4, T, F)` -> `(B, 4, 5)`.
- 학습 모드에서는 로짓을 반환하여 `CrossEntropyLoss`에 사용하고, 평가 모드에서는 softmax 확률을 반환합니다. `output` 인자로 명시적으로 선택할 수도 있습니다.
- 역할 라벨은 입력이 아니라 학습 보조 함수 `role_loss`에서만 받습니다. `forward`는 라벨을 받지 않습니다.
- `RoleFusion`은 이 모듈 내부가 아니라 이후 단계로 분리됩니다("confirmed_role_fusion: deferred to RoleFusion").

### 1.4 EmotionInferenceModule (`emotion_inference.py`)

공개 이벤트 스트림만으로 상대 네 명의 연속 감정 값을 추론합니다. 출력은 Sigmoid를 거쳐 `[0, 1]` 범위의 연속 값이며, 손실은 MSE입니다. 라벨은 DummyAgent의 "행동 직전(behavior-immediately-before-action)" 상태에 정렬됩니다. `target_player_count`는 4로 고정됩니다.

- 입력이 모두 무효인(비어 있는 이력) 스트림은 학습된 값 대신 설정된 `initial_baseline`을 사용합니다.
- 감정 차원 수(`emotion_count`)와 스키마 버전(`emotion_schema_version`)은 `EmotionConfig`에서 옵니다.
- README 서술과 동일하게, 출력은 "estimated emotion"으로 표시되며 인간 심리에 대한 주장이 아닙니다.

### 1.5 RoleFusion (`role_fusion.py`)

상태가 없는(stateless) 보정 계층입니다. `IntentInferenceModule`이 만든 raw role distribution에서 공개로 이미 확정된 역할만 하드 오버라이드로 교체합니다. 남은 역할 수 투영(role count projection)이나 역할 배정 구성은 하지 않습니다.

- 표준 출력 형상은 상대 네 명에 대한 `(4, 5)`이며, 배치 `(B, 4, 5)`도 받습니다.
- 오버라이드 근거(override sources)는 `self_role`, `sheriff_idx`, `revealed_roles`, `confirmed_role_distribution`입니다.
- 확정된 역할은 one-hot으로, 미확정 역할은 정규화만 거친 분포로 남습니다. 특권 역할 입력은 거부합니다.

### 1.6 PPOModule (`ppo_module.py`)

공개 관측 처리, RoleFusion, EmotionInference 이후의 최종 결정 경계입니다. 공유 트렁크(shared trunk) 뒤에 정책 헤드와 가치 헤드가 붙은 범주형 정책/가치 추정기입니다.

입력 특징 블록의 순서(input contract)는 다음과 같습니다.

1. `self_state` (상태 벡터)
2. `board_state` (플레이어 보드)
3. `confirmed_role_features` (확정 역할 특징)
4. `fused_intent` (RoleFusion 출력; raw IntentInference 출력은 금지)
5. `emotion_features` (감정 특징)

행동 선택과 정책 통계 계산에는 `action_mask`가 반드시 필요합니다. 무효 행동 로짓은 `-inf`로 마스킹되어 선택되지 않습니다. 정책 출력 크기(`POLICY_OUTPUT_SIZE`)는 6, 가치 출력(`VALUE_OUTPUT_SIZE`)은 1입니다.

---

## 2. 아키텍처 세부: 고정 형상

메인 AI의 아키텍처 형상은 `bang_rl/config/schema.py`의 `ModelArchitectureSnapshot`에 고정 리터럴로 박혀 있으며, 잡/아티팩트 메타데이터에 그대로 복사됩니다. 이 값들은 변경 가능한 설정이 아니라 계약입니다.

| 구성 요소 | 고정 값 |
| --- | --- |
| `intent_encoder` / `emotion_encoder` | `GRU` |
| `intent_hidden_size` / `emotion_hidden_size` | 128 |
| `intent_num_layers` / `emotion_num_layers` | 1 |
| `intent_batch_first` / `emotion_batch_first` | True |
| `intent_dropout` / `emotion_dropout` | 0.0 |
| `intent_role_head` | `Linear(128,64)->ReLU->Linear(64,5)` |
| `emotion_head` | `Linear(128,64)->ReLU->Linear(64,E)->Sigmoid` |
| `ppo_shared_trunk` | `Linear(input_dim,256)->ReLU->Linear(256,256)->ReLU` |
| `ppo_policy_head` | `Linear(256,6)` |
| `ppo_value_head` | `Linear(256,1)` |

위에서 `E`는 `EmotionConfig.emotion_count`로 정해지는 감정 차원 수입니다.

옵티마이저 계약은 `OptimizerSnapshot`과 `FixedOptimizerConfig`에 고정됩니다. 세 모듈 모두 Adam을 쓰며 학습률만 다릅니다.

| 모듈 | 옵티마이저 | learning_rate |
| --- | --- | --- |
| Intent | Adam | 1e-3 (0.001) |
| Emotion | Adam | 1e-3 (0.001) |
| PPO | Adam | 3e-4 (0.0003) |

`FixedOptimizerConfig.learning_rate`는 `Literal[0.001, 0.0003]`로 제한되고, `OptimizerSnapshot`의 검증기는 Intent/Emotion이 1e-3, PPO가 3e-4가 아니면 오류를 냅니다. 이 학습률은 코드에서 하드코딩으로 덮어쓰지 않고 항상 설정에서 전달됩니다.

고정 형상은 각 모듈의 생성자에서도 재확인됩니다. 예를 들어 Intent/Emotion 모듈은 인코더 `output_dim`이 128이 아니면 생성 시점에 거부하고, PPO 모듈은 구조화 입력 블록이 선언된 `input_dim`으로 정확히 이어붙지 않으면 거부합니다.

---

## 3. 학습 파이프라인: 단계별 흐름

"AI 학습" 버튼을 누르면 `TrainingService`가 러너를 실행합니다. 주입된 러너가 없으면 기본 러너로 `bang_rl/training/pipeline_runner.py`의 `default_training_runner`가 자동으로 연결됩니다(지연 import). 이 러너는 이미 테스트된 각 단계 구성 요소를 재구현하지 않고 조합(compose)하는 "총괄 코드"입니다. 각 단계의 실제 API 청사진은 `bang_rl/tests/test_pipeline_integration.py`에 있습니다.

러너는 Qt를 import하지 않는 평범한 콜러블이며 `TrainingService._execute`가 데몬 스레드에서 실행하므로 UI 스레드를 막지 않습니다. 진행 상황은 `run`의 카운터(`ppo_updates`, `transitions`, `environment_steps`, `episodes_completed` 등)를 `run.lock` 아래에서 갱신하여 GUI 폴링에 반영하고, 동시에 결과 Mapping을 반환해 학습 리포트에 접힙니다. 취소는 협력적이며, 러너가 단계 사이에서 `run.status == 'cancelling'`을 확인하면 조기 종료합니다.

단계 순서와 각 단계의 출력이 다음 단계로 어떻게 이어지는지는 다음과 같습니다.

1. 게임 기록 수집 (COLLECT). `BangEnvironment`를 한 스텝씩 구동하면서 `EmotionRuntime`, `RewardRuntime`(및 `attach_rewards`), `EpisodeCollector`로 공개 정식(canonical) 에피소드를 만듭니다. 감정 런타임의 사전 행동 경계(before_action) 계약을 지켜 `environment_step`을 정확히 1씩 전진시키고 직전 행동의 이벤트를 전달합니다. 완성된 각 에피소드는 `ParquetCanonicalShardWriter`에 append 되고, 리스트로도 모읍니다. 산출물은 정식 에피소드 목록과 Parquet 샤드입니다.
2. 분할 (SPLIT). 수집한 에피소드를 `EpisodeSplitter`로 train/validation/test로 에피소드 단위 분할합니다(기본 비율은 대략 60/20/20이며 README의 기본 정책은 80/10/10 분할입니다). 산출물은 분할 결과와 매니페스트입니다.
3. 텐서 캐시 (CACHE). `NPZTensorCacheWriter`가 분할 결과로 각 split의 NPZ 텐서 캐시를 쓰고, `NPZTensorCacheLoader`가 split별 로더를 제공합니다. 정상(normal) 에피소드만 ML 캐시에 들어가며, 잘린(truncated) 에피소드는 정식 데이터셋과 로그에는 남지만 캐시와 PPO 업데이트에서는 제외됩니다. 산출물은 train/validation/test 캐시 로더입니다.
4. 지도학습 사전 학습 (SL PRETRAIN). 텐서 캐시를 입력으로 `IntentInferenceSLTrainer`(`IntentSLConfig`)와 `EmotionInferenceSLTrainer`(`EmotionSLConfig`, `emotion_config`)를 각각 `fit`합니다. 학습률은 설정이 제공하는 고정 계약(Intent/Emotion 1e-3)을 그대로 씁니다. 산출물은 두 트레이너의 학습 이력입니다.
5. Phase-2 아티팩트 저장/로드. `save_phase2_inference_artifact`로 Intent/Emotion 모듈을 `.phase2model`로 저장하고 `load_phase2_inference_artifact`로 다시 불러옵니다. 이렇게 라운드트립된 Phase-2 번들은 다음 PPO 단계의 초기화 입력(frozen 번들)으로만 쓰입니다. 산출물은 로드된 Phase-2 아티팩트입니다.
6. PPO 학습 (PPO). `PPOModule`을 만들고 `PPOTrainer(ppo, phase2_artifact=phase2, config=ppo_config, snapshot, seed)`에 Phase-2 번들을 넣습니다. 수집한 정식 스텝을 `PPOTransition`으로 투영한 뒤 `collect_rollout(transitions)`로 롤아웃을 모으고 `update()`로 정책을 갱신합니다. PPO 학습률은 `FixedOptimizerConfig`를 통해 `PPOConfig`에서 3e-4로 고정됩니다. 이어 학습(resume) 시에는 로드한 최종 아티팩트의 PPO 가중치만 복사해 가중치 전용 연속으로 시작합니다. 산출물은 학습된 PPO 모듈과 카운터입니다.
7. 최종 아티팩트 저장. `save_final_inference_artifact`로 PPO 모듈과 Phase-2 번들, 잡 스냅샷, 설정 스냅샷, 학습 카운터를 묶어 `.bangmodel`을 `TrainingService.start`가 미리 정한 목적지 경로(`ARTIFACTS_DIR` 아래)에 저장합니다. 산출물은 최종 게임 아티팩트 파일 경로입니다.
8. 평가 (EVALUATE). 저장된 아티팩트로 Win Rate와 지표를 얻습니다. 기본 평가 프로토콜 `deterministic_v1`은 500게임(좌석당 100게임)으로 고정된 불변 벤치마크여서 게임 수를 줄일 수 없습니다. "AI 학습" 버튼을 눌렀을 때 평가가 오래 걸리지 않도록, 사용자가 기본 프로토콜을 그대로 둔 경우 러너는 500게임 벤치마크 대신 게임 수를 조절할 수 있는 `custom_v1` 프로토콜로 짧은 평가(`_DEFAULT_EVALUATION_GAMES`게임)를 `run_custom_evaluation`으로 실행합니다. 사용자가 `custom_v1`/`stochastic_v1` 같은 비기본 프로토콜을 명시적으로 설정한 경우에는 그 `EvaluationConfig`를 그대로 존중합니다. 산출물은 평가 결과(win_rate, metrics)입니다.

러너는 마지막에 `environment_steps`, `main_agent_transitions`, `episodes_completed`, `truncated_episodes`, `normal_episodes`, `win_rate`, `metrics`, `logs`, `records`, `artifact_path` 키를 가진 Mapping을 반환하고, `_execute`가 이를 학습 리포트에 접습니다.

> 참고: 실제 torch/numpy 기반 학습 경로는 CI 전용입니다. 제한된 샌드박스에서는 numpy/pydantic/torch를 설치할 수 없으므로 전체 학습을 실행할 수 없고, 샌드박스 게이트는 `python -m compileall -q bang_rl`과 순수 배선(wiring) 테스트입니다. 전체 학습 경로는 CI의 `test_pipeline_integration.py`가 검증합니다.

---

## 4. 아티팩트 형식과 저장 위치

메인 AI 학습은 성격이 다른 두 아티팩트를 만듭니다. 이 내용은 README "Artifacts and compatibility" 절과 일치합니다.

- Phase-2 모듈 아티팩트: 확장자 `.phase2model`, 타입 `phase2_inference_module`, 포맷 `1.0`. Phase-3 추론 모듈을 초기화하는 용도로만 쓰입니다. 상수는 `bang_rl/training/inference_artifact.py`의 `PHASE2_ARTIFACT_EXTENSION`, `PHASE2_ARTIFACT_TYPE`, `PHASE2_ARTIFACT_FORMAT_VERSION`에 있습니다.
- 최종 게임 아티팩트: 확장자 `.bangmodel`, 타입 `inference`, 포맷 `1.0`. 루트에 정확히 `manifest.json`, `metadata.json`, `model.safetensors` 세 파일만 담은 ZIP 아카이브입니다. 상수는 `bang_rl/training/final_inference_artifact.py`의 `FINAL_ARTIFACT_EXTENSION`, `FINAL_ARTIFACT_TYPE`, `FINAL_ARTIFACT_FORMAT_VERSION`에 있습니다.

최종 `.bangmodel`의 ZIP 구성은 다음과 같습니다.

- `model.safetensors`: 가중치. 네임스페이스는 `intent`, `emotion`, `ppo` 세 개입니다.
- `metadata.json`: 아키텍처, 버전, 설정/스냅샷 해시, 학습 상태 등 메타데이터.
- `manifest.json`: 아티팩트 타입, 포맷 버전, 파일 목록, 그리고 `weights_sha256`/`metadata_sha256` 같은 SHA-256 체크섬. 매니페스트에는 `weight_namespaces`로 `["intent", "emotion", "ppo"]`가 기록됩니다.

로드 시에는 ZIP 루트가 정확히 이 세 파일 집합인지 확인하고, 가중치와 메타데이터의 SHA-256을 매니페스트 값과 대조하며, 아키텍처/스키마/포맷 호환성을 검사합니다. 최종 아티팩트는 추론 전용이며, 옵티마이저/롤아웃/워커/진행 중 환경/이어하기 상태는 저장하지 않습니다.

저장 위치는 다음과 같습니다.

- `ARTIFACTS_DIR`: 저장소 루트의 `artifacts/` 디렉터리(`bang_rl/paths.py`). `TrainingService.start`는 최종 아티팩트 목적지를 기본적으로 `ARTIFACTS_DIR/<job_id>.bangmodel`로 정합니다(`resolve_training_file`).
- 설정 기본값: `ArtifactConfig.output_directory`는 `"models"`입니다. 설정에서 모델 출력 디렉터리를 지정하면 그 경로(예: `bang_rl/models`)가 쓰입니다.
- 학습 리포트: JSON 리포트는 `reports/`(`REPORTS_DIR`)에 저장됩니다. 러너의 중간 산출물(정식 샤드, 캐시, 분할, 평가 리포트)은 실행별 출력 루트(`OUTPUTS_DIR` 아래 `<job_id>-pipeline`)에 격리되어 다른 잡과 충돌하지 않습니다.

데이터 산출물(정식 Parquet 샤드는 `bang_rl/datasets/canonical`, NPZ 캐시는 `bang_rl/datasets/cache`)과 아티팩트는 기본적으로 gitignore 되며, `.gitkeep` 플레이스홀더만 커밋됩니다.

---

## 5. 아티팩트별 학습 상태 읽기와 이어 학습(resume)

### 5.1 학습 상태 읽기

최종 아티팩트의 `metadata.json`에는 학습 상태가 기록됩니다. `save_final_inference_artifact`가 `_build_metadata`로 만드는 `training` 블록에는 다음이 들어갑니다(`bang_rl/training/final_inference_artifact.py`).

- `ppo_config`: PPO 설정 덤프
- `optimizer`: PPO 옵티마이저 메타데이터(Adam, 3e-4 등)
- `counters`: 학습 카운터 모음
- `total_environment_steps`: 누적 Environment Step 수(`counters.environment_steps`)
- `total_main_agent_transitions`: 누적 MainAgent transition 수(`counters.main_agent_transitions`)
- `total_training_time_seconds`: 누적 학습 시간(일시정지 시간 제외, `paused_time_excluded`는 True)
- `opponent`, `preset`: 학습 상대/프리셋 정보

러너는 저장 시 `training_metadata`로 `counters`(`environment_steps`, `main_agent_transitions`, `ppo_updates`)를 전달하고, 이 카운터가 위 메타데이터에 반영됩니다. 메타데이터와 매니페스트만 읽어 체크섬 검증까지 하려면 `read_final_artifact_metadata`를 쓸 수 있습니다.

README 서술대로 메인 AI 학습은 Environment Step과 MainAgent transition을 따로 셉니다. GUI 진행 표시는 MainAgent transition이며, PPO update는 세 번째 단위입니다.

### 5.2 이어 학습(resume)

이어 학습은 "가중치 전용 연속(weights-only continuation)"입니다. 흐름은 다음과 같습니다(`bang_rl/app/services.py` `TrainingService.start`).

1. `latest_artifact(options.resume_artifact_path)`로 이어 학습할 아티팩트를 찾습니다(`bang_rl/paths.py`). 인자가 파일이면 그 파일을, 디렉터리면 그 안에서 가장 최근 수정된 `.bangmodel`을 반환합니다. 기본 `resume_artifact_path`는 `ARTIFACTS_DIR`입니다.
2. 찾은 파일을 `load_final_artifact_for_training`으로 로드합니다. 이 로더는 번들을 검증한 뒤 PPO를 train 모드로 두고 가중치의 `requires_grad`를 켜며, Intent/Emotion은 eval 모드로 동결합니다. 새 옵티마이저는 이후 `PPOTrainer`가 생성합니다.
3. `start`는 이 아티팩트를 `run.resume_artifact`로 러너에 전달합니다. 러너는 `resume_artifact.ppo.state_dict()`를 새 `PPOModule`에 복사하는 방식으로 PPO 가중치만 이어받습니다. 복사가 실패하면 로그를 남기고 건너뜁니다.

중요한 제약은 `.bangmodel`에 옵티마이저/롤아웃/RNG/환경 상태가 저장되지 않는다는 점입니다. 따라서 종료된 프로세스를 그대로 재개하는 것이 아니라 가중치만 이어서 새로 학습을 시작합니다. README의 "Deliberate limitations" 절대로, 디스크 체크포인트나 프로세스 종료 후 학습 재개는 제공하지 않으며 GUI의 일시정지/재개는 메모리 상의 현재 잡만 조율합니다.

---

## 6. 학습된 아티팩트로 플레이

학습된 `.bangmodel`은 게임 실행에 쓸 수 있습니다. 로드와 어댑터 연결 흐름은 다음과 같습니다(`bang_rl/app/services.py`).

- `CompatibilityService.load`가 `load_final_inference_artifact`로 아티팩트를 로드하고 현재 스냅샷에 대해 호환성을 검사합니다.
- 로드된 아티팩트는 `_ArtifactAgent` 어댑터로 감싸집니다. `_ArtifactAgent.act`는 공개 관측을 `preprocessor.preprocess`로 처리하고, `intent.predict_proba`와 `emotion.predict_proba`를 호출한 뒤, `role_fusion.fuse`로 역할 분포를 보정하고, 이 다섯 특징 블록(`self_state`, `board_state`, `confirmed_role_features`, `fused_intent`, `emotion_features`)을 `ppo.act(..., deterministic=True)`에 넘겨 하나의 행동을 고릅니다.

메인 AI가 참여하는 두 모드는 다음과 같습니다(`GameService._build_agents`).

- 사람 대 메인(Human vs Main): 모드 이름 `human_main_dummy`. 좌석 0은 사람(`HumanAgent`), 좌석 1이 메인 AI(`_ArtifactAgent`), 나머지 좌석은 DummyAgent입니다. 이 모드는 추론 아티팩트가 반드시 필요하며, 없으면 "human_main_dummy mode requires an inference artifact" 오류를 냅니다. 입력 별칭으로 `human_vs_main`, `human_main`도 받습니다.
- 메인 대 더미(Main vs Dummy): 모드 이름 `main_one_dummy_four`. 좌석 0이 메인 AI(`_ArtifactAgent`), 나머지 네 좌석은 DummyAgent인 관전(spectator) 모드입니다. 이 모드도 아티팩트가 반드시 필요하며, 없으면 "main_one_dummy_four mode requires an inference artifact" 오류를 냅니다. 입력 별칭으로 `main_1_dummy_4`도 받습니다. 관전 모드에서는 사람이 BANG/PASS를 할 수 없고 AI 턴 속도 슬라이더와 로그만 조작할 수 있습니다(README "GUI game modes and replay formats" 참고).

---

## 참고: 관련 README 절

- "Artifacts and compatibility": 아티팩트 타입/형식/호환성 규칙
- "Logs and evaluation": 구조화 JSON Lines 로그 채널(`training`, `evaluation` 등)과 기본 평가 프로토콜 `deterministic_v1`, Win Rate 정의
- "GUI game modes and replay formats": 플레이 모드와 리플레이 형식
- "Deliberate limitations": 체크포인트/학습 재개 비제공 등 명시적 한계

이 문서의 수치와 규칙은 위 절들 및 소스 코드와 모순되지 않습니다.
