# Requirements — Bang RL Training System

> 작성 기준: EARS (Easy Approach to Requirements Syntax)  
> 버전: 0.7  
> 작성일: 2026-09-21  
> 용어 규약: 의무 요구사항은 "~해야 한다"로 기술한다. 사용자에게 제공되는 선택적 기능만 "~할 수 있어야 한다"로 기술한다.  
> 변경 이력: v0.2 — MainAgent 아키텍처, GUI 메뉴, 3-Phase 파이프라인, 하드웨어 수동 설정, 프리셋 Export/Import 추가  
> 변경 이력: v0.3 — 초기 결정사항 반영. 부관 팀 승패 공유, Invalid Action 로깅, 배신자 자유 행동+1대1 우선 정책, 비정상 긴 게임 로깅, K=무제한 이벤트 이력, Parquet+NPZ 역할 분리, PyQt6/GRU 확정, 감정 레이블 Schema Version, HUD 4명 한정  
> 변경 이력: v0.4 — 승패 조건 전면 재정의(보안관 사망 시 배신자 단독 생존→배신자 승리, 그 외 생존자 존재→무법자팀 승리, 결투 상태 개념 제거). Invalid Action 처리 방식 변경(PASS 자동 변환 제거, Action Mask 우선 예방, Environment 최종 검증 후 거부+로그). Action Space와 Action Mask 명시적 분리  
> 변경 이력: v0.5 — 환경 Step·max_episode_steps·truncated 구분, 역할 공개 및 확정/추론 분리, 실제 역할 사용 범위, DummyAgent 역할 추정, Invalid Action 상태 불변, PPO 단위, 평가, 감정 시간 정렬, 데이터 분리, Config 변경, 추론 모델 저장, 일시정지, dtype 및 인수 기준 정비  
> 변경 이력: v0.6 — Truncated 에피소드의 PPO 학습·Bootstrap 완전 제외, EmotionInference 0~1 연속값 및 MSE 확정, 모든 Reward Shaping 항목의 Config 가변성과 기본 숫자 미확정 원칙 확정  
> 변경 이력: v0.7 — Hardware Benchmark·자동 Worker/Batch 최적화 제거. CPU/RAM/GPU·Worker·Batch·Buffer 사용자 직접 설정, 입력 검증 및 GUI 권장 도움말로 정리

---

## 목차

1. [시스템 아키텍처 및 모듈 독립성](#1-시스템-아키텍처-및-모듈-독립성)
2. [게임 환경 (BangEnvironment)](#2-게임-환경-bangenvironment)
3. [환경 Step 및 게임 종료](#3-환경-step-및-게임-종료)
4. [Action Interface](#4-action-interface)
5. [Observation Interface 및 역할 공개](#5-observation-interface-및-역할-공개)
6. [역할별 승리/패배 조건](#6-역할별-승리패배-조건)
7. [더미 AI (Dummy Agent)](#7-더미-ai-dummy-agent)
8. [메인 학습 AI (Main Agent)](#8-메인-학습-ai-main-agent)
9. [프로그램 기능 및 GUI 요구사항](#9-프로그램-기능-및-gui-요구사항)
10. [Config Management 및 프리셋](#10-config-management-및-프리셋)
11. [학습 파이프라인 (3-Phase Pipeline)](#11-학습-파이프라인-3-phase-pipeline)
12. [PPO 학습 단위와 Trajectory 및 학습 신호 분리](#12-ppo-학습-단위와-trajectory-및-학습-신호-분리)
13. [Reward Function](#13-reward-function)
14. [데이터 저장 및 dtype](#14-데이터-저장-및-dtype)
15. [감정 상태와 학습 Label 시간 정렬](#15-감정-상태와-학습-label-시간-정렬)
16. [학습/검증/테스트 분리](#16-학습검증테스트-분리)
17. [하드웨어 리소스 및 성능 최적화](#17-하드웨어-리소스-및-성능-최적화)
18. [재현성 및 버전 관리](#18-재현성-및-버전-관리)
19. [평가 프로토콜](#19-평가-프로토콜)
20. [Config 변경과 Emotion Schema](#20-config-변경과-emotion-schema)
21. [추론 모델 저장 및 호환성](#21-추론-모델-저장-및-호환성)
22. [일시정지 및 재개](#22-일시정지-및-재개)
23. [결정 사항 요약](#23-결정-사항-요약)
24. [미결 사항](#24-미결-사항)
25. [인수 기준](#25-인수-기준)

---

## 1. 시스템 아키텍처 및 모듈 독립성

### REQ-ARCH-001
**[Ubiquitous]** 시스템은 게임 환경(BangEnvironment), 더미 AI(DummyAgent), 메인 학습 AI(MainAgent), 랜덤 AI(RandomAgent), GUI(UserInterface) 를 서로 독립된 모듈로 제공해야 한다.

### REQ-ARCH-002
**[Ubiquitous]** BangEnvironment는 플레이어 슬롯에 배치된 Agent의 구체적인 구현 타입(사람, RandomAgent, DummyAgent, MainAgent)을 내부적으로 참조하거나 구분해서는 안 된다.

### REQ-ARCH-003
**[Ubiquitous]** 시스템은 5개의 플레이어 슬롯 각각에 동일한 Agent 인터페이스를 구현한 임의의 Agent를 배치할 수 있어야 한다.

### REQ-ARCH-004
**[Ubiquitous]** 모든 Agent는 BangEnvironment가 제공하는 표준 Observation 및 Action 인터페이스를 통해서만 게임과 상호작용해야 한다.

### REQ-ARCH-005
**[Ubiquitous]** GUI는 BangEnvironment의 내부 자료구조에 직접 접근해서는 안 되며, 별도의 Application/Service 계층 또는 공개 API를 통해서만 게임 상태를 조회하고 명령을 전달해야 한다.

### REQ-ARCH-006
**[Ubiquitous]** 각 모듈(BangEnvironment, DummyAgent, MainAgent, RandomAgent, GUI)은 다른 모듈 없이 독립적으로 단위 테스트할 수 있어야 한다.

---

## 2. 게임 환경 (BangEnvironment)

### REQ-ENV-001
**[Ubiquitous]** BangEnvironment는 PettingZoo AEC API 규격을 준수하는 Python 모듈로 구현해야 한다.

### REQ-ENV-002
**[Ubiquitous]** BangEnvironment는 정확히 5명의 플레이어를 지원해야 하며, 역할 구성은 보안관 1명, 부관 1명, 무법자 2명, 배신자 1명으로 고정된다.

### REQ-ENV-003
**[Ubiquitous]** BangEnvironment는 게임 시작 시 보안관의 초기 체력을 5로, 그 외 모든 역할의 초기 체력을 4로 설정해야 한다.

### REQ-ENV-004
**[Ubiquitous]** BangEnvironment는 각 플레이어의 차례에 해당 플레이어가 생존한 다른 플레이어 1명을 대상으로 Bang을 사용하거나 Pass를 선택할 수 있도록 허용해야 한다.

### REQ-ENV-005
**[Ubiquitous]** BangEnvironment는 Bang 행동이 실행되면 대상 플레이어의 체력을 정확히 1 감소시켜야 한다.

### REQ-ENV-006
**[Ubiquitous]** BangEnvironment는 플레이어의 체력이 0 이하가 되는 즉시 해당 플레이어를 탈락 처리하고, 이후 그 플레이어에게 행동 기회를 부여해서는 안 된다.

### REQ-ENV-007
**[Ubiquitous]** BangEnvironment는 특수 능력, 카드 패, 거리 제한 규칙을 구현해서는 안 된다.

### REQ-ENV-008
**[When]** 정상 승패 Terminal Condition이 만족될 때, BangEnvironment는 게임을 즉시 종료(terminated)해야 한다.

### REQ-ENV-009
**[Ubiquitous]** Terminal Condition 판정 로직은 BangEnvironment가 단독으로 소유하며, Agent나 GUI는 해당 로직을 직접 구현하거나 호출해서는 안 된다.

---

## 3. 환경 Step 및 게임 종료

### 3.1 환경 Step 정의

#### REQ-STEP-001 — 환경 Step 정의
**[Ubiquitous]** 환경 Step은 살아 있는 플레이어의 유효한 게임 행동 1회로 정의되어야 한다. BANG 1회는 환경 Step 1회, PASS 1회는 환경 Step 1회로 계산한다.

#### REQ-STEP-002 — Invalid Action은 Step 미증가
**[When]** Invalid Action이 전달될 때, BangEnvironment는 환경 Step 카운터를 증가시켜서는 안 된다.

#### REQ-STEP-003 — AEC 정리 호출은 Step 미포함
**[Ubiquitous]** AEC API 준수를 위한 탈락·종료 Agent 정리 호출(`step(None)` 등)은 환경 Step으로 계산해서는 안 된다.

#### REQ-STEP-004 — 시간축 기준 통일
**[Ubiquitous]** 턴 제한, 경고 기준, 공개 이벤트 시간축은 모두 환경 Step을 기준으로 해야 한다. 문서 내 "턴 수" 표현은 환경 Step을 의미하며, 모든 생존 플레이어가 1회씩 행동하는 단위는 "라운드"로 별도 표기한다.

### 3.2 최대 에피소드 길이와 종료 유형

#### REQ-STEP-005 — PASS 상시 허용
**[Ubiquitous]** PASS는 활성 생존 플레이어의 행동 선택 시 항상 허용되어야 한다.

#### REQ-STEP-006 — 최대 환경 Step Config
**[Ubiquitous]** BangEnvironment는 최대 환경 Step 수 `max_episode_steps`를 Config로 관리해야 하며, 기본값은 2,000 환경 Step이어야 한다.

#### REQ-STEP-007 — 종료 평가 순서
**[When]** 유효 행동이 처리된 직후, BangEnvironment는 먼저 정상 승패 Terminal Condition(§6)을 평가하고, 정상 승패가 결정되지 않았으며 환경 Step 수가 `max_episode_steps`에 도달한 경우에 한하여 에피소드를 제한 종료(truncated)해야 한다.

#### REQ-STEP-008 — 종료 우선순위
**[When]** 동일한 행동에서 정상 승패 조건 충족과 최대 Step 도달이 동시에 발생할 때, BangEnvironment는 정상 승패 종료(terminated)를 우선 적용해야 한다.

#### REQ-STEP-009 — 종료 유형 구분
**[Ubiquitous]** BangEnvironment는 종료 유형을 다음 두 가지로 구분하여 표시해야 한다:
- 정상 승패 종료: `terminated=True`
- 최대 Step에 따른 제한 종료: `truncated=True`

#### REQ-STEP-010 — 제한 종료의 성격
**[Ubiquitous]** 제한 종료(truncated)는 게임 규칙상의 무승부가 아니라 `max_episode_steps` 도달에 따른 운영상 강제 종료이며, 정상적인 게임 결과가 아닌 버려진 게임(discarded game)으로 취급해야 한다. 제한 종료에는 승자·패자를 부여해서는 안 된다.

#### REQ-STEP-011 — 제한 종료 Reward
**[When]** 에피소드가 제한 종료될 때, BangEnvironment는 승리 `+1`·패배 `-1`의 Terminal Reward를 부여해서는 안 된다. 단, 제한 종료 시점까지 이미 발생한 Reward Shaping 값은 유지해야 한다.

#### REQ-STEP-012 — 제한 종료 통계 분리
**[Ubiquitous]** 제한 종료 에피소드는 정상 승패 승률의 분모에서 제외해야 하며, 제한 종료 횟수와 비율은 승률과 별도로 반드시 보고해야 한다.

#### REQ-STEP-013 — termination/truncation 학습 구분
**[Ubiquitous]** 정상 승패 종료(terminated)와 제한 종료(truncated)를 학습에서 구분해야 한다. Truncated 에피소드는 정상 승패 또는 패배로 취급하지 않으며, 해당 에피소드의 transition을 PPO 학습 대상에서 제외해야 한다.

#### REQ-STEP-014 — 종료 Bootstrap 금지
**[Ubiquitous]** 정상 승패 종료와 truncated 종료 모두 종료 이후의 Value Bootstrap을 적용해서는 안 된다. 시스템은 존재하지 않는 다음 행동 상태를 Bootstrap용으로 생성해서는 안 되며, MainAgent가 이미 탈락한 경우에도 그러하다.

### 3.3 제한 종료 및 진단 로그

#### REQ-STEP-015 — 제한 종료 이벤트 보존
**[When]** 에피소드가 제한 종료될 때, BangEnvironment는 실행·로깅 계층이 전체 진행을 재구성할 수 있도록 다음 환경 이벤트와 상태를 제공해야 한다:
- 에피소드 ID 및 Random Seed
- 초기 좌석·역할·체력 상태
- 시작부터 종료까지 모든 유효 행동과 공개 이벤트
- 행동별 체력 변화
- 탈락 및 역할 공개 이벤트
- 최종 체력과 생존 상태
- 종료 원인 및 적용된 `max_episode_steps`

#### REQ-STEP-015-A — 제한 종료 실행 로그 결합
**[When]** 제한 종료 로그가 저장될 때, 실행·로깅 계층은 Environment가 제공한 이벤트·상태에 Environment Version, Agent Preset, Reward Config Version 등 버전·설정 식별자와 평가·실험 식별자를 결합해야 한다. 실제 비공개 역할이 포함된 특권 진단 로그는 Agent Observation 및 공개 이벤트 채널과 분리해야 한다.

#### REQ-STEP-016 — 긴 게임 경고 로그
**[When]** 환경 Step 수가 `warn_turn_threshold`(기본값 200)를 초과할 때, BangEnvironment는 에피소드 ID, 현재 환경 Step 수, 생존 플레이어 목록 및 각 체력, 최근 N개 이벤트 요약을 포함하는 WARNING 로그를 기록해야 한다.

#### REQ-STEP-017 — 긴 게임 심각 로그
**[When]** 환경 Step 수가 `critical_turn_threshold`(기본값 500)를 초과할 때, BangEnvironment는 CRITICAL 수준의 로그를 기록해야 한다.

#### REQ-STEP-018 — 임계값 순서 검증
**[Ubiquitous]** 시스템은 Config 검증 시 `0 < warn_turn_threshold < critical_turn_threshold < max_episode_steps` 순서가 성립하는지 확인해야 하며, 성립하지 않으면 Config 오류를 보고해야 한다.

#### REQ-STEP-019 — 로그 반복 억제
**[Ubiquitous]** 동일 임계값 초과 로그는 에피소드별·임계값별 1회만 기록되어야 하며, 해당 에피소드에서 임계값을 초과한 이후 매 Step 무제한 반복 기록해서는 안 된다.

#### REQ-STEP-020 — 특권 로그 채널 분리
**[Ubiquitous]** 실제 비공개 역할이 포함된 특권 진단 로그는 Agent Observation 및 Agent가 읽는 공개 이벤트 채널과 물리적으로 분리된 채널에 기록되어야 한다.

---

## 4. Action Interface

### REQ-ACT-001 — Action Space 정의
**[Ubiquitous]** BangEnvironment는 모든 Agent에게 동일한 `Discrete(6)` Action Space를 제공해야 한다. 각 인덱스의 의미는 다음과 같이 고정된다:
- `0`: PASS
- `1`: PLAYER_0 공격
- `2`: PLAYER_1 공격
- `3`: PLAYER_2 공격
- `4`: PLAYER_3 공격
- `5`: PLAYER_4 공격

### REQ-ACT-002 — Action Space와 Action Mask의 분리
**[Ubiquitous]** Action Space(`Discrete(6)`)는 논리적으로 존재하는 행동의 전체 집합을 정의한다. 실제로 선택 가능한 행동은 매 스텝 제공되는 Action Mask에 의해 결정된다. 두 개념은 명확히 분리하여 설계해야 한다.

### REQ-ACT-003 — Action Mask 규칙
**[Ubiquitous]** BangEnvironment는 매 스텝 현재 상태에서 선택 가능한 행동을 나타내는 Action Mask(shape `(6,)`, dtype `uint8`, 값 0 또는 1)를 Observation에 포함해야 한다. Action Mask는 다음 규칙을 따른다:
- `PASS`(인덱스 0)는 항상 선택 가능하다 (mask = 1).
- 이미 탈락한 플레이어를 대상으로 하는 공격 행동은 선택 불가능하다 (mask = 0).
- 자기 자신을 대상으로 하는 공격 행동은 선택 불가능하다 (mask = 0).
- 그 외의 생존한 타 플레이어를 대상으로 하는 공격 행동은 선택 가능하다 (mask = 1).

### REQ-ACT-004 — Agent의 Action Mask 준수 의무
**[Ubiquitous]** 모든 Agent는 Action Mask를 준수하여 mask = 1인 행동만 Environment에 전달해야 한다. Action Mask는 유효하지 않은 행동을 사전에 차단하는 주된 수단이다.

### REQ-ACT-005 — Environment의 최종 검증
**[Ubiquitous]** BangEnvironment는 Agent로부터 전달받은 Action이 현재 Action Mask에 의해 허용된 행동인지 최종 검증해야 한다.

### REQ-ACT-006 — Invalid Action 처리 및 로깅
**[When]** BangEnvironment가 Action Mask상 허용되지 않은 Action을 전달받을 때, 다음 규칙을 따른다:
- 해당 Action을 정상적인 게임 행동으로 처리하지 않는다.
- PASS로 자동 변환하지 않는다.
- 다음 정보를 포함하는 오류 로그를 즉시 기록한다:
  - 발생 시각 및 에피소드 ID
  - 현재 턴 수 및 행동 플레이어 인덱스
  - 전달된 유효하지 않은 행동 값
  - 현재 Action Mask 상태 (6비트)
  - 현재 게임 상태 요약 (생존 플레이어 목록, 각 체력)

### REQ-ACT-007 — Invalid Action 로그 채널
**[Ubiquitous]** Invalid Action 로그는 일반 게임 로그와 구분되는 별도 채널(`invalid_action` 로그 레벨 또는 별도 파일 `logs/invalid_actions/`)에 기록되어야 하며, 게임 로직 오류 및 Agent 구현 오류를 진단하는 데 활용할 수 있어야 한다.

### REQ-ACT-008 — Agent 유형별 Mask 적용
**[Ubiquitous]** 모든 Agent는 정상 행동 선택 경로에서 Action Mask를 적용해야 한다:
- 사람(HumanAgent): mask = 0인 행동 버튼을 비활성화해야 한다.
- RandomAgent: mask = 1인 행동에서만 샘플링해야 한다.
- DummyAgent: 최종 선택 전에 Mask를 적용하고 확률을 재정규화해야 한다.
- MainAgent: Mask 적용 후 행동을 선택해야 한다.
- PPO: rollout 수집 당시의 Mask를 보존하고, 행동 확률 재계산 시 동일한 Mask를 적용해야 한다.

### REQ-ACT-009 — Invalid Action 상태 불변 계약
**[When]** BangEnvironment가 Invalid Action(Mask 미허용, 범위 밖 값, 잘못된 타입 포함)을 전달받을 때, 다음을 모두 보장해야 한다:
- 게임 상태를 변경하지 않는다.
- 현재 차례(agent_selection)를 변경하지 않는다.
- 환경 Step을 증가시키지 않는다.
- 정상 공개 행동 이벤트를 생성하지 않는다.
- DummyAgent의 감정 갱신·감쇠를 발생시키지 않는다.
- REQ-ACT-006의 오류 로그를 기록한 뒤 명시적 예외를 발생시킨다.
- PASS 자동 변환이나 묵시적 재선택을 하지 않는다.

### REQ-ACT-010 — AEC 정리 호출 구분
**[Ubiquitous]** AEC API에 따른 탈락·종료 Agent의 `step(None)` 정리 호출은 정상 게임 행동 및 일반 Invalid Action과 구분되어야 한다. 활성 생존 플레이어에 대한 Action Mask 규칙을 정리 호출에 그대로 적용해서는 안 된다.

---

## 5. Observation Interface 및 역할 공개

### 5.1 Observation 기본 구조

#### REQ-OBS-001
**[Ubiquitous]** BangEnvironment는 각 Agent에게 자신의 시점에서 관찰 가능한 정보만을 포함하는 Observation을 제공해야 한다.

#### REQ-OBS-002 — 공개 정보
**[Ubiquitous]** Observation은 다음 공개 정보를 포함해야 한다:
- 5개 플레이어 슬롯의 현재 체력
- 5개 플레이어 슬롯의 생존/탈락 상태
- 현재 차례의 플레이어 인덱스 및 고정된 플레이어 턴 순서
- 현재까지의 게임 진행 환경 Step 수
- 각 플레이어의 과거 공개 행동 이력 (누가, 누구를, Bang/Pass)
- 공격으로 인한 체력 변화 이력
- 플레이어 탈락 이벤트 이력
- 보안관의 공개 신분 (플레이어 인덱스)
- 탈락한 플레이어가 공개한 실제 역할 (REQ-OBS-009)

#### REQ-OBS-003 — 비공개 정보
**[Ubiquitous]** Observation은 해당 Agent 자신의 실제 역할 정보를 포함해야 한다.

#### REQ-OBS-006
**[Ubiquitous]** BangEnvironment는 특정 AI 구현에 종속되지 않는 canonical observation을 제공해야 하며, 구조는 다음 4개 구성 요소를 포함해야 한다:
- Current State Vector
- Player State Matrix
- Action History / Event Sequence
- Action Mask (현재 스텝에서 선택 가능한 행동을 나타내는 마스크, shape `(6,)`)

#### REQ-OBS-007
**[Ubiquitous]** BangEnvironment는 행동 이력을 원본 형태로 제공해야 하며, GRU/1D-CNN 입력을 위한 길이 조정, padding, truncation 등의 전처리는 Agent/Model 계층에서 담당해야 한다.

#### REQ-OBS-008 — 이벤트 이력 길이 설정
**[Ubiquitous]** BangEnvironment는 Observation에 포함할 최근 이벤트 이력의 최대 개수 K를 Config로 관리해야 한다. K의 초기 기본값은 무제한(unlimited)이며, 이는 해당 에피소드 시작부터 현재까지의 모든 이벤트를 원본 형태로 제공한다는 의미이다. 추후 Config를 통해 K를 유한값으로 변경할 수 있어야 한다.

### 5.2 탈락 시 역할 공개

#### REQ-OBS-009 — 탈락 시 역할 공개
**[When]** 플레이어가 탈락할 때, BangEnvironment는 해당 플레이어의 실제 역할을 공개해야 하며, 공개된 역할을 이후 Observation과 공개 이벤트 이력에 포함해야 한다.

#### REQ-OBS-010 — 보안관 신분 공개
**[Ubiquitous]** 보안관의 역할은 게임 시작부터 모든 Agent에게 공개되어야 한다.

#### REQ-OBS-011 — 진영별 생존자 수 계산 가능성
**[Ubiquitous]** 고정 역할 구성(보안관 1, 부관 1, 무법자 2, 배신자 1)과 공개된 탈락자 역할로부터, 공개 정보만으로 역할별·진영별 생존자 수를 정확히 계산할 수 있어야 한다.

### 5.3 확정 역할과 추론 역할의 분리

#### REQ-OBS-012 — 확정 역할 정의
**[Ubiquitous]** 다음 역할은 확정값으로 취급해야 한다:
- 자신의 역할
- 보안관의 공개 역할
- 탈락자가 공개한 역할
- 공개 정보와 남은 역할 수만으로 논리적으로 확정되는 역할

#### REQ-OBS-013 — 추론 대상 한정
**[Ubiquitous]** 실제로 미공개인 역할만 역할 추론의 대상으로 취급해야 한다. 확정된 역할은 one-hot 등 확정 분포로 사용해야 한다.

#### REQ-OBS-014 — 확정 정보 우선
**[Ubiquitous]** 모델 출력과 확정 정보를 결합하는 계층에서는 확정 정보가 모델 추론값보다 우선해야 한다. 어떤 경우에도 실제 비공개 역할을 읽어서 역할을 확정해서는 안 된다.

#### REQ-OBS-015 — 미공개 역할 성능 별도 측정
**[Ubiquitous]** 역할 분류 평가에서는 미공개 역할에 대한 분류 성능을 확정 역할과 분리하여 별도로 측정해야 한다.

### 5.4 실제 역할 사용 범위

#### REQ-OBS-016 — 실제 역할 허용 용도
**[Ubiquitous]** 비공개 실제 역할은 다음 용도로만 사용할 수 있다:
- Environment의 규칙 및 승패 판정
- 탈락 시 역할 공개
- 지도학습 정답 Label 생성
- 명시된 학습용 Reward 계산
- 평가 및 특권 진단 로그

#### REQ-OBS-017 — 실제 역할 금지 경로
**[Ubiquitous]** 비공개 실제 역할은 다음 경로에 포함되어서는 안 된다:
- Agent의 Observation
- DummyAgent의 행동 결정 입력
- MainAgent의 역할·감정 추론 입력
- MainAgent의 PPO 정책 및 가치 예측 입력
- Agent가 읽는 공개 이벤트 이력

#### REQ-OBS-018 — 추정값·확정값 허용
**[Ubiquitous]** 공개 정보로부터 계산한 역할 추정값과 논리적 확정값은 Agent 입력에 사용할 수 있다.

---

## 6. 역할별 승리/패배 조건

> 승패 판정은 플레이어가 탈락한 직후 BangEnvironment가 단독으로 수행한다.  
> 정상 승패 종료(terminated) 시에는 어떤 생존자 조합에서도 승자가 정의되지 않는 미결 상태가 발생해서는 안 된다.  
> 정상 승패가 성립하지 않은 채 `max_episode_steps`에 도달하면 제한 종료(truncated, §3)로 처리하며, 이는 무승부가 아니라 운영상 중단이다.

### REQ-WIN-001 — 보안관팀 승리
**[When]** 보안관이 생존한 상태에서 모든 무법자와 배신자가 탈락하면, BangEnvironment는 보안관팀(보안관, 부관)의 승리를 즉시 판정하고 게임을 종료해야 한다.

### REQ-WIN-001-A — 부관의 팀 승패 공유
**[Ubiquitous]** 부관은 자신의 생존 여부와 무관하게 보안관팀의 최종 결과를 항상 공유한다. 보안관팀 승리 시 부관은 탈락 상태라도 승리로 판정하며, 보안관팀 패배 시 부관도 패배로 판정한다.

### REQ-WIN-002 — 보안관 탈락 시 승패 판정
**[When]** 보안관이 탈락하면, BangEnvironment는 즉시 다음 규칙에 따라 승패를 판정하고 게임을 종료해야 한다:
- 배신자만 생존한 경우 → 배신자 승리. 보안관·부관·무법자 패배.
- 배신자 이외의 생존자가 한 명이라도 존재하는 경우(무법자 포함) → 무법자팀 승리. 보안관·부관·배신자 패배.

### REQ-WIN-003 — 배신자 승리
**[When]** REQ-WIN-002의 보안관 탈락 분기에서 배신자만 생존한 경우, BangEnvironment는 배신자의 승리를 판정해야 한다.

### REQ-WIN-004 — 무법자팀 승리
**[When]** REQ-WIN-002의 보안관 탈락 분기에서 배신자 이외의 생존자가 한 명이라도 존재하는 경우, BangEnvironment는 무법자팀의 승리를 판정해야 한다.

### REQ-WIN-005 — 배신자 패배 (보안관팀 승리)
**[When]** REQ-WIN-001에 따라 보안관팀 승리가 판정될 때, BangEnvironment는 배신자의 패배를 판정해야 한다.

### REQ-WIN-006 — 결투 상태 미적용
**[Ubiquitous]** 보안관과 배신자만 생존하는 상황은 별도의 강제 결투 상태로 전환하지 않는다. 두 플레이어는 정상적인 턴 순서에 따라 행동을 계속 진행한다. 보안관이 배신자를 탈락시키면 REQ-WIN-001에 따라 보안관팀 승리, 배신자가 보안관을 탈락시키면 REQ-WIN-003에 따라 배신자 승리로 판정한다.

### REQ-WIN-007 — Terminal Condition 평가 순서
**[Ubiquitous]** BangEnvironment는 플레이어 탈락 직후 다음 순서로 Terminal Condition을 평가해야 한다:
1. **보안관이 탈락한 경우**
   - 배신자만 생존 → 배신자 승리 (Terminal)
   - 그 외 생존자 존재 → 무법자팀 승리 (Terminal)
2. **보안관이 생존한 경우**
   - 모든 무법자와 배신자가 탈락 → 보안관팀 승리 (Terminal)
3. **그 외** → 게임 계속 진행

### REQ-WIN-008 — 팀 승패 공유
**[Ubiquitous]** 팀 최종 결과는 개인 생존 여부와 무관하게 공유되어야 한다. 보안관팀 승리 시 부관은 탈락 상태라도 승리이며, 무법자팀 승리 시 무법자는 생존 여부와 무관하게 무법자팀의 최종 결과를 공유한다. 무법자 2명이 모두 탈락했더라도 보안관 탈락 시 배신자 이외의 생존자가 존재하면 REQ-WIN-004에 따라 무법자팀 승리로 판정한다.

> REQ-STEP-006~014에 따라 정상 승패 없이 최대 환경 Step에 도달하면 제한 종료된다.

---

## 7. 더미 AI (Dummy Agent)

### REQ-DUMMY-001
**[Ubiquitous]** DummyAgent는 BangEnvironment와 독립된 모듈로 구현되어야 하며, IntentModule, EmotionModule, ActionSelector의 세 하위 모듈로 구성되어야 한다.

### REQ-DUMMY-002 — IntentModule
**[Ubiquitous]** IntentModule은 자신의 역할별 승리 목표를 최우선으로 하는 기본 행동 정책을 생성해야 한다.

### REQ-DUMMY-003
**[Ubiquitous]** IntentModule의 역할별 공격 대상 우선순위, 행동 조건, 예외 행동 규칙은 Config 파일로 관리되어야 한다.

### REQ-DUMMY-003-A — DummyAgent 역할 추정 기본값
**[Ubiquitous]** DummyAgent의 역할 추정은 학습 모델에 의존하지 않는 단순 기본 추정기를 사용해야 하며, 다음 절차를 따른다:
1. 자신, 보안관, 탈락자의 알려진(확정) 역할을 반영한다.
2. 미확정 슬롯에 배정될 남은 역할 수를 계산한다.
3. 미확정 생존자에게 남은 역할 비율에 따른 기본 확률을 부여한다.
4. 공개 정보만으로 역할이 확정되면 해당 슬롯을 확정값으로 전환한다.

이 추정기는 실제 비공개 역할을 참조해서는 안 되며, 향후 공개 행동 기반 휴리스틱으로 교체·확장할 수 있어야 한다.

### REQ-DUMMY-003-B — MainAgent 추론 모듈 비의존
**[Ubiquitous]** DummyAgent는 MainAgent의 학습된 추론 모듈을 필수 의존성으로 사용해서는 안 된다.

### REQ-DUMMY-003-C — 배신자 진영 세력 판단
**[Ubiquitous]** 배신자 DummyAgent는 보안관 진영과 무법자 진영의 세력을 다음 순서로 비교해야 한다:
1. 공개 정보로 정확히 계산한 진영별 생존자 수가 더 많은 진영을 강한 진영으로 판단한다.
2. 생존자 수가 같으면 추정 체력 합산이 더 높은 진영을 강한 진영으로 판단한다. 추정 체력은 `진영 추정 체력 = Σ(생존 플레이어의 공개 체력 × 해당 진영일 추정 확률)`로 계산한다.
3. 두 값이 모두 같으면 Config에 정의된 동률 정책을 적용한다.

### REQ-DUMMY-003-D — 배신자 공격 우선순위
**[Ubiquitous]** 배신자 DummyAgent의 기본 행동 우선순위는 다음을 따라야 한다:
- 강한 진영을 공격하는 방향으로 우선순위를 생성한다.
- 공격 대상 선택에는 실제 비공개 역할이 아니라 역할 추정값을 사용한다.
- 보안관과 배신자만 생존한 1대1 상황이 아니면 보안관 직접 공격의 우선순위 가중치를 낮게 둔다.
- 보안관과 배신자만 생존하면 보안관 공격을 가장 높은 우선순위 가중치로 둔다.
- 세부 가중치와 동률 처리 정책은 Config로 관리한다.

### REQ-DUMMY-003-E — 배신자 행동 제한 범위
**[Ubiquitous]** 배신자에게 역할에 따른 추가 행동 제한을 적용해서는 안 된다. 단, 자기 자신 및 탈락한 플레이어 공격 금지 등 공통 Action Mask 규칙(REQ-ACT-003)은 준수해야 한다. 위 진영 판단 정책은 IntentModule Config에서 조정 가능한 기본값이며, Environment 레벨에서 강제되지 않는다.

### REQ-DUMMY-004
**[Ubiquitous]** IntentModule은 BangEnvironment의 내부 구현을 직접 참조해서는 안 되며, Observation만을 통해 행동 정책을 생성해야 한다.

### REQ-DUMMY-005
**[When]** 일반 게임 실행 중 사용자가 GUI 또는 Config 파일을 통해 IntentModule의 비구조 설정을 변경할 때, 시스템은 변경된 설정을 다음 행동 선택 경계부터 반영해야 한다(REQ-CFG-005). 진행 중인 수집·학습 작업에는 즉시 적용하지 않는다(REQ-CFG-006).

### REQ-DUMMY-006 — EmotionModule
**[Ubiquitous]** EmotionModule은 감정 상태를 N개의 독립적인 수치형 변수로 관리해야 한다.

### REQ-DUMMY-007
**[Ubiquitous]** EmotionModule의 감정 항목(이름, 값의 범위, 초기값, 변화량, 감쇠/회복 설정)은 코드에 하드코딩되어서는 안 되며, Config를 통해 동적으로 추가·삭제·수정할 수 있어야 한다.

### REQ-DUMMY-008
**[Ubiquitous]** 각 감정이 실제 행동에 미치는 영향(영향 방향, 영향 크기, 적용 조건)은 코드에 하드코딩되어서는 안 되며, 별도의 Config로 관리되어야 한다.

### REQ-DUMMY-009
**[When]** 사용자가 EmotionModule 설정을 편집할 때, 시스템은 변경 유형에 따라 다르게 적용해야 한다:
- 감정 변화량·감쇠 등 비구조 동작 파라미터 변경: 일반 게임에서는 다음 행동 경계부터 적용(REQ-CFG-005), 진행 중 학습 작업에는 다음 작업부터 적용(REQ-CFG-006). Preset 버전으로 추적한다(REQ-CFG-008).
- 감정 이름·개수·순서·값 범위·정규화 등 구조 변경: 새 Emotion Schema Version을 요구한다(REQ-CFG-007).

### REQ-DUMMY-010 — Intent + Emotion 결합
**[Ubiquitous]** DummyAgent는 다음 순서로 행동을 결정해야 한다:
1. Observation → IntentModule → 기본 행동 우선순위/분포 생성
2. EmotionModule → 감정에 따른 행동 확률/우선순위 변형
3. ActionSelector → 최종 Action 선택

---

## 8. 메인 학습 AI (Main Agent)

### REQ-MAIN-001 — 3단계 샌드위치 구조
**[Ubiquitous]** MainAgent는 의도 추론 모듈(IntentInferenceModule), 감정 추론 모듈(EmotionInferenceModule), 최종 판단 모듈(PPOModule)의 3단계 구조로 구현되어야 한다.

### REQ-MAIN-002 — 의도 추론 모듈 (IntentInferenceModule)
**[Ubiquitous]** 의도 추론 모듈은 타 플레이어들의 행동 이력 시퀀스를 입력받아 각 플레이어의 역할 확률 분포를 출력해야 한다.

### REQ-MAIN-003 — 시퀀스 인코더: GRU 기본, 교체 가능 구조
**[Ubiquitous]** 의도 추론 모듈의 기본 시퀀스 인코더는 GRU로 구현한다. 단, 인코더 구현체는 `SequenceEncoder` 인터페이스로 추상화하여 추후 GRU 외의 인코더(1D-CNN 등)로 교체 가능한 구조로 설계해야 한다.

### REQ-MAIN-004 — 감정 추론 모듈 (EmotionInferenceModule)
**[Ubiquitous]** 감정 추론 모듈은 타 플레이어의 공개 행동 이력, 공격/피격 기록, 체력 변화, 생존 상태 등 MainAgent가 실제 게임 중 관찰 가능한 공개 정보만을 입력으로 받아, 해당 플레이어의 현재 감정 상태 벡터를 출력해야 한다. DummyAgent의 실제 Emotion State는 입력 Feature로 사용하지 않는다.

### REQ-MAIN-004-A
**[Ubiquitous]** 감정 추론 모듈의 출력 차원은 현재 적용된 Emotion Schema Version의 감정 항목 수와 일치해야 한다. Emotion Schema가 변경되면 감정 추론 모듈의 재학습이 필요하다.

### REQ-MAIN-004-B — 연속 감정값 출력
**[Ubiquitous]** EmotionInferenceModule은 감정을 범주형 class가 아닌 연속적인 실수값 벡터로 예측해야 하며, 감정 변수가 N개이면 출력은 `[e1, e2, ..., eN]` 형태여야 한다. 모든 `ei`는 `0.0 <= ei <= 1.0` 범위에 있어야 한다.

### REQ-MAIN-004-C — EmotionInference Loss
**[Ubiquitous]** EmotionInferenceModule의 기본 지도학습 Loss는 예측 벡터와 DummyAgent의 실제 EmotionModule 상태 벡터 사이의 MSE(Mean Squared Error)로 정의해야 한다.

### REQ-MAIN-005 — 최종 판단 모듈 (PPOModule)
**[Ubiquitous]** 최종 판단 모듈은 현재 게임 상태(Self + Board State)와 의도 추론 모듈 및 감정 추론 모듈의 출력 벡터를 결합(Concatenate)한 입력으로부터 최종 행동을 결정해야 한다.

### REQ-MAIN-006 — 모듈 결합 방식
**[Ubiquitous]** 두 추론 모듈의 출력은 최종 판단 모듈의 입력으로 Concatenate되어야 한다.

### REQ-MAIN-007 — Phase 2 지도학습
**[Ubiquitous]** 의도 추론 모듈과 감정 추론 모듈은 Phase 1에서 수집된 대국 데이터를 사용하여 지도학습(Supervised Learning)으로 사전 학습되어야 한다.

### REQ-MAIN-008 — Phase 3 강화학습 시 모듈 동결
**[When]** Phase 3 강화학습이 시작될 때, 시스템은 Phase 2에서 학습된 의도 추론 모듈과 감정 추론 모듈의 가중치를 동결(Freeze)하고, 최종 판단 모듈(PPO)만 학습시켜야 한다.

### REQ-MAIN-009 — 역할·감정 추론 디버그 출력
**[When]** 디버그 뷰가 활성화된 상태에서 게임이 진행될 때, 시스템은 MainAgent가 추론한 타 플레이어 4명(자신 제외)의 역할 확률 분포와 추정 감정 상태를 GUI HUD에 표시해야 한다. MainAgent 자신의 역할은 Observation에 포함된 정보를 그대로 사용하며 추론 대상이 아니다.

---

## 9. 프로그램 기능 및 GUI 요구사항

### REQ-GUI-001 — 메인 화면 메뉴 구성
**[Ubiquitous]** 프로그램 메인 화면은 다음 메인 메뉴 항목을 포함해야 한다:
- 게임 실행 모드
- AI 학습 모드
- 설정 메뉴
- 프로그램 종료

### REQ-GUI-002 — 게임 실행 모드 선택
**[Ubiquitous]** 게임 실행 모드는 다음 세 가지 대전 구성을 지원해야 한다:
- 사람 1명 vs 랜덤 AI 4명
- 사람 1명 vs 더미 AI 4명
- 사람 1명 vs 학습 AI 1명 + 더미 AI 3명

### REQ-GUI-003 — 학습 AI 선택 모달
**[When]** 사용자가 '사람 1명 vs 학습 AI 1명 + 더미 AI 3명' 모드를 선택할 때, 시스템은 저장된 AI 모델 목록을 표시하는 선택 모달을 제공해야 하며, 각 모델의 Metadata(생성 일시, 학습 상대 설정, 총 학습 Step, 최종 승률)를 미리보기로 표시해야 한다.

### REQ-GUI-004 — 개발자 디버그 HUD
**[When]** 게임 실행 중 디버그 뷰 옵션이 활성화된 상태에서 MainAgent가 배치된 경우, 시스템은 MainAgent가 추론하고 있는 타 플레이어의 역할 확률 분포 및 감정 상태를 인게임 HUD로 실시간 표시해야 한다.

### REQ-GUI-005 — AI 학습 모드 진입
**[Ubiquitous]** AI 학습 모드에서 사용자는 다음을 설정할 수 있어야 한다:
- 학습 상대 선택: 랜덤 AI 또는 특정 의도/감정 프리셋의 더미 AI
- 학습 분량 설정: 목표 MainAgent transition 수 또는 학습 시간

### REQ-GUI-006 — Headless 고속 학습
**[When]** AI 학습 모드가 실행될 때, 시스템은 인게임 그래픽 연산을 비활성화하고 메모리 상에서 고속 시뮬레이션을 수행해야 한다.

### REQ-GUI-007 — 학습 실시간 시각화
**[When]** AI 학습 모드가 실행 중일 때, 시스템은 별도 GUI 창 또는 위젯에서 학습 진행도(Progress Bar)와 시간에 따른 AI 승률 변화 그래프를 실시간으로 렌더링해야 한다.

### REQ-GUI-008 — 학습 일시정지·재개·조기 종료
**[Ubiquitous]** AI 학습 모드는 다음 기능을 제공해야 한다:
- 학습 일시정지 및 재개 (메모리 기반, §22)
- 조기 종료

학습 중간 체크포인트 저장 및 기존 모델/Metadata 기반 학습 이어하기(Resume)는 v0.5에서 제거되었다(§21).

### REQ-GUI-009 — 학습 완료·조기 종료 시 추론용 모델 저장
**[When]** 학습이 정상 완료되거나 사용자가 조기 종료할 때, 시스템은 게임 실행용 추론 모델을 REQ-INFER-001의 Metadata와 함께 저장해야 한다. 이 저장물은 학습 Resume용 체크포인트가 아니라 추론 전용 모델이다.

### REQ-GUI-010 — 설정 메뉴 — 더미 AI 설정
**[Ubiquitous]** 설정 메뉴는 더미 AI의 의도 모듈 규칙 및 감정 모듈 파라미터를 수정하고 프리셋을 저장·불러오기 할 수 있는 UI를 제공해야 한다.

### REQ-GUI-011 — 설정 메뉴 — Reward Function 수정
**[Ubiquitous]** 설정 메뉴는 중간 보상(Reward Shaping) 수치를 사용자가 GUI에서 직접 수정할 수 있는 UI를 제공해야 한다.

### REQ-GUI-012 — 설정 메뉴 — 하드웨어 및 학습 자원 수동 설정
**[Ubiquitous]** 설정 메뉴는 사용자가 CPU 코어 수, RAM 크기, GPU 사용 여부(CUDA/CPU), Worker 수, Batch Size 및 학습 Buffer 크기를 직접 입력하여 학습 환경에 적용할 수 있는 UI를 제공해야 한다.

### REQ-GUI-013 — GUI 프레임워크 명시
**[Ubiquitous]** GUI는 PyQt6로 구현해야 하며, 학습 진행도·승률 변화 등 실시간 그래프는 pyqtgraph로 렌더링해야 한다.

### REQ-GUI-014 — Headless 학습 중 GUI 표시 허용
**[Ubiquitous]** REQ-GUI-006·REQ-HW-004의 Headless 렌더링 제거는 인게임 그래픽 연산 제거를 의미하며, 학습 진행도·승률 그래프 등 학습 모니터링 GUI 표시(REQ-GUI-007)를 금지하지 않는다.

---

## 10. Config Management 및 프리셋

### REQ-CFG-001
**[Ubiquitous]** 더미 AI의 의도 모듈 및 감정 모듈 설정은 JSON 또는 YAML 프리셋 파일로 Export할 수 있어야 한다.

### REQ-CFG-002
**[Ubiquitous]** 시스템은 외부 JSON 또는 YAML 프리셋 파일을 Import하여 더미 AI 설정에 적용할 수 있어야 한다. 적용 시점은 REQ-CFG-005·REQ-CFG-006을 따른다: 일반 게임에서는 다음 행동 경계, 진행 중 학습 작업에는 다음 작업부터 적용한다.

### REQ-CFG-003
**[When]** 프리셋 파일을 Import할 때, 시스템은 파일의 유효성을 검사하고 오류가 있는 경우 구체적인 오류 내용을 사용자에게 표시해야 한다.

### REQ-CFG-004
**[Ubiquitous]** Reward Shaping 설정은 버전 정보를 포함하는 Config 파일로 관리되어야 하며, GUI에서 가중치를 변경하고 저장할 수 있어야 한다.

---

## 11. 학습 파이프라인 (3-Phase Pipeline)

### REQ-PIPE-001 — Phase 1: 데이터 수집
**[Ubiquitous]** 시스템은 더미 AI 간 Headless 고속 대국 시뮬레이션을 통해 대량의 대국 로그를 수집하는 기능을 제공해야 한다.

### REQ-PIPE-002
**[Ubiquitous]** Phase 1 수집 데이터의 Canonical Dataset은 `.parquet` Shard 형태로 저장한다. Parquet은 전체 게임 로그 및 Feature를 보존하며 대규모 분석·필터링·재처리에 사용한다.

### REQ-PIPE-002-A
**[Ubiquitous]** Phase 2 지도학습에 필요한 데이터는 별도 전처리 단계에서 Parquet으로부터 추출하여 `.npz` 또는 이에 준하는 Tensor Cache Shard 형식으로 변환한다. NPZ 캐시는 원본 Parquet에서 파생된 재생성 가능한 캐시이며, 역할 분리 및 재생성 규칙은 REQ-DATA-004를 따른다.

### REQ-PIPE-002-B
**[Ubiquitous]** 데이터가 대규모로 증가하는 것을 고려하여 Parquet 원본 및 NPZ 학습 캐시 모두 Sharding을 지원해야 한다.

### REQ-PIPE-003 — Phase 2: 추론 모듈 지도학습
**[Ubiquitous]** 시스템은 Phase 1에서 수집된 데이터셋을 기반으로 의도 추론 모듈(역할 분류)과 감정 추론 모듈(감정 상태 추측)을 지도학습으로 학습시키는 기능을 제공해야 한다.

### REQ-PIPE-004
**[Ubiquitous]** Phase 2 지도학습의 학습 진행도 및 결과(정확도, 손실값)는 로그로 저장되어야 한다.

### REQ-PIPE-005 — Phase 3: PPO 강화학습
**[When]** Phase 3 강화학습이 실행될 때, 시스템은 Phase 2에서 학습된 두 추론 모듈의 가중치를 동결(Freeze)한 상태에서 최종 판단 모듈(PPO)만을 학습시켜야 한다.

### REQ-PIPE-006
**[When]** Phase 3 강화학습이 실행될 때, 시스템은 설정된 하드웨어 자원과 Reward Function 설정에 따라 병렬 학습을 수행해야 한다.

### REQ-PIPE-007 — Truncated 데이터 학습 제외
**[When]** Phase 1 Canonical Dataset이 Phase 2 지도학습 또는 Phase 3 PPO 학습용 Dataset으로 변환될 때, `termination_status=truncated`인 Episode와 그 transition을 학습 대상에서 제외해야 한다.

---

## 12. PPO 학습 단위와 Trajectory 및 학습 신호 분리

### 12.1 학습 단위 정의

#### REQ-PPO-001 — 세 가지 단위 구분
**[Ubiquitous]** 시스템은 다음 세 단위를 명확히 구분하여 정의·기록해야 한다:
- 환경 Step: 생존 플레이어의 유효한 게임 행동 1회 (REQ-STEP-001과 동일)
- MainAgent transition: MainAgent의 한 행동부터 그 MainAgent의 다음 행동 기회 또는 에피소드 종료까지의 구간
- PPO update: 수집한 transition으로 수행하는 파라미터 갱신 작업

#### REQ-PPO-002 — 구간 보상 합산
**[Ubiquitous]** 하나의 MainAgent transition 구간 동안 해당 MainAgent에게 발생한 보상은 합산되어 그 transition의 보상으로 사용되어야 한다.

#### REQ-PPO-003 — gamma 적용 단위
**[Ubiquitous]** 할인율 gamma는 환경 Step 단위가 아니라 MainAgent transition 단위로 적용되어야 한다.

#### REQ-PPO-004 — GUI 목표 학습 Step 단위
**[Ubiquitous]** GUI의 "목표 학습 Step"은 MainAgent transition 수를 의미하며, GUI는 이 단위를 화면에 명확히 표시해야 한다.

#### REQ-PPO-005 — 이중 카운터 기록
**[Ubiquitous]** 메타데이터에는 환경 Step 수와 MainAgent transition 수를 모두 별도로 기록해야 한다. 병렬 학습의 누적 수치는 전체 Worker 합산 기준임을 명시해야 한다.

#### REQ-PPO-006 — 경계 구분
**[Ubiquitous]** 학습 기록은 rollout 경계, 정상 승패 종료(terminated), 제한 종료(truncated)를 서로 구분하여 표시해야 한다.

### 12.2 탈락 후 Trajectory

#### REQ-PPO-007 — 탈락 후 행동 미생성
**[When]** MainAgent가 탈락한 후, 시스템은 해당 MainAgent에게 추가 행동을 요청하거나 가짜 PASS transition을 생성해서는 안 된다.

#### REQ-PPO-008 — 정상 종료 transition 보존
**[When]** 에피소드가 정상 승패 종료될 때, MainAgent의 마지막 미완료 transition은 최종 팀 보상이 반영될 때까지 학습 기록으로 유지되어야 한다. 이 규칙은 모든 PPO 업데이트를 게임 종료까지 금지한다는 의미가 아니라, 최종 결과가 확정되기 전 transition을 임의의 terminal transition으로 확정하지 말라는 의미이다.

#### REQ-PPO-009 — 정상 종료 학습 계약
**[When]** 에피소드가 정상 승패 종료될 때, 시스템은 해당 MainAgent transition을 PPO 학습 대상에 포함하고 최종 팀 결과를 반영한 Terminal Reward를 기록해야 한다. 정상 종료 transition 이후에는 Bootstrap을 적용해서는 안 된다.

#### REQ-PPO-010 — Truncated 학습 제외
**[When]** 에피소드가 `max_episode_steps`에 도달하여 truncated 종료될 때, 시스템은 해당 에피소드의 모든 MainAgent transition을 PPO 학습 데이터와 PPO update 대상에서 제외해야 한다. Truncated transition에는 Terminal Reward와 Value Bootstrap을 모두 적용해서는 안 된다.

#### REQ-PPO-011 — Truncated 기록 보존
**[When]** 에피소드가 truncated 종료될 때, 시스템은 게임 Episode Log와 통계 기록은 보존해야 하지만 해당 기록을 정상 승패 결과나 PPO 학습 transition으로 취급해서는 안 된다.

#### REQ-PPO-012 — AEC 정리와 학습 기록 분리
**[Ubiquitous]** AEC API상의 Agent 종료·정리 처리와 학습 기록 보존은 별개 계층에서 처리되어야 한다. AEC 정리 호출은 게임 행동이나 학습 transition으로 기록되어서는 안 된다.

#### REQ-PPO-013 — Bootstrap 상태 미생성
**[Ubiquitous]** 시스템은 존재하지 않는 다음 행동 상태를 Bootstrap용으로 생성해서는 안 된다. 정상 종료와 truncated 종료 모두 종료 이후의 가짜 다음 행동 상태를 생성해서는 안 되며, MainAgent가 탈락한 경우에도 그러하다.

#### REQ-PPO-014 — Rollout 경계와 종료 상태
**[Ubiquitous]** PPO 학습 기록은 rollout 경계, 정상 종료(terminated), truncated 종료를 구분해야 하며, truncated 에피소드의 기록은 PPO update 입력 버퍼에서 제외해야 한다.

정상 종료는 Terminal Reward·PPO 학습 포함·Bootstrap 없음으로 처리한다. Truncated는 Terminal Reward 없음·Bootstrap 없음·PPO 학습 제외로 처리하며, Episode Log와 통계만 보존한다.

### 12.3 학습 신호와 행동 입력 분리

#### REQ-PPO-015 — 행동 입력 금지 항목
**[Ubiquitous]** Role Label, 실제 Emotion Label, Reward 값·Reward 구성 요소·특권 로그는 어떤 Agent의 행동 결정 입력 Feature에도 포함되어서는 안 된다.

#### REQ-PPO-016 — Reward의 학습 사용
**[Ubiquitous]** Reward 및 Reward 구성 요소는 정상 종료 transition의 PPO 학습 계산(advantage, value target 등)에 사용할 수 있다. Truncated Episode의 transition은 PPO 학습에서 제외한다. "Agent는 Observation과 Action만으로 상호작용한다"는 원칙은 행동 결정 경로에 한하며, 학습기가 Reward를 수신하는 것을 금지하지 않는다.

#### REQ-PPO-017 — Feature/Label 분리
**[Ubiquitous]** 지도학습에서 입력 Feature와 정답 Label은 명시적으로 분리되어야 하며, Label(Role Label, 실제 Emotion Label)을 추론 모델의 입력 Feature에 포함해서는 안 된다.

#### REQ-PPO-018 — 경로 구분
**[Ubiquitous]** Agent 인터페이스의 Observation/Action 경로와 학습기의 Reward 수신 경로는 코드 상 구분되어야 한다.

---

## 13. Reward Function

### REQ-RWD-001
**[When]** 에피소드가 정상 승패 종료(terminated)될 때에 한하여, Reward Function은 승리 `+1`, 패배 `-1`의 Terminal Reward를 독립적인 구성 요소로 부여해야 한다. 제한 종료(truncated)에는 Terminal Reward를 부여하지 않는다(REQ-STEP-011).

### REQ-RWD-002
**[Ubiquitous]** Reward Shaping 구성 요소는 Terminal Reward와 독립적으로 관리되어야 한다.

### REQ-RWD-003
**[When]** 사용자가 GUI 또는 Config를 통해 Reward Shaping의 종류 또는 수치를 변경할 때, 시스템은 변경된 설정을 다음 학습 작업부터 반영해야 한다(진행 중 작업에는 즉시 적용하지 않음, REQ-CFG-006).

### REQ-RWD-004
**[Ubiquitous]** Reward Shaping은 최소한 다음 항목을 Config로 정의할 수 있어야 한다:
- Bang 적중 보상
- 무법자의 보안관 공격 보상
- 부관의 보안관 공격 감점
- 적 플레이어 처치 보상
- 역할·진영별 공격 관련 보상 또는 패널티
- `pass_penalty`: PASS 관련 보상 또는 패널티

### REQ-RWD-005
**[Ubiquitous]** Reward Shaping의 각 항목별 숫자 값은 코드에 고정하지 않고 Config에서 개별적으로 변경 가능해야 한다. Requirements 단계에서는 특정 기본 숫자를 영구적인 설계 결정으로 확정하지 않는다.

### REQ-RWD-006
**[Ubiquitous]** 최종 Reward 계산식과 각 구성 요소(Terminal Reward, 각 Shaping 항목)는 개별적으로 Logging되어야 한다.

### REQ-RWD-007
**[Ubiquitous]** Reward 설정은 버전 정보를 가져야 하며, 학습된 모델 Metadata에 실제 사용된 Reward Config 버전과 설정 식별자를 저장해야 한다.

### REQ-RWD-008 — Reward와 승패 규칙 분리
**[Ubiquitous]** Reward Config 변경은 게임의 승패 규칙 또는 Terminal Condition을 변경해서는 안 된다. Reward Shaping과 승패 판정은 별도 구성 요소로 관리해야 한다.

### REQ-RWD-009 — 정상 종료 최종 보상 기록
**[When]** 정상 승패 종료가 발생할 때, 시스템은 최종 승패와 이에 따른 Terminal Reward를 Episode Log 및 Reward Breakdown에 명확히 기록해야 한다.

> `pass_penalty`의 기본 숫자값은 Requirements 단계에서 확정하지 않는다. PASS는 전략적으로 유효할 수 있으므로 무조건 잘못된 행동으로 취급해서는 안 된다.

---

## 14. 데이터 저장 및 dtype

### REQ-DATA-001 — 필드별 dtype
**[Ubiquitous]** 데이터는 필드별 표현 범위에 맞는 dtype으로 저장해야 한다:
- 체력(HP), 역할 인덱스, 행동 인덱스, 플레이어 인덱스: `uint8`
- Action Mask: `uint8`, shape `(6,)`, 값 0 또는 1
- 생존 상태: `bool` 또는 `uint8`
- 환경 Step 번호, 이벤트 번호: 범위를 충족하는 정수형(`uint32` 이상)
모든 discrete 값을 무조건 8비트로 제한해서는 안 된다.

### REQ-DATA-002
**[Ubiquitous]** 연속형 Feature 및 감정값은 정밀도 검증 후 `float16` 또는 `float32`를 사용해야 하며, EmotionModule 상태와 EmotionInference Target/Output은 `0.0~1.0` 범위로 정규화해야 한다.

### REQ-DATA-003
**[Ubiquitous]** Role Label은 `uint8` dtype으로 저장해야 한다.

### REQ-DATA-004 — Parquet 원본 / NPZ 캐시 역할 분리
**[Ubiquitous]** Parquet Shard는 기준 원본 데이터이며, NPZ Tensor Cache Shard는 학습에 필요한 파생 Tensor를 담는 재생성 가능한 캐시이다. Phase 1 수집 데이터는 Parquet Shard로 저장하고, Phase 2 학습용 데이터는 Parquet에서 전처리하여 NPZ Tensor Cache Shard로 생성한다.

### REQ-DATA-004-A — 캐시 메타데이터
**[Ubiquitous]** NPZ Tensor Cache Shard는 원본 식별자, 전처리 버전, Emotion Schema Version, 데이터 분할 정보를 포함해야 하며, 삭제 후 원본 Parquet으로부터 재생성할 수 있어야 한다.

### REQ-DATA-004-B — 이력 복제 금지
**[Ubiquitous]** 시스템은 매 환경 Step마다 전체 누적 이력을 다시 복제해서는 안 된다. 원본 이벤트는 한 번씩만 저장하고, 필요한 학습 시퀀스는 이로부터 구성해야 한다.

### REQ-DATA-004-C — PASS 대상 표현
**[Ubiquitous]** PASS 등 대상이 없는 이벤트의 대상 플레이어 필드는 nullable 또는 명시적 sentinel 값(예: `255`)으로 정의하고 문서화해야 한다.

> Parquet 원본과 NPZ 캐시의 "동일 데이터 절대 중복 금지" 문구는 v0.5에서 제거되었다. NPZ는 원본에서 파생된 재생성 가능한 캐시로서 파생 표현의 중복은 허용된다.

### REQ-DATA-005 — Emotion Schema Version
**[Ubiquitous]** EmotionModule의 감정 항목 구성 및 해석(이름, 수, 순서, 값 범위, 정규화 방식)은 별도의 Emotion Schema Version으로 관리되어야 한다. Schema가 변경되면 새 Schema에 대응하는 Dataset 생성과 추론 모델 재학습이 필요하며, 기존 데이터·모델과 자동 호환되어서는 안 된다.

### REQ-DATA-006
**[Ubiquitous]** Phase 1 데이터 수집 시 각 스텝에서 해당 DummyAgent의 실제 EmotionModule 상태 벡터를 Emotion Label로 기록해야 한다. 단, Emotion Label은 관찰 불가 정보이므로 MainAgent의 Observation에는 포함하지 않는다.

### REQ-DATA-007 — 감정 레이블 시간 정렬
**[Ubiquitous]** 감정 레이블은 REQ-EMO-001의 행동 결정 순서에 따라, "행동 직전에 기록된 감정 상태"를 예측 대상 행동의 Label로 사용해야 한다. 예측 대상 행동과 그 이후에 생성된 이벤트는 해당 Label의 입력 Feature에 포함되어서는 안 된다.

### REQ-DATA-008 — 완료 게임 Shard 저장 및 메모리 해제
**[When]** 게임이 종료될 때, 시스템은 완료된 게임 데이터를 Shard로 저장하고 해당 에피소드가 점유한 불필요한 메모리를 해제해야 한다.

### REQ-DATA-009 — 종료 상태 Metadata
**[Ubiquitous]** Episode Dataset은 각 게임의 종료 상태를 `termination_status` 필드로 기록해야 하며, 값은 최소한 `normal` 또는 `truncated` 중 하나여야 한다.

### REQ-DATA-010 — Truncated 학습 Dataset 제외
**[Ubiquitous]** `termination_status=truncated`인 Episode와 그 transition은 Phase 2 지도학습 및 Phase 3 PPO 학습용 Dataset에 포함해서는 안 된다. 해당 Episode의 원본 로그와 통계는 별도로 보존해야 한다.

---

## 15. 감정 상태와 학습 Label 시간 정렬

### REQ-EMO-001 — 행동 결정 순서
**[Ubiquitous]** 감정 처리와 행동 결정은 환경 Step을 시간 기준으로 다음 순서로 수행되어야 한다:
1. 아직 반영하지 않은 공개 이벤트를 감정 상태에 반영
2. 감정 갱신 및 감쇠 적용
3. 행동 직전 감정 상태를 Label로 기록
4. 행동 선택
5. 행동 실행
6. 새로운 공개 이벤트 생성

### REQ-EMO-002 — 이벤트 1회 반영
**[Ubiquitous]** 각 공개 이벤트는 고유 이벤트 ID를 가져야 하며, 이벤트 ID를 통해 관련 생존 DummyAgent의 감정 상태에 정확히 한 번만 반영되어야 한다.

### REQ-EMO-003 — 타 플레이어 감정 Label 정렬
**[Ubiquitous]** 타 플레이어 감정 추론을 위한 데이터에도 동일한 관측 기준 시점의 타 플레이어 감정 Label을 정렬하여 기록해야 한다. 비행동 플레이어의 감정도 필요 시점에 일관되게 갱신·조회할 수 있어야 한다.

### REQ-EMO-004 — 감쇠 적용 단위
**[Ubiquitous]** 감정 감쇠는 환경 Step마다 한 번만 적용되어야 하며, 에피소드 초기 시점에는 존재하지 않는 이전 Step에 대한 감쇠를 추가로 적용해서는 안 된다.

### REQ-EMO-005 — 미래 정보 차단
**[Ubiquitous]** 예측 대상 행동과 그 이후 이벤트는 해당 감정 Label의 입력 Feature에 포함되어서는 안 된다.

### REQ-EMO-006 — Invalid Action·재시도 시 감정 불변
**[When]** Invalid Action 전달 또는 행동 재시도가 발생할 때, 시스템은 감정 갱신·감쇠를 추가로 실행해서는 안 된다. 단, 이는 행동 선택 전에 정상적으로 반영한 감정 상태를 되돌리라는 의미가 아니다.

### REQ-EMO-007 — 감정 관리 책임 계층
**[Ubiquitous]** 감정 상태 관리는 Agent/실행 계층의 책임이며, BangEnvironment는 DummyAgent의 내부 감정 상태를 제어해서는 안 된다.

### REQ-EMO-008 — 감정 추론의 의미 한정
**[Ubiquitous]** 감정 추론은 DummyAgent 감정 체계의 수치를 추정하는 모델이며, 사람이나 다른 Agent의 실제 심리 상태를 측정한다고 표현해서는 안 된다. 관련 출력은 "추정 감정"으로 표기한다.

---

## 16. 학습/검증/테스트 분리

### REQ-SPLIT-001 — 에피소드 단위 분리
**[Ubiquitous]** 데이터는 에피소드 ID 단위로 학습/검증/테스트로 분리되어야 하며, 기본 비율은 학습 80% / 검증 10% / 테스트 10%이다.

### REQ-SPLIT-002 — 동일 에피소드 동일 집합
**[Ubiquitous]** 같은 에피소드의 모든 관측·시퀀스·Label은 동일한 분할 집합에 속해야 한다.

### REQ-SPLIT-003 — 분할 재현성
**[Ubiquitous]** 데이터 분할 Seed와 분할 결과는 재현 가능하게 기록·관리되어야 한다.

### REQ-SPLIT-004 — 전처리 통계 출처 제한
**[Ubiquitous]** 전처리 통계는 학습 데이터로만 계산해야 한다.

### REQ-SPLIT-005 — 검증·테스트 정보 누출 금지
**[Ubiquitous]** 검증·테스트 집합의 정보를 학습 Feature 또는 전처리 통계 계산에 사용해서는 안 된다.

---

## 17. 하드웨어 리소스 및 성능 최적화

### REQ-HW-001 — 수동 설정 확정
**[When]** 사용자가 설정 메뉴에서 하드웨어 및 학습 자원을 설정할 때, 시스템은 CPU 코어 수, RAM 크기, GPU 사용 여부(CUDA/CPU), Worker 수, Batch Size 및 학습 Buffer 크기를 직접 입력받아 학습 환경에 적용해야 한다.

### REQ-HW-002 — 수동 자원 설정 적용
**[When]** 사용자가 하드웨어 및 학습 자원 설정을 저장할 때, 시스템은 사용자가 입력한 Worker 수, Batch Size 및 학습 Buffer 크기를 적용해야 한다. 시스템은 입력값을 자동으로 최적화하거나 특정 값으로 강제해서는 안 된다.

### REQ-HW-003 — 입력 검증 및 GUI 권장 도움말
**[Ubiquitous]** 시스템은 사용자가 입력한 CPU, RAM, GPU, Worker, Batch Size 및 Buffer 설정의 형식과 명백한 범위 오류를 검증해야 하며, 자동 Benchmark나 성능 측정을 수행하지 않고 일반적인 자원 사용 권장사항과 주의사항을 GUI 도움말로 제공해야 한다.

### REQ-HW-004
**[When]** 강화학습 학습 모드가 실행될 때, 시스템은 Headless 환경을 사용하여 인게임 게임 렌더링 연산을 제거해야 한다. 이 요구사항은 학습 진행도와 승률 그래프를 표시하는 모니터링 GUI(REQ-GUI-007, REQ-GUI-014)를 금지하지 않는다.

### REQ-HW-005
**[When]** 강화학습 학습 모드가 실행될 때, 시스템은 사용자가 설정한 Worker 수에 따라 Multiprocessing 또는 Vectorized Environment를 구성하여 다수의 게임을 병렬 실행해야 한다.

### REQ-HW-006
**[Ubiquitous]** 시스템은 사용자가 설정한 프로그램 및 하위 Worker의 자원 상한을 초과하여 자원을 사용해서는 안 된다. 시스템은 RAM, Worker, Buffer 등의 입력값을 검증하고 명백한 범위 오류를 사용자에게 표시해야 한다.

### REQ-HW-007
**[Ubiquitous]** 시스템은 학습 실행 중 실제로 처리한 Environment Step, MainAgent transition 및 경과 시간 등의 실행 통계를 기록할 수 있어야 한다. 별도의 Benchmark를 수행하여 성능을 자동 최적화해서는 안 된다.

---

## 18. 재현성 및 버전 관리

### REQ-REPR-001
**[Ubiquitous]** 시스템은 환경, Agent, Reward, Dataset, Model의 주요 버전 정보를 관리해야 한다.

### REQ-REPR-002
**[Ubiquitous]** 시스템은 Random Seed를 명시적으로 관리해야 하며, 사용자가 Seed 값을 직접 지정할 수 있어야 한다.

### REQ-REPR-003
**[When]** 동일한 Environment Version·Reward Config Version·Emotion Schema Version과 동일한 Seed 및 Configuration이 주어질 때, 시스템은 검증 범위 내에서 동일한 실험 결과를 재현할 수 있어야 한다.

### REQ-REPR-004
**[Ubiquitous]** 저장된 추론 모델은 다음 Metadata를 포함해야 한다:
- 생성 일시
- Environment Version
- Observation Schema Version
- Action Space Version
- Reward Config Version
- Emotion Schema Version
- Inference Model Artifact Format Version
- Dummy AI Preset 정보 및 버전
- 학습 상대 설정
- 총 환경 Step 수 및 총 MainAgent transition 수
- 총 학습 시간 (일시정지 시간 제외)
- 최종 승률 및 제한 종료율
- Random Seed 및 데이터 분할 Seed
- 사용한 하드웨어 설정값 (CPU 코어 수, RAM, GPU 여부, Worker 수, Batch Size, Buffer 크기) 및 실행 통계

---

## 19. 평가 프로토콜

### REQ-EVAL-001 — GUI 기본 평가 입력
**[Ubiquitous]** GUI의 기본 평가 입력은 평가 상대, 상대 프리셋, 평가 게임 수로 단순화되어야 한다.

### REQ-EVAL-002 — 고정·기록되는 평가 정책
**[Ubiquitous]** 평가 Seed 집합, 상대 프리셋 버전, 역할 및 좌석 배정 정책, 확률적/결정적 행동 선택 방식은 자동 설정하거나 고정된 평가 프로토콜로 관리하고 기록해야 한다.

### REQ-EVAL-003 — 좌석 균등 배정
**[Ubiquitous]** 평가 시 MainAgent의 좌석은 균등하게 배정되어야 한다.

### REQ-EVAL-004 — 승률 기록 및 집계 공식
**[Ubiquitous]** 평가는 전체 승률과 역할별 승률을 기록해야 한다. 전체 승률은 `정상 승리 게임 수 / 정상 승패 종료 게임 수`로 계산하며, 제한 종료 게임은 분모에서 제외한다. 역할별 승률도 해당 역할의 정상 승리 게임 수를 해당 역할의 정상 승패 종료 게임 수로 나누어 계산해야 한다. 각 집계의 분모와 정상 종료·제한 종료 건수를 결과에 명시해야 한다.

### REQ-EVAL-005 — 종료 통계 분리 집계
**[Ubiquitous]** 평가 결과는 전체 게임 수, 정상 종료 게임 수, truncated 게임 수, 정상 승리 게임 수, 정상 패배 게임 수를 별도로 기록해야 한다. Truncated Rate는 `truncated 게임 수 / 전체 게임 수`로 계산하고, Win Rate 분모에는 정상 종료 게임만 포함해야 한다.

### REQ-EVAL-006 — 정상 종료 0건 처리
**[When]** 정상 승패 종료 에피소드가 0건일 때, 시스템은 승률을 0%로 표시해서는 안 되며 "계산 불가(N/A)"로 표시해야 한다.

### REQ-EVAL-007 — 평가 조건 재사용
**[Ubiquitous]** 모델 비교 시 동일한 평가 조건을 재사용할 수 있어야 한다. 서로 다른 평가 조건에서 얻은 결과를 동일 조건의 연속 성능 변화처럼 표시해서는 안 된다.

### REQ-EVAL-008 — 미공개 역할 성능·추정 감정 표기
**[Ubiquitous]** 평가는 미공개 역할 분류 성능을 확정 역할과 분리하여 별도로 측정해야 하며, 감정 관련 출력은 "추정 감정"으로 표기해야 한다.

---

## 20. Config 변경과 Emotion Schema

### REQ-CFG-005 — 일반 게임의 비구조 변경 적용 경계
**[When]** 일반 게임 실행 중 구조를 바꾸지 않는 설정(예: 감정이 행동 확률에 미치는 가중치)이 변경될 때, 시스템은 다음 행동 선택 경계부터 변경을 적용하고 적용된 Config 버전과 적용 시점을 기록해야 한다.

### REQ-CFG-006 — 학습 작업 설정 고정
**[Ubiquitous]** 데이터 수집 및 학습 작업의 설정은 작업 시작 시점에 고정되어야 하며, 작업 중 편집된 설정은 다음 작업부터 적용되어야 한다.

### REQ-CFG-007 — Schema 변경 요건
**[When]** 감정 이름·개수·순서·값 범위·정규화 등 값 해석 또는 구조에 영향을 주는 변경이 발생할 때, 시스템은 새 Emotion Schema Version을 부여해야 하며, 새 Schema에는 대응하는 Dataset 생성과 추론 모델 재학습이 필요하다. 호환되지 않는 기존 모델을 그대로 실행해서는 안 된다.

### REQ-CFG-008 — 동작 파라미터 vs 구조 변경 구분
**[Ubiquitous]** 감정 변화량·감쇠 등 동작 파라미터 변경은 Preset 버전으로 추적하며, 이는 Emotion Schema 구조적 비호환과 구분되어야 한다.

> REQ-DUMMY-005, REQ-DUMMY-009, REQ-CFG-002, REQ-RWD-003의 "즉시 적용" 표현은 위 REQ-CFG-005~007과 일치하도록 해석한다: 일반 게임에서는 다음 행동 경계 적용, 학습 작업에서는 다음 작업 적용, 구조 변경은 Schema Version 요구.

---

## 21. 추론 모델 저장 및 호환성

### REQ-INFER-001 — 추론용 모델 저장
**[When]** 학습이 정상 완료되거나 사용자가 조기 종료할 때, 시스템은 게임 실행용 추론 모델을 저장해야 하며, 다음을 함께 보존해야 한다:
- 모델 가중치 및 구조 정보
- 필요한 전처리 정보
- 두 추론 모듈(IntentInference, EmotionInference)의 식별 정보 또는 가중치
- 관련 Schema·버전·Metadata (REQ-REPR-004)

### REQ-INFER-002 — 추론 모델 불러오기
**[Ubiquitous]** 시스템은 저장된 추론 모델을 불러와 게임 실행에 사용할 수 있어야 한다. 단, 저장 모델을 불러와 학습을 이어가는 기능(Resume)은 제공하지 않는다.

### REQ-INFER-003 — 체크포인트·Resume 미제공
**[Ubiquitous]** 시스템은 다음을 구현해서는 안 된다:
- 중간 학습 체크포인트 저장
- 프로그램 종료 후 학습 상태 복원
- 기존 모델 또는 Metadata 기반 학습 Resume
- optimizer, rollout buffer, 진행 중 환경 상태 등 학습 복원용 디스크 저장
- Resume 대상 선택 UI

### REQ-COMPAT-001 — 추론 모델 호환성 검사
**[When]** 저장된 추론 모델을 불러올 때, 시스템은 다음 항목의 호환성을 검사해야 한다:
- Model Architecture
- Environment Version
- Observation Schema Version
- Action Space Version
- Reward Config Version
- Emotion Schema Version
- Inference Model Artifact Format Version

### REQ-COMPAT-002 — 호환 불가 처리
**[When]** 불러온 추론 모델이 현재 시스템과 호환되지 않을 때, 시스템은 해당 모델을 게임 실행 대상으로 사용해서는 안 되며, 사용자에게 구체적인 호환성 오류 내용을 표시해야 한다.

---

## 22. 일시정지 및 재개

### REQ-PAUSE-001 — 안전 경계 정지
**[When]** 일시정지가 요청될 때, 시스템은 환경 행동 및 학습 연산의 안전한 경계에서 정지해야 하며, 진행 중인 단일 연산을 위험하게 강제 중단해서는 안 된다.

### REQ-PAUSE-002 — 전체 Worker 정지 확인
**[When]** 일시정지가 진행될 때, 시스템은 모든 Worker의 정지를 확인한 후 일시정지 완료를 표시해야 한다.

### REQ-PAUSE-003 — 정지 상태 불변
**[Ubiquitous]** 일시정지 완료 상태에서는 환경 Step과 PPO update가 증가해서는 안 되며, 모델 및 환경 상태는 메모리에 유지되어야 한다.

### REQ-PAUSE-004 — 재개
**[When]** 재개가 요청될 때, 시스템은 메모리에 유지된 상태에서 학습을 계속 진행해야 한다.

### REQ-PAUSE-005 — 디스크 체크포인트 미생성
**[Ubiquitous]** 시스템은 일시정지를 위해 디스크 체크포인트를 생성해서는 안 된다.

### REQ-PAUSE-006 — 일시정지 시간 제외
**[Ubiquitous]** 일시정지 시간은 목표 학습 시간 및 실제 학습 시간 측정에서 제외되어야 한다.

### REQ-PAUSE-007 — 종료 시 재개 불가 안내
**[When]** 프로그램이 종료될 때, 시스템은 메모리 상태가 사라지며 종료 후 재개가 불가능함을 사용자에게 안내해야 하며, 필요한 경우 추론용 모델 저장 여부를 명확히 처리해야 한다.

---

## 23. 결정 사항 요약

v0.3~v0.6에서 확정된 주요 결정 사항이다. 결정 내용은 각 요구사항 본문에 직접 반영되어 있다.

| ID | 결정 내용 | 반영 요구사항 |
|----|---------|------------|
| D-001 | 부관·무법자는 탈락 여부와 무관하게 팀 최종 결과를 공유 | REQ-WIN-001-A, REQ-WIN-008, REQ-PPO-009 |
| D-002 | Invalid Action: PASS 자동 변환 없음. Mask 사전 차단 + Environment 검증 + 상태 불변 + 예외 | REQ-ACT-003~010, REQ-ACT-009 |
| D-003 | 배신자 DummyAgent: 공개 정보 기반 진영 세력 비교(생존자 수 → 추정 체력 → 동률 정책). 실제 역할 미참조 | REQ-DUMMY-003-A~E |
| D-004 | 무제한 에피소드 진행을 제거하고 `max_episode_steps`(기본 2,000)에서 truncated 종료. 정상 승패 종료를 우선 | REQ-STEP-005~019, REQ-STEP-015-A |
| D-005 | 이벤트 이력 K 기본 무제한(에피소드 시작~현재), Config로 유한값 변경 | REQ-OBS-008 |
| D-006 | Parquet 원본 + NPZ 재생성 가능 캐시. 역할 분리, 파생 표현 중복 허용 | REQ-DATA-004~004-C, REQ-DATA-008 |
| D-007 | GUI: PyQt6, 실시간 그래프: pyqtgraph. Headless 학습 중 모니터링 GUI는 허용 | REQ-GUI-013, REQ-GUI-014, REQ-HW-004 |
| D-008 | 시퀀스 인코더 GRU 기본, SequenceEncoder 인터페이스로 교체 가능 | REQ-MAIN-003 |
| D-009 | 감정 Label: DummyAgent 실제 EmotionModule 수치. Emotion Schema Version 관리. 관측 순서 기반 정렬. 값은 0~1 연속값이며 기본 Loss는 MSE | REQ-MAIN-004~004-C, REQ-DATA-002, REQ-DATA-005~007, REQ-EMO-001~008 |
| D-010 | 디버그 HUD: 타 플레이어 4명 대상, 자신 제외, 감정은 추정 감정 표기 | REQ-MAIN-009, REQ-EVAL-008 |
| D-011 | 탈락 시 역할 공개. 확정 역할 우선, 미공개 역할만 추론 | REQ-OBS-009~018 |
| D-012 | 실제 비공개 역할은 판정·Label·Reward·평가·특권 로그에만 사용, 행동 입력 금지 | REQ-OBS-016~018, REQ-PPO-015~018 |
| D-013 | PPO 단위: 환경 Step / MainAgent transition / PPO update. gamma는 transition 단위 | REQ-PPO-001~018 |
| D-014 | 체크포인트·학습 Resume 제거, 추론용 모델 저장·불러오기만 유지 | REQ-INFER-001~003, REQ-GUI-008/009 |
| D-015 | 일시정지는 메모리 기반, 일시정지 시간은 학습 시간에서 제외 | REQ-PAUSE-001~007 |
| D-016 | 학습/검증/테스트 에피소드 단위 80/10/10 분리, 전처리 통계는 학습 데이터로만 | REQ-SPLIT-001~005 |
| D-017 | Truncated 정책: Normal Termination은 Terminal Reward·PPO 학습 포함·Bootstrap 없음. Truncated는 Terminal Reward·Bootstrap·모든 모델 학습을 제외하고 Log/통계만 보존 | REQ-STEP-013~015, REQ-STEP-015-A, REQ-PPO-008~014, REQ-PIPE-007, REQ-DATA-009~010, REQ-EVAL-004~006 |
| D-018 | EmotionInference 정책: Emotion 값 0~1 정규화, N차원 연속 벡터, 기본 Loss는 MSE, Label은 DummyAgent 실제 상태 | REQ-MAIN-004-A~C, REQ-DATA-002, REQ-DATA-005~007, REQ-EMO-001~008 |
| D-019 | Reward Config 정책: 모든 Reward Shaping 항목은 Config·GUI에서 변경 가능. 실제 숫자 기본값은 Requirements에서 확정하지 않으며 Reward Config Version으로 기록 | REQ-RWD-002~009, REQ-CFG-004, REQ-REPR-004, REQ-GUI-011 |
| D-020 | Hardware 정책: Benchmark 및 자동 Worker/Batch 최적화는 구현하지 않으며, CPU/RAM/GPU·Worker·Batch·Buffer는 사용자 직접 설정, 입력 검증 및 GUI 권장 도움말만 제공 | REQ-GUI-012, REQ-HW-001~007 |

---

## 24. 미결 사항

현재 Requirements 범위에서 별도로 남겨둔 미결 사항은 없다. 다음 단계에서는 이 문서의 확정 요구사항을 기반으로 Design을 작성한다.

---

## 25. 인수 기준

각 인수 기준(AC)은 관련 요구사항 ID와 연결된다. 모든 AC는 자동화 테스트로 검증 가능해야 한다.

| AC ID | 검증 시나리오 | 기대 결과 | 관련 요구사항 |
|-------|-------------|----------|------------|
| AC-01 | 모든 Agent가 매 턴 PASS | 2,000 환경 Step에서 `truncated=True`로 제한 종료 | REQ-STEP-005, REQ-STEP-006, REQ-STEP-007 |
| AC-02 | 마지막 허용 Step에 승패 확정 + 최대 Step 동시 도달 | `terminated=True`(정상 종료)가 우선 | REQ-STEP-008 |
| AC-03 | 제한 종료된 에피소드 로그 재구성 | Environment 이벤트와 실행·로깅 계층의 설정 식별자를 결합하여 초기 상태부터 종료까지 전체 진행 재구성 가능 | REQ-STEP-015, REQ-STEP-015-A |
| AC-04 | Truncated 시 보상·학습 확인 | Terminal Reward 없음, Bootstrap 없음, PPO 학습 대상 transition 없음, Episode Log·통계는 보존 | REQ-STEP-011, REQ-STEP-013~015, REQ-PPO-010~011, REQ-DATA-009~010 |
| AC-05 | 평가 통계 표시 | 전체 게임 수·정상 종료 수·Truncated 수·정상 승리/패배 수 기록. Win Rate는 정상 승리/정상 종료, Truncated Rate는 Truncated/전체로 계산하며 정상 종료 0건 시 N/A | REQ-EVAL-004~006 |
| AC-06 | 플레이어 탈락 | 실제 역할 공개, 이후 Observation·이벤트 이력에 포함 | REQ-OBS-009 |
| AC-07 | Agent 입력 검사 | 비공개 실제 역할·실제 Emotion Label·Reward 내역이 행동 입력에 없음 | REQ-OBS-017, REQ-PPO-015~018 |
| AC-08 | 공개/확정 역할 사용 | 확정 역할은 확정값(우선)으로, 미공개만 추론 | REQ-OBS-012, REQ-OBS-014 |
| AC-09 | DummyAgent 진영 계산 | 실제 비공개 역할 없이 진영별 생존자 수·추정 체력 계산 | REQ-DUMMY-003-A, REQ-DUMMY-003-C |
| AC-10 | 정상 행동 경로 | 모든 Agent 유형에서 Action Mask 적용 | REQ-ACT-008 |
| AC-11 | Invalid Action 강제 주입 | 상태·차례·환경 Step·감정 불변 + 오류 로그 + 예외 발생 | REQ-ACT-009, REQ-STEP-002 |
| AC-12 | AEC 정리 호출 | 게임 Step·학습 transition으로 기록되지 않고 가짜 PASS transition 미생성 | REQ-STEP-003, REQ-PPO-007, REQ-PPO-012 |
| AC-13 | 정상 승패 종료 후 탈락한 부관·무법자 | 게임 종료 후 각 팀의 최종 보상(+1/-1)이 생존 여부와 무관하게 반영 | REQ-WIN-008, REQ-PPO-009 |
| AC-14 | 감정 Label 구성 | 이벤트 ID 기반 1회 반영, 감쇠 Step당 1회, 해당 행동·미래 이벤트가 Label 입력에 미포함 | REQ-EMO-002, REQ-EMO-004~006 |
| AC-15 | 데이터 분리 | 동일 에피소드가 학습·검증·테스트에 중복되지 않음 | REQ-SPLIT-001, REQ-SPLIT-002 |
| AC-16 | 일시정지 완료 | 환경 Step·PPO update 증가하지 않음 | REQ-PAUSE-003 |
| AC-17 | 재개 | 동일 메모리 상태에서 계속 진행 | REQ-PAUSE-004 |
| AC-18 | 모델 저장/불러오기 | 추론용 모델 저장·불러오기 동작, 체크포인트·학습 Resume 미제공 | REQ-INFER-001, REQ-INFER-002, REQ-INFER-003 |
| AC-19 | 학습 중 Config | 학습 작업 설정 고정, Emotion Schema 비호환 모델 실행 차단 | REQ-CFG-006, REQ-CFG-007, REQ-COMPAT-002 |
| AC-20 | 정상 종료 PPO 계약 | Terminal Reward 기록, MainAgent transition을 PPO 학습에 포함, 종료 후 Bootstrap 없음 | REQ-RWD-001, REQ-PPO-009, REQ-PPO-013 |
| AC-21 | Truncated PPO 계약 | Terminal Reward 없음, Bootstrap 없음, 해당 Episode와 transition이 Phase 2·Phase 3 학습에서 제외 | REQ-PPO-010~011, REQ-PIPE-007, REQ-DATA-010 |
| AC-22 | Dataset 종료 상태 | Episode에 `termination_status=normal|truncated` 기록하고 Truncated 원본 Log·통계는 보존 | REQ-DATA-009~010 |
| AC-23 | EmotionInference 출력·Loss | 감정값 0~1, N차원 연속 벡터, DummyAgent 실제 상태 Label, 기본 Loss MSE | REQ-MAIN-004-A~C, REQ-DATA-002, REQ-DATA-006~007 |
| AC-24 | Emotion Schema 호환성 | 감정 이름·개수·순서·범위·정규화 변경 시 새 Schema·Dataset·재학습 요구 및 기존 모델 호환 차단 | REQ-DATA-005, REQ-CFG-007, REQ-COMPAT-001~002 |
| AC-25 | Reward Config | 모든 Reward Shaping 항목을 Config·GUI에서 수정 가능하고 실제 사용 Version이 기록되며 승패 규칙은 변경되지 않음 | REQ-RWD-004~009, REQ-CFG-004, REQ-GUI-011, REQ-REPR-004 |
