"""Lightweight internationalization for the GUI (English / Korean).

Translation lives here rather than in the validated ``SystemConfig`` so that
choosing a UI language never touches training/reward configuration or an active
job snapshot.  The chosen language is remembered in a small JSON file in the
user's home directory.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable


LANGUAGES: tuple[str, ...] = ("en", "ko")
DEFAULT_LANGUAGE = "en"

# Human-readable names for the language picker, keyed by language code.
LANGUAGE_LABELS: dict[str, str] = {"en": "English", "ko": "한국어"}

_SETTINGS_FILE = Path.home() / ".bang_rl_gui.json"

# key -> {language_code: text}
_STRINGS: dict[str, dict[str, str]] = {
    # Main menu / window
    "app.title": {"en": "Bang RL", "ko": "Bang RL"},
    "menu.training": {"en": "AI Training", "ko": "AI 학습"},
    "menu.settings": {"en": "Settings", "ko": "설정"},
    "menu.exit": {"en": "Exit", "ko": "종료"},
    "common.back": {"en": "← Back to menu", "ko": "← 메뉴로 돌아가기"},

    # Game mode selector
    "mode.group": {"en": "Game mode", "ko": "게임 모드"},
    "mode.human_random": {"en": "Human 1 vs Random AI 4", "ko": "사람 1 vs 랜덤 AI 4"},
    "mode.human_dummy": {"en": "Human 1 vs Dummy AI 4", "ko": "사람 1 vs 더미 AI 4"},
    "mode.human_main_dummy": {"en": "Human 1 vs Main AI 1 + Dummy AI 3", "ko": "사람 1 vs 메인 AI 1 + 더미 AI 3"},
    # Item 6: all-AI spectator modes.
    "mode.random_five": {"en": "Random AI x5", "ko": "랜덤 AI 5명"},
    "mode.dummy_five": {"en": "Dummy AI x5", "ko": "더미 AI 5명"},
    "mode.main_one_dummy_four": {"en": "Main AI 1 vs Dummy AI 4", "ko": "메인 AI 1 vs 더미 AI 4"},

    # Game screen
    "game.title": {"en": "Bang RL — Game", "ko": "Bang RL — 게임"},
    "game.public_state": {"en": "Public game state", "ko": "공개 게임 상태"},
    "game.opponents": {"en": "Opponents", "ko": "상대"},
    "game.log": {"en": "Game log", "ko": "게임 로그"},
    "game.you_are": {"en": "You are Player {player} · Role: {role}", "ko": "당신은 플레이어 {player} · 역할: {role}"},
    "game.you_suffix": {"en": " (you)", "ko": " (나)"},
    "game.hp": {"en": "HP", "ko": "체력"},
    "game.alive": {"en": "alive", "ko": "생존"},
    "game.eliminated": {"en": "eliminated", "ko": "탈락"},
    "game.no_session": {"en": "No game session", "ko": "진행 중인 게임이 없습니다"},
    "game.loading": {"en": "Loading...", "ko": "로딩 중..."},
    "game.error_generic": {
        "en": "The operation could not be completed.",
        "ko": "작업을 완료할 수 없습니다.",
    },
    "game.status_line": {
        "en": "Step {step} · turn Player {player} · {status}",
        "ko": "스텝 {step} · 플레이어 {player} 차례 · {status}",
    },
    "game.action.pass": {"en": "PASS", "ko": "패스"},
    "game.action.bang": {"en": "BANG Player {player}", "ko": "뱅 플레이어 {player}"},
    "game.pause": {"en": "Pause", "ko": "일시정지"},
    "game.resume": {"en": "Resume", "ko": "재개"},
    "game.ai_speed": {"en": "AI turn speed", "ko": "AI 턴 속도"},
    "game.ai_speed_value": {
        "en": "AI turn speed: {seconds:.1f}s per turn",
        "ko": "AI 턴 속도: 턴당 {seconds:.1f}초",
    },
    # Item 3 (FEAT-002): manual turn-advance mode. The speed slider at its
    # minimum means "manual": auto-advance stops and the user presses Next turn.
    "game.next_turn": {"en": "Next turn", "ko": "다음 턴"},
    # Item 3 (FEAT-003): spectator playback controls mirror the replay screen
    # (prev turn / pause-play / next turn), so the game screen gets its own
    # prev-turn and play labels. game.pause / game.next_turn already exist.
    "game.prev_turn": {"en": "Previous turn", "ko": "이전 턴"},
    "game.play": {"en": "Play", "ko": "재생"},
    "game.ai_speed_manual": {
        "en": "AI turn speed: Manual (press Next turn)",
        "ko": "AI 턴 속도: 수동 (다음 턴 버튼)",
    },
    "game.speed_manual_label": {"en": "Manual", "ko": "수동"},
    "game.you_name": {"en": "You", "ko": "나"},
    "game.player_name": {"en": "Player {player}", "ko": "플레이어 {player}"},
    "game.click_hint": {
        "en": "Your turn: click an opponent to BANG, click yourself to PASS.",
        "ko": "당신 차례: 상대를 눌러 뱅, 자신을 눌러 패스하세요.",
    },
    "game.hp_label": {"en": "HP {health}", "ko": "체력 {health}"},
    "game.paused_msg": {"en": "Game paused.", "ko": "게임이 일시정지되었습니다."},
    "game.finished_normally_faction": {
        "en": "Game finished normally. {faction} wins.",
        "ko": "게임이 정상적으로 종료되었습니다. {faction} 승리.",
    },
    "game.finished_normally": {
        "en": "Game finished normally.",
        "ko": "게임이 정상적으로 종료되었습니다.",
    },
    "game.finished_truncated": {
        "en": "Game stopped at the episode limit.",
        "ko": "게임이 에피소드 한도에서 중단되었습니다.",
    },
    "game.winner_banner": {
        "en": "{faction} wins!",
        "ko": "{faction} 승리!",
    },
    "game.no_winner_banner": {
        "en": "No winner (time limit)",
        "ko": "승자 없음 (시간 초과)",
    },
    "game.time_travel_hint": {
        "en": "Reviewing step {step} (read-only). Leave the log to return to the live game.",
        "ko": "스텝 {step} 시점을 보는 중입니다 (읽기 전용). 로그에서 벗어나면 진행 중인 게임으로 돌아갑니다.",
    },
    "game.time_travel_hint_terminal": {
        "en": "Reviewing step {step} of the finished game (read-only).",
        "ko": "종료된 게임의 스텝 {step} 시점을 보는 중입니다 (읽기 전용).",
    },
    "game.spectator_no_actions": {
        "en": "Spectator mode: you are watching the AI play and cannot take actions.",
        "ko": "관전 모드: AI의 플레이를 지켜보는 중이며 조작할 수 없습니다.",
    },
    # Item 6: spectator hint shown in place of the click hint.
    "game.spectating": {
        "en": "Spectating - AI players only",
        "ko": "관전 중 - AI만 플레이합니다",
    },
    # Item 7: buttons shown under the winner banner when a game is over.
    "game.restart": {"en": "Restart", "ko": "재시작"},
    "game.return_menu": {"en": "Return to menu", "ko": "메뉴로 돌아가기"},

    # Winning factions (game-over message)
    "faction.sheriff_team": {"en": "Sheriff team", "ko": "보안관 팀"},
    "faction.outlaw_team": {"en": "Outlaw team", "ko": "무법자 팀"},
    "faction.traitor_team": {"en": "Traitor", "ko": "배신자"},

    # HUD table headers
    "hud.player": {"en": "Player", "ko": "플레이어"},
    "hud.hp": {"en": "HP", "ko": "체력"},
    "hud.status": {"en": "Status", "ko": "상태"},
    "hud.role": {"en": "Role info", "ko": "역할 정보"},

    # Roles
    "role.sheriff": {"en": "Sheriff", "ko": "보안관"},
    "role.deputy": {"en": "Deputy", "ko": "부관"},
    "role.outlaw_a": {"en": "Outlaw A", "ko": "무법자 A"},
    "role.outlaw_b": {"en": "Outlaw B", "ko": "무법자 B"},
    "role.traitor": {"en": "Traitor", "ko": "배신자"},
    "role.hidden": {"en": "Hidden", "ko": "숨김"},
    # Merged outlaw label for the spectator / replay role-inference list. The
    # schema splits outlaw_a / outlaw_b, but the inference merges the two slots
    # into a single bucket, so the hover list shows one "무법자" / "Outlaw" entry.
    "role.outlaw_merged": {"en": "Outlaw", "ko": "무법자"},

    # Event log labels
    "event.bang": {"en": "BANG", "ko": "뱅"},
    "event.pass": {"en": "PASS", "ko": "패스"},
    "event.player_eliminated": {"en": "Player eliminated", "ko": "플레이어 탈락"},
    "event.role_revealed": {"en": "Role revealed", "ko": "역할 공개"},
    "event.game_started": {"en": "Game started", "ko": "게임 시작"},
    "event.game_ended": {"en": "Game ended", "ko": "게임 종료"},
    "event.turn_started": {"en": "Turn started", "ko": "차례 시작"},
    "event.by": {"en": "by Player {player}", "ko": "플레이어 {player}가"},
    "event.target": {"en": "→ Player {player}", "ko": "→ 플레이어 {player}"},
    "event.eliminated": {"en": "(Player {player} eliminated)", "ko": "(플레이어 {player} 탈락)"},
    "event.role": {"en": "[role: {role}]", "ko": "[역할: {role}]"},
    "event.step": {"en": "Turn {step} ", "ko": "{step}턴 "},

    # Bang! hit / elimination status shown during the shot animation.
    "board.hit": {
        "en": "Player {player} was hit!",
        "ko": "플레이어 {player}가 명중당했습니다!",
    },
    "board.eliminated": {
        "en": "Player {player} was eliminated!",
        "ko": "플레이어 {player}가 탈락했습니다!",
    },
    # Spectator / replay role-inference hover: fallback shown when a hovered
    # seat is not a dummy AI or its belief row is unavailable.
    "board.no_inference": {
        "en": "No inference info",
        "ko": "추론 정보 없음",
    },

    # Settings screen
    "settings.language": {"en": "Language", "ko": "언어"},
    "settings.ppo_clip": {"en": "PPO clip epsilon", "ko": "PPO 클립 엡실론"},
    "settings.emotion_baseline": {"en": "Emotion initial baseline", "ko": "감정 초기 기준값"},
    "settings.cpu": {"en": "CPU cores", "ko": "CPU 코어"},
    "settings.ram": {"en": "RAM (GB)", "ko": "RAM (GB)"},
    "settings.workers": {"en": "Workers", "ko": "워커 수"},
    "settings.batch": {"en": "Batch size", "ko": "배치 크기"},
    "settings.buffer": {"en": "Buffer size", "ko": "버퍼 크기"},
    "settings.gpu": {"en": "GPU", "ko": "GPU"},
    "settings.gpu_check": {"en": "Use GPU/CUDA", "ko": "GPU/CUDA 사용"},
    "settings.reward_json": {"en": "Reward Config (JSON)", "ko": "보상 설정 (JSON)"},
    "settings.save": {"en": "Validate and save", "ko": "검증 후 저장"},
    "settings.saved": {"en": "Settings saved for the next training job.", "ko": "다음 학습 작업을 위해 설정이 저장되었습니다."},
    "settings.language_saved": {
        "en": "Language saved. Reopen a screen to see all changes.",
        "ko": "언어가 저장되었습니다. 화면을 다시 열면 모든 변경이 적용됩니다.",
    },
    "settings.resource_help": {
        "en": (
            "Set resources manually. Leaving 1–2 CPU cores available usually keeps the "
            "operating system and GUI responsive. No benchmark, auto-tuning, or OOM search "
            "is performed."
        ),
        "ko": (
            "자원을 직접 설정하세요. CPU 코어 1~2개를 남겨두면 보통 운영체제와 GUI가 원활하게 "
            "동작합니다. 벤치마크나 자동 튜닝, OOM 탐색은 수행하지 않습니다."
        ),
    },

    # Dummy AI emotion configuration (Settings screen + dialog)
    "settings.emotion_initial_values": {
        "en": "Emotion initial values", "ko": "감정 초기 기준값",
    },
    "settings.emotion_initial_help": {
        "en": "Set the starting value (0-1) for each of the dummy AI's emotions.",
        "ko": "더미 AI의 각 감정에 대한 초기값(0~1)을 설정하세요.",
    },
    "settings.dummy_ai": {"en": "Dummy AI settings", "ko": "더미 AI 설정"},
    "settings.dummy_ai_title": {
        "en": "Dummy AI emotion configuration", "ko": "더미 AI 감정 설정",
    },
    "settings.dummy_ai_help": {
        "en": (
            "Configure how many emotions the dummy AI has, their names, how they "
            "increase or decrease, and their effects when high or low."
        ),
        "ko": (
            "더미 AI가 갖는 감정의 개수, 이름, 증감 조건, 감정이 높거나 낮을 때의 영향을 "
            "설정하세요."
        ),
    },
    "settings.emotion_count": {"en": "Number of emotions", "ko": "감정 개수"},
    "settings.emotion_schema_version": {"en": "Schema version", "ko": "스키마 버전"},
    "settings.emotion_initial_baseline": {
        "en": "Initial baseline", "ko": "초기 기준값",
    },
    "settings.emotion_name": {"en": "Name", "ko": "이름"},
    "settings.emotion_initial": {"en": "Initial value", "ko": "감정 초기값"},
    "settings.emotion_change_rate": {"en": "Change rate", "ko": "증가율"},
    "settings.emotion_decay_rate": {"en": "Decay rate", "ko": "감소율"},
    "settings.emotion_event_updates": {
        "en": "Event conditions (JSON)", "ko": "이벤트 증감 조건 (JSON)",
    },
    "settings.emotion_action_updates": {
        "en": "Action conditions (JSON)", "ko": "행동 증감 조건 (JSON)",
    },
    "settings.emotion_reward_updates": {
        "en": "Reward conditions (JSON)", "ko": "보상 증감 조건 (JSON)",
    },
    "settings.emotion_terminal_updates": {
        "en": "Terminal conditions (JSON)", "ko": "종료 증감 조건 (JSON)",
    },
    "settings.emotion_effects": {
        "en": "Effects when high/low (JSON)", "ko": "감정 효과 높음/낮음 (JSON)",
    },
    "settings.emotion_number": {"en": "Emotion {index}", "ko": "감정 {index}"},
    "settings.dummy_ai_apply": {"en": "Apply", "ko": "적용"},
    "settings.dummy_ai_cancel": {"en": "Cancel", "ko": "취소"},
    "settings.dummy_ai_applied": {
        "en": "Dummy AI emotions updated.", "ko": "더미 AI 감정이 업데이트되었습니다.",
    },

    # Dummy AI emotion save/load to a standalone external file
    "settings.emotion_save_file": {
        "en": "Save emotions to file", "ko": "감정 설정 파일로 저장",
    },
    "settings.emotion_load_file": {
        "en": "Load emotions from file", "ko": "감정 설정 파일 불러오기",
    },
    "settings.emotion_file_filter": {
        "en": "Emotion config (*.yaml *.yml *.json)",
        "ko": "감정 설정 파일 (*.yaml *.yml *.json)",
    },
    "settings.emotion_file_saved": {
        "en": "Dummy AI emotions saved to {path}.",
        "ko": "더미 AI 감정을 {path}에 저장했습니다.",
    },
    "settings.emotion_file_loaded": {
        "en": "Dummy AI emotions loaded from {path}.",
        "ko": "더미 AI 감정을 {path}에서 불러왔습니다.",
    },
    "tooltip.emotion_save_file": {
        "en": "Save the complete DummyAgent emotion configuration, including schema, names, initial values, update rules, and effects, to a portable file for review or reuse.",
        "ko": "스키마·이름·초기값·갱신 규칙·효과를 포함한 DummyAgent 전체 감정 설정을 검토하거나 재사용할 수 있는 파일로 저장합니다.",
    },
    "tooltip.emotion_load_file": {
        "en": "Load and validate an emotion configuration created outside the GUI. The file must match the supported schema and its values are applied to future DummyAgent sessions.",
        "ko": "GUI 밖에서 만든 감정 설정을 읽고 검증합니다. 지원하는 스키마와 일치해야 하며 불러온 값은 이후 DummyAgent 세션에 적용됩니다.",
    },

    # Training monitor
    "training.mode": {"en": "Training goal", "ko": "학습 목적"},
    "training.mode_data_collection": {"en": "Collect training data (Dummy Five)", "ko": "학습 데이터 수집 (Dummy 5명)"},
    "training.mode_inference": {"en": "Train inference models", "ko": "추론 모델 학습"},
    "training.mode_policy": {"en": "Train action policy (PPO)", "ko": "행동 정책 학습 (PPO)"},
    "training.episodes": {"en": "Episodes", "ko": "에피소드"},
    "training.target": {"en": "Target MainAgent transitions", "ko": "목표 MainAgent 전이 수"},
    "training.update_mode": {"en": "PPO update mode", "ko": "PPO 갱신 모드"},
    "training.update_online": {"en": "Online (one transition)", "ko": "온라인 (전이 1개마다)"},
    "training.update_batch": {"en": "Batch (rollout batch)", "ko": "배치 (롤아웃 배치)"},
    "training.update_semibatch": {"en": "Semibatch (custom size)", "ko": "세미배치 (사용자 지정 크기)"},
    "training.ppo_batch_size": {"en": "Semibatch size", "ko": "세미배치 크기"},
    "training.seed": {"en": "Seed", "ko": "시드"},
    "training.report_path": {"en": "Report path", "ko": "리포트 경로"},
    "training.output_path": {"en": "Output path", "ko": "출력 경로"},
    "training.artifact_path": {"en": "Artifact output path", "ko": "아티팩트 출력 경로"},
    "training.resume_artifact_path": {"en": "Continue from artifact", "ko": "이어 학습할 아티팩트"},
    "training.replay_source_path": {"en": "Full replay source path", "ko": "Full 리플레이 경로"},
    "training.phase2_artifact_path": {"en": "Phase-2 artifact for PPO", "ko": "PPO용 Phase-2 아티팩트"},
    "training.sl_epochs": {"en": "SL epochs", "ko": "SL 학습 epoch 수"},
    "training.start": {"en": "Start", "ko": "시작"},
    "training.pause": {"en": "Pause", "ko": "일시정지"},
    "training.resume": {"en": "Resume", "ko": "재개"},
    "training.stop": {"en": "Stop", "ko": "정지"},
    "training.no_job": {"en": "No training job", "ko": "진행 중인 학습 작업이 없습니다"},
    "training.snapshot_not_started": {"en": "Snapshot: not started", "ko": "스냅샷: 시작 안 됨"},
    "training.graph_needs_pyqtgraph": {
        "en": "Win-rate graph requires optional pyqtgraph.",
        "ko": "승률 그래프에는 pyqtgraph(선택 설치)가 필요합니다.",
    },

    # Settings field help tooltips (shown on mouse-over of each field/label)
    "tooltip.language": {
        "en": "Language used by labels, buttons, tooltips, and messages throughout the application. Changing it affects newly rendered GUI text without changing stored training data.",
        "ko": "애플리케이션 전체의 라벨, 버튼, 도움말, 메시지에 사용할 언어입니다. 언어를 바꿔도 저장된 학습 데이터의 내용은 변경되지 않습니다.",
    },
    "tooltip.ppo_clip": {
        "en": "PPO clipping epsilon limits how far the policy distribution may move during one update. Lower values are more conservative; higher values allow larger changes but can destabilize training.",
        "ko": "PPO 클리핑 엡실론은 한 번의 정책 갱신에서 정책 분포가 변할 수 있는 최대 폭을 제한합니다. 낮을수록 보수적이고, 높을수록 큰 변화가 가능하지만 학습이 불안정해질 수 있습니다.",
    },
    "tooltip.cpu": {
        "en": "Number of CPU cores reserved for training workers and preprocessing. More cores can increase throughput, but the configured worker count and workload still determine actual usage.",
        "ko": "학습 워커와 전처리에 사용할 CPU 코어 수입니다. 코어를 늘리면 처리량이 늘 수 있지만 실제 사용량은 워커 수와 작업량에 따라 달라집니다.",
    },
    "tooltip.ram": {
        "en": "Amount of memory, in gigabytes, reserved as the training resource budget. This is a user-declared limit and is recorded in the job snapshot; it is not an automatic benchmark result.",
        "ko": "학습 자원 예산으로 지정하는 메모리 용량(GB)입니다. 작업 스냅샷에 기록되는 사용자가 선언한 제한이며 자동 벤치마크 결과가 아닙니다.",
    },
    "tooltip.workers": {
        "en": "Number of independent workers that execute game episodes in parallel. More workers can collect data faster, but they also consume more CPU/RAM and produce separate deterministic seeds.",
        "ko": "게임 에피소드를 병렬로 실행할 독립 워커 수입니다. 워커를 늘리면 수집 속도가 빨라질 수 있지만 CPU·RAM 사용량이 늘고 워커별 시드가 따로 생성됩니다.",
    },
    "tooltip.batch": {
        "en": "Number of samples used inside one optimizer update. In PPO this is related to rollout/minibatch behavior; it is different from the number of complete episodes collected.",
        "ko": "한 optimizer 갱신 내부에서 사용할 샘플 수입니다. PPO의 롤아웃·미니배치 동작과 관련되며, 수집할 완전한 에피소드 수와는 다른 값입니다.",
    },
    "tooltip.buffer": {
        "en": "Maximum number of transitions retained in the in-memory experience buffer before PPO consumes or discards them. Larger buffers use more memory and do not by themselves guarantee more frequent updates.",
        "ko": "PPO가 소비하기 전 메모리에 보관할 전이의 최대 수입니다. 버퍼가 크면 메모리를 더 사용하며, 버퍼 크기만 바꾼다고 갱신 빈도가 자동으로 바뀌지는 않습니다.",
    },
    "tooltip.gpu": {
        "en": "Whether CUDA/GPU is used when the selected Python environment supports it. GPU availability does not change the model architecture, but it can affect training speed and memory use.",
        "ko": "현재 Python 환경에서 CUDA/GPU를 사용할지 정합니다. GPU 사용 가능 여부는 모델 구조를 바꾸지 않지만 학습 속도와 메모리 사용량에 영향을 줄 수 있습니다.",
    },
    "tooltip.reward_json": {
        "en": "JSON reward-shaping configuration. It changes diagnostic/shaping signals while the environment's terminal win conditions remain defined by the game rules.",
        "ko": "JSON 형식의 보상 shaping 설정입니다. 진단·shaping 보상은 바꾸지만 환경의 승패 종료 조건은 게임 규칙에 의해 별도로 결정됩니다.",
    },
    "tooltip.emotion_initial_values": {
        "en": "Initial normalized value between 0 and 1 for each DummyAgent emotion. This is the starting target state; it is separate from later event-driven updates and decay.",
        "ko": "각 DummyAgent 감정의 시작값(0~1)입니다. 초기 상태를 정하며 이후 이벤트에 따른 변화와 감쇠 설정과는 별개입니다.",
    },
    "tooltip.emotion_initial_baseline_field": {
        "en": "Shared baseline used when DummyAgent emotions are initialized. It affects the starting point of every configured emotion and must remain within the validated range.",
        "ko": "DummyAgent 감정을 초기화할 때 모든 감정에 공통으로 적용하는 기준값입니다. 설정된 모든 감정의 시작점에 영향을 주며 유효 범위를 지켜야 합니다.",
    },
    "tooltip.dummy_ai": {
        "en": "Open the dummy AI emotion configuration dialog.",
        "ko": "더미 AI 감정 설정 창을 여는 버튼입니다.",
    },

    # AI-training settings menu (TrainingSetupPanel) field/label tooltips
    "tooltip.training_mode": {
        "en": "Choose what this training job is intended to produce. Data collection runs Dummy Five games and saves full replay targets; inference training learns role and emotion estimators from those replays; policy training runs PPO for a MainAgent against four DummyAgents.",
        "ko": "이번 학습 작업의 목적을 선택합니다. 학습 데이터 수집은 Dummy 5명 게임과 full replay 정답을 만들고, 추론 모델 학습은 역할·감정 모델을 학습하며, 행동 정책 학습은 MainAgent 1명과 Dummy 4명으로 PPO를 수행합니다.",
    },
    "tooltip.training_episodes": {
        "en": "Number of complete game episodes to run during data collection or policy training. In inference-model training this field is disabled because the replay source determines the data size.",
        "ko": "학습 데이터 수집 또는 행동 정책 학습에서 실행할 완전한 게임 수입니다. 추론 모델 학습에서는 리플레이 경로가 데이터 양을 정하므로 이 항목이 비활성화됩니다.",
    },
    "tooltip.training_target": {
        "en": "Minimum number of PPO-ready MainAgent transitions. The run can overshoot this value because transitions are collected from complete episodes; this field is disabled for data collection and supervised inference training.",
        "ko": "PPO에 사용할 MainAgent 전이의 최소 목표 수입니다. 완전한 에피소드 단위로 수집하므로 목표를 조금 초과할 수 있으며, 데이터 수집과 추론 모델 학습에서는 비활성화됩니다.",
    },
    "tooltip.training_update_mode": {
        "en": "Controls when PPO applies an optimizer update. Online updates after one eligible transition, batch waits for the configured rollout capacity, and semibatch updates after the user-selected semibatch size. It affects policy training only.",
        "ko": "PPO optimizer 갱신 시점을 정합니다. 온라인은 유효 전이 1개마다, 배치는 설정된 롤아웃 용량마다, 세미배치는 사용자가 입력한 크기마다 갱신합니다. 행동 정책 학습에서만 사용됩니다.",
    },
    "tooltip.training_ppo_batch_size": {
        "en": "Number of PPO-ready transitions consumed by one semibatch update. This input is enabled only for semibatch policy training; smaller values react faster but usually produce noisier updates.",
        "ko": "세미배치 PPO 갱신 한 번에 사용할 전이 수입니다. 세미배치 행동 정책 학습에서만 활성화되며, 작은 값은 빠르게 반응하지만 갱신 변동이 커질 수 있습니다.",
    },
    "tooltip.training_seed": {
        "en": "Random seed used to make the training run reproducible.",
        "ko": "학습을 재현 가능하게 만드는 난수 시드를 의미합니다.",
    },
    "tooltip.training_report_path": {
        "en": "Folder where the training report is written.",
        "ko": "학습 리포트가 기록되는 폴더를 의미합니다.",
    },
    "tooltip.training_output_path": {
        "en": "Folder where training output files are written.",
        "ko": "학습 출력 파일이 기록되는 폴더를 의미합니다.",
    },
    "tooltip.training_artifact_path": {
        "en": "Folder where the trained model artifact is saved.",
        "ko": "학습된 모델 아티팩트가 저장되는 폴더를 의미합니다.",
    },
    "tooltip.training_resume_artifact_path": {
        "en": "Existing artifact folder to continue training from.",
        "ko": "이어서 학습할 기존 아티팩트 폴더를 의미합니다.",
    },
    "tooltip.training_replay_source_path": {
        "en": "A full replay JSON file or folder used for role and emotion supervised learning.",
        "ko": "역할·감정 지도학습에 사용할 full 리플레이 JSON 파일 또는 폴더입니다.",
    },
    "tooltip.training_phase2_artifact_path": {
        "en": "Phase-2 role/emotion artifact used to build PPO inputs and save a final game model.",
        "ko": "PPO 입력과 최종 게임 모델 저장에 사용할 Phase-2 역할·감정 아티팩트입니다.",
    },
    "tooltip.training_sl_epochs": {
        "en": "Number of complete passes over the training cache for both role and emotion inference models. More epochs create more validation points, but can overfit small replay sets.",
        "ko": "역할·감정 추론 모델이 학습 cache를 반복해서 학습할 횟수입니다. epoch를 늘리면 validation 그래프 점이 늘지만 작은 리플레이 데이터에서는 과적합될 수 있습니다.",
    },

    # Dummy AI emotion dialog field/label tooltips
    "tooltip.emotion_schema_version": {
        "en": "Schema version for the emotion vector: names, order, count, and value interpretation. Changing the schema makes old emotion labels incompatible with new training data.",
        "ko": "감정 벡터의 이름·순서·개수·값 해석을 정의하는 스키마 버전입니다. 스키마를 바꾸면 기존 감정 레이블은 새 학습 데이터와 호환되지 않습니다.",
    },
    "tooltip.emotion_count": {
        "en": "Number of emotion dimensions each DummyAgent stores and the inference model predicts. Changing it changes label tensor shape and requires a new compatible dataset.",
        "ko": "각 DummyAgent가 저장하고 추론 모델이 예측할 감정 차원 수입니다. 값을 바꾸면 레이블 tensor 모양이 바뀌므로 호환되는 새 데이터셋이 필요합니다.",
    },
    "tooltip.emotion_name": {
        "en": "Unique display and label name for this emotion. Keep the spelling stable across replay collection and model training so columns and metrics refer to the same signal.",
        "ko": "이 감정의 고유 이름입니다. 리플레이 수집과 모델 학습에서 같은 신호로 인식되도록 표기를 바꾸지 않는 것이 좋습니다.",
    },
    "tooltip.emotion_change_rate": {
        "en": "Amount added to this emotion when its configured trigger is observed. Values are applied according to the module's normalization and clamping rules.",
        "ko": "설정한 조건이 발생했을 때 이 감정에 더할 변화량입니다. 감정 모듈의 정규화와 범위 제한 규칙에 따라 적용됩니다.",
    },
    "tooltip.emotion_decay_rate": {
        "en": "Per-step decay toward the configured baseline. A larger value makes the emotion return to baseline faster after events stop affecting it.",
        "ko": "매 턴 기준값으로 돌아가는 감쇠량입니다. 값이 클수록 이벤트가 사라진 뒤 감정이 기준값으로 더 빠르게 돌아갑니다.",
    },
    "tooltip.emotion_event_updates": {
        "en": "JSON map of game events to how much they change this emotion. Relationship keys such as bang_at_me / bang_by_me are also supported and take precedence over the plain event key.",
        "ko": "게임 이벤트별로 이 감정을 얼마나 변화시키는지 나타내는 JSON 맵을 의미합니다. bang_at_me / bang_by_me 같은 관계 기반 키도 지원되며, 일반 이벤트 키보다 우선 적용됩니다.",
    },
    "tooltip.emotion_action_updates": {
        "en": "JSON mapping from action/event relationship names to emotion changes. Use this for behavior-specific effects such as bang_at_me or bang_by_me; malformed JSON is rejected before saving.",
        "ko": "행동·이벤트 관계 이름을 감정 변화량에 연결하는 JSON 맵입니다. bang_at_me, bang_by_me 같은 행동별 효과를 설정하며 잘못된 JSON은 저장 전에 거부됩니다.",
    },
    "tooltip.emotion_reward_updates": {
        "en": "JSON mapping from reward signal names to emotion changes. Keep these values separate from the PPO reward itself; they describe DummyAgent internal emotion updates.",
        "ko": "보상 신호 이름을 감정 변화량에 연결하는 JSON 맵입니다. PPO 보상 자체와 분리된 DummyAgent 내부 감정 갱신 규칙입니다.",
    },
    "tooltip.emotion_terminal_updates": {
        "en": "JSON mapping for terminal outcomes such as win, loss, or elimination. These updates are applied only at the episode boundary and are recorded with the emotion schema.",
        "ko": "승리·패배·탈락 같은 종료 결과에 대한 JSON 맵입니다. 에피소드 종료 경계에서만 적용되며 감정 스키마와 함께 기록됩니다.",
    },
    "tooltip.emotion_effects": {
        "en": "JSON map describing how this emotion affects DummyAgent behavior or derived signals when its value is high or low. It is configuration data, not a model target.",
        "ko": "감정값이 높거나 낮을 때 DummyAgent 행동과 파생 신호에 미치는 영향을 설명하는 JSON 맵입니다. 모델 정답이 아니라 동작 설정 데이터입니다.",
    },

    # Replay viewer (reuses the graphical board + log + speed slider)
    "menu.replay": {"en": "Replay", "ko": "리플레이"},
    "replay.title": {"en": "Bang RL - Replay", "ko": "Bang RL - 리플레이"},
    "replay.play": {"en": "Play", "ko": "재생"},
    "replay.pause": {"en": "Pause", "ko": "일시정지"},
    "replay.next": {"en": "Next turn →", "ko": "다음 턴 →"},
    "replay.prev": {"en": "← Previous turn", "ko": "← 이전 턴"},
    "replay.autoplay": {"en": "Auto-play", "ko": "자동 재생"},
    "replay.speed": {
        "en": "Playback speed: {seconds:.1f}s per turn",
        "ko": "재생 속도: 턴당 {seconds:.1f}초",
    },
    # Item 3 (FEAT-002): manual mode indication for the replay speed slider at
    # its minimum. The user advances with the existing Next turn / Prev buttons.
    "replay.speed_manual": {
        "en": "Playback: Manual (press Next turn)",
        "ko": "재생: 수동 (다음 턴 버튼)",
    },
    "replay.speed_manual_label": {"en": "Manual", "ko": "수동"},
    "replay.status": {"en": "Turn {index}/{total}", "ko": "턴 {index}/{total}"},
    "replay.no_episode": {
        "en": "No replay loaded. Open a recorded episode to watch it.",
        "ko": "불러온 리플레이가 없습니다. 기록된 에피소드를 열어 감상하세요.",
    },
    "replay.open_file": {"en": "Open replay file", "ko": "리플레이 파일 열기"},
    "replay.browser": {"en": "Saved replays", "ko": "저장된 리플레이"},
    "replay.repath": {"en": "Re-point replay folder", "ko": "리플레이 경로 재지정"},
    "replay.folder_title": {"en": "Choose replay folder", "ko": "리플레이 폴더 선택"},
    "replay.file_filter": {
        "en": "Recorded episode (*.json *.jsonl)",
        "ko": "기록된 에피소드 (*.json *.jsonl)",
    },
    # Item 10: rename / delete a saved replay from the browser.
    "replay.rename": {"en": "Rename", "ko": "이름 바꾸기"},
    "replay.delete": {"en": "Delete", "ko": "삭제"},
    "replay.rename_prompt": {
        "en": "Enter a new name for this replay:",
        "ko": "이 리플레이의 새 이름을 입력하세요:",
    },
    "replay.rename_title": {"en": "Rename replay", "ko": "리플레이 이름 바꾸기"},
    "replay.delete_confirm": {
        "en": "Delete this replay permanently? This cannot be undone.",
        "ko": "이 리플레이를 영구적으로 삭제하시겠습니까? 되돌릴 수 없습니다.",
    },
    "replay.delete_title": {"en": "Delete replay", "ko": "리플레이 삭제"},
    "replay.select_first": {
        "en": "Select a replay in the list first.",
        "ko": "먼저 목록에서 리플레이를 선택하세요.",
    },
    "replay.delete_failed": {
        "en": "Could not delete the replay file.",
        "ko": "리플레이 파일을 삭제하지 못했습니다.",
    },
    # Item 5: browser labels for the two replay formats.
    "replay.format_public": {"en": "Public", "ko": "공개"},
    "replay.format_full": {"en": "Full state", "ko": "전체 상태"},

    # Item 6: read-only info panel shown while a replay plays back. Surfaces the
    # recorded mode / outcome / seed and the per-seat AI + version breakdown.
    "replay.info_title": {"en": "Replay info", "ko": "리플레이 정보"},
    "replay.info_mode": {"en": "Mode: {mode}", "ko": "모드: {mode}"},
    "replay.info_format": {"en": "Format: {format}", "ko": "포맷: {format}"},
    "replay.info_outcome": {"en": "Outcome: {outcome}", "ko": "결과: {outcome}"},
    "replay.info_seed": {"en": "Seed: {seed}", "ko": "시드: {seed}"},
    "replay.info_seat": {"en": "Seat {seat}: {ai}", "ko": "좌석 {seat}: {ai}"},
    "replay.ai_random": {"en": "Random AI", "ko": "랜덤 AI"},
    "replay.ai_dummy": {"en": "Dummy AI", "ko": "더미 AI"},
    "replay.ai_main": {"en": "Main AI {version}", "ko": "메인 AI {version}"},
    "replay.ai_unknown": {"en": "Unknown AI", "ko": "알 수 없는 AI"},

    # Model selection dialog
    "model.title": {"en": "Select inference model", "ko": "추론 모델 선택"},
    "model.select_prompt": {"en": "Select a compatible model.", "ko": "호환되는 모델을 선택하세요."},
    "model.compatible": {"en": "Compatible", "ko": "호환됨"},
    "model.not_compatible": {"en": "Not compatible", "ko": "호환 안 됨"},
    "model.incompatible_msg": {
        "en": "This model is incompatible and cannot be executed.",
        "ko": "이 모델은 호환되지 않아 실행할 수 없습니다.",
    },
    "model.no_metadata": {"en": "No metadata available.", "ko": "사용 가능한 메타데이터가 없습니다."},
}


class _Translator:
    """Holds the current language and notifies observers when it changes."""

    def __init__(self) -> None:
        self._language = self._load()
        self._observers: list[Callable[[str], None]] = []

    @property
    def language(self) -> str:
        return self._language

    def set_language(self, language: str) -> None:
        if language not in LANGUAGES:
            raise ValueError(f"unsupported language: {language!r}")
        if language == self._language:
            return
        self._language = language
        self._save(language)
        for observer in tuple(self._observers):
            observer(language)

    def subscribe(self, observer: Callable[[str], None]) -> Callable[[], None]:
        self._observers.append(observer)

        def unsubscribe() -> None:
            if observer in self._observers:
                self._observers.remove(observer)

        return unsubscribe

    def translate(self, key: str, /, **kwargs: object) -> str:
        entry = _STRINGS.get(key)
        if entry is None:
            return key
        text = entry.get(self._language) or entry.get(DEFAULT_LANGUAGE) or key
        if kwargs:
            try:
                return text.format(**kwargs)
            except (KeyError, IndexError):
                return text
        return text

    @staticmethod
    def _load() -> str:
        try:
            data = json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
            language = str(data.get("language", DEFAULT_LANGUAGE))
            return language if language in LANGUAGES else DEFAULT_LANGUAGE
        except (OSError, ValueError):
            return DEFAULT_LANGUAGE

    @staticmethod
    def _save(language: str) -> None:
        try:
            _SETTINGS_FILE.write_text(json.dumps({"language": language}), encoding="utf-8")
        except OSError:
            # Persisting the preference is best-effort; the in-memory choice
            # still applies for this session.
            pass


# Process-wide translator instance shared by every GUI screen.
translator = _Translator()


def tr(key: str, /, **kwargs: object) -> str:
    """Translate ``key`` into the current language, formatting ``kwargs``."""

    return translator.translate(key, **kwargs)


def set_language(language: str) -> None:
    translator.set_language(language)


def current_language() -> str:
    return translator.language


__all__ = [
    "DEFAULT_LANGUAGE",
    "LANGUAGES",
    "LANGUAGE_LABELS",
    "current_language",
    "set_language",
    "tr",
    "translator",
]
