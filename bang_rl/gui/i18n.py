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


LANGUAGE_LABELS: dict[str, str] = {"en": "English", "ko": "한국어"}

_SETTINGS_FILE = Path.home() / ".bang_rl_gui.json"


_STRINGS: dict[str, dict[str, str]] = {

    "app.title": {"en": "Bang RL", "ko": "Bang RL"},
    "menu.training": {"en": "AI Training", "ko": "AI 학습"},
    "menu.settings": {"en": "Settings", "ko": "설정"},
    "menu.exit": {"en": "Exit", "ko": "종료"},
    "common.back": {"en": "← Back to menu", "ko": "← 메뉴로 돌아가기"},


    'menu.intent_training': {'en': 'Intent Module Training', 'ko': '의도 모듈 학습'},
    'intent.title': {'en': 'Intent Module Training', 'ko': '의도 모듈 학습'},
    'intent.subtitle': {'en': 'Train each role independently (100 per role = 400 games). BANG/PASS only; emotions are excluded.', 'ko': '역할별로 독립 학습합니다 (역할별 100회 = 총 400게임). BANG/PASS만 사용하며 감정 모듈은 제외합니다.'},
    'intent.games': {'en': 'Training games per role', 'ko': '역할별 학습 게임 수'},
    'intent.steps': {'en': 'Maximum environment steps per game', 'ko': '게임당 최대 환경 Step'},
    'intent.seed': {'en': 'Random seed', 'ko': '난수 시드'},
    'intent.random': {'en': 'Random AI opponents (%)', 'ko': '상대방 Random AI 비율 (%)'},
    'intent.batch': {'en': 'Completed games per policy update', 'ko': '정책 업데이트당 종료 게임 수'},
    'intent.eval': {'en': 'Evaluation games per role', 'ko': '역할별 사전/사후 평가 게임 수'},
    'intent.output': {'en': 'Output model file', 'ko': '모델 저장 경로'},
    'intent.browse_output': {'en': 'Save as...', 'ko': '저장 위치...'},
    'intent.resume': {'en': 'Start from existing model (optional)', 'ko': '기존 모델에서 추가 학습 (선택)'},
    'intent.resume_hint': {'en': 'New model: leave blank. To continue training, select a previous .intentmodel. Original will be preserved. Optimizer state is reset.', 'ko': '새 학습: 비워두세요. 추가 학습: 기존 .intentmodel을 선택하세요. 원본은 보존되며 옵티마이저 상태는 초기화됩니다.'},
    'intent.browse_resume': {'en': 'Choose model...', 'ko': '기존 모델 선택...'},
    'intent.clear_resume': {'en': 'New training', 'ko': '새로 학습'},
    'intent.start': {'en': 'Start training', 'ko': '학습 시작'},
    'intent.cancel': {'en': 'Stop training', 'ko': '학습 중단'},
    'intent.load': {'en': 'Load model for analysis / continuation', 'ko': '분석·추가 학습용 모델 불러오기'},
    'intent.explain': {'en': 'Explain example decision', 'ko': '학습 결과 예시 분석'},
    'intent.before': {'en': 'Ready', 'ko': '시작 전'},
    'intent.baseline': {'en': 'Evaluating existing rule-based intent (without emotion)...', 'ko': '기존 규칙형 의도 모델 평가 중 (감정 제외)...'},
    'intent.training': {'en': 'Training {done}/{total} games | {roles}', 'ko': '학습 {done}/{total}게임 | {roles}'},
    'intent.checkpoint': {'en': 'Checkpoint saved: {path}', 'ko': '중간 모델 저장: {path}'},
    'intent.eval_result': {'en': 'Evaluation [{role}]: win rate {rate} / completed {normal} / truncated {truncated}', 'ko': '평가 [{role}]: 승률 {rate} / 정상 {normal} / 중단 {truncated}'},
    'intent.saved_model': {'en': 'Model saved: {path}', 'ko': '모델 저장: {path}'},
    'intent.saved_report': {'en': 'Report saved: {path}', 'ko': '보고서 저장: {path}'},
    'intent.comparison': {'en': 'Rule AI vs learned AI (win rate, normally completed games)', 'ko': '규칙형 AI와 학습형 AI 비교 (정상 종료 게임 승률)'},
    'intent.initial_compare': {'en': 'Loaded model before/after additional training', 'ko': '불러온 모델의 추가 학습 전후 비교'},
    'intent.compare_role': {'en': '{role}: before {before} → after {after}', 'ko': '{role}: 이전 {before} → 이후 {after}'},
    'intent.allfive': {'en': 'All-five-agent evaluation (team wins)', 'ko': '5인 전원 적용 평가 (팀별 승률)'},
    'intent.allfive_rule': {'en': 'Rule AI x5', 'ko': '기존 규칙 AI 5명'},
    'intent.allfive_learned': {'en': 'Learned AI x5', 'ko': '학습 의도 AI 5명'},
    'intent.allfive_detail': {'en': '{label}: {rates} (completed {normal}, truncated {truncated})', 'ko': '{label}: {rates} (정상 {normal}, 중단 {truncated})'},
    'intent.cumulative': {'en': 'Cumulative training: {games} games per role', 'ko': '누적 학습: 역할별 {games}게임'},
    'intent.complete': {'en': 'Training finished. Model and report saved.', 'ko': '학습 완료. 모델과 보고서가 저장됐습니다.'},
    'intent.cancel_request': {'en': 'Stopping after the current game...', 'ko': '현재 게임 종료 후 학습 중단 처리 중...'},
    'intent.cancelled': {'en': 'Stopped. Existing checkpoints remain.', 'ko': '학습 중단. 기존 체크포인트는 남아 있습니다.'},
    'intent.error': {'en': 'Training error: {error}', 'ko': '학습 오류: {error}'},
    'intent.config_error': {'en': 'Invalid training configuration', 'ko': '학습 설정 오류'},
    'intent.load_title': {'en': 'Choose trained intent model', 'ko': '학습된 의도 모델 선택'},
    'intent.loaded': {'en': 'Model loaded for analysis and additional training: {path}', 'ko': '분석·추가 학습용 모델 불러옴: {path}'},
    'intent.load_failed': {'en': 'Cannot load intent model', 'ko': '모델 불러오기 실패'},
    'intent.analysis_failed': {'en': 'Cannot analyze intent model', 'ko': '의도 모델 분석 실패'},
    'intent.analysis_title': {'en': 'Example: sheriff opening state', 'ko': '분석 예시: 보안관 초기 상태'},
    'intent.analysis_action': {'en': 'Chosen action: {action}', 'ko': '선택 행동: {action}'},
    'intent.analysis_probabilities': {'en': 'Action probabilities', 'ko': '행동 확률'},
    'intent.analysis_contributions': {'en': 'Local gradient contributions (approximation)', 'ko': '국소적 기여도 (근사)'},
    'intent.analysis_note': {'en': 'This is a local neural approximation, not a fixed decision formula.', 'ko': '고정된 의도식이 아니라 신경망의 해당 상태에 대한 근사적 설명입니다.'},
    'intent.file_filter': {'en': 'Learned Intent Model (*.intentmodel)', 'ko': '학습 의도 모델 (*.intentmodel)'},
    'intent.preserve_error': {'en': 'Choose an output file different from the input model.', 'ko': '기존 모델을 보존하도록 저장 경로를 다르게 선택해주세요.'},
    'settings.intent_group': {'en': 'Dummy AI intent model selection', 'ko': '더미 AI 의도 모듈 선택'},
    'settings.intent_note': {'en': 'Choose a default and optionally override each player seat (0–4). Only Dummy AI seats are affected; learned intent currently runs without emotion.', 'ko': '전체 기본값을 정하고 플레이어 좌석(0~4)을 개별 변경할 수 있습니다. 더미 AI에만 적용되며 학습형 의도 모듈은 현재 감정 모듈 없이 실행됩니다.'},
    'settings.intent_default': {'en': 'Default for all Dummy AI', 'ko': '전체 더미 AI 기본값'},
    'settings.intent_rule': {'en': 'Original rule intent', 'ko': '기존 규칙형 의도 모듈'},
    'settings.intent_learned': {'en': 'Learned intent model', 'ko': '학습된 의도 모듈'},
    'settings.intent_follow': {'en': 'Follow default', 'ko': '전체 기본값 따르기'},
    'settings.intent_seat': {'en': 'Player {seat} (seat {index})', 'ko': '플레이어 {seat} (좌석 {index})'},
    'settings.intent_path': {'en': 'Learned model file', 'ko': '학습 모델 파일'},
    'settings.intent_browse': {'en': 'Browse...', 'ko': '찾아보기...'},
    'settings.intent_saved': {'en': 'Dummy AI intent selection saved.', 'ko': '더미 AI 의도 모듈 설정을 저장했습니다.'},
    'settings.intent_save_failed': {'en': 'Unable to save intent settings: {error}', 'ko': '의도 모듈 설정 저장 실패: {error}'},


    "mode.group": {"en": "Game mode", "ko": "게임 모드"},
    "mode.human_random": {"en": "Human 1 vs Random AI 4", "ko": "사람 1 vs 랜덤 AI 4"},
    "mode.human_dummy": {"en": "Human 1 vs Dummy AI 4", "ko": "사람 1 vs 더미 AI 4"},
    "mode.human_main_dummy": {"en": "Human 1 vs Main AI 1 + Dummy AI 3", "ko": "사람 1 vs 메인 AI 1 + 더미 AI 3"},

    "mode.random_five": {"en": "Random AI x5", "ko": "랜덤 AI 5명"},
    "mode.dummy_five": {"en": "Dummy AI x5", "ko": "더미 AI 5명"},
    "mode.main_one_dummy_four": {"en": "Main AI 1 vs Dummy AI 4", "ko": "메인 AI 1 vs 더미 AI 4"},


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


    "game.next_turn": {"en": "Next turn", "ko": "다음 턴"},



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

    "game.spectating": {
        "en": "Spectating - AI players only",
        "ko": "관전 중 - AI만 플레이합니다",
    },

    "game.restart": {"en": "Restart", "ko": "재시작"},
    "game.return_menu": {"en": "Return to menu", "ko": "메뉴로 돌아가기"},


    "faction.sheriff_team": {"en": "Sheriff team", "ko": "보안관 팀"},
    "faction.outlaw_team": {"en": "Outlaw team", "ko": "무법자 팀"},
    "faction.traitor_team": {"en": "Traitor", "ko": "배신자"},


    "hud.player": {"en": "Player", "ko": "플레이어"},
    "hud.hp": {"en": "HP", "ko": "체력"},
    "hud.status": {"en": "Status", "ko": "상태"},
    "hud.role": {"en": "Role info", "ko": "역할 정보"},


    "role.sheriff": {"en": "Sheriff", "ko": "보안관"},
    "role.deputy": {"en": "Deputy", "ko": "부관"},
    "role.outlaw_a": {"en": "Outlaw A", "ko": "무법자 A"},
    "role.outlaw_b": {"en": "Outlaw B", "ko": "무법자 B"},
    "role.traitor": {"en": "Traitor", "ko": "배신자"},
    "role.hidden": {"en": "Hidden", "ko": "숨김"},



    "role.outlaw_merged": {"en": "Outlaw", "ko": "무법자"},


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


    "board.hit": {
        "en": "Player {player} was hit!",
        "ko": "플레이어 {player}가 명중당했습니다!",
    },
    "board.eliminated": {
        "en": "Player {player} was eliminated!",
        "ko": "플레이어 {player}가 탈락했습니다!",
    },


    "board.no_inference": {
        "en": "No inference info",
        "ko": "추론 정보 없음",
    },


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
    "settings.saved": {"en": "Settings saved locally for the next training job.", "ko": "다음 학습 작업을 위한 로컬 설정이 저장되었습니다."},
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
        "en": "Save the dummy AI's emotion configuration to a standalone file you can edit by hand.",
        "ko": "더미 AI의 감정 설정을 직접 편집할 수 있는 별도의 파일로 저장합니다.",
    },
    "tooltip.emotion_load_file": {
        "en": "Load a dummy AI emotion configuration file that was saved or edited outside the GUI.",
        "ko": "GUI 밖에서 저장하거나 편집한 더미 AI 감정 설정 파일을 불러옵니다.",
    },


    "training.mode": {"en": "Mode", "ko": "모드"},
    "training.mode.data_collection": {"en": "Collect training data", "ko": "학습 데이터 수집"},
    "training.mode.module_training": {"en": "Train intent and emotion modules", "ko": "의도·감정 추론 모듈 학습"},
    "training.mode.main_ai_training": {"en": "Train Main AI (coming soon)", "ko": "메인 AI 학습 (준비 중)"},
    "training.episodes": {"en": "Episodes", "ko": "에피소드"},
    "training.epochs": {"en": "Epochs", "ko": "에폭"},
    "training.epochs_blank": {"en": "Blank (limit by time)", "ko": "비움 (시간으로 제한)"},
    "training.graph_interval": {"en": "Graph point every N batches", "ko": "그래프 점 간격 (batch)"},
    "training.max_minutes": {"en": "Max training time (min)", "ko": "최대 학습 시간(분)"},
    "training.max_minutes_placeholder": {"en": "blank = unlimited", "ko": "비우면 무제한"},
    "training.hardware_group": {"en": "Hardware & AI training", "ko": "하드웨어 · AI 학습 설정"},
    "training.target": {"en": "Target MainAgent transitions", "ko": "목표 MainAgent 전이 수"},
    "training.seed": {"en": "Seed", "ko": "시드"},
    "training.report_path": {"en": "Report path", "ko": "리포트 경로"},
    "training.output_path": {"en": "Output path", "ko": "출력 경로"},
    "training.artifact_path": {"en": "Artifact output path", "ko": "아티팩트 출력 경로"},
    "training.resume_artifact_path": {"en": "Continue Main AI from (.bangmodel)", "ko": "이어 학습할 메인 AI (.bangmodel)"},
    "training.resume_module": {"en": "Continue modules from", "ko": "이어 학습할 추론 모듈"},
    "training.resume_module_none": {"en": "None (train from scratch)", "ko": "없음 (새로 학습)"},

    "perf.open": {"en": "Saved model performance", "ko": "저장된 모델 성능 보기"},
    "perf.title": {"en": "Saved inference module performance", "ko": "저장된 추론 모듈 성능"},
    "perf.empty": {"en": "No saved intent/emotion modules yet. Run module training first.", "ko": "저장된 의도·감정 모듈이 없습니다. 먼저 모듈 학습을 실행하세요."},
    "perf.refresh": {"en": "Refresh", "ko": "새로고침"},
    "perf.close": {"en": "Close", "ko": "닫기"},
    "perf.use_for_resume": {"en": "Continue training from this model", "ko": "이 모델로 이어 학습"},
    "perf.resume_selected": {"en": "Module training will continue from {job}.", "ko": "모듈 학습을 {job}에서 이어서 진행합니다."},
    "perf.parts_both": {"en": "intent + emotion", "ko": "의도+감정"},
    "perf.parts_intent": {"en": "intent only", "ko": "의도만"},
    "perf.parts_emotion": {"en": "emotion only", "ko": "감정만"},
    "perf.job": {"en": "Job: {job}", "ko": "작업: {job}"},
    "perf.saved_at": {"en": "Saved: {time}", "ko": "저장 시각: {time}"},
    "perf.report": {"en": "Report: {path}", "ko": "리포트: {path}"},
    "perf.intent_section": {"en": "■ Intent (role) inference", "ko": "■ 의도(역할) 추론"},
    "perf.emotion_section": {"en": "■ Emotion inference", "ko": "■ 감정 추론"},
    "perf.not_saved": {"en": "Not saved in this run.", "ko": "이 작업에서는 저장되지 않았습니다."},
    "perf.error": {"en": "Cannot read this model: {error}", "ko": "모델을 읽을 수 없습니다: {error}"},
    "perf.epochs": {"en": "Epochs: {run} in this run, {total} in total", "ko": "학습 에폭: 이번 {run} / 누적 {total}"},
    "perf.resumed_from": {"en": "Continued from: {name}", "ko": "이어 학습 원본: {name}"},
    "perf.best_epoch": {"en": "Saved weights: best validation epoch {epoch}", "ko": "저장된 가중치: 검증 최고 에폭 {epoch}"},
    "perf.best_is_start": {"en": "Saved weights: the starting model (further training did not improve validation)", "ko": "저장된 가중치: 시작 모델 (추가 학습이 검증 성능을 넘지 못함)"},
    "perf.test_intent": {"en": "Test accuracy {accuracy} (loss {loss})", "ko": "테스트 정확도 {accuracy} (loss {loss})"},
    "perf.test_emotion": {"en": "Test accuracy {accuracy} (within ±{tolerance}), MSE {mse}", "ko": "테스트 정확도 {accuracy} (허용 오차 ±{tolerance}), MSE {mse}"},
    "perf.validation": {"en": "Validation accuracy {accuracy}", "ko": "검증 정확도 {accuracy}"},
    "perf.split": {"en": "Episodes: train {train} / validation {validation} / test {test}", "ko": "에피소드: 학습 {train} / 검증 {validation} / 테스트 {test}"},
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


    "tooltip.language": {
        "en": "Choose the language used by buttons, labels, menus, and help messages in this window. Changing it does not change the game rules or a running job.",
        "ko": "이 창의 버튼, 라벨, 메뉴, 도움말에 사용할 언어를 선택합니다. 언어를 바꿔도 게임 규칙이나 진행 중인 학습 작업은 바뀌지 않습니다.",
    },
    "tooltip.ppo_clip": {
        "en": "PPO clip epsilon limits how much the AI policy may change in one update. Smaller values make learning more cautious; larger values allow bigger changes but can make training unstable.",
        "ko": "PPO 클립 엡실론은 한 번의 학습에서 AI 행동 정책을 얼마나 바꿀 수 있는지 제한합니다. 작은 값은 조심스럽게 학습하고, 큰 값은 빠르게 바뀌지만 학습이 불안정해질 수 있습니다.",
    },
    "tooltip.cpu": {
        "en": "The maximum number of CPU cores that training may use. More cores can run more work at once, but leaving some cores free helps Windows and the GUI remain responsive.",
        "ko": "학습에 사용할 수 있는 CPU 코어의 최대 개수입니다. 코어를 많이 사용하면 작업을 동시에 처리할 수 있지만, 일부 코어를 남겨 두어야 Windows와 GUI가 원활하게 동작합니다.",
    },
    "tooltip.ram": {
        "en": "The memory budget for training, measured in gigabytes. Larger datasets and larger batches use more memory. If the value is too small, a training workflow may fail before it starts.",
        "ko": "학습에 사용할 메모리 한도(GB)입니다. 데이터셋과 배치가 클수록 메모리를 더 사용하므로, 너무 작게 설정하면 학습이 시작되기 전에 실패할 수 있습니다.",
    },
    "tooltip.settings_save": {
        "en": "Save the current settings to the local Git-ignored file artifacts/local-settings/settings.json. Pulling another version will not overwrite this machine's saved settings.",
        "ko": "현재 설정을 Git에서 제외된 로컬 파일 artifacts/local-settings/settings.json에 저장합니다. 다른 버전을 pull해도 이 컴퓨터의 저장 설정은 덮어쓰지 않습니다.",
    },
    "tooltip.workers": {
        "en": "The number of separate workers that run game environments. One worker is simplest. More workers can collect experience faster, but they use more CPU and memory.",
        "ko": "게임 환경을 별도로 실행할 작업자 수입니다. 1개가 가장 단순하고 안정적입니다. 작업자를 늘리면 경험 데이터를 더 빨리 모을 수 있지만 CPU와 메모리를 더 사용합니다.",
    },
    "tooltip.batch": {
        "en": "A batch is a group of examples processed together before the model is updated. Larger batches are usually steadier but need more memory; smaller batches use less memory but can make learning noisier.",
        "ko": "배치는 모델을 한 번 갱신하기 전에 함께 처리하는 학습 예시 묶음입니다. 큰 배치는 학습이 안정적일 수 있지만 메모리를 더 사용하고, 작은 배치는 메모리는 적게 쓰지만 결과가 흔들릴 수 있습니다.",
    },
    "tooltip.buffer": {
        "en": "The maximum number of game transitions kept for training. A transition is one action and its result. A larger buffer gives the learner more experience but consumes more memory.",
        "ko": "학습을 위해 보관하는 게임 전이의 최대 개수입니다. 전이는 한 번의 행동과 그 결과를 뜻합니다. 버퍼가 크면 더 많은 경험을 사용할 수 있지만 메모리를 더 사용합니다.",
    },
    "tooltip.gpu": {
        "en": "Use a CUDA-compatible graphics card when available. A GPU can speed up neural-network training, but it is optional and may require additional drivers and memory.",
        "ko": "가능하면 CUDA를 지원하는 그래픽카드를 학습에 사용합니다. GPU는 신경망 학습을 빠르게 할 수 있지만 선택 사항이며 별도의 드라이버와 메모리가 필요할 수 있습니다.",
    },
    "tooltip.reward_json": {
        "en": "A JSON description of rewards. Rewards are numeric feedback: positive values encourage an action or result, and negative values discourage it. Invalid JSON prevents the settings from being saved.",
        "ko": "보상 규칙을 JSON으로 입력합니다. 보상은 행동이나 결과에 대한 숫자 피드백입니다. 양수는 해당 행동이나 결과를 권장하고 음수는 억제합니다. JSON 형식이 잘못되면 설정을 저장할 수 없습니다.",
    },
    "tooltip.emotion_initial_values": {
        "en": "The starting strength of this emotion, from 0 to 1. Zero means it starts inactive; one means it starts fully active. In the Formula model, the named bars are derived from confidence, aggression, and anxiety.",
        "ko": "이 감정의 시작 세기(0~1)입니다. 0은 처음에 거의 작동하지 않음을, 1은 처음부터 강하게 작동함을 뜻합니다. Formula 모델에서는 표시되는 감정 막대가 자신감·공격성·불안 값으로 계산됩니다.",
    },
    "tooltip.emotion_initial_baseline_field": {
        "en": "The quiet resting level used when an emotion starts or recovers. After an event changes an emotion, the value gradually moves back toward this baseline.",
        "ko": "감정이 시작하거나 회복할 때 기준이 되는 안정 상태의 값입니다. 사건으로 감정이 변한 뒤 시간이 지나면 이 기준값을 향해 돌아옵니다.",
    },
    "tooltip.dummy_ai": {
        "en": "Open the Dummy AI settings. These settings describe how the non-learning rule-based players react to public events, actions, rewards, and the current Formula emotion state.",
        "ko": "더미 AI 설정을 엽니다. 이 설정에서는 학습되지 않은 규칙 기반 플레이어가 공개 사건, 행동, 보상, 현재 Formula 감정 상태에 어떻게 반응하는지 정합니다.",
    },


    "tooltip.training_mode": {
        "en": "Choose what this screen should do. Collect training data plays Dummy AI games and saves examples. Module training learns the Intent and Emotion inference modules from saved data. Main AI training is a reserved placeholder for the later PPO phase.",
        "ko": "이 화면에서 수행할 작업을 선택합니다. 학습 데이터 수집은 더미 AI 게임을 실행해 예시를 저장합니다. 추론 모듈 학습은 저장된 데이터로 의도·감정 모듈을 학습합니다. 메인 AI 학습은 나중에 PPO를 연결하기 위한 준비 모드입니다.",
    },
    "tooltip.training_episodes": {
        "en": "How many complete games to run. One episode means one game from its initial setup until a team wins or the step limit is reached. More episodes provide more examples but take longer.",
        "ko": "실행할 게임 수입니다. 에피소드 하나는 게임을 초기화한 뒤 승패가 결정되거나 제한 시간에 도달할 때까지 진행한 한 판을 뜻합니다. 많이 실행할수록 예시는 늘지만 시간이 더 걸립니다.",
    },
    "tooltip.training_epochs": {
        "en": "How many times module training passes over the whole dataset; training stops after exactly this many epochs. Set it to 0 (shown as Blank) to limit training by time instead, using the Max training time field. Only module training uses this value.",
        "ko": "모듈 학습이 전체 데이터셋을 몇 번 반복할지입니다. 정확히 이 에폭 수만큼 학습하고 멈춥니다. 0(비움)으로 두면 에폭 대신 '최대 학습 시간' 값으로 학습을 제한합니다. 모듈 학습에서만 사용됩니다.",
    },
    "tooltip.training_graph_interval": {
        "en": "Store one accuracy point on the graph every N training batches (the first batch is always shown). Larger values keep the graph and the GUI fast on long runs; 1 plots every batch.",
        "ko": "학습 batch N개마다 정확도 점을 하나씩만 그래프에 저장합니다(첫 batch는 항상 표시). 값이 클수록 오래 학습할 때 그래프와 GUI가 빨라지고, 1이면 모든 batch를 찍습니다.",
    },
    "tooltip.training_max_minutes": {
        "en": "Time limit in minutes, used only when Epochs is Blank. Leave this blank (or 0) to train until you press Stop. Intent and emotion modules train in turn, one epoch each, until the time is up; then the best models are saved. Pausing does not use up the time, and Stop takes effect within the current batch.",
        "ko": "에폭이 '비움'일 때만 쓰이는 시간 제한(분)입니다. 이 칸도 비우거나 0이면 정지 버튼을 누를 때까지 학습합니다. 의도·감정 모듈이 에폭 하나씩 번갈아 학습하다가 시간이 되면 멈추고 최적 모델을 저장합니다. 일시정지한 시간은 제한 시간에 포함되지 않으며, 정지는 현재 batch 안에서 반영됩니다.",
    },
    "tooltip.training_target": {
        "en": "A legacy Main AI transition target used by reinforcement-learning workflows. For data collection, use Episodes instead; this field does not determine how many Parquet games are collected.",
        "ko": "강화학습 workflow에서 사용하는 기존 Main AI 전이 목표입니다. 학습 데이터 수집에서는 에피소드 수를 사용하며, 이 값은 수집할 Parquet 게임 수를 결정하지 않습니다.",
    },
    "tooltip.training_seed": {
        "en": "The starting random number seed. Using the same seed, configuration, and episode count makes the generated games repeatable. Change it to create a different dataset sample.",
        "ko": "난수 생성의 시작값입니다. 같은 시드·설정·에피소드 수를 사용하면 생성되는 게임을 다시 재현할 수 있습니다. 다른 데이터 표본을 만들려면 시드를 바꾸세요.",
    },
    "tooltip.training_report_path": {
        "en": "Where the final JSON report is saved. The report contains the job status, counters, configuration snapshot, errors, and workflow metrics. It is written when the job finishes.",
        "ko": "최종 JSON 리포트를 저장할 위치입니다. 리포트에는 작업 상태, 카운터, 설정 스냅샷, 오류, workflow 지표가 들어가며 작업이 끝날 때 저장됩니다.",
    },
    "tooltip.training_output_path": {
        "en": "A general folder for workflow output. Dataset files and model files use their dedicated dataset/model folders; this path is used for the job's general output metadata.",
        "ko": "workflow의 일반 출력 폴더입니다. 데이터셋과 모델은 전용 데이터셋·모델 폴더를 사용하며, 이 경로에는 작업의 일반 출력 정보가 저장됩니다.",
    },
    "tooltip.training_artifact_path": {
        "en": "The destination for a final Main AI artifact in workflows that create one. Data collection and Phase-2 module training do not create a final .bangmodel file.",
        "ko": "최종 Main AI 아티팩트를 만드는 workflow에서 사용할 저장 위치입니다. 데이터 수집과 Phase-2 추론 모듈 학습에서는 최종 .bangmodel 파일을 만들지 않습니다.",
    },
    "tooltip.training_resume_module": {
        "en": "Continue module training from intent/emotion modules saved by an earlier run instead of starting from random weights. The weights are reused; the optimizer starts fresh. The starting model is evaluated first and kept unless further training improves validation loss. A model whose emotion schema differs from the current data cannot be continued.",
        "ko": "새 가중치 대신 이전 작업에서 저장한 의도·감정 모듈을 이어서 학습합니다. 가중치는 이어받고 optimizer는 새로 시작합니다. 시작 모델을 먼저 검증해 두고, 추가 학습이 검증 loss를 개선하지 못하면 시작 모델이 그대로 저장됩니다. 감정 스키마가 현재 데이터와 다른 모델은 이어 학습할 수 없습니다.",
    },
    "tooltip.perf_open": {
        "en": "Browse saved intent/emotion modules and see their final test and validation performance.",
        "ko": "저장된 의도·감정 모듈 목록과 최종 테스트·검증 성능을 확인합니다.",
    },
    "tooltip.perf_use_for_resume": {
        "en": "Close this window and select the chosen run in 'Continue modules from'.",
        "ko": "창을 닫고 선택한 작업을 '이어 학습할 추론 모듈'에 지정합니다.",
    },
    "tooltip.training_resume_artifact_path": {
        "en": "An existing final .bangmodel file or folder from which a future Main AI training run can continue. It is not used by data collection or Phase-2 module training.",
        "ko": "향후 메인 AI 학습을 이어갈 때 사용할 기존 .bangmodel 파일 또는 폴더입니다. 데이터 수집과 Phase-2 추론 모듈 학습에서는 사용하지 않습니다.",
    },


    "tooltip.emotion_schema_version": {
        "en": "A label for the format of this emotion configuration. Keep it unchanged unless you intentionally create a new format, because saved data and trained modules use it to check compatibility.",
        "ko": "이 감정 설정 형식의 이름표입니다. 의도적으로 새 형식을 만드는 경우가 아니라면 그대로 두세요. 저장된 데이터와 학습 모듈이 서로 호환되는지 확인할 때 사용합니다.",
    },
    "tooltip.emotion_count": {
        "en": "The number of rule-table emotion values. Each value has its own name and update rules. The production Formula model uses confidence, aggression, anxiety, and derived calm instead.",
        "ko": "rule-table 방식에서 사용할 감정 값의 개수입니다. 각 감정에는 이름과 변화 규칙이 있습니다. 현재 production Formula 모델은 자신감·공격성·불안과 여기서 계산되는 평온함을 사용합니다.",
    },
    "tooltip.emotion_name": {
        "en": "A short unique name used to identify this emotion in saved settings and logs. Choose a name that describes a feeling or behavior clearly.",
        "ko": "저장된 설정과 로그에서 이 감정을 구분하는 짧고 고유한 이름입니다. 감정이나 행동 성향을 쉽게 알 수 있는 이름을 사용하세요.",
    },
    "tooltip.emotion_change_rate": {
        "en": "The amount added when this emotion's update condition occurs. Positive values increase it; negative values decrease it. The final value is kept between 0 and 1.",
        "ko": "감정 변화 조건이 발생했을 때 더하거나 빼는 양입니다. 양수는 감정을 높이고 음수는 낮춥니다. 최종 값은 항상 0~1 사이로 제한됩니다.",
    },
    "tooltip.emotion_decay_rate": {
        "en": "The amount by which the emotion moves back toward its baseline during later game steps. A larger value means the emotion returns to normal faster.",
        "ko": "게임이 진행되는 동안 감정이 기준값으로 돌아가는 변화량입니다. 값이 클수록 감정이 더 빨리 평상 상태로 회복됩니다.",
    },
    "tooltip.emotion_event_updates": {
        "en": "A JSON map from public game events to emotion changes. For example, bang_at_me can react when this player is shot. Relationship-specific keys are checked before general event names.",
        "ko": "공개 게임 이벤트와 감정 변화량을 연결하는 JSON 맵입니다. 예를 들어 bang_at_me는 자신이 공격당했을 때 반응하도록 만들 수 있습니다. 관계 기반 키가 일반 이벤트 이름보다 먼저 적용됩니다.",
    },
    "tooltip.emotion_action_updates": {
        "en": "A JSON map from valid actions such as pass or bang to emotion changes. Invalid or rejected actions do not update the emotion.",
        "ko": "pass나 bang처럼 허용된 행동과 감정 변화량을 연결하는 JSON 맵입니다. 잘못되었거나 거부된 행동은 감정을 변화시키지 않습니다.",
    },
    "tooltip.emotion_reward_updates": {
        "en": "A JSON map from reward feedback to emotion changes. Positive, negative, or zero reward can be handled separately. Rewards are training feedback, not private information shown in the public game view.",
        "ko": "보상 피드백과 감정 변화량을 연결하는 JSON 맵입니다. 양의 보상, 음의 보상, 0 보상을 각각 다르게 처리할 수 있습니다. 보상은 학습용 피드백이며 공개 게임 화면에 표시되는 비공개 정보가 아닙니다.",
    },
    "tooltip.emotion_terminal_updates": {
        "en": "A JSON map from game-ending results such as win, loss, terminated, or truncated to emotion changes. Use this for a final reaction at the end of a game.",
        "ko": "승리, 패배, 정상 종료, 제한 시간 종료 같은 게임 종료 결과와 감정 변화량을 연결하는 JSON 맵입니다. 게임이 끝났을 때의 최종 감정 반응을 설정할 때 사용합니다.",
    },
    "tooltip.emotion_effects": {
        "en": "A JSON map that turns an emotion into an action preference. A positive effect can raise an action score and a negative effect can lower it; it changes preference, not the legal-action rules.",
        "ko": "감정값을 행동 선호도로 바꾸는 JSON 맵입니다. 양의 효과는 행동 점수를 높이고 음의 효과는 낮춥니다. 감정은 행동 선호만 바꾸며, 불법 행동을 허용하지는 않습니다.",
    },


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

    "replay.format_public": {"en": "Public", "ko": "공개"},
    "replay.format_full": {"en": "Full state", "ko": "전체 상태"},



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


            pass



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
