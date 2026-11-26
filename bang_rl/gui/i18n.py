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
    # Data library: delete (moves to artifacts/trash, never permanent)
    "library.action.delete": {"en": "Delete...", "ko": "삭제..."},
    "tooltip.library_action.delete": {
        "en": "Move the selected entries to the trash folder (artifacts/trash). Nothing is deleted permanently; the trash folder records where each file came from.",
        "ko": "선택한 항목을 휴지통 폴더(artifacts/trash)로 옮깁니다. 영구 삭제하지 않으며, 휴지통에 각 파일의 원래 위치가 기록됩니다.",
    },
    "library.delete_title": {"en": "Delete entries", "ko": "항목 삭제"},
    "library.delete_prompt": {
        "en": "Move these {count} entries to the trash?",
        "ko": "다음 {count}개 항목을 휴지통으로 옮길까요?",
    },
    "library.delete_size": {"en": "Total size: {size}", "ko": "전체 용량: {size}"},
    "library.delete_dependents": {
        "en": "Other entries were made from these (they stay, but lose this source):",
        "ko": "이 항목으로 만든 다른 항목이 있습니다 (남아 있지만 원본을 잃습니다):",
    },
    "library.delete_saved_warning": {
        "en": "Some entries are in saved/ (shared through git). Deleting them also removes them for others after you commit.",
        "ko": "saved/ 폴더(git으로 공유되는) 항목이 포함되어 있습니다. 커밋하면 다른 사용자에게서도 사라집니다.",
    },
    "library.delete_trash_note": {
        "en": "Files can be restored from artifacts/trash (see trash-record.json in each folder).",
        "ko": "artifacts/trash 폴더에서 복원할 수 있습니다 (각 폴더의 trash-record.json 참고).",
    },
    "library.delete_done": {"en": "Moved {count} entries to the trash.", "ko": "{count}개 항목을 휴지통으로 옮겼습니다."},
    "library.delete_failed": {"en": "Some entries were not deleted: {errors}", "ko": "일부 항목을 삭제하지 못했습니다: {errors}"},
    # Main menu / window
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
    "menu.group": {"en": "Training and tools", "ko": "학습 · 도구"},
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

    # Dummy AI emotion configuration (Settings screen + dialog)
    "settings.dummy_ai": {"en": "Dummy AI settings", "ko": "더미 AI 설정"},
    "settings.dummy_ai_title": {
        "en": "Dummy AI emotion configuration", "ko": "더미 AI 감정 설정",
    },
    "settings.dummy_ai_help": {
        "en": (
            "The dummy AI always acts on nine emotions: calm, confidence, aggression, "
            "anxiety and one relation value per seat. Edit the emotion descriptions and "
            "the formula parameters that drive them."
        ),
        "ko": (
            "더미 AI는 항상 9개의 감정(평온, 자신감, 공격성, 불안, 좌석별 관계 5개)으로만 "
            "행동합니다. 감정 설명과 감정을 계산하는 공식 파라미터를 편집할 수 있습니다."
        ),
    },
    "settings.emotion_description": {"en": "Description", "ko": "설명"},
    "settings.emotion_formula_json": {
        "en": "Formula parameters (JSON)", "ko": "감정 공식 파라미터 (JSON)",
    },
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
        "en": "Save the dummy AI's emotion configuration to a standalone file you can edit by hand.",
        "ko": "더미 AI의 감정 설정을 직접 편집할 수 있는 별도의 파일로 저장합니다.",
    },
    "tooltip.emotion_load_file": {
        "en": "Load a dummy AI emotion configuration file that was saved or edited outside the GUI.",
        "ko": "GUI 밖에서 저장하거나 편집한 더미 AI 감정 설정 파일을 불러옵니다.",
    },

    # Training monitor
    "training.mode": {"en": "Mode", "ko": "모드"},
    "training.mode.data_collection": {"en": "Collect training data", "ko": "학습 데이터 수집"},
    "training.mode.module_training": {"en": "Train intent and emotion modules", "ko": "의도·감정 추론 모듈 학습"},
    "training.mode.main_ai_training": {"en": "Train Main AI", "ko": "메인 AI 학습"},
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
    "training.resume_module_main": {"en": "Inference modules to use", "ko": "사용할 추론 모듈"},
    "training.resume_module_auto": {"en": "Auto (newest compatible modules)", "ko": "자동 (가장 최근 호환 모듈)"},
    "training.resume_module_incompatible": {"en": " (incompatible)", "ko": " (호환 안 됨)"},
    "training.resume_module_reason.missing_module": {"en": "This run does not have both an intent and an emotion module.", "ko": "이 작업에는 의도 모듈과 감정 모듈이 모두 있지 않습니다."},
    "training.resume_module_reason.intent_roles": {"en": "The intent module does not predict 4 role classes.", "ko": "의도 모듈이 역할 4종을 예측하지 않습니다."},
    "training.resume_module_reason.emotion_schema": {"en": "The emotion module does not use the dummy AI's 9-entry formula-1.0+label9 labels. Retrain it.", "ko": "감정 모듈이 더미 AI의 9개 항목(formula-1.0+label9) 라벨을 쓰지 않습니다. 다시 학습하세요."},
    "training.resume_module_reason.unreadable": {"en": "The module file cannot be read.", "ko": "모듈 파일을 읽을 수 없습니다."},
    "training.games": {"en": "Training games", "ko": "학습 게임 수"},
    "training.graph_interval_main": {"en": "Graph point every N games/updates", "ko": "그래프 점 간격 (게임/업데이트)"},
    "training.main_ai_bangmodel_active": {"en": "A .bangmodel is set, so its own modules are used and the module choice is ignored.", "ko": ".bangmodel이 지정되어 그 안의 모듈을 사용하며 모듈 선택은 무시됩니다."},
    # Data collection players, dataset selection, inference refinement setup
    "training.mode.inference_refinement": {"en": "Refine inference modules (PPO fixed)", "ko": "추론 모듈 개선 (PPO 고정)"},
    "training.collection_players": {"en": "Players", "ko": "수집할 게임"},
    "training.collection_players.dummy5": {"en": "Dummy AI x5", "ko": "더미 AI 5인"},
    "training.collection_players.main1_dummy4": {"en": "Main AI x1 + Dummy AI x4", "ko": "메인 AI 1 + 더미 AI 4"},
    "training.base_model": {"en": "Main AI model (.bangmodel)", "ko": "메인 AI 모델 (.bangmodel)"},
    "training.base_model_none": {"en": "Choose a model", "ko": "모델을 선택하세요"},
    "training.base_model_refresh": {"en": "Refresh", "ko": "새로고침"},
    "training.base_model_reason.unreadable": {"en": "The model file cannot be read.", "ko": "모델 파일을 읽을 수 없습니다."},
    "training.base_model_reason.emotion_schema": {"en": "The emotion module does not use the dummy AI's 9-entry formula-1.0+label9 labels.", "ko": "감정 모듈이 더미 AI의 9개 항목(formula-1.0+label9) 라벨을 쓰지 않습니다."},
    "training.main_action_mode": {"en": "Main AI action mode", "ko": "메인 AI 행동 방식"},
    "training.action_mode.deterministic": {"en": "Deterministic (same as game and evaluation)", "ko": "결정적 (게임·평가와 동일)"},
    "training.action_mode.stochastic": {"en": "Stochastic (sampled)", "ko": "확률적 (샘플링)"},
    "training.datasets": {"en": "Datasets to train on", "ko": "학습에 쓸 데이터셋"},
    "training.datasets_refresh": {"en": "Refresh datasets", "ko": "데이터셋 새로고침"},
    "training.datasets_empty": {"en": "No datasets yet. Collect data first.", "ko": "데이터셋이 없습니다. 먼저 데이터를 수집하세요."},
    "training.datasets_all": {"en": "None checked (use every dataset)", "ko": "선택 안 함 (모든 데이터셋 사용)"},
    "training.resume_main_ai_none": {"en": "None (new PPO on the chosen modules)", "ko": "없음 (선택한 모듈로 PPO 새로 학습)"},
    # Shared pickers (module pair, main AI model, datasets) and path fields
    "picker.refresh": {"en": "Refresh", "ko": "새로고침"},
    "picker.open_folder": {"en": "Open", "ko": "열기"},
    "picker.browse": {"en": "Browse for a file...", "ko": "파일 찾아보기..."},
    "picker.browse_title": {"en": "Choose a Main AI model", "ko": "메인 AI 모델 선택"},
    "picker.browse_folder_title": {"en": "Choose a folder", "ko": "폴더 선택"},
    "picker.bangmodel_filter": {"en": "Main AI model (*.bangmodel)", "ko": "메인 AI 모델 (*.bangmodel)"},
    "picker.selected_count": {"en": "{count} selected", "ko": "{count}개 선택됨"},
    "tooltip.picker_refresh": {"en": "Reload the list from disk.", "ko": "디스크에서 목록을 다시 읽습니다."},
    "tooltip.picker_open_folder": {"en": "Open the folder in the file explorer.", "ko": "탐색기에서 폴더를 엽니다."},
    "tooltip.picker_browse": {"en": "Pick a .bangmodel file outside the listed folders.", "ko": "목록에 없는 폴더의 .bangmodel 파일을 고릅니다."},
    "tooltip.path_browse": {"en": "Choose a folder.", "ko": "폴더를 고릅니다."},
    # Save locations group and the result path line
    "training.storage_group": {"en": "Save locations", "ko": "저장 위치"},
    "training.storage_report": {"en": "report", "ko": "리포트"},
    "training.storage_artifact": {"en": "artifact", "ko": "아티팩트"},
    "training.storage_default": {"en": "default folders", "ko": "기본 폴더"},
    "training.storage_changed": {"en": "changed", "ko": "변경됨"},
    "training.storage_naming": {
        "en": "File names: date-time_kind_conditions_shortid (for example 20261117-0348_mainai_30000g_ba97bb). Each .bangmodel has a .about.md next to it with its kind, sources and notes.",
        "ko": "파일 이름 규칙: 날짜-시각_종류_조건_짧은ID (예: 20261117-0348_mainai_30000g_ba97bb). 각 .bangmodel 옆의 .about.md에 종류, 원본, 비고가 적혀 있습니다.",
    },
    "tooltip.training_storage": {
        "en": "Show or hide the folders where reports and models are written.",
        "ko": "리포트와 모델을 저장할 폴더를 펼치거나 접습니다.",
    },
    "training.result_saving": {"en": "Saving to:", "ko": "저장 위치:"},
    "training.result_saved": {"en": "Saved:", "ko": "저장됨:"},
    "training.result_copy": {"en": "Copy path", "ko": "경로 복사"},
    "tooltip.training_result_copy": {"en": "Copy the full path to the clipboard.", "ko": "전체 경로를 클립보드에 복사합니다."},
    "training.games_per_round": {"en": "Games per round", "ko": "라운드당 수집 게임 수"},
    "training.refine_rounds": {"en": "Max rounds", "ko": "최대 라운드 수"},
    "training.refine_epochs": {"en": "Epochs per round", "ko": "라운드당 에폭"},
    "training.refine_patience": {"en": "Patience (rounds without gain)", "ko": "허용할 개선 없는 라운드 수 (patience)"},
    "training.refine_eval_games": {"en": "Evaluation games", "ko": "평가 게임 수"},
    "tooltip.training_collection_players": {
        "en": "Dummy AI x5 collects games played by five dummies. Main AI x1 + Dummy AI x4 collects games where the chosen .bangmodel plays one seat (the seat rotates every game) among four dummies. Only the main AI's own decisions are stored, with the dummies' emotions and all roles as labels, because that is the situation the main AI infers in.",
        "ko": "더미 AI 5인은 더미 다섯이 두는 게임을 수집합니다. 메인 AI 1 + 더미 AI 4는 고른 .bangmodel이 한 좌석(게임마다 좌석이 바뀝니다)을 맡고 나머지를 더미 넷이 맡는 게임을 수집합니다. 메인 AI가 추론하는 실제 상황이므로 메인 AI가 결정한 시점의 기록만 저장하고, 더미의 감정과 모든 역할이 정답으로 함께 저장됩니다.",
    },
    "tooltip.training_base_model": {
        "en": "The main AI model that plays the games. For refinement its PPO stays fixed and is copied unchanged into the new model; only the intent and emotion modules change. Models whose modules do not fit are disabled.",
        "ko": "게임을 두는 메인 AI 모델입니다. 개선 모드에서는 PPO가 고정된 채 새 모델에 그대로 복사되고, 의도·감정 모듈만 바뀝니다. 모듈이 맞지 않는 모델은 선택할 수 없습니다.",
    },
    "tooltip.training_main_action_mode": {
        "en": "Deterministic always picks the most likely action, like a game in the app and like evaluation. Stochastic samples actions, which gives more varied games.",
        "ko": "결정적은 앱의 게임과 평가처럼 가장 확률이 높은 행동을 항상 고릅니다. 확률적은 행동을 샘플링하므로 더 다양한 게임이 나옵니다.",
    },
    "tooltip.training_datasets": {
        "en": "Check the datasets to train on. Dummy AI x5 and Main AI x1 + Dummy AI x4 datasets, local or saved, can be mixed; they are combined without extra weight, duplicates are used once and the train, validation and test split is made per game. Nothing checked means every dataset found (refinement: every Dummy AI x5 dataset).",
        "ko": "학습에 쓸 데이터셋을 체크하세요. 더미 AI 5인과 메인 AI 1 + 더미 AI 4 데이터셋을 로컬·저장본 구분 없이 섞을 수 있습니다. 가중치 없이 그대로 합치고, 중복은 한 번만 쓰며, 학습·검증·테스트 분할은 게임 단위로 합니다. 아무것도 체크하지 않으면 찾은 모든 데이터셋을 씁니다 (개선 모드는 더미 AI 5인 데이터셋 전부).",
    },
    "tooltip.training_games_per_round": {
        "en": "How many main AI + 4 dummy games each refinement round collects with the current best modules. The seat rotates every game, so a multiple of 5 gives an even seat mix.",
        "ko": "개선의 각 라운드에서 현재 최고 모듈로 수집할 메인 AI + 더미 4 게임 수입니다. 좌석이 게임마다 바뀌므로 5의 배수로 두면 좌석이 고르게 섞입니다.",
    },
    "tooltip.training_refine_rounds": {
        "en": "Most rounds to run. Each round collects games, trains the modules, evaluates them with the fixed PPO and keeps the new modules only if they are better. Round 0 evaluates the starting modules as the reference.",
        "ko": "실행할 최대 라운드 수입니다. 각 라운드는 게임 수집, 모듈 학습, 고정된 PPO로 평가를 거쳐 더 좋을 때만 새 모듈을 채택합니다. 0라운드는 시작 모듈을 평가한 기준입니다.",
    },
    "tooltip.training_refine_epochs": {
        "en": "Training epochs of the modules in each round. They continue from the current best modules.",
        "ko": "각 라운드에서 모듈을 학습하는 에폭 수입니다. 현재 최고 모듈에서 이어서 학습합니다.",
    },
    "tooltip.training_refine_patience": {
        "en": "Stop after this many rounds in a row whose new modules were not accepted. The previous best stays.",
        "ko": "새 모듈이 연속으로 이 라운드 수만큼 채택되지 않으면 멈춥니다. 이전 최고 모듈이 그대로 남습니다.",
    },
    "tooltip.training_refine_eval_games": {
        "en": "Games of the fixed evaluation deals used to compare modules under the fixed PPO. More games make the comparison steadier. The same deals are used every round.",
        "ko": "고정된 PPO로 모듈을 비교할 때 쓰는 평가 판(고정된 deal)의 게임 수입니다. 많을수록 비교가 안정적입니다. 매 라운드 같은 deal을 씁니다.",
    },
    "training.max_games": {"en": "Max games to use", "ko": "사용할 최대 게임 수"},
    "training.max_games_unlimited": {"en": "No limit", "ko": "제한 없음"},
    "tooltip.training_max_games": {
        "en": "Reads only a random subset of this many games from the selected datasets, so a huge dataset is never loaded whole. The subset is different on every run, so repeated continue-training runs cover more of the data. Roughly 0.3 GB of RAM per 1000 games. 0 loads everything and may run out of memory with large datasets.",
        "ko": "선택한 데이터셋에서 이 수만큼의 게임만 무작위로 읽어서, 매우 큰 데이터셋도 통째로 불러오지 않습니다. 실행할 때마다 다른 부분 집합을 고르므로 이어서 학습을 반복하면 더 많은 데이터를 다룹니다. 1000게임당 RAM 약 0.3GB가 필요합니다. 0은 전부 불러오며, 큰 데이터셋에서는 메모리가 부족할 수 있습니다.",
    },
    "tooltip.training_max_minutes_refine": {
        "en": "Optional time limit in minutes for the whole refinement. Blank means no limit. The best model found so far is saved when it ends. Pausing does not use up the time.",
        "ko": "개선 전체의 선택적 시간 제한(분)입니다. 비우면 제한이 없습니다. 끝나면 그때까지의 최고 모델이 저장됩니다. 일시정지한 시간은 포함되지 않습니다.",
    },
    # Inference refinement monitor
    "refine.progress": {"en": "Round {round} / {total} · {stage} · best round: {best} · main AI games collected: {games}", "ko": "라운드 {round} / {total} · {stage} · 최고 라운드: {best} · 수집한 메인 AI 게임: {games}판"},
    "refine.stage.refine_base_eval": {"en": "evaluating the starting modules", "ko": "시작 모듈 평가 중"},
    "refine.stage.refine_collecting": {"en": "collecting main AI + 4 dummy games", "ko": "메인 AI + 더미 4 게임 수집 중"},
    "refine.stage.refine_training": {"en": "training the modules", "ko": "모듈 학습 중"},
    "refine.stage.refine_evaluating": {"en": "evaluating the new modules", "ko": "새 모듈 평가 중"},
    "refine.stage.done": {"en": "finished", "ko": "완료"},
    "refine.stage.none": {"en": "preparing", "ko": "준비 중"},
    "refine.base": {"en": "Round 0 (starting modules): hidden-role accuracy {hidden} · role accuracy {role} · emotion MSE {mse} · win rate {win}", "ko": "0라운드 (시작 모듈): 숨겨진 역할 정확도 {hidden} · 역할 정확도 {role} · 감정 MSE {mse} · 승률 {win}"},
    "refine.best": {"en": "Best (round {round}): hidden-role accuracy {hidden} · role accuracy {role} · emotion MSE {mse} · win rate {win}", "ko": "최고 (라운드 {round}): 숨겨진 역할 정확도 {hidden} · 역할 정확도 {role} · 감정 MSE {mse} · 승률 {win}"},
    "refine.round_line": {"en": "Round {round}: {verdict} ({reason}) · hidden-role {hidden} · role {role} · MSE {mse} · win {win}", "ko": "라운드 {round}: {verdict} ({reason}) · 숨겨진 역할 {hidden} · 역할 {role} · MSE {mse} · 승률 {win}"},
    "refine.verdict.accepted": {"en": "accepted", "ko": "채택"},
    "refine.verdict.rejected": {"en": "rejected", "ko": "기각"},
    "refine.reason.base_reference": {"en": "reference", "ko": "기준"},
    "refine.reason.improved_hidden_role_accuracy": {"en": "hidden-role accuracy improved", "ko": "숨겨진 역할 정확도 향상"},
    "refine.reason.improved_emotion_mse": {"en": "emotion MSE improved", "ko": "감정 MSE 개선"},
    "refine.reason.improved_both": {"en": "both improved", "ko": "둘 다 개선"},
    "refine.reason.hidden_role_accuracy_worse": {"en": "hidden-role accuracy got worse", "ko": "숨겨진 역할 정확도 하락"},
    "refine.reason.emotion_mse_worse": {"en": "emotion MSE got worse", "ko": "감정 MSE 악화"},
    "refine.reason.no_meaningful_improvement": {"en": "no meaningful improvement", "ko": "의미 있는 개선 없음"},
    "refine.reason.metric_missing": {"en": "a metric could not be measured", "ko": "측정하지 못한 지표가 있음"},
    "refine.stop.max_rounds": {"en": "Stopped: reached the max rounds.", "ko": "종료: 최대 라운드에 도달했습니다."},
    "refine.stop.patience": {"en": "Stopped: no improvement for the allowed number of rounds.", "ko": "종료: 허용한 라운드 동안 개선이 없었습니다."},
    "refine.stop.user_stop_or_time_budget": {"en": "Stopped by the user or the time limit.", "ko": "사용자 중지 또는 시간 제한으로 종료했습니다."},
    "refine.rule": {"en": "Acceptance: hidden-role accuracy must not drop by more than 0.005, emotion MSE must not rise by more than 1%, and at least one must improve beyond that margin.", "ko": "채택 기준: 숨겨진 역할 정확도가 0.005보다 더 떨어지지 않고, 감정 MSE가 1%보다 더 오르지 않으며, 둘 중 하나는 그 폭을 넘어 좋아져야 합니다."},
    "refine.saved": {"en": "Refined model: {path}", "ko": "개선된 모델: {path}"},
    "refine.not_saved": {"en": "Refined model: saved after the first accepted round and at the end", "ko": "개선된 모델: 처음 채택된 라운드와 종료 시점에 저장됩니다"},
    "refine.progress_bar": {"en": "Round %v / %m", "ko": "라운드 %v / %m"},
    "refine.axis.round": {"en": "Round (0 = starting modules)", "ko": "라운드 (0 = 시작 모듈)"},
    "refine.axis.accuracy": {"en": "Accuracy / win rate", "ko": "정확도 / 승률"},
    "refine.axis.mse": {"en": "Emotion MSE", "ko": "감정 MSE"},
    "refine.curve.hidden": {"en": "Hidden-role accuracy", "ko": "숨겨진 역할 정확도"},
    "refine.curve.role": {"en": "Role accuracy", "ko": "역할 정확도"},
    "refine.curve.win": {"en": "Win rate (secondary)", "ko": "승률 (부가 지표)"},
    "refine.curve.mse": {"en": "Emotion MSE", "ko": "감정 MSE"},
    # Inference accuracy evaluation dialog
    "infer.open": {"en": "Evaluate inference accuracy", "ko": "추론 정확도 평가"},
    "tooltip.infer_open": {
        "en": "Compare intent and emotion module pairs under the fixed PPO of one main AI model on the fixed evaluation games.",
        "ko": "메인 AI 모델 하나의 고정된 PPO 아래에서 의도·감정 모듈 쌍들을 고정된 평가 게임으로 비교합니다.",
    },
    "infer.title": {"en": "Inference accuracy evaluation", "ko": "추론 정확도 평가"},
    "infer.model": {"en": "Main AI model (fixed PPO)", "ko": "메인 AI 모델 (PPO 고정)"},
    "infer.pairs": {"en": "Module pairs to compare", "ko": "비교할 모듈 쌍"},
    "infer.include_base": {"en": "Include the model's own modules", "ko": "모델에 들어 있는 모듈도 평가"},
    "infer.games": {"en": "Evaluation games", "ko": "평가 게임 수"},
    "infer.run": {"en": "Run", "ko": "실행"},
    "infer.stop": {"en": "Stop", "ko": "중지"},
    "infer.note": {"en": "Swapping modules changes what the PPO receives, so its play may shift. That is the real deployment effect.", "ko": "모듈을 바꾸면 PPO가 받는 입력이 달라져 플레이가 바뀔 수 있습니다. 실제로 쓸 때 일어나는 효과입니다."},
    "infer.col.name": {"en": "Modules", "ko": "모듈"},
    "infer.col.hidden": {"en": "Hidden-role accuracy", "ko": "숨겨진 역할 정확도"},
    "infer.col.role": {"en": "Role accuracy", "ko": "역할 정확도"},
    "infer.col.mse": {"en": "Emotion MSE", "ko": "감정 MSE"},
    "infer.col.win": {"en": "Win rate (secondary)", "ko": "승률 (부가)"},
    "infer.col.by_role": {"en": "Role accuracy by target role", "ko": "대상 역할별 정확도"},
    "infer.base_label": {"en": "Modules inside the model", "ko": "모델에 들어 있는 모듈"},
    "infer.no_model": {"en": "Choose a main AI model.", "ko": "메인 AI 모델을 고르세요."},
    "infer.nothing": {"en": "Nothing to evaluate: check a module pair or include the model's own modules.", "ko": "평가할 것이 없습니다. 모듈 쌍을 체크하거나 모델의 모듈을 포함하세요."},
    "infer.running": {"en": "Evaluating {done} / {total}: {label}", "ko": "평가 중 {done} / {total}: {label}"},
    "infer.finished": {"en": "Finished.", "ko": "완료했습니다."},
    "infer.stopped": {"en": "Stopped.", "ko": "중지했습니다."},
    "infer.failed": {"en": "Failed: {error}", "ko": "실패: {error}"},
    # Main AI training monitor
    "mainai.role.sheriff": {"en": "Sheriff", "ko": "보안관"},
    "mainai.role.deputy": {"en": "Deputy", "ko": "부관"},
    "mainai.role.outlaw": {"en": "Outlaw", "ko": "무법자"},
    "mainai.role.traitor": {"en": "Traitor", "ko": "배신자"},
    "mainai.progress": {"en": "Games completed: {games} · PPO updates: {updates} · Training win rate (moving average): {win}", "ko": "완료한 게임: {games}판 · PPO 업데이트: {updates}회 · 학습 승률(이동 평균): {win}"},
    "mainai.baselines": {"en": "Baselines on the evaluation deals, overall win rate: dummy {dummy} · random {random}", "ko": "평가 판 기준선 전체 승률: 더미 {dummy} · 랜덤 {random}"},
    "mainai.baselines_pending": {"en": "Baselines: being computed (cached after the first run)", "ko": "기준선: 계산 중 (처음 한 번 계산하면 저장해 재사용)"},
    "mainai.latest": {"en": "Latest evaluation:", "ko": "최근 평가:"},
    "mainai.latest.line": {"en": "Latest evaluation (PPO update {update}, {games} games trained): win rate {win} · normalized score {score} · role accuracy {accuracy} · emotion MSE {mse}", "ko": "최근 평가 (PPO 업데이트 {update}회, 학습 {games}판): 승률 {win} · 정규화 점수 {score} · 역할 정확도 {accuracy} · 감정 MSE {mse}"},
    "mainai.best": {"en": "Best evaluation (saved model):", "ko": "최고 평가 (저장된 모델):"},
    "mainai.best.line": {"en": "Best evaluation, saved model (PPO update {update}, {games} games trained): win rate {win} · normalized score {score} · role accuracy {accuracy} · emotion MSE {mse}", "ko": "최고 평가, 저장된 모델 (PPO 업데이트 {update}회, 학습 {games}판): 승률 {win} · 정규화 점수 {score} · 역할 정확도 {accuracy} · 감정 MSE {mse}"},
    "mainai.no_eval": {"en": "none yet", "ko": "아직 없음"},
    "mainai.by_role": {"en": "Win rate by role (main / dummy / random):", "ko": "역할별 승률 (메인 / 더미 / 랜덤):"},
    "mainai.saved": {"en": "Saved model: {path}", "ko": "저장된 모델: {path}"},
    "mainai.not_saved": {"en": "Saved model: none yet (only the best evaluated model is saved)", "ko": "저장된 모델: 아직 없음 (평가가 가장 좋은 모델만 저장됩니다)"},
    "mainai.truncated_warning": {"en": "Truncated rate is above 1%: some games reached the step limit.", "ko": "잘린 게임 비율이 1%를 넘었습니다. 일부 게임이 단계 제한에 도달했습니다."},
    "mainai.curve.train_win": {"en": "Training win rate (moving avg)", "ko": "학습 승률 (이동 평균)"},
    "mainai.curve.eval_win": {"en": "Evaluation win rate", "ko": "평가 승률"},
    "mainai.curve.dummy": {"en": "Dummy baseline", "ko": "더미 기준선"},
    "mainai.curve.random": {"en": "Random baseline", "ko": "랜덤 기준선"},
    "mainai.curve.policy_loss": {"en": "Policy loss", "ko": "정책 loss"},
    "mainai.curve.value_loss": {"en": "Value loss", "ko": "가치 loss"},
    "mainai.curve.entropy": {"en": "Entropy", "ko": "엔트로피"},
    "mainai.progress_bar": {"en": "Games: %v / %m", "ko": "게임: %v / %m"},
    "mainai.progress_bar_time": {"en": "Games: %v (limited by time)", "ko": "게임: %v (시간 기준)"},
    "mainai.axis.win_rate": {"en": "Win rate", "ko": "승률"},
    "mainai.axis.games": {"en": "Games", "ko": "게임"},
    "mainai.axis.loss": {"en": "Loss / entropy", "ko": "Loss / 엔트로피"},
    "mainai.axis.updates": {"en": "PPO update", "ko": "PPO 업데이트"},
    # Saved inference-module performance dialog
    "perf.open": {"en": "Saved model performance", "ko": "저장된 모델 성능 보기"},
    "perf.title": {"en": "Saved model performance", "ko": "저장된 모델 성능"},
    "perf.empty": {"en": "No saved models yet. Run module training or main AI training first.", "ko": "저장된 모델이 없습니다. 먼저 모듈 학습이나 메인 AI 학습을 실행하세요."},
    "perf.main_ai_item": {"en": "[Main AI] {time} · {id}", "ko": "[메인 AI] {time} · {id}"},
    "perf.main_ai_section": {"en": "■ Main AI (PPO)", "ko": "■ 메인 AI (PPO)"},
    "perf.main_ai_file": {"en": "File: {path}", "ko": "파일: {path}"},
    "perf.main_ai_not_trained": {"en": "This file has no main AI training record.", "ko": "이 파일에는 메인 AI 학습 기록이 없습니다."},
    "perf.main_ai_training": {"en": "Trained on {games} games, {updates} PPO updates, {seconds} s", "ko": "학습 {games}판, PPO 업데이트 {updates}회, {seconds}초"},
    "perf.main_ai_source": {"en": "Inference modules: {origin}{job}", "ko": "추론 모듈: {origin}{job}"},
    "perf.main_ai_source_job": {"en": " (module job {job})", "ko": " (모듈 작업 {job})"},
    "perf.main_ai_origin.phase2_pair": {"en": "selected intent + emotion pair", "ko": "선택한 의도+감정 모듈"},
    "perf.main_ai_origin.latest_saved_modules": {"en": "newest compatible saved modules", "ko": "가장 최근 호환 모듈"},
    "perf.main_ai_origin.bangmodel": {"en": "continued from a .bangmodel", "ko": ".bangmodel에서 이어 학습"},
    "perf.main_ai_eval": {"en": "Best evaluation ({games} games): win rate {win} · normalized score {score}", "ko": "최고 평가 ({games}판): 승률 {win} · 정규화 점수 {score}"},
    "perf.main_ai_eval_extra": {"en": "Role accuracy {accuracy} · emotion MSE {mse} · truncated games {truncated}", "ko": "역할 정확도 {accuracy} · 감정 MSE {mse} · 잘린 게임 {truncated}"},
    "perf.main_ai_baselines": {"en": "Baselines (win rate): dummy {dummy} · random {random}", "ko": "기준선 승률: 더미 {dummy} · 랜덤 {random}"},
    "perf.main_ai_normalized": {"en": "Normalized score by role (0 = random level, 1 = dummy level):", "ko": "역할별 정규화 점수 (0 = 랜덤 수준, 1 = 더미 수준):"},
    "perf.refresh": {"en": "Refresh", "ko": "새로고침"},
    "perf.close": {"en": "Close", "ko": "닫기"},
    "perf.use_for_resume": {"en": "Continue training from this model", "ko": "이 모델로 이어 학습"},
    "perf.resume_selected": {"en": "Module training will continue from {job}.", "ko": "모듈 학습을 {job}에서 이어서 진행합니다."},
    "perf.resume_selected_main": {"en": "Main AI training will use the modules of {job}.", "ko": "메인 AI 학습에 {job}의 모듈을 사용합니다."},
    "perf.resume_unusable": {"en": "{job} cannot be used in the current mode (incompatible modules).", "ko": "{job}은(는) 현재 모드에서 사용할 수 없습니다 (호환되지 않는 모듈)."},
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

    # Settings field help tooltips (shown on mouse-over of each field/label)
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
    "tooltip.dummy_ai": {
        "en": "Open the Dummy AI settings. These settings describe how the non-learning rule-based players react to public events, actions, rewards, and the current Formula emotion state.",
        "ko": "더미 AI 설정을 엽니다. 이 설정에서는 학습되지 않은 규칙 기반 플레이어가 공개 사건, 행동, 보상, 현재 Formula 감정 상태에 어떻게 반응하는지 정합니다.",
    },

    # AI-training settings menu (TrainingSetupPanel) field/label tooltips
    "tooltip.training_mode": {
        "en": "Choose what this screen should do. Collect training data plays Dummy AI games and saves examples. Module training learns the Intent and Emotion inference modules from saved data. Main AI training plays games against Dummy AIs and trains the PPO policy on top of frozen intent and emotion modules, then saves the best evaluated model.",
        "ko": "이 화면에서 수행할 작업을 선택합니다. 학습 데이터 수집은 더미 AI 게임을 실행해 예시를 저장합니다. 추론 모듈 학습은 저장된 데이터로 의도·감정 모듈을 학습합니다. 메인 AI 학습은 고정된 의도·감정 모듈 위에서 더미 AI와 게임을 하며 PPO 정책을 학습하고, 평가가 가장 좋은 모델을 저장합니다.",
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
        "en": "The folder where Main AI training saves its model as a .bangmodel file named after the job ID. Only the best evaluated model is saved. Data collection and module training do not create a .bangmodel file.",
        "ko": "메인 AI 학습이 모델을 작업 ID 이름의 .bangmodel 파일로 저장할 폴더입니다. 평가가 가장 좋은 모델만 저장됩니다. 데이터 수집과 모듈 학습에서는 .bangmodel 파일을 만들지 않습니다.",
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
    "tooltip.training_resume_artifact_path_main": {
        "en": "Optional. A saved Main AI model (.bangmodel) to continue training from: its PPO weights are kept, the optimizer starts fresh, and the intent and emotion modules stored inside it are used. When a model is chosen here, the module choice above is ignored. Choose None to start a fresh PPO on the chosen modules. Use 'Browse for a file...' for a model outside the listed folders. Incompatible models are listed but cannot be chosen.",
        "ko": "선택 사항입니다. 이어 학습할 저장된 메인 AI 모델(.bangmodel)입니다. PPO 가중치는 이어받고 optimizer는 새로 시작하며, 파일 안에 저장된 의도·감정 모듈을 그대로 사용합니다. 여기서 모델을 고르면 위의 모듈 선택은 무시됩니다. '없음'을 고르면 선택한 모듈 위에서 PPO를 새로 학습합니다. 목록에 없는 폴더의 모델은 '파일 찾아보기...'로 고릅니다. 호환되지 않는 모델은 목록에 보이지만 고를 수 없습니다.",
    },
    "tooltip.training_resume_module_main": {
        "en": "The intent and emotion modules the Main AI is built on. They stay frozen during training and only the PPO policy learns. Auto picks the newest saved run that is compatible. Runs marked incompatible cannot be used: the intent module must predict 4 roles and the emotion module must use the Dummy AI's 9-entry formula-1.0+label9 labels. Ignored when a .bangmodel is set below.",
        "ko": "메인 AI가 사용할 의도·감정 모듈입니다. 학습하는 동안 모듈은 고정되고 PPO 정책만 학습합니다. 자동은 저장된 작업 중 가장 최근의 호환되는 것을 고릅니다. 호환 안 됨으로 표시된 작업은 쓸 수 없습니다. 의도 모듈은 역할 4종을, 감정 모듈은 더미 AI의 9개 항목(formula-1.0+label9) 라벨을 예측해야 합니다. 아래에 .bangmodel을 지정하면 무시됩니다.",
    },
    "tooltip.training_games": {
        "en": "How many games Main AI training plays. The main agent plays sheriff, deputy, outlaw and traitor in turn, so use a multiple of 4 for an even role mix. Set it to 0 (shown as Blank) to limit training by time instead, using the Max training time field.",
        "ko": "메인 AI 학습이 진행할 게임 수입니다. 메인 에이전트가 보안관·부관·무법자·배신자 역할을 차례로 맡으므로 4의 배수로 두면 역할이 고르게 섞입니다. 0(비움)으로 두면 게임 수 대신 '최대 학습 시간' 값으로 학습을 제한합니다.",
    },
    "tooltip.training_max_minutes_main": {
        "en": "Time limit in minutes, used only when Training games is Blank. Leave this blank (or 0) to train until you press Stop. The best model found so far is saved when training ends. Pausing does not use up the time.",
        "ko": "학습 게임 수가 '비움'일 때만 쓰이는 시간 제한(분)입니다. 이 칸도 비우거나 0이면 정지 버튼을 누를 때까지 학습합니다. 학습이 끝나면 그때까지 찾은 최고 모델이 저장됩니다. 일시정지한 시간은 제한 시간에 포함되지 않습니다.",
    },
    "tooltip.training_graph_interval_main": {
        "en": "Store one graph point every N games (win rate) or every N PPO updates (losses). The first point is always shown. Larger values keep the graph fast on long runs; 1 plots every game and update.",
        "ko": "게임 N판마다(승률) 또는 PPO 업데이트 N번마다(loss) 그래프 점을 하나씩 저장합니다. 첫 점은 항상 표시됩니다. 값이 클수록 오래 학습할 때 그래프가 빨라지고, 1이면 모든 게임과 업데이트를 찍습니다.",
    },

    # Dummy AI emotion dialog field/label tooltips
    "tooltip.emotion_description": {
        "en": "Text shown as a tooltip on this emotion's bar in spectator mode and replay.",
        "ko": "관전과 리플레이에서 이 감정 막대에 마우스를 올렸을 때 표시되는 설명입니다.",
    },
    "tooltip.emotion_formula": {
        "en": "A JSON object with 'params' (formula coefficients) and 'personality_by_role' (per-role overrides for sheriff, deputy, outlaw, traitor). Unknown names are rejected.",
        "ko": "'params'(공식 계수)와 'personality_by_role'(보안관·부관·무법자·배신자별 덮어쓰기)로 이루어진 JSON 객체입니다. 알 수 없는 이름은 거부됩니다.",
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

    # Save outputs (copy a training result into the shared saved/ folder)
    "save.button": {"en": "Save outputs...", "ko": "산출물 저장..."},
    "tooltip.save_button": {
        "en": "Copy this run's outputs to the saved folder under a name you choose. Saved copies are not git-ignored, so teammates can use them.",
        "ko": "이번 실행의 산출물을 원하는 이름으로 saved 폴더에 복사합니다. 저장본은 git에서 제외되지 않아 다른 사용자가 쓸 수 있습니다.",
    },
    "save.title": {"en": "Save outputs", "ko": "산출물 저장"},
    "save.category.datasets": {"en": "Training dataset", "ko": "학습 데이터"},
    "save.category.modules": {"en": "Inference modules (intent + emotion)", "ko": "추론 모듈 (의도 + 감정)"},
    "save.category.main_ai": {"en": "Main AI model", "ko": "메인 AI 모델"},
    "save.files": {"en": "Files to copy", "ko": "복사할 파일"},
    "save.total": {"en": "Total: {size}  ({count} files)", "ko": "합계: {size} (파일 {count}개)"},
    "save.name": {"en": "Name", "ko": "이름"},
    "save.name_hint": {
        "en": "Default follows the naming rule. You can type any name, Korean is fine.",
        "ko": "기본값은 이름 규칙입니다. 한글 등 원하는 이름으로 바꿀 수 있습니다.",
    },
    "save.notes": {"en": "Notes", "ko": "비고"},
    "save.target": {"en": "Saved to: {path}", "ko": "저장 위치: {path}"},
    "save.warn_50": {
        "en": "Warning: {file} is {size}. GitHub warns for files over 50 MB.",
        "ko": "주의: {file} 파일이 {size}입니다. GitHub는 50MB를 넘는 파일에 경고를 표시합니다.",
    },
    "save.strong_100": {
        "en": "Warning: {file} is {size}. It cannot be pushed to GitHub (100 MB per file limit).",
        "ko": "경고: {file} 파일이 {size}입니다. GitHub에 올릴 수 없습니다 (파일당 100MB 제한).",
    },
    "save.confirm_strong": {
        "en": "{file} is {size} and cannot be pushed to GitHub (100 MB per file limit). Save it locally anyway?",
        "ko": "{file} 파일이 {size}라서 GitHub에 올릴 수 없습니다 (파일당 100MB 제한). 그래도 이 컴퓨터에 저장할까요?",
    },
    "save.confirm_title": {"en": "File too large for GitHub", "ko": "GitHub 용량 제한 초과"},
    "save.name_exists": {
        "en": "A saved item named '{name}' already exists. Choose another name.",
        "ko": "'{name}' 이름의 저장본이 이미 있습니다. 다른 이름을 입력하세요.",
    },
    "save.name_invalid": {"en": "Enter a valid name.", "ko": "올바른 이름을 입력하세요."},
    "save.done": {"en": "Saved to {path}", "ko": "저장했습니다: {path}"},
    "save.failed": {"en": "Could not save: {error}", "ko": "저장하지 못했습니다: {error}"},
    "save.nothing": {"en": "No outputs to save for this run.", "ko": "이번 실행에는 저장할 산출물이 없습니다."},
    "save.save": {"en": "Save", "ko": "저장"},
    "save.cancel": {"en": "Cancel", "ko": "취소"},
    "notes.title": {"en": "Edit notes", "ko": "비고 편집"},
    "notes.prompt": {
        "en": "Free-text notes stored inside the file. Renaming the file does not lose them.",
        "ko": "파일 안에 저장되는 자유 메모입니다. 파일 이름을 바꿔도 사라지지 않습니다.",
    },
    "perf.edit_notes": {"en": "Edit notes...", "ko": "비고 편집..."},
    "tooltip.perf_edit_notes": {
        "en": "Edit the free-text notes stored inside the selected model files.",
        "ko": "선택한 모델 파일 안에 저장된 자유 메모를 수정합니다.",
    },
    "perf.save_copy": {"en": "Save a copy...", "ko": "복사본 저장..."},
    "tooltip.perf_save_copy": {
        "en": "Copy the selected model to the saved folder so it can be shared through git.",
        "ko": "선택한 모델을 saved 폴더에 복사해 git으로 공유할 수 있게 합니다.",
    },
    "perf.name": {"en": "Name: {name}", "ko": "이름: {name}"},
    "perf.notes_line": {"en": "Notes: {notes}", "ko": "비고: {notes}"},
    "perf.created": {"en": "Created: {time}", "ko": "생성: {time}"},
    # --- data library (데이터 보관함) ---------------------------------------------
    "menu.library": {"en": "Data library", "ko": "데이터 보관함"},
    "library.open": {"en": "Data library...", "ko": "데이터 보관함..."},
    "tooltip.library_open": {
        "en": "Browse every stored dataset, module pair, model, report and replay in one place, and use them for training or play.",
        "ko": "저장된 학습 데이터, 모듈, 모델, 보고서, 리플레이를 한곳에서 보고 학습이나 게임에 바로 쓸 수 있습니다.",
    },
    "library.title": {"en": "Data library", "ko": "데이터 보관함"},
    "library.empty": {
        "en": "Nothing is stored yet. Collect data or train a model first, then press Refresh.",
        "ko": "아직 저장된 항목이 없습니다. 데이터를 수집하거나 모델을 학습한 뒤 새로고침을 누르세요.",
    },
    "library.select_prompt": {
        "en": "Select an item to see its details. Hold Ctrl or Shift to select several datasets.",
        "ko": "항목을 선택하면 자세한 내용이 보입니다. 학습 데이터는 Ctrl 또는 Shift로 여러 개를 고를 수 있습니다.",
    },
    "library.count": {"en": "{shown} of {total}", "ko": "{shown} / {total}개"},
    "library.filter_kind": {"en": "Kind", "ko": "종류"},
    "library.filter_location": {"en": "Where", "ko": "위치"},
    "library.search_placeholder": {"en": "Search name, notes, conditions...", "ko": "이름, 비고, 조건 검색..."},
    "tooltip.library_search": {
        "en": "Shows only entries whose name, notes, conditions or results contain every word typed here.",
        "ko": "이름, 비고, 조건, 결과에 입력한 단어가 모두 들어 있는 항목만 보여 줍니다.",
    },
    "tooltip.library_refresh": {
        "en": "Scan the folders again, for example after a training job has finished.",
        "ko": "폴더를 다시 읽습니다. 학습이 끝난 뒤에 눌러 새 결과를 확인하세요.",
    },
    "tooltip.library_kind": {
        "en": "Show only one kind of item. Saved models means module pairs and main AI models.",
        "ko": "한 종류만 보여 줍니다. 저장된 모델은 모듈 쌍과 메인 AI 모델입니다.",
    },
    "tooltip.library_location": {
        "en": "Generated means files made on this computer. Saved means the copies kept in the saved folder.",
        "ko": "생성됨은 이 컴퓨터에서 만든 파일, 저장됨은 saved 폴더에 보관한 복사본입니다.",
    },
    "library.kind.all": {"en": "All", "ko": "전체"},
    "library.kind.dataset": {"en": "Dataset", "ko": "학습 데이터"},
    "library.kind.module_pair": {"en": "Module pair", "ko": "모듈 쌍"},
    "library.kind.main_ai": {"en": "Main AI", "ko": "메인 AI"},
    "library.kind.report": {"en": "Report", "ko": "보고서"},
    "library.kind.replay": {"en": "Replay", "ko": "리플레이"},
    "library.kind.models": {"en": "Saved models (modules + main AI)", "ko": "저장된 모델 (모듈 + 메인 AI)"},
    "library.location.all": {"en": "All", "ko": "전체"},
    "library.location.local": {"en": "Generated", "ko": "생성됨"},
    "library.location.saved": {"en": "Saved", "ko": "저장됨"},
    "library.col.kind": {"en": "Kind", "ko": "종류"},
    "library.col.name": {"en": "Name", "ko": "이름"},
    "library.col.created": {"en": "Created", "ko": "생성 시각"},
    "library.col.conditions": {"en": "Conditions", "ko": "조건"},
    "library.col.summary": {"en": "Result", "ko": "결과 요약"},
    "library.col.location": {"en": "Where", "ko": "위치"},
    "library.col.notes": {"en": "Notes", "ko": "비고"},
    "library.detail.kind": {"en": "Kind", "ko": "종류"},
    "library.detail.name": {"en": "Name", "ko": "이름"},
    "library.detail.created": {"en": "Created", "ko": "생성 시각"},
    "library.detail.created_by": {"en": "Made by", "ko": "생성 방법"},
    "library.detail.location": {"en": "Where", "ko": "위치"},
    "library.detail.id": {"en": "ID", "ko": "고유 ID"},
    "library.detail.path": {"en": "Path", "ko": "경로"},
    "library.detail.size": {"en": "Size", "ko": "크기"},
    "library.detail.conditions": {"en": "Conditions", "ko": "초기 조건"},
    "library.detail.summary": {"en": "Result summary", "ko": "결과 요약"},
    "library.detail.sources": {"en": "Made from", "ko": "만든 재료 (원본)"},
    "library.detail.used_by": {"en": "Used by", "ko": "이것을 사용한 항목"},
    "library.detail.related": {"en": "Related", "ko": "관련 항목"},
    "library.detail.notes": {"en": "Notes", "ko": "비고"},
    "library.detail.performance": {"en": "Performance", "ko": "성능"},
    "library.detail.none": {"en": "None", "ko": "없음"},
    "library.detail.not_stored": {"en": "(not stored here)", "ko": "(보관함에 없음)"},
    "library.detail.multi": {"en": "{count} items selected", "ko": "{count}개 선택됨"},
    "library.action.module_training_data": {"en": "Use for module training", "ko": "모듈 학습에 사용"},
    "library.action.refinement_data": {"en": "Use as refinement data", "ko": "개선 루프 데이터로 사용"},
    "library.action.resume_modules": {"en": "Continue module training", "ko": "모듈 이어 학습"},
    "library.action.main_ai_train": {"en": "Use for main AI training", "ko": "메인 AI 학습에 사용"},
    "library.action.evaluate": {"en": "Evaluate inference accuracy", "ko": "추론 정확도 평가"},
    "library.action.play": {"en": "Use in a game", "ko": "게임에서 사용"},
    "library.action.continue_ppo": {"en": "Continue PPO training", "ko": "PPO 이어 학습"},
    "library.action.refine_base": {"en": "Use as refinement base model", "ko": "추론 모듈 개선 기본 모델로"},
    "library.action.collect": {"en": "Use for match data collection", "ko": "대전 데이터 수집에 사용"},
    "library.action.view_replay": {"en": "View replay", "ko": "리플레이 보기"},
    "library.action.open_report": {"en": "Open report", "ko": "보고서 열기"},
    "library.action.edit_notes": {"en": "Edit notes...", "ko": "비고 편집..."},
    "library.action.save_copy": {"en": "Save a copy...", "ko": "복사본 저장..."},
    "library.action.open_folder": {"en": "Open folder", "ko": "폴더 열기"},
    "tooltip.library_action.module_training_data": {
        "en": "Open the training screen in module training mode with exactly the selected datasets checked.",
        "ko": "선택한 학습 데이터만 체크된 상태로 모듈 학습 화면을 엽니다.",
    },
    "tooltip.library_action.refinement_data": {
        "en": "Open the training screen in inference refinement mode with the selected datasets checked.",
        "ko": "선택한 학습 데이터를 체크한 채 추론 모듈 개선 화면을 엽니다.",
    },
    "tooltip.library_action.resume_modules": {
        "en": "Open module training and continue from this intent and emotion pair.",
        "ko": "이 의도·감정 모듈 쌍에서 이어서 학습하도록 모듈 학습 화면을 엽니다.",
    },
    "tooltip.library_action.main_ai_train": {
        "en": "Open main AI training with this module pair as the frozen inference modules.",
        "ko": "이 모듈 쌍을 고정된 추론 모듈로 쓰도록 메인 AI 학습 화면을 엽니다.",
    },
    "tooltip.library_action.evaluate": {
        "en": "Open the inference accuracy evaluation with this item preselected.",
        "ko": "이 항목을 미리 선택한 상태로 추론 정확도 평가를 엽니다.",
    },
    "tooltip.library_action.play": {
        "en": "Watch this model play, or play against it.",
        "ko": "이 모델이 플레이하는 것을 관전하거나 직접 대전합니다.",
    },
    "tooltip.library_action.continue_ppo": {
        "en": "Open main AI training and continue PPO training from this model.",
        "ko": "이 모델에서 PPO 학습을 이어 하도록 메인 AI 학습 화면을 엽니다.",
    },
    "tooltip.library_action.refine_base": {
        "en": "Open inference refinement with this model as the base. Its PPO stays fixed.",
        "ko": "이 모델을 기본 모델로 추론 모듈 개선 화면을 엽니다. PPO는 고정됩니다.",
    },
    "tooltip.library_action.collect": {
        "en": "Open data collection with this model playing against four dummy AIs.",
        "ko": "이 모델이 더미 AI 4명과 대전하는 데이터 수집 화면을 엽니다.",
    },
    "tooltip.library_action.view_replay": {
        "en": "Open this replay in the replay viewer.",
        "ko": "이 리플레이를 리플레이 화면에서 엽니다.",
    },
    "tooltip.library_action.open_report": {
        "en": "Show the report contents. You can also open it in the default app from there.",
        "ko": "보고서 내용을 보여 줍니다. 거기서 기본 앱으로도 열 수 있습니다.",
    },
    "tooltip.library_action.edit_notes": {
        "en": "Edit the free text note stored with this item.",
        "ko": "이 항목에 함께 저장된 자유 메모를 수정합니다.",
    },
    "tooltip.library_action.save_copy": {
        "en": "Copy this item to the saved folder so it can be shared through git.",
        "ko": "이 항목을 saved 폴더에 복사해 git으로 공유할 수 있게 합니다.",
    },
    "tooltip.library_action.open_folder": {
        "en": "Show the folder that holds this item in the file explorer.",
        "ko": "이 항목이 들어 있는 폴더를 파일 탐색기로 엽니다.",
    },
    "library.reason.select_first": {"en": "Select an item first.", "ko": "먼저 항목을 선택하세요."},
    "library.reason.select_one": {
        "en": "Select a single item. Only datasets can be applied together.",
        "ko": "항목을 하나만 선택하세요. 학습 데이터만 여러 개를 함께 적용할 수 있습니다.",
    },
    "library.reason.other_kind": {"en": "Not available for this kind of item.", "ko": "이 종류의 항목에는 쓸 수 없습니다."},
    "library.reason.incompatible": {"en": "This item is not compatible.", "ko": "호환되지 않는 항목입니다."},
    "library.reason.no_id": {"en": "This item has no ID.", "ko": "고유 ID가 없는 항목입니다."},
    "library.reason.no_info": {
        "en": "This old file has no info block, so notes cannot be stored.",
        "ko": "정보 블록이 없는 옛 파일이라 비고를 저장할 수 없습니다.",
    },
    "library.reason.no_copy_plan": {
        "en": "Copies can be saved for datasets, module pairs and main AI models.",
        "ko": "복사본은 학습 데이터, 모듈 쌍, 메인 AI 모델만 저장할 수 있습니다.",
    },
    "library.reason.already_saved": {"en": "This item is already in the saved folder.", "ko": "이미 saved 폴더에 있는 항목입니다."},
    "library.busy": {
        "en": "A training job is running, so nothing can be applied to the setup now. Try again when it has finished.",
        "ko": "학습이 진행 중이라 지금은 설정에 적용할 수 없습니다. 끝난 뒤 다시 시도하세요.",
    },
    "library.yes": {"en": "yes", "ko": "예"},
    "library.no": {"en": "no", "ko": "아니오"},
    "library.players.dummy5": {"en": "Dummy AI x5", "ko": "더미 AI 5인"},
    "library.players.main1_dummy4": {"en": "Main AI x1 + Dummy AI x4", "ko": "메인 AI 1 + 더미 AI 4"},
    "library.players.human1_dummy4": {"en": "Human + Dummy AI x4", "ko": "사람 + 더미 AI 4"},
    "library.players.human1_random4": {"en": "Human + Random AI x4", "ko": "사람 + 랜덤 AI 4"},
    "library.players.human1_main1_dummy3": {"en": "Human + Main AI + Dummy AI x3", "ko": "사람 + 메인 AI + 더미 AI 3"},
    "library.players.random5": {"en": "Random AI x5", "ko": "랜덤 AI 5인"},
    "library.fact.players": {"en": "{value}", "ko": "{value}"},
    "library.fact.games": {"en": "{value} games", "ko": "{value}판"},
    "library.fact.action_mode": {"en": "actions: {value}", "ko": "행동: {value}"},
    "library.fact.rows": {"en": "{value} rows", "ko": "{value}행"},
    "library.fact.win_rate": {"en": "win rate {value}", "ko": "승률 {value}"},
    "library.fact.truncated": {"en": "{value} truncated", "ko": "중단 {value}판"},
    "library.fact.epochs": {"en": "{value} epochs", "ko": "{value} 에폭"},
    "library.fact.datasets": {"en": "{value} datasets", "ko": "데이터 {value}개"},
    "library.fact.parts": {"en": "{value}", "ko": "{value}"},
    "library.fact.intent_accuracy": {"en": "intent acc {value}", "ko": "의도 정확도 {value}"},
    "library.fact.emotion_accuracy": {"en": "emotion acc {value}", "ko": "감정 정확도 {value}"},
    "library.fact.emotion_mse": {"en": "emotion MSE {value}", "ko": "감정 MSE {value}"},
    "library.fact.main_ai_ready.ok": {"en": "usable for main AI", "ko": "메인 AI 학습 가능"},
    "library.fact.main_ai_ready.no": {"en": "not usable for main AI", "ko": "메인 AI 학습 불가"},
    "library.fact.base_model": {"en": "base {value}", "ko": "기본 모델 {value}"},
    "library.fact.score": {"en": "score {value}", "ko": "점수 {value}"},
    "library.fact.inference_accuracy": {"en": "inference acc {value}", "ko": "추론 정확도 {value}"},
    "library.fact.workflow": {"en": "{value}", "ko": "{value}"},
    "library.fact.status": {"en": "{value}", "ko": "{value}"},
    "library.fact.mode": {"en": "{value}", "ko": "{value}"},
    "library.fact.outcome": {"en": "{value}", "ko": "{value}"},
    "library.fact.winner": {"en": "winner {value}", "ko": "승리 {value}"},
    "library.fact.human_won": {"en": "human won: {value}", "ko": "사람 승리: {value}"},
    "library.apply_unknown_mode": {"en": "Unknown training mode: {mode}", "ko": "알 수 없는 학습 모드: {mode}"},
    "library.apply_dataset_missing": {"en": "Dataset not found: {path}", "ko": "학습 데이터를 찾을 수 없습니다: {path}"},
    "library.apply_module_unusable": {
        "en": "Module run {job} cannot be used in this mode.",
        "ko": "모듈 {job}은(는) 이 모드에서 쓸 수 없습니다.",
    },
    "library.apply_model_missing": {"en": "Model file not found: {path}", "ko": "모델 파일을 찾을 수 없습니다: {path}"},
    "library.apply_model_unusable": {"en": "Model cannot be used here: {path}", "ko": "이 모델은 여기서 쓸 수 없습니다: {path}"},
    "library.applied_datasets": {"en": "{count} dataset(s) checked", "ko": "학습 데이터 {count}개 체크"},
    "library.applied_module": {"en": "module run {job} selected", "ko": "모듈 {job} 선택"},
    "library.applied_continue": {"en": "continue from {name}", "ko": "{name}에서 이어 학습"},
    "library.applied_base": {"en": "model {name} selected", "ko": "모델 {name} 선택"},
    "library.applied": {"en": "Applied from the library. {mode}: {details}", "ko": "보관함에서 적용했습니다. {mode}: {details}"},
    "library.play_title": {"en": "Use in a game", "ko": "게임에서 사용"},
    "library.play_prompt": {"en": "How do you want to use {name}?", "ko": "{name}을(를) 어떻게 사용할까요?"},
    "library.play_spectate": {"en": "Watch (Main AI + Dummy AI x4)", "ko": "관전 (메인 AI 1 + 더미 AI 4)"},
    "library.play_human": {"en": "Play against it (Human + Main AI + Dummy AI x3)", "ko": "직접 대전 (사람 + 메인 AI + 더미 AI 3)"},
    "library.play_incompatible": {"en": "{name} is not compatible with the game.", "ko": "{name}은(는) 게임과 호환되지 않습니다."},
    "library.open_external": {"en": "Open in default app", "ko": "기본 앱으로 열기"},
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
