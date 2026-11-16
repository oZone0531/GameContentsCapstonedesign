"""Emotion-free, role-balanced intent training UI with warm-start continuation.

GUI strings follow the persisted language setting from gui.i18n. The training
worker never touches Qt widgets; all UI updates occur through queued events.
"""
from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Callable

from ..i18n import tr
from ..qt_compat import (
    QFileDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QSpinBox, QTimer, QVBoxLayout, QWidget, QT_AVAILABLE, require_qt,
)

if QT_AVAILABLE:

    class IntentTrainingScreen(QWidget):
        def __init__(self, *, on_back: Callable[[], None],
                     rule_policy_config: object = None, parent: QWidget | None = None) -> None:
            require_qt()
            super().__init__(parent)
            self._on_back = on_back
            self._rule_policy_config = rule_policy_config
            self._events: queue.Queue[dict] = queue.Queue()
            self._stop = threading.Event()
            self._worker: threading.Thread | None = None
            self._trained_model: Path | None = None
            layout = QVBoxLayout(self)
            title = QLabel(tr("intent.title"), self)
            title.setStyleSheet("font-size: 22pt; font-weight: bold;")
            layout.addWidget(title)
            subtitle = QLabel(tr("intent.subtitle"), self)
            subtitle.setWordWrap(True)
            layout.addWidget(subtitle)

            form = QFormLayout()
            self.episodes = QSpinBox(self)
            self.episodes.setRange(1, 1_000_000)
            self.episodes.setValue(5000)
            self.episodes.setSingleStep(100)
            form.addRow(tr("intent.games"), self.episodes)
            self.steps = QSpinBox(self)
            self.steps.setRange(2, 2000)
            self.steps.setValue(200)
            form.addRow(tr("intent.steps"), self.steps)
            self.seed = QSpinBox(self)
            self.seed.setRange(0, 2_147_483_000)
            self.seed.setValue(12345)
            form.addRow(tr("intent.seed"), self.seed)
            self.random_ratio = QDoubleSpinBox(self)
            self.random_ratio.setRange(0.0, 100.0)
            self.random_ratio.setSingleStep(5.0)
            self.random_ratio.setValue(35.0)
            self.random_ratio.setSuffix(" %")
            form.addRow(tr("intent.random"), self.random_ratio)
            self.batch = QSpinBox(self)
            self.batch.setRange(1, 1024)
            self.batch.setValue(16)
            form.addRow(tr("intent.batch"), self.batch)
            self.eval_games = QSpinBox(self)
            self.eval_games.setRange(1, 1000)
            self.eval_games.setValue(20)
            form.addRow(tr("intent.eval"), self.eval_games)

            self.resume_edit = QLineEdit("", self)
            self.resume_edit.setReadOnly(True)
            self.resume_edit.setPlaceholderText(tr("intent.resume_hint"))
            self.resume_browse = QPushButton(tr("intent.browse_resume"), self)
            self.resume_browse.clicked.connect(self._load_model)
            self.resume_clear = QPushButton(tr("intent.clear_resume"), self)
            self.resume_clear.clicked.connect(self._clear_resume)
            resume_row = QHBoxLayout()
            resume_row.addWidget(self.resume_edit)
            resume_row.addWidget(self.resume_browse)
            resume_row.addWidget(self.resume_clear)
            form.addRow(tr("intent.resume"), resume_row)

            self.path_edit = QLineEdit("bang_rl/models/intent/trained.intentmodel", self)
            browse = QPushButton(tr("intent.browse_output"), self)
            browse.clicked.connect(self._browse)
            output_row = QHBoxLayout()
            output_row.addWidget(self.path_edit)
            output_row.addWidget(browse)
            form.addRow(tr("intent.output"), output_row)
            layout.addLayout(form)
            self.progress = QProgressBar(self)
            self.progress.setRange(0, 100)
            layout.addWidget(self.progress)
            self.status = QLabel(tr("intent.before"), self)
            layout.addWidget(self.status)
            self.log = QPlainTextEdit(self)
            self.log.setReadOnly(True)
            layout.addWidget(self.log, 1)
            buttons = QHBoxLayout()
            self.start_button = QPushButton(tr("intent.start"), self)
            self.start_button.clicked.connect(self._start)
            buttons.addWidget(self.start_button)
            self.cancel_button = QPushButton(tr("intent.cancel"), self)
            self.cancel_button.setEnabled(False)
            self.cancel_button.clicked.connect(self._cancel)
            buttons.addWidget(self.cancel_button)
            self.load_button = QPushButton(tr("intent.load"), self)
            self.load_button.clicked.connect(self._load_model)
            buttons.addWidget(self.load_button)
            self.explain_button = QPushButton(tr("intent.explain"), self)
            self.explain_button.setEnabled(False)
            self.explain_button.clicked.connect(self._explain_sample)
            buttons.addWidget(self.explain_button)
            self.back_button = QPushButton(tr("common.back"), self)
            self.back_button.clicked.connect(self._on_back)
            buttons.addWidget(self.back_button)
            layout.addLayout(buttons)
            self._timer = QTimer(self)
            self._timer.setInterval(100)
            self._timer.timeout.connect(self._poll)

        def _browse(self) -> None:
            selected, _ = QFileDialog.getSaveFileName(
                self, tr("intent.browse_output"), self.path_edit.text(), tr("intent.file_filter"))
            if selected:
                self.path_edit.setText(selected if selected.lower().endswith(".intentmodel") else selected + ".intentmodel")

        def _load_model(self) -> None:
            from bang_rl.intent_training import LearnedIntentPolicy
            selected, _ = QFileDialog.getOpenFileName(
                self, tr("intent.load_title"), self.resume_edit.text() or self.path_edit.text(), tr("intent.file_filter"))
            if not selected:
                return
            try:
                policy = LearnedIntentPolicy.load(selected)
                self._trained_model = Path(selected)
                self.resume_edit.setText(selected)
                self.explain_button.setEnabled(True)


                source = Path(selected)
                self.path_edit.setText(str(source.with_name(source.stem + "_continued.intentmodel")))
                self.log.appendPlainText(tr("intent.loaded", path=selected))
                prior = policy.metadata.get("cumulative_episodes_per_role", policy.metadata.get("training_config", {}).get("episodes_per_role"))
                if prior is not None:
                    self.log.appendPlainText(tr("intent.cumulative", games=prior))
            except Exception as exc:
                QMessageBox.warning(self, tr("intent.load_failed"), str(exc))

        def _clear_resume(self) -> None:
            self.resume_edit.clear()
            self._trained_model = None
            self.explain_button.setEnabled(False)

        def _start(self) -> None:
            from bang_rl.intent_training import IntentTrainingConfig, IntentTrainer
            try:
                cfg = IntentTrainingConfig(
                    episodes_per_role=self.episodes.value(),
                    max_steps=self.steps.value(),
                    seed=self.seed.value(),
                    random_opponent_probability=self.random_ratio.value() / 100,
                    batch_games=self.batch.value(),
                    eval_games_per_role=self.eval_games.value(),
                    output_path=self.path_edit.text().strip(),
                    initial_model_path=self.resume_edit.text().strip() or None,
                )
                if cfg.initial_model_path and not Path(cfg.initial_model_path).is_file():
                    raise ValueError("Input .intentmodel was not found")
            except (ValueError, TypeError) as exc:
                QMessageBox.warning(self, tr("intent.config_error"), str(exc))
                return
            self._events = queue.Queue()
            self._stop = threading.Event()
            self.progress.setValue(0)
            self.log.clear()
            self.start_button.setEnabled(False)
            self.load_button.setEnabled(False)
            self.resume_browse.setEnabled(False)
            self.resume_clear.setEnabled(False)
            self.cancel_button.setEnabled(True)
            self.explain_button.setEnabled(False)
            self.back_button.setEnabled(False)
            self.status.setText(tr("intent.baseline"))

            def worker() -> None:
                try:
                    outcome = IntentTrainer(cfg, callback=self._events.put,
                                            stop_event=self._stop,
                                            rule_policy_config=self._rule_policy_config).run()
                    self._events.put({"phase": "finished", "result": outcome})
                except InterruptedError:
                    self._events.put({"phase": "cancelled"})
                except Exception as exc:
                    self._events.put({"phase": "error", "error": f"{type(exc).__name__}: {exc}"})

            self._worker = threading.Thread(target=worker, name="intent-training", daemon=True)
            self._worker.start()
            self._timer.start()

        def _cancel(self) -> None:
            self._stop.set()
            self.cancel_button.setEnabled(False)
            self.status.setText(tr("intent.cancel_request"))

        @staticmethod
        def _rate(value: object) -> str:
            return "N/A" if value is None else f"{value:.1%}"

        def _poll(self) -> None:
            for _ in range(1000):
                try:
                    event = self._events.get_nowait()
                except queue.Empty:
                    break
                phase = event.get("phase")
                if phase == "training":
                    completed, total = event["completed"], event["total"]
                    self.progress.setValue(int(100 * completed / total))
                    roles = " / ".join(f"{k}: {v}" for k, v in event["played"].items())
                    self.status.setText(tr("intent.training", done=completed, total=total, roles=roles))
                    if completed == total or completed % max(1, total // 25) == 0:
                        self.log.appendPlainText(self.status.text())
                elif phase == "baseline":
                    self.log.appendPlainText(tr("intent.baseline"))
                elif phase == "checkpoint":
                    self.log.appendPlainText(tr("intent.checkpoint", path=event["path"]))
                elif phase == "evaluation":
                    if "results" in event:
                        r = event["results"]
                        self.log.appendPlainText(tr("intent.eval_result", role=event["role"],
                            rate=self._rate(r["win_rate"]), normal=r["normal_games"], truncated=r["truncated"]))
                    else:


                        self.status.setText(tr("intent.comparison"))
                elif phase == "done":
                    self._trained_model = Path(event["path"])
                    self.log.appendPlainText(tr("intent.saved_model", path=event["path"]))
                    self.log.appendPlainText(tr("intent.saved_report", path=event["report_path"]))
                    self.explain_button.setEnabled(True)
                elif phase == "finished":
                    result = event["result"]
                    self.log.appendPlainText(tr("intent.comparison"))
                    for role, base in result["baseline_evaluation"].items():
                        after = result["learned_evaluation"][role]
                        self.log.appendPlainText(tr("intent.compare_role", role=role,
                            before=self._rate(base["win_rate"]), after=self._rate(after["win_rate"])))
                    if result.get("initial_model_evaluation"):
                        self.log.appendPlainText("\n" + tr("intent.initial_compare"))
                        for role, before in result["initial_model_evaluation"].items():
                            after = result["learned_evaluation"][role]
                            self.log.appendPlainText(tr("intent.compare_role", role=role,
                                before=self._rate(before["win_rate"]), after=self._rate(after["win_rate"])))
                    self.log.appendPlainText("\n" + tr("intent.allfive"))
                    for label, key in ((tr("intent.allfive_rule"), "all_five_rule_evaluation"),
                                       (tr("intent.allfive_learned"), "all_five_learned_evaluation")):
                        data = result[key]
                        rates = " | ".join(f"{team}: {self._rate(rate)}"
                                           for team, rate in data["team_win_rates"].items())
                        self.log.appendPlainText(tr("intent.allfive_detail", label=label, rates=rates,
                            normal=data["normal_games"], truncated=data["truncated"]))
                    self.log.appendPlainText(tr("intent.cumulative", games=result["cumulative_episodes_per_role"]))
                    self.status.setText(tr("intent.complete"))
                    self._complete()
                elif phase == "cancelled":
                    self.status.setText(tr("intent.cancelled"))
                    self._complete()
                elif phase == "error":
                    message = tr("intent.error", error=event["error"])
                    self.status.setText(message)
                    self.log.appendPlainText(message)
                    self._complete()

        def _complete(self) -> None:
            self._timer.stop()
            self.start_button.setEnabled(True)
            self.load_button.setEnabled(True)
            self.resume_browse.setEnabled(True)
            self.resume_clear.setEnabled(True)
            self.cancel_button.setEnabled(False)
            self.back_button.setEnabled(True)

        def _explain_sample(self) -> None:
            from bang_rl.intent_training import LearnedIntentPolicy
            from bang_rl.agents.dummy_agent.role_belief import RoleBeliefEstimator
            from bang_rl.env.state import EnvironmentState
            try:
                if self._trained_model is None:
                    return
                model = LearnedIntentPolicy.load(self._trained_model)
                state = EnvironmentState.initial("sample-intent-explain", self.seed.value())
                viewer = state.sheriff_idx
                observation = state.to_public(viewer)
                beliefs = RoleBeliefEstimator(player_idx=viewer).update(observation)
                mask = observation["action_mask"]
                explanation = model.explain(observation, beliefs, mask)
                names = ["PASS"] + [f"BANG P{i}" for i in range(5)]
                probabilities = "\n".join(
                    f"{names[i]}: {p:.1%}" for i, p in enumerate(explanation["probabilities"]) if mask[i])
                influences = "\n".join(
                    f"{item['feature']}: {item['gradient_times_input']:+.3f}"
                    for item in explanation["local_contributions"])
                QMessageBox.information(self, tr("intent.analysis_title"),
                    tr("intent.analysis_action", action=names[explanation["chosen_action"]]) + "\n\n" +
                    tr("intent.analysis_probabilities") + ":\n" + probabilities + "\n\n" +
                    tr("intent.analysis_contributions") + ":\n" + influences + "\n\n" +
                    tr("intent.analysis_note"))
            except Exception as exc:
                QMessageBox.warning(self, tr("intent.analysis_failed"), str(exc))

        def closeEvent(self, event: object) -> None:
            self._stop.set()
            super().closeEvent(event)

else:

    class IntentTrainingScreen:
        def __init__(self, *args: object, **kwargs: object) -> None:
            require_qt()


__all__ = ["IntentTrainingScreen"]
