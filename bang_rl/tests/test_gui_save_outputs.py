"""Qt checks for the Save outputs button/dialog and note display (real QApplication)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from bang_rl.gui.qt_compat import QApplication, QT_AVAILABLE

pytestmark = pytest.mark.skipif(not QT_AVAILABLE, reason="PyQt6 is not installed")

_APP = QApplication.instance() or QApplication([]) if QT_AVAILABLE else None

from bang_rl.artifact_info import KIND_DATASET, new_info, write_info  # noqa: E402
from bang_rl.gui.dialogs.module_performance import (  # noqa: E402
    ModulePerformanceDialog,
    main_ai_model_label,
    module_run_label,
)
from bang_rl.saved_artifacts import plan_dataset  # noqa: E402
from bang_rl.training.module_artifacts import SavedModuleRun  # noqa: E402

MIB = 1024 * 1024


def _dataset(tmp_path: Path) -> Path:
    folder = tmp_path / "ds"
    folder.mkdir()
    (folder / "shard-0000.parquet").write_bytes(b"x" * 10)
    write_info(
        folder,
        new_info(KIND_DATASET, id="3f9a1c77", created_by="data_collection", conditions={"players": "dummy5", "games": 5}),
    )
    return folder


def test_training_monitor_enables_save_only_when_outputs_exist(tmp_path: Path) -> None:
    from bang_rl.gui.screens.training_monitor import TrainingMonitor

    monitor = TrainingMonitor()
    assert not monitor.save_button.isEnabled()
    assert monitor.save_button.toolTip() and "—" not in monitor.save_button.toolTip()
    folder = _dataset(tmp_path)
    running = SimpleNamespace(status="running", output_path=str(folder), artifact_path=None, report_path=None, metrics={})
    monitor._update_save_button(running, "data_collection")
    assert not monitor.save_button.isEnabled()
    done = SimpleNamespace(
        status="finished", output_path=str(folder), artifact_path=None, report_path=None, metrics={"dataset_path": str(folder)}
    )
    monitor._update_save_button(done, "data_collection")
    assert monitor.save_button.isEnabled() and len(monitor._save_plans) == 1
    empty = SimpleNamespace(status="cancelled", output_path=None, artifact_path=None, report_path=None, metrics={})
    monitor._update_save_button(empty, "module_training")
    assert not monitor.save_button.isEnabled()
    monitor.deleteLater()


def test_save_dialog_prefills_rule_name_and_shows_size_warnings(tmp_path: Path) -> None:
    from bang_rl.gui.dialogs.save_outputs import SaveOutputsDialog

    plan = plan_dataset(_dataset(tmp_path), saved_root=tmp_path / "saved")
    dialog = SaveOutputsDialog(plan)
    assert dialog.name_edit.text() == plan.default_name and "_dataset_dummy5_5g_" in plan.default_name
    assert dialog.warning.isHidden() and "shard-0000.parquet" in dialog.files.toPlainText()
    plan.size_of = lambda path: 120 * MIB if path.suffix == ".parquet" else 1
    strong = SaveOutputsDialog(plan)
    assert not strong.warning.isHidden() and "GitHub" in strong.warning.text() and "100" in strong.warning.text()
    # Saving with the name and notes typed in the dialog (confirmation is stubbed).
    strong._confirm = lambda text: True  # type: ignore[method-assign]
    strong.name_edit.setText("더미끼리 5판")
    strong.notes_edit.setPlainText("테스트 메모")
    strong._save()
    assert (tmp_path / "saved" / "datasets" / "더미끼리 5판" / "shard-0000.parquet").is_file()
    clash = SaveOutputsDialog(plan)
    clash.name_edit.setText("더미끼리 5판")
    clash._confirm = lambda text: True  # type: ignore[method-assign]
    clash._save()
    assert "더미끼리 5판" in clash.message.text()
    dialog.deleteLater()
    strong.deleteLater()
    clash.deleteLater()


def test_performance_dialog_shows_names_and_notes() -> None:
    run = SavedModuleRun(
        job_id="b8ecacce", intent_path=Path("i"), emotion_path=Path("e"), modified=1_700_000_000.0,
        name="20261110-0252_modules_e5_b8ecac", notes="첫 번째 실험 메모입니다 아주 길게 써 보는 비고 문장",
    )
    label = module_run_label(run)
    assert "첫 번째" in label and "…" in label
    assert run.tooltip.splitlines()[0] == run.name and "첫 번째" in run.tooltip
    model = SimpleNamespace(modified_text="2026-11-16 09:10", short_id="c81e55", notes="좋음")
    assert main_ai_model_label(model).endswith("좋음")

    dialog = ModulePerformanceDialog(
        list_runs=lambda _root: [run],
        summarize=lambda r, reports_dir=None: {"job_id": r.job_id, "name": r.name, "notes": r.notes, "modified": "x", "intent": None, "emotion": None},
        list_main_ai=lambda _dirs: [],
        main_ai_dirs=[],
    )
    assert dialog.list.item(0).toolTip().startswith(run.name)
    assert run.name in dialog.details.toPlainText() and "첫 번째" in dialog.details.toPlainText()
    assert dialog.notes_button.isEnabled() and dialog.save_copy_button.isEnabled()
    dialog.deleteLater()
