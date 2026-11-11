from __future__ import annotations

from bang_rl.config import ConfigLoader, HardwareConfig, SystemConfig


def test_local_settings_round_trip_preserves_hardware(tmp_path) -> None:
    path = tmp_path / "settings.json"
    source = SystemConfig(
        hardware=HardwareConfig(
            cpu_cores=12,
            ram_gb=16,
            use_gpu=True,
            workers=4,
            batch_size=128,
            buffer_size=4096,
        )
    )

    ConfigLoader().save_local(source, path)
    restored = ConfigLoader().load_local(path)

    assert restored.hardware == source.hardware


def test_local_settings_overlay_preserves_new_base_values(tmp_path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        '{"hardware": {"cpu_cores": 12, "ram_gb": 16, "use_gpu": true}}',
        encoding="utf-8",
    )
    base = SystemConfig(seed=91, hardware=HardwareConfig(workers=3, batch_size=32))

    restored = ConfigLoader().load_local(path, base=base)

    assert restored.seed == 91
    assert restored.hardware.cpu_cores == 12
    assert restored.hardware.ram_gb == 16
    assert restored.hardware.use_gpu is True
    assert restored.hardware.workers == 3
    assert restored.hardware.batch_size == 32


def test_missing_local_settings_use_base_defaults(tmp_path) -> None:
    base = SystemConfig(seed=17, hardware=HardwareConfig(cpu_cores=8, ram_gb=32))

    restored = ConfigLoader().load_local(tmp_path / "missing.json", base=base)

    assert restored.seed == 17
    assert restored.hardware.cpu_cores == 8
    assert restored.hardware.ram_gb == 32


def test_outdated_local_schema_keeps_valid_hardware_values(tmp_path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        '{"removed_future_field": true, "hardware": {"cpu_cores": 12, "ram_gb": 16}}',
        encoding="utf-8",
    )
    base = SystemConfig(hardware=HardwareConfig(workers=2))

    restored = ConfigLoader().load_local(path, base=base)

    assert restored.hardware.cpu_cores == 12
    assert restored.hardware.ram_gb == 16
    assert restored.hardware.workers == 2
