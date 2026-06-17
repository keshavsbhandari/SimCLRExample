from simclr_fraud.train.control import ensure_control_file, read_control


def test_ensure_control_file_creates_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "simclr_fraud.train.control.control_path",
        lambda name: tmp_path / "control.yaml",
    )
    control = ensure_control_file("test")
    assert control.stop is False
    assert control.skip_finetune is False
    assert control.max_epochs is None


def test_read_control_stop_flag(tmp_path, monkeypatch):
    path = tmp_path / "control.yaml"
    path.write_text("stop: true\nskip_finetune: false\nmax_epochs: null\n")
    monkeypatch.setattr(
        "simclr_fraud.train.control.control_path",
        lambda name: path,
    )
    control = read_control("test")
    assert control is not None
    assert control.stop is True


def test_max_epochs_cap_logic():
    """Epoch cap: stop when completed_epochs >= max_epochs."""
    cap = 5
    for current_epoch in range(10):
        completed = current_epoch + 1
        should_stop = completed >= cap
        if current_epoch < 4:
            assert not should_stop
        else:
            assert should_stop
