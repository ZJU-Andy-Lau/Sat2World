from __future__ import annotations

import errno
import os
from types import SimpleNamespace

import pytest

from engine.tensorboard_vis import TensorBoardMonitor


class _FakeWriter:
    def __init__(self, *, add_scalar_exc: Exception | None = None, flush_exc: Exception | None = None, close_exc: Exception | None = None):
        self.add_scalar_exc = add_scalar_exc
        self.flush_exc = flush_exc
        self.close_exc = close_exc
        self.add_scalar_calls = 0
        self.flush_calls = 0
        self.close_calls = 0

    def add_scalar(self, tag, value, step):
        self.add_scalar_calls += 1
        if self.add_scalar_exc is not None:
            raise self.add_scalar_exc

    def flush(self):
        self.flush_calls += 1
        if self.flush_exc is not None:
            raise self.flush_exc

    def close(self):
        self.close_calls += 1
        if self.close_exc is not None:
            raise self.close_exc


def _ok_statvfs():
    return SimpleNamespace(f_bavail=1024 * 1024, f_frsize=1024)


def test_init_writer_enospc_does_not_crash(monkeypatch, tmp_path):
    def _raise_writer(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr("engine.tensorboard_vis.SummaryWriter", _raise_writer)
    monitor = TensorBoardMonitor(log_dir=str(tmp_path / "tb"), is_enabled=True)
    assert monitor.is_enabled is True
    assert monitor.writer is None


def test_add_scalar_edquot_closes_writer_but_not_disable(monkeypatch, tmp_path):
    writer = _FakeWriter(add_scalar_exc=OSError(errno.EDQUOT, "Disk quota exceeded"))
    monkeypatch.setattr("engine.tensorboard_vis.SummaryWriter", lambda *args, **kwargs: writer)
    monkeypatch.setattr(os, "statvfs", lambda *_: _ok_statvfs())

    monitor = TensorBoardMonitor(log_dir=str(tmp_path / "tb"), is_enabled=True)
    monitor.log_scalars("train", {"loss_total": 1.0}, global_step=1)

    assert writer.add_scalar_calls == 1
    assert monitor.writer is None
    assert monitor.is_enabled is True
    assert monitor._writer_closed_by_disk_error is True


def test_low_disk_skip_and_warning_throttle(monkeypatch, tmp_path):
    writer = _FakeWriter()
    monkeypatch.setattr("engine.tensorboard_vis.SummaryWriter", lambda *args, **kwargs: writer)
    monkeypatch.setattr(os, "statvfs", lambda *_: SimpleNamespace(f_bavail=10, f_frsize=1024))

    monitor = TensorBoardMonitor(
        log_dir=str(tmp_path / "tb"),
        is_enabled=True,
        min_free_mb=100.0,
        disk_check_interval_sec=0.0,
        low_disk_warn_interval_sec=60.0,
    )
    with pytest.warns(RuntimeWarning) as rec:
        monitor.log_scalars("train", {"loss_total": 1.0}, global_step=1)
        monitor.log_scalars("train", {"loss_total": 2.0}, global_step=2)
    low_disk_msgs = [str(w.message) for w in rec if "low disk space" in str(w.message)]
    assert len(low_disk_msgs) == 1
    assert writer.add_scalar_calls == 0


def test_reopen_after_disk_recovery(monkeypatch, tmp_path):
    writer1 = _FakeWriter(add_scalar_exc=OSError(errno.ENOSPC, "No space left on device"))
    writer2 = _FakeWriter()
    writers = iter([writer1, writer2])
    monkeypatch.setattr("engine.tensorboard_vis.SummaryWriter", lambda *args, **kwargs: next(writers))
    monkeypatch.setattr(os, "statvfs", lambda *_: _ok_statvfs())

    monitor = TensorBoardMonitor(log_dir=str(tmp_path / "tb"), is_enabled=True, reopen_interval_sec=0.0, disk_check_interval_sec=0.0)
    monitor.log_scalars("train", {"loss_total": 1.0}, global_step=1)
    assert monitor.writer is None
    monitor.log_scalars("train", {"loss_total": 2.0}, global_step=2)
    assert monitor.writer is writer2
    assert writer2.add_scalar_calls == 1


def test_flush_and_close_disk_errors_do_not_raise(monkeypatch, tmp_path):
    writer = _FakeWriter(flush_exc=OSError(errno.ENOSPC, "No space left on device"), close_exc=OSError(errno.EDQUOT, "Disk quota exceeded"))
    monkeypatch.setattr("engine.tensorboard_vis.SummaryWriter", lambda *args, **kwargs: writer)
    monkeypatch.setattr(os, "statvfs", lambda *_: _ok_statvfs())

    monitor = TensorBoardMonitor(log_dir=str(tmp_path / "tb"), is_enabled=True)
    monitor.flush()
    monitor.close()
    assert monitor.writer is None
