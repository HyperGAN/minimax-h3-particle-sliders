"""FOUR new, UNRUN software-only cases for the sibling runner boundary.

Synthetic owners/sessions only; no Torch, native/model load, old proof or cache.
This explicit filename is outside default pytest test_*.py discovery.
"""
import os
import fcntl
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from minimax_h3_runner95 import ownership, api


class RunnerBoundary(unittest.TestCase):
    def test_actual_held_flock_modes_and_identity(self):
        # Genuinely new owned temp resources only; no GPU/global policy lease.
        with tempfile.TemporaryDirectory(prefix='h3-runner-new-lease-case-') as directory:
            path = Path(directory) / 'owned.lock'
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                ownership.HeldLease(str(path), fd, 'SH').check()
                with self.assertRaises(RuntimeError):
                    ownership.HeldLease(str(path), fd, 'EX').check()
                fcntl.flock(fd, fcntl.LOCK_UN)
                with self.assertRaises(RuntimeError):
                    ownership.HeldLease(str(path), fd, 'SH').check()
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                ownership.HeldLease(str(path), fd, 'EX').check()
                with self.assertRaises(RuntimeError):
                    ownership.HeldLease(str(path), fd, 'SH').check()
                # Same pathname now names a different inode; held old FD stays real.
                replacement = Path(directory) / 'replacement.lock'
                replacement.write_text('new owned inode')
                os.replace(replacement, path)
                with self.assertRaises(RuntimeError):
                    ownership.HeldLease(str(path), fd, 'EX').check()
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)

    def test_policy_refusal_is_not_an_admission(self):
        owner = ownership.RunnerAdmission('synthetic', ownership.UUIDS, (), None, lambda caps, identity: False)
        with patch.object(ownership.RunnerAdmission, 'structural_check'):
            with self.assertRaises(RuntimeError):
                owner.check()

    def test_foreign_marker_refuses_before_close_and_retry_is_pending_only(self):
        owner = ownership.RunnerAdmission('synthetic', ownership.UUIDS, (), None, Mock(return_value=True))
        runner = api.Runner(owner)
        runner._entered = runner._owns_lock = True
        runner._identity = Mock()
        native = Mock()
        session = api._Training(runner, native)
        runner._sessions = [session]
        self.assertTrue(api._PROCESS_SCOPE.acquire(blocking=False))
        try:
            with patch.dict(os.environ, {api._FLAG: 'foreign', 'CUDA_VISIBLE_DEVICES': ','.join(ownership.UUIDS)}), patch.object(ownership.RunnerAdmission, 'structural_check'):
                with self.assertRaises(RuntimeError):
                    runner.close()
                native.close.assert_not_called()
                self.assertIs(runner._sessions[0], session)
                os.environ[api._FLAG] = '1'
                native.close.side_effect = [RuntimeError('synthetic cleanup failure'), None]
                with self.assertRaises(BaseExceptionGroup):
                    runner.close()
                self.assertIn(api._FLAG, os.environ)
                self.assertIs(runner._sessions[0], session)
                runner.close()
                self.assertNotIn(api._FLAG, os.environ)
                self.assertEqual(native.close.call_count, 2)
                self.assertEqual(runner._sessions, [])
        finally:
            if runner._owns_lock:
                runner._owns_lock = False
                api._PROCESS_SCOPE.release()

    def test_official_train_and_step_delegate_once(self):
        runner = Mock()
        native = Mock()
        session = api._Training(runner, native)
        session.step()
        native.step.assert_called_once_with()
        session.train(stop_after=2)
        native.train.assert_called_once()
        kwargs = native.train.call_args.kwargs
        self.assertEqual(kwargs['stop_after'], 2)
        kwargs['on_step']({'step': 1})
        self.assertGreaterEqual(runner.check.call_count, 4)
