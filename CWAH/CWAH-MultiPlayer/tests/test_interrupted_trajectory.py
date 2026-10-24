"""Interrupt and checkpoint checks without Unity or API calls."""
import os
import pickle
import signal
import tempfile
import unittest
from types import SimpleNamespace

from algos.arena_mp2 import ArenaMP


class InterruptedTrajectoryTests(unittest.TestCase):
    def test_sigint_requests_stop_and_restores_handler(self):
        arena = ArenaMP.__new__(ArenaMP)
        original = signal.getsignal(signal.SIGINT)

        def episode(*args):
            os.kill(os.getpid(), signal.SIGINT)
            self.assertEqual(arena._stop_signal, signal.SIGINT)
            return False, 7, {'interrupted': True}

        arena._run_episode = episode
        result = arena.run()
        self.assertEqual(result[1], 7)
        self.assertEqual(signal.getsignal(signal.SIGINT), original)

    def test_checkpoint_retains_last_complete_actions_and_trace(self):
        arena = ArenaMP.__new__(ArenaMP)
        arena.env = SimpleNamespace(continuous_subtasks_enabled=False)
        state = SimpleNamespace(to_dict=lambda: {'intent': 'unknown'})
        trace = SimpleNamespace(
            ledger=SimpleNamespace(events=[]), update_log=[], state=state,
            archive_fields=lambda: {'final_belief': state.to_dict(),
                                   'last_decision_belief': state.to_dict()},
        )
        arena.agents = [SimpleNamespace(trace=trace)]
        with tempfile.TemporaryDirectory() as directory:
            arena.record_dir = directory
            arena._save_checkpoint({'action': {0: ['grab'], 1: [None]},
                                    'termination_reason': None})
            with open(os.path.join(directory, 'trajectory_checkpoint.pik'), 'rb') as stream:
                saved = pickle.load(stream)
            self.assertEqual(saved['environment_steps'], 1)
            self.assertEqual(saved['action'][0], ['grab'])
            self.assertEqual(saved['trace']['final_belief'], {'intent': 'unknown'})
            self.assertFalse(os.path.exists(os.path.join(directory, 'trajectory_checkpoint.pik.tmp')))
