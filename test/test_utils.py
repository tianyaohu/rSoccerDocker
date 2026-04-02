"""
Tests for src/utils.py — algo registry, env/model builders, checkpoint callback.

Run: pytest tests/test_utils.py -v
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils import (
    ALGOS,
    OFF_POLICY,
    RotatingCheckpointCallback,
    build_model,
    create_run_dir,
    make_env,
)


# ===========================================================================
# Algo registry
# ===========================================================================

class TestAlgoRegistry:

    def test_all_algos_present(self):
        assert set(ALGOS.keys()) == {"sac", "ddpg", "ppo"}

    def test_off_policy_set(self):
        assert OFF_POLICY == {"sac", "ddpg"}
        assert "ppo" not in OFF_POLICY


# ===========================================================================
# create_run_dir
# ===========================================================================

class TestCreateRunDir:

    def test_creates_directory(self, tmp_path):
        run_dir = create_run_dir(tmp_path, "sac", "SSLDribbling-v0", 0)
        assert run_dir.exists()
        assert run_dir.is_dir()

    def test_correct_structure(self, tmp_path):
        run_dir = create_run_dir(tmp_path, "sac", "SSLDribbling-v0", 0)
        assert run_dir.parent.name == "SAC"
        assert run_dir.parent.parent.name == "experiments"
        assert "SSLDribbling-v0" in run_dir.name
        assert "seed0" in run_dir.name

    def test_algo_uppercased(self, tmp_path):
        for algo in ["sac", "ddpg", "ppo"]:
            run_dir = create_run_dir(tmp_path, algo, "VSS-v0", 0)
            assert run_dir.parent.name == algo.upper()

    def test_different_seeds(self, tmp_path):
        r0 = create_run_dir(tmp_path, "sac", "VSS-v0", 0)
        r1 = create_run_dir(tmp_path, "sac", "VSS-v0", 1)
        assert "seed0" in r0.name
        assert "seed1" in r1.name
        assert r0 != r1


# ===========================================================================
# make_env
# ===========================================================================

class TestMakeEnv:

    @patch("utils.Monitor")
    @patch("utils.gym.make")
    def test_off_policy_single_env(self, mock_gym_make, mock_monitor):
        mock_env = MagicMock()
        mock_gym_make.return_value = mock_env

        for algo in ["sac", "ddpg"]:
            make_env("SSLDribbling-v0", algo, n_envs=4, seed=0)
            mock_gym_make.assert_called_with("SSLDribbling-v0")
            mock_monitor.assert_called_with(mock_env)

    @patch("utils.make_vec_env")
    def test_ppo_vec_env(self, mock_vec):
        make_env("VSS-v0", "ppo", n_envs=8, seed=42)
        mock_vec.assert_called_with("VSS-v0", n_envs=8, seed=42)

    @patch("utils.make_vec_env")
    def test_ppo_passes_n_envs(self, mock_vec):
        for n in [1, 2, 4, 16]:
            make_env("VSS-v0", "ppo", n_envs=n, seed=0)
            assert mock_vec.call_args[1]["n_envs"] == n


# ===========================================================================
# build_model
# ===========================================================================

class TestBuildModel:

    def test_fresh_creates_with_correct_args(self):
        for algo_name in ["sac", "ddpg", "ppo"]:
            mock_class = MagicMock()
            mock_env = MagicMock()

            with patch.dict("utils.ALGOS", {algo_name: mock_class}):
                build_model(algo_name, mock_env, Path("/tmp/logs"), seed=42, resume=None)

            mock_class.assert_called_once_with(
                "MlpPolicy", mock_env, verbose=1, seed=42,
                tensorboard_log="/tmp/logs"
            )

    def test_resume_calls_load(self):
        mock_class = MagicMock()
        mock_env = MagicMock()

        with patch.dict("utils.ALGOS", {"sac": mock_class}):
            build_model("sac", mock_env, Path("/tmp/logs"), seed=42,
                        resume="/path/to/ckpt.zip")

        mock_class.load.assert_called_once_with(
            "/path/to/ckpt.zip", env=mock_env, tensorboard_log="/tmp/logs"
        )
        # Should NOT create a fresh model
        mock_class.assert_not_called()

    def test_fresh_does_not_call_load(self):
        mock_class = MagicMock()
        with patch.dict("utils.ALGOS", {"sac": mock_class}):
            build_model("sac", MagicMock(), Path("/tmp"), seed=0, resume=None)
        mock_class.load.assert_not_called()

    def test_seed_is_passed_through(self):
        mock_class = MagicMock()
        with patch.dict("utils.ALGOS", {"ddpg": mock_class}):
            build_model("ddpg", MagicMock(), Path("/tmp"), seed=99, resume=None)
        _, kwargs = mock_class.call_args
        assert kwargs["seed"] == 99


# ===========================================================================
# RotatingCheckpointCallback
# ===========================================================================

class TestRotatingCheckpointCallback:

    def test_no_save_before_freq(self, tmp_path):
        cb = RotatingCheckpointCallback(
            save_freq=100, save_dir=tmp_path,
            name_prefix="sac_test", max_keep=5, verbose=0,
        )
        cb.model = MagicMock()
        cb.num_timesteps = 50
        cb.n_calls = 50
        cb._on_step()
        cb.model.save.assert_not_called()

    def test_saves_at_freq(self, tmp_path):
        cb = RotatingCheckpointCallback(
            save_freq=100, save_dir=tmp_path,
            name_prefix="sac_test", max_keep=5, verbose=0,
        )
        cb.model = MagicMock()
        cb.num_timesteps = 100
        cb.n_calls = 100
        cb._on_step()
        cb.model.save.assert_called_once()
        # Verify the path includes the step number
        saved_path = cb.model.save.call_args[0][0]
        assert "00000100" in saved_path

    def test_rotates_oldest(self, tmp_path):
        cb = RotatingCheckpointCallback(
            save_freq=1, save_dir=tmp_path,
            name_prefix="sac_test", max_keep=2, verbose=0,
        )
        cb.model = MagicMock()

        # Create 2 existing (at limit)
        (tmp_path / "sac_test_00000001.zip").write_bytes(b"fake")
        (tmp_path / "sac_test_00000002.zip").write_bytes(b"fake")

        cb.n_calls = 1
        cb.num_timesteps = 3
        cb._on_step()

        assert not (tmp_path / "sac_test_00000001.zip").exists(), "oldest should be deleted"
        assert (tmp_path / "sac_test_00000002.zip").exists(), "second should remain"

    def test_under_limit_no_delete(self, tmp_path):
        cb = RotatingCheckpointCallback(
            save_freq=1, save_dir=tmp_path,
            name_prefix="sac_test", max_keep=5, verbose=0,
        )
        cb.model = MagicMock()

        (tmp_path / "sac_test_00000001.zip").write_bytes(b"fake")

        cb.n_calls = 1
        cb.num_timesteps = 2
        cb._on_step()

        assert (tmp_path / "sac_test_00000001.zip").exists()

    def test_creates_save_dir(self, tmp_path):
        save_dir = tmp_path / "nested" / "checkpoints"
        cb = RotatingCheckpointCallback(
            save_freq=1, save_dir=save_dir,
            name_prefix="test", max_keep=5, verbose=0,
        )
        assert save_dir.exists()