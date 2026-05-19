"""
Tests for src/train.py — CLI parsing, resume wiring, end-to-end main().

Run: pytest tests/test_train.py -v
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from train import parse_args, resolve_device, resolve_resume


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_COMMAND = (
    "python src/train.py --env SSLDribbling-v0 --algo sac --seed 0 "
    "--timesteps 10000000 --save-freq 100000 --max-checkpoints 5 --n-envs 4\n"
)


def make_run(experiments: Path, algo: str, name: str,
             command_txt: str = SAMPLE_COMMAND,
             checkpoints: list[str] | None = None,
             final: bool = False) -> Path:
    run_dir = experiments / algo.upper() / name
    run_dir.mkdir(parents=True, exist_ok=True)
    if command_txt:
        (run_dir / "command.txt").write_text(command_txt)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    for ckpt in (checkpoints or []):
        (ckpt_dir / ckpt).write_bytes(b"fake")
    if final:
        (ckpt_dir / "sac_ssldribbling_v0_final.zip").write_bytes(b"fake")
    return run_dir


# ===========================================================================
# parse_args
# ===========================================================================

class TestParseArgs:

    def test_defaults(self):
        args = parse_args([])
        assert args.env is None
        assert args.algo is None
        assert args.timesteps == 500_000
        assert args.save_freq == 100_000
        assert args.max_checkpoints == 5
        assert args.n_envs == 4
        assert args.seed == 0
        assert args.device == "auto"
        assert args.resume is None

    def test_all_explicit(self):
        args = parse_args([
            "--env", "VSS-v0", "--algo", "ppo",
            "--timesteps", "10000000", "--save-freq", "200000",
            "--max-checkpoints", "3", "--n-envs", "8", "--seed", "42",
            "--device", "cuda",
        ])
        assert args.env == "VSS-v0"
        assert args.algo == "ppo"
        assert args.timesteps == 10_000_000
        assert args.save_freq == 200_000
        assert args.max_checkpoints == 3
        assert args.n_envs == 8
        assert args.seed == 42
        assert args.device == "cuda"

    @pytest.mark.parametrize("device", ["auto", "cpu", "cuda"])
    def test_device_choices(self, device):
        assert parse_args(["--device", device]).device == device

    def test_resume_latest(self):
        assert parse_args(["--resume", "latest"]).resume == "latest"

    def test_resume_explicit_name(self):
        assert parse_args(["--resume", "some_run"]).resume == "some_run"

    def test_invalid_algo_exits(self):
        with pytest.raises(SystemExit):
            parse_args(["--algo", "td3"])

    def test_invalid_device_exits(self):
        with pytest.raises(SystemExit):
            parse_args(["--device", "mps"])


class TestResolveDevice:

    @patch("train.torch.cuda.is_available", return_value=False)
    def test_auto_falls_back_to_cpu(self, _mock_cuda):
        assert resolve_device("auto") == "cpu"

    @patch("train.torch.cuda.is_available", return_value=True)
    def test_auto_uses_cuda_when_available(self, _mock_cuda):
        assert resolve_device("auto") == "cuda"

    @patch("train.torch.cuda.is_available", return_value=False)
    def test_explicit_cuda_unavailable_exits(self, _mock_cuda):
        with pytest.raises(SystemExit):
            resolve_device("cuda")


# ===========================================================================
# resolve_resume
# ===========================================================================

class TestResolveResume:

    def test_fresh_run_applies_defaults(self):
        args = parse_args([])
        mode, run_dir, resume_result = resolve_resume(args)

        assert mode is None
        assert run_dir is None
        assert resume_result is None
        assert args.env == "SSLDribbling-v0"
        assert args.algo == "sac"

    def test_fresh_run_keeps_explicit_args(self):
        args = parse_args(["--env", "VSS-v0", "--algo", "ppo"])
        mode, _, resume_result = resolve_resume(args)

        assert mode is None
        assert resume_result is None
        assert args.env == "VSS-v0"
        assert args.algo == "ppo"

    def test_latest_inherits_all_config(self, tmp_path):
        experiments = tmp_path / "experiments"
        make_run(experiments, "sac", "2026-02-25_100000_SSLDribbling-v0_seed0",
                 checkpoints=["sac_SSLDribbling-v0_00200000.zip"])

        args = parse_args(["--resume", "latest", "--root", str(tmp_path)])
        mode, run_dir, resume_result = resolve_resume(args)

        assert mode == "latest"
        assert run_dir is not None
        assert resume_result is not None
        assert resume_result.start_step == 200_000
        assert args.env == "SSLDribbling-v0"
        assert args.algo == "sac"
        assert args.timesteps == 10_000_000
        assert args.save_freq == 100_000
        assert args.seed == 0
        assert args.max_checkpoints == 5
        assert args.n_envs == 4
        assert "00200000" in args.resume

    def test_explicit_inherits_all_config(self, tmp_path):
        experiments = tmp_path / "experiments"
        name = "2026-02-25_100000_SSLDribbling-v0_seed0"
        make_run(experiments, "sac", name,
                 checkpoints=["sac_SSLDribbling-v0_00200000.zip"])

        args = parse_args(["--resume", name, "--root", str(tmp_path)])
        mode, run_dir, resume_result = resolve_resume(args)

        assert mode == "explicit"
        assert run_dir.name == name
        assert resume_result is not None
        assert resume_result.start_step == 200_000
        assert args.env == "SSLDribbling-v0"
        assert args.timesteps == 10_000_000

    def test_latest_no_runs_raises(self, tmp_path):
        (tmp_path / "experiments").mkdir()
        args = parse_args(["--resume", "latest", "--root", str(tmp_path)])

        from resume import ResumeError
        with pytest.raises(ResumeError):
            resolve_resume(args)

    def test_explicit_not_found_raises(self, tmp_path):
        (tmp_path / "experiments").mkdir()
        args = parse_args(["--resume", "nonexistent", "--root", str(tmp_path)])

        from resume import ResumeError
        with pytest.raises(ResumeError):
            resolve_resume(args)


# ===========================================================================
# End-to-end: main() with mocked training deps
# ===========================================================================

class TestMainWiring:
    """Call main() with mocked SB3/wandb. Verify settings reach the model."""

    @patch("train.wandb")
    @patch("train.WandbCallback")
    @patch("utils.Monitor")
    @patch("utils.gym.make")
    def test_fresh_sac_wiring(self, mock_gym, mock_monitor,
                              mock_wandb_cb, mock_wandb, tmp_path):
        mock_env = MagicMock()
        mock_env.num_envs = 1
        mock_gym.return_value = mock_env
        mock_monitor.return_value = mock_env
        mock_model = MagicMock()
        mock_wandb.init.return_value = MagicMock(id="test123")

        with patch("utils.SAC") as MockSAC, \
             patch.dict("utils.ALGOS", {"sac": MockSAC}), \
             patch("train.torch.cuda.is_available", return_value=False):
            MockSAC.return_value = mock_model

            from train import main
            main([
                "--env", "SSLDribbling-v0", "--algo", "sac",
                "--timesteps", "1000", "--seed", "7",
                "--root", str(tmp_path),
            ])

        # Correct algo with correct seed
        MockSAC.assert_called_once()
        assert MockSAC.call_args[1]["seed"] == 7
        assert MockSAC.call_args[1]["device"] == "cpu"

        # Correct env
        mock_gym.assert_called_with("SSLDribbling-v0")

        # Correct timesteps and fresh run flag
        learn_kw = mock_model.learn.call_args[1]
        assert learn_kw["total_timesteps"] == 1000
        assert learn_kw["reset_num_timesteps"] is True

        # W&B project name
        assert mock_wandb.init.call_args[1]["project"] == "ssldribbling"
        assert mock_wandb.init.call_args[1]["config"]["device"] == "cpu"

    @patch("train.wandb")
    @patch("train.WandbCallback")
    @patch("utils.Monitor")
    @patch("utils.gym.make")
    def test_resume_latest_wiring(self, mock_gym, mock_monitor,
                                  mock_wandb_cb, mock_wandb, tmp_path):
        # Set up a resumable run
        experiments = tmp_path / "experiments"
        cmd = ("python src/train.py --env SSLDribbling-v0 --algo sac "
               "--seed 42 --timesteps 5000000 --save-freq 200000 "
               "--max-checkpoints 3 --n-envs 2\n")
        run_dir = make_run(experiments, "sac",
                           "2026-02-25_100000_SSLDribbling-v0_seed0",
                           command_txt=cmd,
                           checkpoints=["sac_SSLDribbling-v0_00200000.zip"])
        (run_dir / "wandb_run_id.txt").write_text("prev_run_id")

        mock_env = MagicMock()
        mock_env.num_envs = 1
        mock_gym.return_value = mock_env
        mock_monitor.return_value = mock_env
        mock_model = MagicMock()
        mock_model.num_timesteps = 200_000  # match checkpoint so no step-mismatch warning
        mock_wandb.init.return_value = MagicMock(id="prev_run_id")

        with patch("utils.SAC") as MockSAC, \
             patch.dict("utils.ALGOS", {"sac": MockSAC}):
            MockSAC.load.return_value = mock_model

            from train import main
            main(["--resume", "latest", "--root", str(tmp_path)])

        # Should load, not create
        MockSAC.load.assert_called_once()
        assert "00200000" in MockSAC.load.call_args[0][0]

        # Respect original budget: pass remaining timesteps, not full target
        # Target 5M, checkpoint at 200k → remaining = 4_800_000
        assert mock_model.learn.call_args[1]["total_timesteps"] == 4_800_000

        # reset_num_timesteps=False for resume
        assert mock_model.learn.call_args[1]["reset_num_timesteps"] is False

        # W&B resumed
        wandb_kw = mock_wandb.init.call_args[1]
        assert wandb_kw["id"] == "prev_run_id"
        assert wandb_kw["resume"] == "must"

    @patch("train.wandb")
    @patch("train.WandbCallback")
    @patch("utils.Monitor")
    @patch("utils.gym.make")
    def test_interrupt_saves_checkpoint_not_final(self, mock_gym, mock_monitor,
                                                  mock_wandb_cb, mock_wandb, tmp_path):
        """Ctrl+C during training saves a checkpoint (for resume) but not _final.zip."""
        mock_env = MagicMock()
        mock_env.num_envs = 1
        mock_gym.return_value = mock_env
        mock_monitor.return_value = mock_env
        mock_model = MagicMock()
        mock_model.num_timesteps = 500  # mid-run
        mock_model.learn.side_effect = KeyboardInterrupt
        mock_wandb.init.return_value = MagicMock(id="test123")

        with patch("utils.SAC") as MockSAC, \
             patch.dict("utils.ALGOS", {"sac": MockSAC}):
            MockSAC.return_value = mock_model

            from train import main
            main([
                "--env", "SSLDribbling-v0", "--algo", "sac",
                "--timesteps", "1000",
                "--root", str(tmp_path),
            ])

        # model.save called once with a checkpoint path (for resume), not _final
        assert mock_model.save.call_count == 1
        save_path = mock_model.save.call_args[0][0]
        assert "_final" not in save_path
        assert "checkpoint" in save_path or "500" in save_path

        mock_wandb.init.return_value.finish.assert_called_once()
        mock_env.close.assert_called_once()

    @patch("train.wandb")
    @patch("train.WandbCallback")
    @patch("utils.Monitor")
    @patch("utils.gym.make")
    def test_normal_completion_saves_final(self, mock_gym, mock_monitor,
                                           mock_wandb_cb, mock_wandb, tmp_path):
        """Normal completion should save _final.zip."""
        mock_env = MagicMock()
        mock_env.num_envs = 1
        mock_gym.return_value = mock_env
        mock_monitor.return_value = mock_env
        mock_model = MagicMock()
        mock_wandb.init.return_value = MagicMock(id="test123")

        with patch("utils.SAC") as MockSAC, \
             patch.dict("utils.ALGOS", {"sac": MockSAC}):
            MockSAC.return_value = mock_model

            from train import main
            main([
                "--env", "SSLDribbling-v0", "--algo", "sac",
                "--timesteps", "1000",
                "--root", str(tmp_path),
            ])

        # model.save should be called with _final path
        mock_model.save.assert_called_once()
        save_path = mock_model.save.call_args[0][0]
        assert "_final" in save_path

    @patch("train.wandb")
    @patch("train.WandbCallback")
    @patch("utils.Monitor")
    @patch("utils.gym.make")
    def test_already_at_target_does_not_call_learn_saves_final(self, mock_gym, mock_monitor,
                                                              mock_wandb_cb, mock_wandb, tmp_path):
        """When resumed checkpoint is already at or past target, skip learn() and save _final."""
        experiments = tmp_path / "experiments"
        cmd = ("python src/train.py --env SSLDribbling-v0 --algo sac --seed 0 "
               "--timesteps 500000 --save-freq 100000\n")
        make_run(experiments, "sac", "2026-02-25_100000_SSLDribbling-v0_seed0",
                 command_txt=cmd,
                 checkpoints=["sac_SSLDribbling-v0_00500000.zip"])
        (experiments / "SAC" / "2026-02-25_100000_SSLDribbling-v0_seed0" / "wandb_run_id.txt").write_text("prev_id")

        mock_env = MagicMock()
        mock_env.num_envs = 1
        mock_gym.return_value = mock_env
        mock_monitor.return_value = mock_env
        mock_model = MagicMock()
        mock_model.num_timesteps = 500_000
        mock_wandb.init.return_value = MagicMock(id="prev_id")

        with patch("utils.SAC") as MockSAC, \
             patch.dict("utils.ALGOS", {"sac": MockSAC}):
            MockSAC.load.return_value = mock_model

            from train import main
            main(["--resume", "latest", "--root", str(tmp_path)])

        mock_model.learn.assert_not_called()
        mock_model.save.assert_called_once()
        assert "_final" in mock_model.save.call_args[0][0]
        mock_wandb.init.return_value.finish.assert_called_once()
        mock_env.close.assert_called_once()

    @patch("train.wandb")
    @patch("train.WandbCallback")
    @patch("utils.Monitor")
    @patch("utils.gym.make")
    def test_verify_loaded_step_warns_on_mismatch(self, mock_gym, mock_monitor,
                                                  mock_wandb_cb, mock_wandb, tmp_path, capsys):
        """When model.num_timesteps != start_step from checkpoint filename, warn."""
        experiments = tmp_path / "experiments"
        make_run(experiments, "sac", "2026-02-25_100000_SSLDribbling-v0_seed0",
                 checkpoints=["sac_SSLDribbling-v0_00200000.zip"])
        (experiments / "SAC" / "2026-02-25_100000_SSLDribbling-v0_seed0" / "wandb_run_id.txt").write_text("prev_id")

        mock_env = MagicMock()
        mock_env.num_envs = 1
        mock_gym.return_value = mock_env
        mock_monitor.return_value = mock_env
        mock_model = MagicMock()
        mock_model.num_timesteps = 199_999
        mock_wandb.init.return_value = MagicMock(id="prev_id")

        with patch("utils.SAC") as MockSAC, \
             patch.dict("utils.ALGOS", {"sac": MockSAC}):
            MockSAC.load.return_value = mock_model
            from train import main
            main(["--resume", "latest", "--root", str(tmp_path)])

        out = capsys.readouterr()
        assert "200000" in out.out or "200,000" in out.out or "199999" in out.out
        assert "step" in out.out.lower() or "mismatch" in out.out.lower() or "warning" in out.out.lower()

    @patch("train.wandb")
    @patch("train.WandbCallback")
    @patch("utils.Monitor")
    @patch("utils.gym.make")
    def test_global_progress_callback_included(self, mock_gym, mock_monitor,
                                               mock_wandb_cb, mock_wandb, tmp_path):
        """Callbacks include GlobalProgressCallback so progress bar reflects global progress."""
        mock_env = MagicMock()
        mock_env.num_envs = 1
        mock_gym.return_value = mock_env
        mock_monitor.return_value = mock_env
        mock_model = MagicMock()
        mock_wandb.init.return_value = MagicMock(id="test123")

        with patch("utils.SAC") as MockSAC, \
             patch.dict("utils.ALGOS", {"sac": MockSAC}), \
             patch("train.GlobalProgressCallback") as MockProgress:
            MockSAC.return_value = mock_model

            from train import main
            main([
                "--env", "SSLDribbling-v0", "--algo", "sac",
                "--timesteps", "1000", "--root", str(tmp_path),
            ])

        MockProgress.assert_called_once()
        call_kw = MockProgress.call_args[1]
        assert "run_state" in call_kw
        rs = call_kw["run_state"]
        assert rs.target_total == 1000
        assert rs.start_step == 0
        assert rs.save_freq == 100_000
