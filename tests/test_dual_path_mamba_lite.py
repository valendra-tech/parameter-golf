from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "records" / "track_10min_16mb" / "2026-04-29_DualPathMambaLite" / "train_gpt.py"
BASE_ARTIFACT_BYTES = 15_374_243
BASE_CODE_BYTES = 50_651
ARTIFACT_LIMIT_BYTES = 16_000_000


def _script_text() -> str:
    assert SCRIPT.exists(), f"missing experiment script: {SCRIPT}"
    return SCRIPT.read_text(encoding="utf-8")


class DualPathMambaLiteTest(unittest.TestCase):
    def test_dual_path_mamba_lite_is_declared_and_enabled_by_default(self) -> None:
        text = _script_text()
        self.assertIn('architecture_name = "dual_path_mamba_lite"', text)
        self.assertIn("class MambaLiteBlock", text)
        self.assertIn("class MambaLiteBranch", text)
        self.assertIn('use_mamba_path = bool(int(os.environ.get("USE_MAMBA_PATH", "1")))', text)
        self.assertIn('mamba_dim = int(os.environ.get("MAMBA_DIM", 112))', text)
        self.assertIn('mamba_layers = int(os.environ.get("MAMBA_LAYERS", 3))', text)

    def test_fusion_is_gated_residual_from_transformer_and_mamba_states(self) -> None:
        text = _script_text()
        self.assertIn("self.mamba_branch = MambaLiteBranch", text)
        self.assertIn("self.fusion_gate", text)
        self.assertIn("mamba_out = self.mamba_branch(x0)", text)
        self.assertRegex(text, r"x\s*=\s*x\s*\+\s*gate\s*\*\s*mamba_out")

    def test_default_shape_has_room_under_16mb_artifact_cap(self) -> None:
        text = _script_text()
        match = re.search(r"DUAL_MAMBA_DEFAULT_ARTIFACT_ESTIMATE_BYTES\s*=\s*([0-9_]+)", text)
        self.assertIsNotNone(match, "script should expose a conservative default artifact estimate")
        assert match is not None
        estimate = int(match.group(1).replace("_", ""))
        mamba_dim = 112
        model_dim = 512
        layers = 3
        kernel = 7
        branch_payload = (
            2 * model_dim * mamba_dim * 2
            + layers * ((mamba_dim * 2 * mamba_dim * 2) + (mamba_dim * mamba_dim * 2) + (mamba_dim * kernel * 2) + (3 * mamba_dim * 2))
            + model_dim * 2
        )
        static_estimate = BASE_ARTIFACT_BYTES + (SCRIPT.stat().st_size - BASE_CODE_BYTES) + branch_payload
        self.assertLess(static_estimate, ARTIFACT_LIMIT_BYTES)
        self.assertLessEqual(static_estimate, estimate)
        self.assertLess(estimate, ARTIFACT_LIMIT_BYTES)
        self.assertIn("enforce_artifact_budget", text)


if __name__ == "__main__":
    unittest.main()
