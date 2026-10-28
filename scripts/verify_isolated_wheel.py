"""
Independent clean-room isolated virtual environment installation and smoke test
for TorchShield v0.7.1 release candidate wheel.

This script creates a pristine virtual environment WITHOUT --system-site-packages,
installs the built distribution wheel with full dependency resolution via pip,
and validates imports, version, CLI entrypoint, checkpoint save/load, and tamper detection.
"""

import os
import shutil
import subprocess
import sys
import tempfile


def main():
    print("=" * 70)
    print("TORCHSHIELD v0.7.1 ISOLATED CLEAN-ROOM INSTALLATION VERIFICATION")
    print("=" * 70)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    whl_path = os.path.join(repo_root, "dist", "torchshield-0.7.1-py3-none-any.whl")

    if not os.path.exists(whl_path):
        raise FileNotFoundError(f"Candidate wheel not found at: {whl_path}")

    temp_dir = tempfile.mkdtemp(prefix="torchshield_clean_")
    venv_dir = os.path.join(temp_dir, "clean_venv")

    try:
        print(f"\n[1/5] Creating isolated venv (no --system-site-packages): {venv_dir}")
        subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)

        venv_python = (
            os.path.join(venv_dir, "Scripts", "python.exe")
            if os.name == "nt"
            else os.path.join(venv_dir, "bin", "python")
        )
        venv_cli = (
            os.path.join(venv_dir, "Scripts", "torchshield.exe")
            if os.name == "nt"
            else os.path.join(venv_dir, "bin", "torchshield")
        )

        print(f"\n[2/5] Installing {os.path.basename(whl_path)} with pip dependency resolution...")
        subprocess.run([venv_python, "-m", "pip", "install", whl_path], check=True)

        test_script_content = r"""
import sys
import os
import subprocess

print("--- Running Inside Clean-Room Isolated Environment ---")
import torchshield
print("Installed torchshield file:", torchshield.__file__)
assert "clean_venv" in torchshield.__file__ or "site-packages" in torchshield.__file__, (
    f"Not imported from clean venv: {torchshield.__file__}"
)
assert torchshield.__version__ == "0.7.1", f"Expected version 0.7.1, got {torchshield.__version__}"
print("Version check passed: 0.7.1")

# Test CLI entrypoint executable
cli_exe = sys.argv[1]
if os.path.exists(cli_exe):
    cli_res = subprocess.run([cli_exe, "version"], capture_output=True, text=True, check=True)
    print("CLI Executable Output:", cli_res.stdout.strip())
    assert "v0.7.1" in cli_res.stdout, f"Expected v0.7.1 in CLI output, got: {cli_res.stdout}"
else:
    # Fallback to module execution if Windows script wrapper differs
    cli_res = subprocess.run([sys.executable, "-m", "torchshield.cli", "version"], capture_output=True, text=True, check=True)
    print("CLI Module Output:", cli_res.stdout.strip())
    assert "v0.7.1" in cli_res.stdout

# Checkpoint save/load roundtrip
import torch
import torch.nn as nn
from torchshield.training_state import (
    save_training_checkpoint,
    load_training_checkpoint,
    SafeStateProvider,
    TrainingStateError
)
from torchshield.checkpoint import CheckpointError

class DemoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
    def forward(self, x):
        return self.fc(x)

class CounterTracker(SafeStateProvider):
    def __init__(self, count):
        self.count = count
        self.key_invocations = 0
    def get_state_key(self):
        self.key_invocations += 1
        return "counter_tracker"
    def capture_state(self):
        return {"count": self.count}
    def restore_state(self, state):
        self.count = state["count"]

model = DemoNet()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
tracker_save = CounterTracker(42)
key = os.urandom(32)

work_dir = sys.argv[2]
ckpt_path = os.path.join(work_dir, "cleanroom_checkpoint.shield")

print("\nSaving checkpoint from clean environment...")
save_training_checkpoint(
    output_path=ckpt_path,
    model=model,
    optimizer=optimizer,
    state_providers=[tracker_save],
    key=key
)
assert os.path.exists(ckpt_path), "Checkpoint file was not created!"
print(f"Checkpoint saved successfully ({os.path.getsize(ckpt_path)} bytes)")

print("\nLoading checkpoint into fresh destination in clean environment...")
dest_model = DemoNet()
dest_optimizer = torch.optim.AdamW(dest_model.parameters(), lr=1e-3)
tracker_load = CounterTracker(0)

load_training_checkpoint(
    input_path=ckpt_path,
    model=dest_model,
    optimizer=dest_optimizer,
    state_providers=[tracker_load],
    key=key
)

# Verify parameter identity and equality
for (p1_name, p1_val), (p2_name, p2_val) in zip(model.named_parameters(), dest_model.named_parameters()):
    assert torch.equal(p1_val, p2_val), f"Mismatch in parameter {p1_name}"
assert tracker_load.count == 42, f"Expected tracker count 42, got {tracker_load.count}"
assert tracker_load.key_invocations == 1, (
    f"Expected get_state_key to be called exactly once, called {tracker_load.key_invocations} times"
)
print("Roundtrip state restoration verified bit-for-bit!")

# Tamper detection test
print("\nTesting cryptographic tamper detection in clean environment...")
with open(ckpt_path, "r+b") as f:
    f.seek(-10, os.SEEK_END)
    b = f.read(1)
    f.seek(-10, os.SEEK_END)
    f.write(bytes([b[0] ^ 0xFF]))

tamper_caught = False
try:
    load_training_checkpoint(
        input_path=ckpt_path,
        model=dest_model,
        key=key
    )
except (CheckpointError, TrainingStateError):
    tamper_caught = True

assert tamper_caught, "Tampered checkpoint was unexpectedly accepted!"
print("Cryptographic tamper detection verified: CheckpointError raised!")
print("\n[SUCCESS] ALL CLEAN-ROOM ISOLATED VERIFICATION CHECKS PASSED [100% SUCCESS]!")
"""
        test_script_path = os.path.join(temp_dir, "run_cleanroom_test.py")
        with open(test_script_path, "w", encoding="utf-8") as f:
            f.write(test_script_content)

        print("\n[3/5] Executing clean-room verification in isolated environment...")
        res = subprocess.run(
            [venv_python, test_script_path, venv_cli, temp_dir],
            capture_output=True,
            text=True,
        )
        print(res.stdout)
        if res.returncode != 0:
            print("STDERR:\n", res.stderr)
            sys.exit(res.returncode)

        print("\n[4/5] Cleaning up isolated virtual environment...")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("[5/5] Temporary virtual environment removed.")

    print("\n" + "=" * 70)
    print("GENUINELY ISOLATED CLEAN-ROOM VERIFICATION: COMPLETED WITH 100% SUCCESS")
    print("=" * 70)


if __name__ == "__main__":
    main()
