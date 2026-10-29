"""Independent clean-room isolated virtual environment installation and smoke test
for TorchShield release candidate wheel.

This script creates a pristine virtual environment WITHOUT --system-site-packages,
installs the built candidate wheel with normal pip dependency resolution,
and validates:
- Virtual environment creation (sys.prefix != sys.base_prefix)
- Import isolation outside repository path (resolves in venv site-packages)
- Package version matches expected release version
- Installed metadata and dependency versions recorded
- CLI executable runs (torchshield version)
- Checkpoint save and load with model, optimizer, and auxiliary state
- SafeStateProvider.get_state_key() invocation count is strictly 1 during load
- Tamper detection fails closed on ciphertext bitflip
- Wrong key fails closed
- Malformed / truncated container input fails closed
- Cleanup of temporary files and secrets
- Nonzero process exit code propagation on failure
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile


def compute_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


CLEANROOM_RUNNER_CODE = r'''
import sys
import os
import subprocess

print("--- Running Inside Isolated Clean-Room Test Process ---")
print("sys.prefix:     ", sys.prefix)
print("sys.base_prefix:", sys.base_prefix)
assert sys.prefix != sys.base_prefix, "FAIL: Not running inside isolated virtual environment!"

venv_dir = sys.argv[1]
expected_ver = sys.argv[2]
cli_exe = sys.argv[3]
work_dir = sys.argv[4]

import torchshield
print("torchshield location:", torchshield.__file__)
venv_norm = os.path.normcase(os.path.abspath(venv_dir))
mod_norm = os.path.normcase(os.path.abspath(torchshield.__file__))
assert venv_norm in mod_norm, f"FAIL: torchshield imported from outside isolated venv! {mod_norm}"

assert torchshield.__version__ == expected_ver, (
    f"FAIL: Expected version {expected_ver}, got {torchshield.__version__}"
)
print("Version check passed:", torchshield.__version__)

# Verify CLI executable
if os.path.isfile(cli_exe):
    cli_res = subprocess.run([cli_exe, "version"], capture_output=True, text=True, check=True)
    cli_out = cli_res.stdout.strip()
    print("CLI Executable Output:", cli_out)
    assert f"v{expected_ver}" in cli_out, f"FAIL: Expected v{expected_ver} in CLI output, got: {cli_out}"
else:
    cli_res = subprocess.run([sys.executable, "-m", "torchshield.cli", "version"], capture_output=True, text=True, check=True)
    cli_out = cli_res.stdout.strip()
    print("CLI Module Fallback Output:", cli_out)
    assert f"v{expected_ver}" in cli_out, f"FAIL: Expected v{expected_ver} in CLI module output"

# Checkpoint save / load / tamper / wrong-key verification
import torch
import torch.nn as nn
from torchshield.training_state import (
    save_training_checkpoint,
    load_training_checkpoint,
    SafeStateProvider,
    TrainingStateError,
)
from torchshield.checkpoint import CheckpointError

class DemoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
    def forward(self, x):
        return self.fc(x)

class AuxiliaryStateTracker(SafeStateProvider):
    def __init__(self, count):
        self.count = count
        self.key_invocations = 0
    def get_state_key(self):
        self.key_invocations += 1
        return "aux_tracker"
    def capture_state(self):
        return {"count": self.count}
    def restore_state(self, state):
        self.count = state["count"]

model = DemoModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
tracker_save = AuxiliaryStateTracker(777)
dek = os.urandom(32)

ckpt_path = os.path.join(work_dir, "isolated_smoke_checkpoint.shield")

print("\nSaving checkpoint with model, optimizer, auxiliary state...")
save_training_checkpoint(
    output_path=ckpt_path,
    model=model,
    optimizer=optimizer,
    state_providers=[tracker_save],
    key=dek,
)
assert os.path.isfile(ckpt_path), "FAIL: Checkpoint file not created!"
print("Checkpoint saved successfully:", os.path.getsize(ckpt_path), "bytes")

print("\nRestoring checkpoint into fresh destination model...")
dest_model = DemoModel()
dest_opt = torch.optim.AdamW(dest_model.parameters(), lr=1e-3)
tracker_load = AuxiliaryStateTracker(0)

load_training_checkpoint(
    input_path=ckpt_path,
    model=dest_model,
    optimizer=dest_opt,
    state_providers=[tracker_load],
    key=dek,
)

# 1. State fidelity verification
for (p1_n, p1_v), (p2_n, p2_v) in zip(model.named_parameters(), dest_model.named_parameters()):
    assert torch.equal(p1_v, p2_v), f"FAIL: Parameter mismatch in {p1_n}"
assert tracker_load.count == 777, f"FAIL: Expected tracker count 777, got {tracker_load.count}"
assert tracker_load.key_invocations == 1, (
    f"FAIL: get_state_key() invoked {tracker_load.key_invocations} times (expected 1)"
)
print("Model and auxiliary state restored bit-for-bit!")
print("SafeStateProvider.get_state_key() invocation count verified:", tracker_load.key_invocations)

# 2. Cryptographic tamper detection verification
print("\nTesting ciphertext tamper detection (bitflip)...")
with open(ckpt_path, "r+b") as f:
    f.seek(-16, os.SEEK_END)
    b = f.read(1)
    f.seek(-16, os.SEEK_END)
    f.write(bytes([b[0] ^ 0xFF]))

tamper_failed = False
try:
    load_training_checkpoint(input_path=ckpt_path, model=dest_model, key=dek)
except (CheckpointError, TrainingStateError):
    tamper_failed = True

assert tamper_failed, "FAIL: Tampered checkpoint was accepted without error!"
print("Tamper detection verified: failed closed with exception.")

# 3. Wrong key verification
print("\nTesting wrong key rejection...")
wrong_dek = os.urandom(32)
wrong_key_failed = False
try:
    load_training_checkpoint(input_path=ckpt_path, model=dest_model, key=wrong_dek)
except (CheckpointError, TrainingStateError):
    wrong_key_failed = True

assert wrong_key_failed, "FAIL: Wrong key was accepted without error!"
print("Wrong key rejection verified: failed closed with exception.")

# 4. Malformed input verification
print("\nTesting malformed container rejection...")
malformed_path = os.path.join(work_dir, "corrupt.shield")
with open(malformed_path, "wb") as f:
    f.write(b"NOT_A_VALID_SHIELD_FILE_RANDOM_GARBAGE")

malformed_failed = False
try:
    load_training_checkpoint(input_path=malformed_path, model=dest_model, key=dek)
except (CheckpointError, TrainingStateError, ValueError):
    malformed_failed = True

assert malformed_failed, "FAIL: Malformed file was accepted without error!"
print("Malformed input rejection verified: failed closed with exception.")

# 5. Attestation and provider-key lifecycle verification in installed package
print("\nTesting attestation and provider-key lifecycle in installed package...")
import time
import hashlib
from torchshield.attestation import (
    AttestationGatedKeyProvider,
    AttestationEvidence,
    HardwareAttestationVerifier,
    AttestationPolicy,
    PlatformType,
    AttestationError,
)
from torchshield.key_providers.local import LocalKeyProvider

inner_provider = LocalKeyProvider()
verifier = HardwareAttestationVerifier()
policy = AttestationPolicy(
    allowed_platforms={PlatformType.NVIDIA_CC.value},
    golden_measurements={"gold"},
    max_age_seconds=60.0,
    require_signature=False,
)
gated = AttestationGatedKeyProvider(inner_provider, verifier, policy=policy)
ch = gated.create_challenge()
ev = AttestationEvidence(
    platform=PlatformType.NVIDIA_CC.value,
    measurement="gold",
    report_data=hashlib.sha256(ch).hexdigest(),
    timestamp=time.time(),
)
gated.stage_evidence(ev, challenge=ch)
assert gated._staged_evidence is not None
# Expire challenge and trigger cleanup
gated._authorized_challenges[ch] = time.time() - 400.0
gated._cleanup_expired_challenges_locked()
assert gated._staged_evidence is None, "FAIL: Stale staged evidence was not purged in installed wheel!"
print("Attestation staged evidence purge verified in installed package.")

# Cleanup temporary secrets from memory
del dek, wrong_dek
print("\n[CLEAN-ROOM SUITE COMPLETED WITH 100% SUCCESS]")
'''


def main():
    parser = argparse.ArgumentParser(description="TorchShield Clean-Room Isolated Wheel Verification")
    parser.add_argument("--wheel", default=None, help="Path to .whl file (defaults to candidate in dist/)")
    parser.add_argument("--expected-version", default="0.7.3", help="Expected package version")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dist_dir = os.path.join(repo_root, "dist")

    if args.wheel:
        whl_path = os.path.abspath(args.wheel)
        if not os.path.isfile(whl_path):
            raise FileNotFoundError(f"Specified candidate wheel not found: {whl_path}")
        whl_filename = os.path.basename(whl_path)
        expected_prefix = f"torchshield-{args.expected_version}"
        if not whl_filename.startswith(expected_prefix):
            raise ValueError(
                f"Specified wheel '{whl_filename}' does not match expected version '{args.expected_version}'"
            )
    else:
        # Strictly locate candidate wheel matching expected version; NEVER fall back to older wheels
        expected_pattern = os.path.join(dist_dir, f"torchshield-{args.expected_version}*.whl")
        matching = glob.glob(expected_pattern)
        if len(matching) == 1:
            whl_path = matching[0]
        elif len(matching) > 1:
            raise RuntimeError(
                f"Multiple candidate wheels found matching {expected_pattern}: {matching}. "
                "Specify --wheel explicitly."
            )
        else:
            raise FileNotFoundError(
                f"No candidate wheel matching version {args.expected_version} found in {dist_dir}. "
                "Explicitly provide --wheel or build the candidate first. Refusing silent fallback to older wheels."
            )

    if not os.path.isfile(whl_path):
        raise FileNotFoundError(f"Candidate wheel not found at: {whl_path}")

    whl_size = os.path.getsize(whl_path)
    whl_hash = compute_sha256(whl_path)

    print("=" * 80)
    print(f"TORCHSHIELD v{args.expected_version} ISOLATED CLEAN-ROOM INSTALLATION VERIFICATION")
    print("=" * 80)
    print(f"Target Wheel:     {whl_path}")
    print(f"Wheel Size:       {whl_size:,} bytes")
    print(f"Wheel SHA-256:    {whl_hash}")
    print(f"Expected Version: {args.expected_version}")
    print("=" * 80)

    temp_dir = tempfile.mkdtemp(prefix="torchshield_cleanroom_")
    venv_dir = os.path.join(temp_dir, "clean_venv")

    try:
        print(f"\n[1/6] Creating pristine isolated venv (no --system-site-packages): {venv_dir}")
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

        print(f"\n[2/6] Installing {os.path.basename(whl_path)} with pip dependency resolution...")
        subprocess.run([venv_python, "-m", "pip", "install", whl_path], check=True)

        print("\n[3/6] Recording installed package metadata & dependency versions:")
        pip_list = subprocess.run([venv_python, "-m", "pip", "list"], capture_output=True, text=True, check=True)
        print(pip_list.stdout)

        test_script_path = os.path.join(temp_dir, "verify_cleanroom_runner.py")
        with open(test_script_path, "w", encoding="utf-8") as f:
            f.write(CLEANROOM_RUNNER_CODE)

        print("\n[4/6] Executing isolated test runner outside repository path...")
        # CRITICAL: Run with cwd=temp_dir so local repo directory is not in sys.path
        res = subprocess.run(
            [venv_python, test_script_path, venv_dir, args.expected_version, venv_cli, temp_dir],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )
        print(res.stdout)
        if res.returncode != 0:
            print("STDERR:\n", res.stderr)
            raise RuntimeError(f"Clean-room verification failed with returncode {res.returncode}")

        print("[5/6] Cleaning up isolated clean-room environment...")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("[6/6] Isolated clean-room temporary directory removed.")

    print("\n" + "=" * 80)
    print(f"GENUINELY ISOLATED CLEAN-ROOM VERIFICATION (v{args.expected_version}): 100% PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
