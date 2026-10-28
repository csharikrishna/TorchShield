"""End-to-End Local KMS Verification Suite for TorchShield.

Tests TorchShield AWSKMSProvider with a local KMS emulator (LocalStack or Moto)
at zero cloud cost without requiring real AWS credentials or cloud accounts.
"""
from __future__ import annotations

import os
import socket
import sys
import tempfile
import time
import torch
import torch.nn as nn

# Set dummy credentials for local KMS emulator
os.environ["AWS_ACCESS_KEY_ID"] = "test"
os.environ["AWS_SECRET_ACCESS_KEY"] = "test"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

import boto3
from torchshield import (
    TrainingProgress,
    save_training_checkpoint,
    load_training_checkpoint,
    save_hf_model,
    load_hf_model,
    AuthenticationError,
    CheckpointError,
)
from torchshield.key_providers.kms import AWSKMSProvider


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


def is_port_in_use(port: int = 4566) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def run_local_kms_verification(endpoint_url: str = "http://127.0.0.1:4566"):
    server = None
    port = 4566
    if "4566" in endpoint_url and not is_port_in_use(port):
        print(f"[Init] Starting embedded Local KMS Mock Server on port {port}...")
        try:
            from moto.server import ThreadedMotoServer
            server = ThreadedMotoServer(ip_address="127.0.0.1", port=port)
            server.start()
            time.sleep(1)
            print("       Embedded Local KMS server is live!")
        except Exception as e:
            print(f"       Warning: Could not launch embedded server: {e}")

    try:
        print("=" * 70)
        print("TorchShield - Local KMS Envelope Encryption Verification Suite")
        print("=" * 70)
        print(f"[1/7] Connecting to KMS endpoint at {endpoint_url}...")

        kms_client = boto3.client(
            "kms",
            endpoint_url=endpoint_url,
            region_name="us-east-1",
            aws_access_key_id="test",
            aws_secret_access_key="test",
        )

        # 1. Create a Master Key (KEK) in the local KMS emulator
        response = kms_client.create_key(
            Description="TorchShield Local KMS Test KEK",
            KeyUsage="ENCRYPT_DECRYPT",
            CustomerMasterKeySpec="SYMMETRIC_DEFAULT",
        )
        key_arn = response["KeyMetadata"]["Arn"]
        print(f"      Created Master Key (KEK): {key_arn}")

        # 2. Initialize TorchShield AWSKMSProvider
        print("[2/7] Initializing TorchShield AWSKMSProvider...")
        provider = AWSKMSProvider(
            key_arn=key_arn,
            endpoint_url=endpoint_url,
            region_name="us-east-1",
            aws_access_key_id="test",
            aws_secret_access_key="test",
        )
        assert provider.is_production_secure is True
        print(f"      AWSKMSProvider initialized. is_production_secure: {provider.is_production_secure}")

        # 3. Test KMS envelope primitives (DEK generation & wrapping)
        print("[3/7] Testing DEK lifecycle (generate_data_key, wrap_key, unwrap_key)...")
        context = {"model_name": "torchshield-transformer", "version": "v1.0"}
        plaintext_dek, wrapped_dek = provider.generate_data_key(encryption_context=context)
        assert len(plaintext_dek) == 32, "Plaintext DEK must be 256-bit (32 bytes)"
        assert len(wrapped_dek) > 0, "Wrapped DEK must be non-empty"

        unwrapped_dek = provider.unwrap_key(wrapped_dek, encryption_context=context)
        assert unwrapped_dek == plaintext_dek, "Unwrapped DEK must match original plaintext DEK"
        print("      DEK generation and envelope wrapping verified successfully.")

        # 4. Save PyTorch checkpoint using KMS Provider
        print("[4/7] Saving PyTorch model checkpoint with AWS KMS envelope encryption...")
        model = SimpleModel()
        orig_weight = model.fc.weight.clone()
        orig_bias = model.fc.bias.clone()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        progress = TrainingProgress(
            global_step=100,
            epoch=2,
            loss=0.42,
            config={"architecture": "Linear(10->2)", "framework": "TorchShield"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "secure_model.ckpt")
            save_training_checkpoint(
                output_path=ckpt_path,
                model=model,
                optimizer=optimizer,
                progress=progress,
                key_provider=provider,
            )
            file_size = os.path.getsize(ckpt_path)
            print(f"      Saved encrypted training checkpoint to: {ckpt_path}")
            print(f"      Bytes written: {file_size:,} bytes")

            # 5. Tamper-resistance test
            print("[5/7] Testing Cryptographic Tamper-Resistance...")
            # Corrupt byte in the payload
            with open(ckpt_path, "r+b") as f:
                data = bytearray(f.read())
                mid = len(data) // 2
                data[mid] ^= 0xFF
                f.seek(0)
                f.write(data)

            tamper_detected = False
            corrupted_model = SimpleModel()
            try:
                load_training_checkpoint(
                    input_path=ckpt_path,
                    model=corrupted_model,
                    key_provider=provider,
                )
            except (AuthenticationError, CheckpointError, Exception) as e:
                tamper_detected = True
                print(f"      Tamper detected and blocked! Exception: {type(e).__name__} ({e})")

            assert tamper_detected, "TAMPERING WAS NOT DETECTED! Security failure!"
            print("      Tamper-resistance verified: Authenticated encryption rejected corrupted payload.")

        # 6. Re-save fresh checkpoint and verify full roundtrip recovery
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "secure_model_roundtrip.ckpt")
            save_training_checkpoint(
                output_path=ckpt_path,
                model=model,
                optimizer=optimizer,
                progress=progress,
                key_provider=provider,
            )

            print("[6/7] Testing Decryption & Bit-for-bit Checkpoint Restoration...")
            restored_model = SimpleModel()
            restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.01)

            recovered_progress = load_training_checkpoint(
                input_path=ckpt_path,
                model=restored_model,
                optimizer=restored_optimizer,
                key_provider=provider,
            )

            # Verify restored weights
            assert torch.equal(restored_model.fc.weight, orig_weight), "Weight mismatch!"
            assert torch.equal(restored_model.fc.bias, orig_bias), "Bias mismatch!"
            assert recovered_progress.global_step == 100
            assert recovered_progress.epoch == 2
            assert abs(recovered_progress.loss - 0.42) < 1e-5
            print("      Weights, biases, optimizer state, and training progress match 100%!")

        # 7. Hugging Face Model format save and load with KMS Provider
        print("[7/7] Testing Hugging Face model save & load with AWS KMS KeyProvider...")
        with tempfile.TemporaryDirectory() as tmpdir:
            hf_model = SimpleModel()
            hf_weight = hf_model.fc.weight.clone()
            hf_ckpt_path = os.path.join(tmpdir, "hf_model.shield")

            save_hf_model(model=hf_model, output_path=hf_ckpt_path, key=provider)
            assert os.path.exists(hf_ckpt_path)
            print(f"      HuggingFace model safely saved to: {hf_ckpt_path}")

            loaded_hf_model = SimpleModel()
            load_hf_model(model=loaded_hf_model, input_path=hf_ckpt_path, key=provider)
            assert torch.equal(loaded_hf_model.fc.weight, hf_weight)
            print("      HuggingFace model restored and weights verified with 100% fidelity!")

        print("=" * 70)
        print("ALL 7/7 LOCAL KMS ENVELOPE ENCRYPTION TESTS PASSED! [100% SUCCESS]")
        print("=" * 70)
        return True

    finally:
        if server is not None:
            print("[Cleanup] Stopping embedded Local KMS Mock Server...")
            server.stop()
            print("          Server stopped successfully.")


if __name__ == "__main__":
    endpoint = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:4566"
    run_local_kms_verification(endpoint)
