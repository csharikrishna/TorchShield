"""Tests for Multi-Cloud KMS Providers: GCPKMSProvider, AzureKeyVaultProvider, HashiCorpVaultProvider."""
import base64
import os
import secrets
import tempfile
import pytest
import torch
import torch.nn as nn

from torchshield.key_providers import (
    GCPKMSProvider,
    AzureKeyVaultProvider,
    HashiCorpVaultProvider,
    KeyProviderError,
)
from torchshield.checkpoint import SecureModelCheckpoint


class MockGCPKMSClient:
    """Mock gRPC client simulating Google Cloud KMS."""
    def __init__(self):
        self.call_count = 0
        self.should_fail_once = False

    def encrypt(self, request=None, **kwargs):
        self.call_count += 1
        if self.should_fail_once:
            self.should_fail_once = False
            raise RuntimeError("UNAVAILABLE: transient network failure")
        req = request or kwargs
        pt = req["plaintext"]
        # Simulated envelope wrap: XOR with mock key
        wrapped = bytes([b ^ 0xAA for b in pt])

        class Resp:
            ciphertext = wrapped
        return Resp()

    def decrypt(self, request=None, **kwargs):
        req = request or kwargs
        ct = req["ciphertext"]
        pt = bytes([b ^ 0xAA for b in ct])

        class Resp:
            plaintext = pt
        return Resp()


class MockAzureCryptoClient:
    """Mock client simulating Azure Key Vault CryptographyClient."""
    def __init__(self):
        self.call_count = 0
        self.should_fail_once = False

    def wrap_key(self, algorithm, key):
        self.call_count += 1
        if self.should_fail_once:
            self.should_fail_once = False
            raise RuntimeError("ThrottlingException: 429 rate limit")

        class Result:
            encrypted_key = bytes([b ^ 0x55 for b in key])
        return Result()

    def unwrap_key(self, algorithm, encrypted_key):
        class Result:
            key = bytes([b ^ 0x55 for b in encrypted_key])
        return Result()


class MockVaultClient:
    """Mock hvac client simulating HashiCorp Vault Transit engine."""
    def __init__(self):
        class Transit:
            def __init__(self):
                self.should_fail_once = False

            def generate_data_key(self, **kwargs):
                if self.should_fail_once:
                    self.should_fail_once = False
                    raise RuntimeError("503 Service Unavailable")
                raw = secrets.token_bytes(32)
                ct = bytes([b ^ 0x33 for b in raw])
                return {
                    "data": {
                        "plaintext": base64.b64encode(raw).decode("utf-8"),
                        "ciphertext": f"vault:v1:{base64.b64encode(ct).decode('utf-8')}",
                    }
                }

            def encrypt_data(self, **kwargs):
                pt = base64.b64decode(kwargs["plaintext"])
                ct = bytes([b ^ 0x33 for b in pt])
                return {"data": {"ciphertext": f"vault:v1:{base64.b64encode(ct).decode('utf-8')}"}}

            def decrypt_data(self, **kwargs):
                ct_b64 = kwargs["ciphertext"].replace("vault:v1:", "")
                ct = base64.b64decode(ct_b64)
                pt = bytes([b ^ 0x33 for b in ct])
                return {"data": {"plaintext": base64.b64encode(pt).decode("utf-8")}}

            def rewrap_data(self, **kwargs):
                # Simulated rewrap under new key
                return {"data": {"ciphertext": kwargs["ciphertext"]}}

        class Secrets:
            transit = Transit()

        self.secrets = Secrets()


def test_gcp_kms_provider_contract_and_checkpoint_roundtrip():
    mock_client = MockGCPKMSClient()
    key_name = "projects/my-proj/locations/global/keyRings/my-ring/cryptoKeys/my-key"
    provider = GCPKMSProvider(key_name=key_name, kms_client=mock_client)

    assert provider.is_production_secure is True
    assert provider.get_key_id() == key_name

    # Test key generation
    dek, wrapped = provider.generate_data_key()
    assert len(dek) == 32
    assert len(wrapped) == 32

    # Test unwrap
    unwrapped = provider.unwrap_key(wrapped)
    assert unwrapped == dek

    # Test retry on transient error
    mock_client.should_fail_once = True
    dek2, wrapped2 = provider.generate_data_key()
    assert len(dek2) == 32

    # End-to-end checkpoint save and load using GCP provider
    with tempfile.NamedTemporaryFile(suffix=".shield", delete=False) as tmp:
        path = tmp.name

    try:
        tensors = {"weight": torch.randn(4, 4), "bias": torch.randn(4)}
        SecureModelCheckpoint.save_checkpoint(
            state_dict=tensors,
            output_path=path,
            key_provider=provider,
            encryption_context={"env": "prod", "cloud": "gcp"},
        )

        restored = SecureModelCheckpoint.load_checkpoint(
            input_path=path,
            key_provider=provider,
            encryption_context={"env": "prod", "cloud": "gcp"},
        )
        for k in tensors:
            assert torch.equal(tensors[k], restored[k])
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_azure_key_vault_provider_contract_and_checkpoint_roundtrip():
    mock_client = MockAzureCryptoClient()
    vault_url = "https://my-vault.vault.azure.net"
    provider = AzureKeyVaultProvider(
        vault_url=vault_url,
        key_name="model-encryption-key",
        crypto_client=mock_client,
    )

    assert provider.is_production_secure is True
    assert "model-encryption-key" in provider.get_key_id()

    # Test generate data key
    dek, wrapped = provider.generate_data_key()
    assert len(dek) == 32
    assert len(wrapped) == 32

    # Test unwrap
    assert provider.unwrap_key(wrapped) == dek

    # Test retry
    mock_client.should_fail_once = True
    dek2, wrapped2 = provider.generate_data_key()
    assert len(dek2) == 32

    # End-to-end checkpoint save and load using Azure provider
    with tempfile.NamedTemporaryFile(suffix=".shield", delete=False) as tmp:
        path = tmp.name

    try:
        tensors = {"layer.weight": torch.randn(8, 8)}
        SecureModelCheckpoint.save_checkpoint(
            state_dict=tensors,
            output_path=path,
            key_provider=provider,
        )

        restored = SecureModelCheckpoint.load_checkpoint(
            input_path=path,
            key_provider=provider,
        )
        assert torch.equal(tensors["layer.weight"], restored["layer.weight"])
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_hashicorp_vault_provider_contract_and_checkpoint_roundtrip():
    mock_client = MockVaultClient()
    provider = HashiCorpVaultProvider(
        key_name="my-transit-key",
        hvac_client=mock_client,
    )

    assert provider.is_production_secure is True
    assert "my-transit-key" in provider.get_key_id()

    # Test generate data key
    dek, wrapped = provider.generate_data_key()
    assert len(dek) == 32

    # Test unwrap
    assert provider.unwrap_key(wrapped) == dek

    # Test rewrap
    rewrapped = provider.rewrap_key(wrapped, new_key_id="my-transit-key-v2")
    assert provider.unwrap_key(rewrapped) == dek

    # End-to-end checkpoint save and load using Vault provider
    with tempfile.NamedTemporaryFile(suffix=".shield", delete=False) as tmp:
        path = tmp.name

    try:
        tensors = {"param": torch.tensor([1.23, 4.56, 7.89])}
        SecureModelCheckpoint.save_checkpoint(
            state_dict=tensors,
            output_path=path,
            key_provider=provider,
        )

        restored = SecureModelCheckpoint.load_checkpoint(
            input_path=path,
            key_provider=provider,
        )
        assert torch.equal(tensors["param"], restored["param"])
    finally:
        if os.path.exists(path):
            os.remove(path)
