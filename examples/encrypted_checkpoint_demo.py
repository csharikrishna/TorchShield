"""Comprehensive demonstration of AI tensor and model checkpoint encryption workflows.

Compares:
A. Plaintext PyTorch checkpoint loading (Performance Floor).
B. Whole-file AES-256-GCM encrypted loading.
C. Chunked authenticated streaming checkpoint loading (SecureModelCheckpoint).
D. Experimental TorchShield tensor encryption (Research Prototype).

Demonstrates timing, memory behavior, layer-by-layer streaming, and tamper detection.
"""
import os
import sys
import tempfile
import time
from pathlib import Path

# Ensure project root is in sys.path
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import torch
import torch.nn as nn

from torchshield.checkpoint import SecureModelCheckpoint
from torchshield.secure import AESGCMEnvelope, AuthenticationError
from torchshield.engine import TorchShield_Engine, TensorCipherWrapper


class DeepTransformerLayer(nn.Module):
    """Synthetic multi-head attention and feed-forward block."""
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ff1 = nn.Linear(d_model, 2048)
        self.ff2 = nn.Linear(2048, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        res = x
        x = self.norm1(x)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (x.shape[-1] ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        x = res + self.out_proj(torch.matmul(attn, v))
        return x + self.ff2(torch.relu(self.ff1(self.norm2(x))))


def run_checkpoint_comparison_demo():
    print("=" * 80)
    print("PHASE 4: END-TO-END AI MODEL CHECKPOINT ENCRYPTION & LOADING DEMO")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target Compute Device : {device}")

    # Build model (~3.15M parameters, ~12.6 MB float32)
    model = DeepTransformerLayer().to(device)
    dummy_input = torch.randn(2, 64, 512, device=device)
    baseline_pred = model(dummy_input)

    key = AESGCMEnvelope.generate_key()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Paths
        plain_path = os.path.join(tmpdir, "model_plain.pt")
        whole_path = os.path.join(tmpdir, "model_whole.aes")
        chunked_path = os.path.join(tmpdir, "model_chunked.shield_ckpt")

        # --- A. Plaintext Baseline ---
        t0 = time.perf_counter()
        torch.save(model.state_dict(), plain_path)
        plain_save_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        plain_state = torch.load(plain_path, map_location=device, weights_only=True)
        plain_load_ms = (time.perf_counter() - t0) * 1000

        # --- B. Whole-File AES-256-GCM ---
        t0 = time.perf_counter()
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        raw_model_bytes = buf.getvalue()
        enc_whole = AESGCMEnvelope.encrypt(raw_model_bytes, key, associated_data=b"model_whole_v1")
        with open(whole_path, "wb") as f:
            f.write(enc_whole)
        whole_save_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        with open(whole_path, "rb") as f:
            enc_read = f.read()
        dec_model_bytes = AESGCMEnvelope.decrypt(enc_read, key, associated_data=b"model_whole_v1")
        whole_state = torch.load(io.BytesIO(dec_model_bytes), map_location=device, weights_only=True)
        whole_load_ms = (time.perf_counter() - t0) * 1000

        # --- C. Chunked Authenticated Streaming Checkpoint ---
        t0 = time.perf_counter()
        SecureModelCheckpoint.save_checkpoint(model.state_dict(), chunked_path, key, chunk_bytes=1024 * 1024)
        chunked_save_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        chunked_state = SecureModelCheckpoint.load_checkpoint(chunked_path, key, device=device)
        chunked_load_ms = (time.perf_counter() - t0) * 1000

        # --- D. Experimental TorchShield Tensor Cipher Wrapper ---
        # Test 1 sample parameter tensor
        torchshield_engine = TorchShield_Engine(N=64, R=4, p=4_294_967_291, device=device)
        torchshield_wrapper = TensorCipherWrapper(torchshield_engine)
        sample_tensor = model.q_proj.weight.detach()
        # Quantize float to int64 field elements
        scale = 10000.0
        q_int = torch.clamp((sample_tensor * scale).round().to(torch.int64), 0, torchshield_engine.p - 1)

        t0 = time.perf_counter()
        ct_torchshield, meta_torchshield = torchshield_wrapper.encrypt_tensor(q_int)
        torchshield_enc_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        pt_torchshield = torchshield_wrapper.decrypt_tensor(ct_torchshield, meta_torchshield)
        torchshield_dec_ms = (time.perf_counter() - t0) * 1000

        # Verify inference equivalence
        test_model = DeepTransformerLayer().to(device)
        test_model.load_state_dict(chunked_state)
        chunked_pred = test_model(dummy_input)
        assert torch.equal(baseline_pred, chunked_pred), "Prediction mismatch on restored model!"

        print(f"Model Parameters   : {sum(p.numel() for p in model.parameters()):,} ({len(raw_model_bytes)/1024/1024:.2f} MiB)")
        print("\nStorage & Loading Latency Benchmark:")
        print("-" * 80)
        print(f"A. Plaintext PyTorch (No Security)       : Save {plain_save_ms:6.2f} ms | Load {plain_load_ms:6.2f} ms")
        print(f"B. Whole-File AES-256-GCM               : Save {whole_save_ms:6.2f} ms | Load {whole_load_ms:6.2f} ms")
        print(f"C. Chunked Streaming AES-GCM Checkpoint : Save {chunked_save_ms:6.2f} ms | Load {chunked_load_ms:6.2f} ms")
        print(f"D. Experimental TorchShield (Single Layer Only): Enc  {torchshield_enc_ms:6.2f} ms | Dec  {torchshield_dec_ms:6.2f} ms")
        print("-" * 80)

        # Demonstrate tamper detection
        print("\nTamper Detection Verification:")
        with open(chunked_path, "r+b") as f:
            f.seek(-100, os.SEEK_END)
            f.write(b"\xFF\xFF\xFF\xFF")  # Corrupt a chunk

        try:
            SecureModelCheckpoint.load_checkpoint(chunked_path, key, device=device)
            print("[FAIL] Tampered checkpoint was silently accepted!")
        except AuthenticationError:
            print("[PASS] Tampered chunk detected and rejected with fail-closed AuthenticationError.")

    print("=" * 80)
    print("END-TO-END MODEL CHECKPOINT WORKFLOW COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    import io
    run_checkpoint_comparison_demo()
