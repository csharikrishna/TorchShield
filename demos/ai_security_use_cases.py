"""Illustrative, synthetic-data demonstrations for the AHCC research prototype:

1. Finite-field encoding, encryption, decryption, and plaintext aggregation of toy gradients.
2. Sequential parameter-tensor round trips for a toy model.

These are correctness demos only. They do not establish confidentiality, secure
aggregation, production model-IP protection, or a cryptographic security guarantee.
"""
from typing import Tuple
import math
import torch
import torch.nn as nn
from ahcc.engine import AHCC_Engine, TensorCipherWrapper


def quantize_fp32_to_fp(tensor: torch.Tensor, p: int, scale: float = 1e6) -> Tuple[torch.Tensor, dict]:
    """Quantize a finite floating tensor to F_p and record dequantization metadata.

    Quantization is intentionally lossy. The helper checks for non-finite inputs
    and int64 overflow rather than silently producing invalid field elements.
    """
    if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
        raise TypeError("tensor must be a floating-point torch.Tensor.")
    if isinstance(p, bool) or not isinstance(p, int) or p not in (2_147_483_579, 4_294_967_291):
        raise ValueError("p must be one of the supported AHCC prime integer moduli.")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(float(scale)) or scale <= 0:
        raise ValueError("scale must be a finite positive number.")
    if tensor.numel() and not bool(torch.isfinite(tensor).all().item()):
        raise ValueError("tensor contains NaN or infinite values.")
    scaled = tensor.to(torch.float64) * float(scale)
    limits = torch.iinfo(torch.int64)
    if scaled.numel() and (bool((scaled >= float(limits.max)).any().item()) or bool((scaled <= float(limits.min)).any().item())):
        raise ValueError("Scaled tensor exceeds the representable int64 range.")
    scaled_int = torch.round(scaled).to(torch.int64)
    signed_limit = p // 2
    if scaled_int.numel() and (bool((scaled_int > signed_limit).any().item()) or bool((scaled_int < -signed_limit).any().item())):
        raise ValueError(
            f"Scaled values must lie within [-{signed_limit}, {signed_limit}] for unambiguous signed decoding."
        )
    field_tensor = scaled_int % p
    return field_tensor, {"scale": float(scale), "p": int(p), "orig_dtype": tensor.dtype}


def dequantize_fp_to_fp32(field_tensor: torch.Tensor, q_meta: dict) -> torch.Tensor:
    """Map quantized F_p residues back to the original floating dtype."""
    if not isinstance(field_tensor, torch.Tensor) or field_tensor.dtype != torch.int64:
        raise TypeError("field_tensor must be a torch.int64 tensor of finite-field elements.")
    if not isinstance(q_meta, dict) or not {"p", "scale", "orig_dtype"}.issubset(q_meta):
        raise ValueError("q_meta must contain p, scale, and orig_dtype.")
    p, scale, orig_dtype = q_meta["p"], q_meta["scale"], q_meta["orig_dtype"]
    if isinstance(p, bool) or not isinstance(p, int) or p not in (2_147_483_579, 4_294_967_291):
        raise ValueError("Unsupported prime integer modulus in q_meta.")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(float(scale)) or scale <= 0:
        raise ValueError("q_meta scale must be finite and positive.")
    if orig_dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        raise ValueError("q_meta orig_dtype must be a floating-point torch dtype.")
    values = field_tensor.to(torch.int64)
    if values.numel() and (bool((values < 0).any().item()) or bool((values >= p).any().item())):
        raise ValueError("field_tensor values must be in [0, p-1].")
    signed = torch.where(values > p // 2, values - p, values)
    return (signed.to(torch.float64) / float(scale)).to(orig_dtype)


def demo_federated_learning_gradient_encryption():
    print("======================================================================")
    print("USE CASE 1: TOY GRADIENT ENCRYPT/DECRYPT ROUND TRIP")
    print("======================================================================")
    p = 2147483579
    engine = AHCC_Engine(N=64, R=4, p=p, master_seed=202601)
    wrapper = TensorCipherWrapper(engine)

    # Simulate 3 Federated Learning clients computing local gradients
    torch.manual_seed(10)
    num_clients = 3
    grad_shape = (64, 128)  # Simulated neural network layer gradient

    client_grads_fp32 = [torch.randn(grad_shape) * 0.05 for _ in range(num_clients)]
    encrypted_payloads = []

    for idx, grad in enumerate(client_grads_fp32):
        q_grad, q_meta = quantize_fp32_to_fp(grad, p)
        ct_grad, shape_meta = wrapper.encrypt_tensor(q_grad)
        encrypted_payloads.append((ct_grad, shape_meta, q_meta))
        print(
            f"Client {idx+1}: Local Gradient {tuple(grad.shape)} encrypted -> "
            f"Ciphertext {tuple(ct_grad.shape)} in F_p"
        )

    # The server decrypts each payload before aggregating plaintext gradients.
    # This is not secure aggregation and AHCC is not homomorphic encryption.
    decrypted_grads = []
    for idx, (ct_grad, shape_meta, q_meta) in enumerate(encrypted_payloads):
        rec_q_grad = wrapper.decrypt_tensor(ct_grad, shape_meta)
        rec_fp32 = dequantize_fp_to_fp32(rec_q_grad.cpu(), q_meta)
        decrypted_grads.append(rec_fp32)

        # Verify reconstruction accuracy
        max_err = (client_grads_fp32[idx] - rec_fp32).abs().max().item()
        assert max_err <= 1e-6, f"Gradient reconstruction error too high: {max_err}"

    aggregated_grad = sum(decrypted_grads) / num_clients
    expected_grad = sum(client_grads_fp32) / num_clients
    agg_err = (aggregated_grad - expected_grad).abs().max().item()
    print(f"Max Federated Aggregation Quantization Error : {agg_err:.2e} (<= 1e-6)")
    print("[PASS] Toy gradient quantization/encryption/decryption/aggregation round trip passed.")
    print("NOTE: Aggregation occurs after plaintext decryption; this is not secure aggregation.")


def demo_on_the_fly_model_weight_decryption():
    print("\n======================================================================")
    print("USE CASE 2: TOY MODEL PARAMETER ENCRYPT/DECRYPT ROUND TRIP")
    print("======================================================================")
    p = 2147483579
    engine = AHCC_Engine(N=64, R=4, p=p, master_seed=888888)
    wrapper = TensorCipherWrapper(engine)

    # Create a dummy 2-layer MLP model
    torch.manual_seed(99)
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )
    sample_input = torch.randn(4, 128)
    with torch.no_grad():
        baseline_output = model(sample_input)

    # Encrypt all model weights offline into an encrypted state dict
    encrypted_vault = {}
    for name, param in model.named_parameters():
        q_param, q_meta = quantize_fp32_to_fp(param.data, p, scale=1e6)
        ct_param, shape_meta = wrapper.encrypt_tensor(q_param)
        encrypted_vault[name] = (ct_param, shape_meta, q_meta)
        # Wipe plaintext weights in memory to simulate zero-trust storage
        param.data.zero_()

    print(f"Encrypted {len(encrypted_vault)} parameter tensors into AHCC Vault (Plaintext weights zeroed).")

    # Decrypt parameter tensors sequentially, then run inference. The model's
    # parameters are ultimately restored to plaintext memory; this is not a
    # protected execution environment or a memory-confidential inference system.
    for name, param in model.named_parameters():
        ct_param, shape_meta, q_meta = encrypted_vault[name]
        rec_q_param = wrapper.decrypt_tensor(ct_param, shape_meta)
        param.data.copy_(dequantize_fp_to_fp32(rec_q_param.cpu(), q_meta))

    with torch.no_grad():
        restored_output = model(sample_input)

    max_output_diff = (baseline_output - restored_output).abs().max().item()
    print(f"Max Inference Output Deviation after Decryption: {max_output_diff:.2e}")
    assert max_output_diff < 1e-4, "Restored model inference output diverged!"
    print("[PASS] Toy model parameter round trip and output consistency passed.")
    print("NOTE: All parameters are restored to plaintext memory before inference.")


if __name__ == "__main__":
    demo_federated_learning_gradient_encryption()
    demo_on_the_fly_model_weight_decryption()
