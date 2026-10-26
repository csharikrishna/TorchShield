"""Research prototype for Adaptive Hyperdimensional Chaotic Cryptography (AHCC).

IMPORTANT SECURITY NOTICE
-------------------------
This module implements the experimental AHCC construction described in the
accompanying research blueprint. It is NOT a production-grade encryption
scheme. Its default round-key schedule derives keys from a 256-bit master key
using HKDF-SHA256; the explicit legacy ``master_seed`` mode retains the older
weak finite-state recurrence for reproducibility only. Neither mode authenticates
ciphertexts, and the full custom construction has not undergone independent
cryptanalysis. Do not use this module to protect sensitive or valuable data.
Use the separate AESGCMEnvelope API (or another vetted AEAD) for real data.

The implementation focuses on exact modular arithmetic and invertible
transformations for research reproducibility. It supports only the two prime
moduli validated by this project and a dense Cauchy diffusion matrix up to
MAX_SUPPORTED_N. The dense implementation is O(N^2) in storage and O(N^2)
per state vector, so large-N use requires a different diffusion design.
"""
from __future__ import annotations

import math
import secrets
from typing import Optional, Tuple

import torch
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


SUPPORTED_PRIMES = (2_147_483_579, 4_294_967_291)  # 2**32 - 5 is the paper's target.
MAX_SUPPORTED_N = 1024
MAX_SUPPORTED_R = 64
_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


class AHCC_Engine:
    """Exact-integer implementation of the experimental AHCC round function.

    State tensors are represented as vectors over F_p. Direct ``encrypt`` and
    ``decrypt`` accept a vector of shape ``(N,)`` or a matrix of shape
    ``(N, batch)``. Matrices with shape ``(batch, N)`` are accepted and returned
    in that orientation when ``batch != N``. If both dimensions equal N, the
    documented ``(N, batch)`` convention takes precedence.

    If neither ``master_key`` nor ``master_seed`` is supplied, a random
    256-bit master key is generated and exposed as ``engine.master_key`` so it
    can be stored securely and used to reconstruct the engine. Supplying an
    integer ``master_seed`` explicitly selects the legacy deterministic chaotic
    recurrence for reproducibility only. Neither mode makes this custom cipher
    suitable for sensitive data: the core construction remains unreviewed and
    the direct ``encrypt``/``decrypt`` API has no authentication.

    This class guarantees round-trip correctness for supported parameters; it
    does not claim cryptographic confidentiality or integrity.
    """

    def __init__(
        self,
        N: int = 64,
        R: int = 4,
        p: int = 2_147_483_579,
        master_seed: Optional[int] = None,
        device: Optional[str] = None,
        master_key: Optional[bytes] = None,
    ):
        if isinstance(N, bool) or not isinstance(N, int) or N < 1:
            raise ValueError("N must be a positive integer.")
        if N > MAX_SUPPORTED_N:
            raise ValueError(
                f"N={N} exceeds this dense implementation's safe limit "
                f"N <= {MAX_SUPPORTED_N}. The Cauchy matrix and its inverse "
                "require O(N^2) memory; use a structured/sparse diffusion layer "
                "for larger dimensions."
            )
        if isinstance(R, bool) or not isinstance(R, int) or R < 1 or R > MAX_SUPPORTED_R:
            raise ValueError(
                f"R must be an integer between 1 and {MAX_SUPPORTED_R}; "
                "R=0 provides no rounds and larger values exceed the supported resource limit."
            )
        if isinstance(p, bool) or not isinstance(p, int) or p not in SUPPORTED_PRIMES:
            allowed = ", ".join(str(v) for v in SUPPORTED_PRIMES)
            raise ValueError(f"Unsupported modulus p={p!r}. Supported prime moduli: {allowed}.")
        if master_seed is not None and (isinstance(master_seed, bool) or not isinstance(master_seed, int)):
            raise TypeError("master_seed must be an integer when supplied.")
        if master_key is not None and not isinstance(master_key, (bytes, bytearray, memoryview)):
            raise TypeError("master_key must be a 32-byte key when supplied.")
        if master_key is not None and len(master_key) != 32:
            raise ValueError("master_key must contain exactly 32 bytes (256 bits).")
        if master_key is not None and master_seed is not None:
            raise ValueError("Supply either master_key or master_seed, not both.")
        # With x_i=i and y_j=N+j, all denominators range from N to 3N-2.
        if 3 * N - 2 >= p:
            raise ValueError("N is too large for the selected Cauchy construction over F_p.")

        requested_device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.device = torch.device(requested_device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError(
                f"Unsupported device type {self.device.type!r}; this implementation supports only CPU and CUDA."
            )
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"CUDA device {self.device} was requested, but CUDA is not available "
                    "in the installed PyTorch runtime. Use device='cpu' or install a CUDA build."
                )
            if self.device.index is None:
                # Pin the engine to the current CUDA device so changing the global
                # device later cannot move newly created tensors to another GPU.
                self.device = torch.device("cuda", torch.cuda.current_device())
            elif self.device.index < 0 or self.device.index >= torch.cuda.device_count():
                raise ValueError(
                    f"CUDA device index {self.device.index} is out of range for "
                    f"{torch.cuda.device_count()} available CUDA device(s)."
                )

        self.N = N
        self.R = R
        self.p = p
        self.bit_length = p.bit_length()
        # The legacy seed mode is retained for reproducible experiments only.
        # Default operation uses 256 bits of key material and HKDF-SHA256 with
        # parameter/round domain separation. This improves key derivation but
        # does not, by itself, establish security of the AHCC round function.
        if master_seed is not None:
            self.master_seed = master_seed
            self.master_key = None
            self.key_schedule_mode = "legacy-chaotic-seed (research-only)"
        else:
            self.master_seed = None
            self.master_key = bytes(master_key) if master_key is not None else secrets.token_bytes(32)
            self.key_schedule_mode = "HKDF-SHA256 (experimental AHCC round keys)"

        # x^3 is a permutation over F_p exactly when gcd(3, p - 1) == 1.
        if math.gcd(3, self.p - 1) != 1:  # Defensive guard if the supported list changes.
            raise ValueError("The cubic S-box is not invertible for this modulus.")
        self.d = pow(3, -1, self.p - 1)

        self.M, self.M_inv = self._build_cauchy_mds_matrices()
        if self.master_seed is not None:
            self.round_keys = self._generate_chaotic_round_keys(self.master_seed)
        else:
            self.round_keys = self._generate_hkdf_round_keys(self.master_key)

    def _build_cauchy_mds_matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the Cauchy matrix and its analytical inverse modulo p."""
        p, N = self.p, self.N
        x = list(range(N))
        y = list(range(N, 2 * N))

        M_list = [
            [pow((x[i] + y[j]) % p, -1, p) for j in range(N)]
            for i in range(N)
        ]

        # For C_ij = 1/(x_i + y_j), this closed form gives the transpose
        # indexing of the inverse (see comments in the project research notes).
        A = [1] * N
        B = [1] * N
        C = [1] * N
        D = [1] * N
        for i, xi in enumerate(x):
            a_prod = c_prod = 1
            for k in range(N):
                a_prod = (a_prod * (xi + y[k])) % p
                if k != i:
                    c_prod = (c_prod * (xi - x[k])) % p
            A[i], C[i] = a_prod, c_prod
        for j, yj in enumerate(y):
            b_prod = d_prod = 1
            for k in range(N):
                b_prod = (b_prod * (x[k] + yj)) % p
                if k != j:
                    d_prod = (d_prod * (yj - y[k])) % p
            B[j], D[j] = b_prod, d_prod

        alpha = [(A[i] * pow(C[i], -1, p)) % p for i in range(N)]
        beta = [(B[j] * pow(D[j], -1, p)) % p for j in range(N)]
        M_inv_list = [
            [(beta[j] * alpha[i] % p) * M_list[i][j] % p for i in range(N)]
            for j in range(N)
        ]

        M = torch.tensor(M_list, dtype=torch.int64, device=self.device)
        M_inv = torch.tensor(M_inv_list, dtype=torch.int64, device=self.device)
        return M, M_inv

    def _generate_hkdf_round_keys(self, master_key: bytes) -> torch.Tensor:
        """Derive round keys from 256-bit key material using HKDF-SHA256.

        This replaces the low-entropy integer seed in the default mode. A
        rejection-sampling step maps each 32-bit word uniformly into F_p.
        The output is deterministic for a given master key and parameter set.
        This is a stronger key-derivation procedure, not a proof that the AHCC
        block transformation is secure.
        """
        rows = []
        salt = b"AHCC-HKDF-SHA256-key-schedule-v1"
        parameter_context = (
            b"N=" + str(self.N).encode("ascii")
            + b";R=" + str(self.R).encode("ascii")
            + b";p=" + str(self.p).encode("ascii")
        )
        for round_index in range(self.R + 1):
            accepted = []
            batch_index = 0
            while len(accepted) < self.N:
                info = (
                    b"AHCC experimental round key;" + parameter_context
                    + b";round=" + str(round_index).encode("ascii")
                    + b";batch=" + str(batch_index).encode("ascii")
                )
                # N*4 is <= 4096 for this implementation, below HKDF's
                # 255*SHA-256-digest output limit.
                material = HKDF(
                    algorithm=hashes.SHA256(),
                    length=self.N * 4,
                    salt=salt,
                    info=info,
                ).derive(master_key)
                for offset in range(0, len(material), 4):
                    value = int.from_bytes(material[offset:offset + 4], "big")
                    if value < self.p:
                        accepted.append(value)
                        if len(accepted) == self.N:
                            break
                batch_index += 1
                if batch_index > 32:  # Defensive bound; failure would be extraordinary.
                    raise RuntimeError("Unable to derive enough field elements from the key material.")
            rows.append(accepted)
        return torch.tensor(rows, dtype=torch.int64, device=self.device).view(self.R + 1, self.N, 1)

    def _generate_chaotic_round_keys(self, seed: int) -> torch.Tensor:
        """Generate deterministic experimental round keys over F_p.

        WARNING: This finite-state recurrence is included for fidelity to the
        research prototype. Sensitivity to a seed change is not a security proof,
        and this generator is not a vetted cryptographic KDF or PRNG.
        """
        p = self.p
        total_elements = (self.R + 1) * self.N
        keys = []
        state = ((seed % (p - 3)) + 3) % p
        a = 2_654_435_761 % p
        b = 1_664_525 % p
        c = 1_013_904_223 % p

        for _ in range(64):
            state = (a * ((state * state) % p) + b * state + c) % p
        for _ in range(total_elements):
            state = (a * ((state * state) % p) + b * state + c) % p
            keys.append(state)
        return torch.tensor(keys, dtype=torch.int64, device=self.device).view(self.R + 1, self.N, 1)

    def _as_integer_tensor(self, value: torch.Tensor, *, name: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if value.dtype not in _INTEGER_DTYPES:
            raise TypeError(
                f"{name} must use an integer dtype; received {value.dtype}. "
                "For floating-point AI tensors, explicitly quantize them to F_p first."
            )
        return value.to(device=self.device, dtype=torch.int64)

    def _normalize_state(self, value: torch.Tensor, *, name: str):
        state = self._as_integer_tensor(value, name=name)
        was_1d = state.dim() == 1
        transposed = False
        if was_1d:
            if state.shape[0] != self.N:
                raise ValueError(f"{name} vector must have exactly N={self.N} elements; got {tuple(state.shape)}.")
            state = state.unsqueeze(1)
        elif state.dim() == 2:
            if state.shape[0] == self.N:
                pass
            elif state.shape[1] == self.N:
                state = state.t().contiguous()
                transposed = True
            else:
                raise ValueError(
                    f"{name} matrix must have shape (N, batch) or (batch, N) with N={self.N}; "
                    f"got {tuple(state.shape)}."
                )
        else:
            raise ValueError(f"{name} must be a 1D vector or 2D batch matrix; got {state.dim()} dimensions.")
        return state % self.p, was_1d, transposed

    def safe_mod_mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute elementwise ``a*b mod p`` without signed-int64 overflow."""
        a = self._as_integer_tensor(a, name="a") % self.p
        b = self._as_integer_tensor(b, name="b") % self.p
        if self.bit_length <= 31:
            return (a * b) % self.p
        # For the 32-bit modulus, split b into 16-bit limbs so each product is < 2^48.
        b_lo = b & 0xFFFF
        b_hi = b >> 16
        term_lo = (a * b_lo) % self.p
        term_hi = (a * b_hi) % self.p
        term_hi_shifted = (term_hi << 16) % self.p
        return (term_hi_shifted + term_lo) % self.p

    def sbox_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the bijective map S(x) = x^3 mod p."""
        x = self._as_integer_tensor(x, name="x") % self.p
        return self.safe_mod_mul(self.safe_mod_mul(x, x), x)

    def sbox_inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Apply S^-1(y) = y^d mod p with exact modular exponentiation."""
        base = self._as_integer_tensor(y, name="y") % self.p
        result = torch.ones_like(base, dtype=torch.int64, device=self.device)
        exp = self.d
        while exp > 0:
            if exp & 1:
                result = self.safe_mod_mul(result, base)
            base = self.safe_mod_mul(base, base)
            exp >>= 1
        return result

    def safe_mod_matmul(self, M: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Compute ``M @ V mod p`` exactly, splitting V into radix-2^11 limbs."""
        M = self._as_integer_tensor(M, name="M") % self.p
        V = self._as_integer_tensor(V, name="V") % self.p
        if M.dim() != 2 or tuple(M.shape) != (self.N, self.N):
            raise ValueError(f"M must have shape ({self.N}, {self.N}); got {tuple(M.shape)}.")
        if V.dim() != 2 or V.shape[0] != self.N:
            raise ValueError(f"V must be a 2D matrix with first dimension N={self.N}; got {tuple(V.shape)}.")

        V_0 = V & 0x7FF
        V_1 = (V >> 11) & 0x7FF
        V_2 = V >> 22

        chunk_size = 32768
        if self.N <= chunk_size:
            prod_0 = self._int64_mm(M, V_0) % self.p
            prod_1 = self._int64_mm(M, V_1) % self.p
            prod_2 = self._int64_mm(M, V_2) % self.p
        else:  # Kept as a defensive path if the dimension cap is raised in future.
            shape = (self.N, V.shape[1])
            prod_0 = torch.zeros(shape, dtype=torch.int64, device=self.device)
            prod_1 = torch.zeros_like(prod_0)
            prod_2 = torch.zeros_like(prod_0)
            for start in range(0, self.N, chunk_size):
                end = min(start + chunk_size, self.N)
                prod_0 = (prod_0 + self._int64_mm(M[:, start:end], V_0[start:end, :])) % self.p
                prod_1 = (prod_1 + self._int64_mm(M[:, start:end], V_1[start:end, :])) % self.p
                prod_2 = (prod_2 + self._int64_mm(M[:, start:end], V_2[start:end, :])) % self.p

        # prod_l < p < 2^32, so these shifts remain safely inside signed int64.
        term_2 = (prod_2 << 22) % self.p
        term_1 = (prod_1 << 11) % self.p
        return (term_2 + term_1 + prod_0) % self.p

    def _int64_mm(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Exact int64 matrix product, using bounded temporary tiles on CUDA."""
        if A.device != B.device:
            raise ValueError("A and B must be on the same device.")
        if A.dim() != 2 or B.dim() != 2 or A.shape[1] != B.shape[0]:
            raise ValueError(f"Incompatible matrix shapes {tuple(A.shape)} and {tuple(B.shape)}.")
        if A.shape[1] == 0:
            return torch.zeros((A.shape[0], B.shape[1]), dtype=torch.int64, device=A.device)
        if B.shape[1] == 0:
            return torch.empty((A.shape[0], 0), dtype=torch.int64, device=A.device)
        if A.device.type == "cpu":
            return torch.matmul(A, B)

        batch_size = B.shape[1]
        # 8M int64 values (~64 MiB) per temporary tile; avoid unbounded allocations.
        tile_b = max(1, (8 * 1024 * 1024) // max(1, A.shape[0] * A.shape[1]))
        out = torch.empty((A.shape[0], batch_size), dtype=torch.int64, device=A.device)
        for b_start in range(0, batch_size, tile_b):
            b_end = min(b_start + tile_b, batch_size)
            out[:, b_start:b_end] = (
                A.unsqueeze(2) * B[:, b_start:b_end].unsqueeze(0)
            ).sum(dim=1)
        return out

    def encrypt(self, plaintext: torch.Tensor) -> torch.Tensor:
        """Transform finite-field state vectors or a batch of vectors.

        This is a deterministic research transform, not authenticated encryption.
        The output is deterministic under a fixed key/configuration; the direct
        method has no nonce, integrity tag, or production security guarantee.
        """
        state, was_1d, transposed = self._normalize_state(plaintext, name="plaintext")
        for r in range(self.R):
            state = (state + self.round_keys[r]) % self.p
            state = self.sbox_forward(state)
            state = self.safe_mod_matmul(self.M, state)
            state = (state + self.round_keys[r + 1]) % self.p
        if was_1d:
            return state.squeeze(1)
        return state.t().contiguous() if transposed else state

    def decrypt(self, ciphertext: torch.Tensor) -> torch.Tensor:
        """Invert ``encrypt`` for the same engine parameters and key schedule."""
        state, was_1d, transposed = self._normalize_state(ciphertext, name="ciphertext")
        for r in range(self.R - 1, -1, -1):
            state = (state - self.round_keys[r + 1]) % self.p
            state = self.safe_mod_matmul(self.M_inv, state)
            state = self.sbox_inverse(state)
            state = (state - self.round_keys[r]) % self.p
        if was_1d:
            return state.squeeze(1)
        return state.t().contiguous() if transposed else state


class TensorCipherWrapper:
    """Flatten/pad/encrypt/recover integer tensors containing F_p elements.

    This wrapper intentionally accepts only ``torch.int64`` tensors whose values
    are already in ``[0, p-1]``. Floating-point tensors must first be quantized
    explicitly (see ``demos/ai_security_use_cases.py``); otherwise silently
    truncating or wrapping values would make recovery ambiguous.
    """

    def __init__(self, engine: AHCC_Engine):
        if not isinstance(engine, AHCC_Engine):
            raise TypeError("engine must be an AHCC_Engine instance.")
        self.engine = engine

    def encrypt_tensor(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("tensor must be a torch.Tensor.")
        if tensor.dtype != torch.int64:
            raise TypeError(
                f"TensorCipherWrapper expects torch.int64 field elements; received {tensor.dtype}. "
                "Quantize float tensors explicitly and convert to int64 first."
            )
        flat = tensor.to(device=self.engine.device).reshape(-1)
        if flat.numel() and (bool(torch.any(flat < 0).item()) or bool(torch.any(flat >= self.engine.p).item())):
            raise ValueError(f"All tensor values must be in the finite-field range [0, {self.engine.p - 1}].")

        orig_shape = tuple(tensor.shape)
        total_len = flat.numel()
        N = self.engine.N
        pad_len = (N - (total_len % N)) % N
        if pad_len:
            flat = torch.cat([flat, torch.zeros(pad_len, dtype=torch.int64, device=self.engine.device)])
        if flat.numel() == 0:
            ciphertext = torch.empty((N, 0), dtype=torch.int64, device=self.engine.device)
        else:
            matrix_view = flat.view(-1, N).t().contiguous()
            ciphertext = self.engine.encrypt(matrix_view)

        meta = {
            "version": 1,
            "orig_shape": orig_shape,
            "orig_dtype": torch.int64,
            "total_len": total_len,
            "pad_len": pad_len,
            "engine_N": self.engine.N,
            "engine_R": self.engine.R,
            "engine_p": self.engine.p,
        }
        return ciphertext, meta

    def decrypt_tensor(self, ciphertext: torch.Tensor, meta: dict) -> torch.Tensor:
        if not isinstance(meta, dict):
            raise TypeError("meta must be the metadata dictionary returned by encrypt_tensor().")
        required = {"version", "orig_shape", "orig_dtype", "total_len", "pad_len", "engine_N", "engine_R", "engine_p"}
        missing = sorted(required - set(meta))
        if missing:
            raise ValueError(f"Metadata is missing required fields: {', '.join(missing)}.")
        if isinstance(meta["version"], bool) or not isinstance(meta["version"], int) or meta["version"] != 1:
            raise ValueError(f"Unsupported ciphertext metadata version: {meta['version']!r}.")
        expected_config = (self.engine.N, self.engine.R, self.engine.p)
        metadata_config = (meta["engine_N"], meta["engine_R"], meta["engine_p"])
        if any(isinstance(value, bool) or not isinstance(value, int) for value in metadata_config):
            raise ValueError("Cipher metadata engine_N, engine_R and engine_p must be integers.")
        if metadata_config != expected_config:
            raise ValueError(
                f"Cipher metadata expects engine (N, R, p)={metadata_config}, "
                f"but the current engine is {expected_config}."
            )
        if meta["orig_dtype"] != torch.int64:
            raise ValueError("Unsupported metadata dtype; only torch.int64 field tensors are supported.")
        shape_raw = meta["orig_shape"]
        if not isinstance(shape_raw, (tuple, list, torch.Size)):
            raise ValueError("Metadata orig_shape must be a sequence of non-negative integers.")
        if any(isinstance(dim, bool) or not isinstance(dim, int) for dim in shape_raw):
            raise ValueError("Metadata orig_shape dimensions must be integers, not booleans or coerced values.")
        shape = tuple(shape_raw)
        if any(dim < 0 for dim in shape):
            raise ValueError("Metadata shape dimensions must be non-negative.")
        shape_len = math.prod(shape) if shape else 1
        total_len_raw = meta["total_len"]
        pad_len_raw = meta["pad_len"]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (total_len_raw, pad_len_raw)):
            raise ValueError("Metadata total_len and pad_len must be integers, not booleans or coerced values.")
        total_len = total_len_raw
        pad_len = pad_len_raw
        if total_len < 0 or total_len != shape_len:
            raise ValueError("Metadata total_len does not match orig_shape.")
        expected_pad = (self.engine.N - (total_len % self.engine.N)) % self.engine.N
        if pad_len != expected_pad:
            raise ValueError(f"Invalid metadata pad_len={pad_len}; expected {expected_pad}.")
        if not isinstance(ciphertext, torch.Tensor) or ciphertext.dtype not in _INTEGER_DTYPES:
            raise TypeError("ciphertext must be an integer torch.Tensor.")
        expected_blocks = (total_len + pad_len) // self.engine.N
        if tuple(ciphertext.shape) != (self.engine.N, expected_blocks):
            raise ValueError(
                f"Ciphertext must have shape ({self.engine.N}, {expected_blocks}); got {tuple(ciphertext.shape)}."
            )

        ciphertext = ciphertext.to(device=self.engine.device, dtype=torch.int64)
        if ciphertext.numel() and (
            bool(torch.any(ciphertext < 0).item()) or bool(torch.any(ciphertext >= self.engine.p).item())
        ):
            raise ValueError(f"Ciphertext values must be in [0, {self.engine.p - 1}].")
        if expected_blocks == 0:
            return torch.empty(shape, dtype=torch.int64, device=self.engine.device)
        decrypted_matrix = self.engine.decrypt(ciphertext)
        flat = decrypted_matrix.t().contiguous().view(-1)
        return flat[:total_len].view(shape)
