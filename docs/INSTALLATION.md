# Installation Guide for TorchShield Secure AI Checkpoint Platform

**Document:** `docs/INSTALLATION.md`  
**Platform Version:** `0.7.0`  

---

## 1. Prerequisites

- **Operating System:** Windows 10/11 or Linux x86_64
- **Python:** 3.10, 3.11, or 3.12 (Validated on Python 3.12.10)
- **PyTorch:** Version 2.0 or newer with CUDA or CPU support (e.g. PyTorch 2.5.1+cu121)
- **Hardware:**
  - *GPU Mode (Recommended for training):* NVIDIA GPU with Compute Capability 7.0+ (Tested on GTX 1650, RTX 30/40 series, A100/H100)
  - *CPU Mode:* Supported for CPU-only training, checkpoint inspection, verification, and migration.
- **Core Dependencies:**
  - `cryptography>=42.0.0`
  - `torch>=2.0.0`
  - `pytest>=8.0.0` (for testing)

---

## 2. Installation Options

### Option A: Install from Pre-Built Wheel (Recommended for Pilot / Production)

Install the standalone distribution wheel without needing build tools:

```bash
# In your target virtual environment:
pip install dist/release_v070/torchshield-0.7.0-py3-none-any.whl
```

Verify checksums against [`dist/release_v070/SHA256SUMS`](../dist/release_v070/SHA256SUMS):
```bash
# PowerShell:
Get-FileHash dist\release_v070\torchshield-0.7.0-py3-none-any.whl -Algorithm SHA256
# Linux:
sha256sum -c dist/release_v070/SHA256SUMS
```

### Option B: Editable Developer Installation

If developing or running repository test suites:

```bash
# 1. Clone and enter repository
cd TorchShield

# 2. Create virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install pytest

# 4. Install in editable mode
pip install -e . --no-build-isolation --no-deps
```

---

## 3. Post-Installation Verification

Run a quick verification command to confirm import isolation and package version:

```python
import torchshield
print(f"TorchShield Platform Version: {torchshield.__version__}")
# Output: TorchShield Platform Version: 0.7.0
```

Verify the CLI tool:
```bash
torchshield --help
```

Run test suite:
```bash
pytest tests
```
