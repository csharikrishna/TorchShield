# Installation Guide for AHCC & Validated Data Protection Engine

## 1. Prerequisites

- **Operating System:** Windows 10/11 or Linux x86_64
- **Python:** Version 3.10, 3.11, or 3.12 (Tested on Python 3.12.10)
- **PyTorch:** Version 2.0 or newer with CUDA support (e.g. PyTorch 2.5.1+cu121)
- **NVIDIA GPU:** Compute Capability 7.0+ recommended for GPU execution (Tested on Turing GTX 1650, CC 7.5)
- **Dependencies:** `cryptography>=42.0`, `torch>=2.0.0`, `pytest>=8.0.0`, `matplotlib`

---

## 2. Environment Setup

### 2.1 CPU-Only Setup
```bash
python -m venv venv
venv\Scripts\activate  # On Linux: source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-build-isolation --no-deps
```

### 2.2 CUDA GPU Setup
1. Verify NVIDIA drivers and CUDA support:
   ```bash
   nvidia-smi
   ```
2. Install PyTorch with CUDA support matching your system:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
3. Install project and development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pytest matplotlib
   pip install -e . --no-build-isolation --no-deps
   ```
4. Verify CUDA runtime availability:
   ```bash
   python -m benchmarks.check_cuda
   ```
