# Phase 4 Resource Allocation Limits, Denial-of-Service Defenses, and Durability Audit

**Document:** `reports/PHASE4_ALLOCATION_AND_DURABILITY.md`  
**Date:** 2026-10-26  
**Auditor:** Senior Systems & Storage Engineer, Security Auditor  
**Scope:** Memory safety, allocation bounding, crash consistency, and filesystem durability semantics  

---

## 1. Allocation Bounds & Denial-of-Service (DoS) Protection

In unauthenticated or poorly bounded checkpoint formats, malicious or corrupted container headers can trigger catastrophic out-of-memory (OOM) crashes, system thrashing, or resource exhaustion before cryptographic authentication occurs.

### The Dimension Limit Defect in Prior Versions
In Phase 2 and Phase 3, the dimension ceiling was defined as `MAX_DIM_SIZE = 2**31 - 1` (2,147,483,647 elements). While overflow during multiplication was prevented, allowing a single dimension of 2 billion elements meant a crafted header could declare a single tensor of $2\times 10^9 \times 4\text{ bytes} \approx 8\text{ GiB}$ or more. On standard AI training nodes with 4 GB to 16 GB VRAM, allocating such a buffer triggers an immediate unhandled `MemoryError` or process abort.

### Phase 4 Defense-in-Depth: Configurable Resource Limits
Phase 4 introduces a formal, immutable `ResourceLimits` configuration dataclass:

```python
@dataclass(frozen=True)
class ResourceLimits:
    max_file_size: int = 128 * 1024 * 1024 * 1024        # 128 GiB max file size
    max_total_raw_bytes: int = 64 * 1024 * 1024 * 1024   # 64 GiB max aggregate decoded tensor bytes
    max_tensor_bytes: int = 16 * 1024 * 1024 * 1024      # 16 GiB max single tensor bytes
    max_manifest_bytes: int = 16 * 1024 * 1024           # 16 MiB max manifest length
    max_tensor_count: int = 50_000                       # 50,000 tensors max
    max_tensor_rank: int = 8                             # rank <= 8
    max_chunk_count: int = 1_000_000                     # 1,000,000 chunks max
    max_chunk_bytes: int = 64 * 1024 * 1024              # 64 MiB max single chunk
    max_dim_size: int = 100_000_000                      # 100,000,000 elements per dim
    max_total_elements: int = 16 * 1024 * 1024 * 1024    # 16G aggregate elements
```

### Allocation Pre-Verification Checklist
Before a single byte of memory is allocated for a tensor or buffer, the following checks are enforced in constant time:
1. **Container Header Limits:**
   - Manifest declared byte length $L_{\text{manifest}}$ is verified against `limits.max_manifest_bytes` (16 MiB default).
   - If the underlying file stream is seekable, total file size is verified against `limits.max_file_size` (128 GiB default).
2. **Manifest Specification Limits (Authenticated First):**
   - Total declared layers $\le \text{limits.max\_tensor\_count}$ (50,000).
   - Total declared raw bytes $\le \text{limits.max\_total\_raw\_bytes}$ (64 GiB).
   - Total declared chunks $\le \text{limits.max\_chunk\_count}$ (1,000,000).
3. **Per-Tensor Dimension Arithmetic:**
   - Rank $\le \text{limits.max\_tensor\_rank}$ (8).
   - Each individual dimension $0 \le \text{dim} \le \text{limits.max\_dim\_size}$ (100,000,000).
   - Cumulative element multiplication checks overflow: $\text{numel} \le \text{limits.max\_total\_elements}$.
   - Exact byte calculation: $\text{expected\_bytes} = \text{numel} \times \text{DTYPE\_ITEMSIZE}[\text{dtype}]$.
   - Single tensor size check: $\text{expected\_bytes} \le \text{limits.max\_tensor\_bytes}$ (16 GiB).
   - Exact length equality: $\text{declared\_raw\_len} == \text{expected\_bytes}$.
   - Only after all checks pass is `bytearray(expected_bytes)` allocated.

---

## 2. Crash Consistency, Atomicity, and Durability Audit

### Write Path Mechanics
Saving a checkpoint in `ahcc.checkpoint.SecureModelCheckpoint` follows an atomic staging protocol:
1. **Staging File Creation:** Writes all header bytes, encrypted manifest, and encrypted chunks into a unique staging file:
   $$\text{staging\_path} = \text{output\_path} + f\text{".tmp."} + \text{uuid.uuid4}().\text{hex}$$
2. **Buffer Flush:** Explicitly calls `f.flush()` to ensure user-space runtime buffers are written to OS file descriptor cache.
3. **Filesystem Synchronization:** Calls `os.fsync(f.fileno())` to instruct the operating system storage stack to commit all dirty metadata and file data blocks to non-volatile physical storage media.
4. **Atomic Replacement:** Calls `os.replace(staging_path, output_path)` to atomically swap the staging file over the destination file.
5. **Fail-Closed Cleanup:** If an exception occurs at any point during writing, `staging_path` is immediately removed via `os.remove` inside the exception handler, leaving no partial or corrupt staging files on disk.

### Operating System & Filesystem Durability Semantics

| Platform / Filesystem | Atomicity (`os.replace`) | Durability (`os.fsync`) | Limitations & Caveats |
| :--- | :--- | :--- | :--- |
| **Windows (NTFS)** | **Atomic** (`SetFileInformationByHandle` / `MoveFileExW`) | **Guaranteed** (`FlushFileBuffers`) | File handles must be cleanly closed before deletion/replacement; open handles raise `PermissionError`. |
| **Linux (ext4, XFS)** | **Atomic** (`renameat2` / POSIX `rename`) | **Guaranteed** (Data + inode journal flushed) | Directory metadata must be fsynced separately (`os.fsync(dir_fd)`) for full directory durability. |
| **macOS (APFS)** | **Atomic** (`renamex_np` / `RENAME_SWAP`) | **Partial** (`fcntl(F_FULLFSYNC)`) | Standard `os.fsync` on macOS flushes to drive cache only; requires `fcntl(fd, F_FULLFSYNC)` for disk platter flush. |
| **NFS / SMB Network Storage** | **Not Guaranteed** (Varies by mount flags) | **Server-Dependent** | `os.replace` over network shares may not be atomic under concurrent clients or aggressive caching. |
| **Cloud Object Storage (S3 / GCS)** | **N/A (Object Model)** | **N/A (Multi-part upload)** | FUSE-based mounts (e.g., s3fs, gcsfuse) do not support POSIX atomic rename semantics. |

### Failure Recovery Boundaries
- **Process Termination (SIGKILL / Crash):** **VERIFIED**. If the saving process is terminated abruptly before step 4, the existing valid checkpoint at `output_path` remains completely untouched. The leftover `.tmp.<uuid>` file is orphaned and can be swept without affecting system integrity.
- **Simulated Disk Full:** **VERIFIED**. If disk space is exhausted during writing, `os.fsync` raises `OSError`, triggering immediate cleanup of the partial staging file.
- **Power Loss / Host Reset:** **PARTIAL / CONDITIONAL**. While `os.fsync` instructs the disk controller to commit buffers, consumer SSDs with volatile write caches may reorder or lose in-flight blocks during sudden power cuts unless backed by battery-backed write cache (BBU) or SuperCAP power-loss protection (PLP).
