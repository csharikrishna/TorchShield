"""Command Line Interface for inspecting, verifying, and managing AHCC Secure Checkpoints.

Provides zero-key container header inspection, inventory verification,
Ed25519 publisher provenance checking, and anti-rollback ledger auditing.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from typing import Optional

from ahcc.checkpoint import SecureModelCheckpoint, CheckpointError
from ahcc.distributed import verify_sharded_checkpoint, DistributedCheckpointError
from ahcc.provenance import Ed25519Verifier, FreshnessLedger, ProvenanceError, RollbackError


def cmd_version(args: argparse.Namespace) -> int:
    from importlib.metadata import version, PackageNotFoundError
    try:
        pkg_ver = version("ahcc-gpu-engine")
    except PackageNotFoundError:
        pkg_ver = "0.6.0-dev"

    print(f"AHCC Secure Checkpoint Engine v{pkg_ver}")
    print("Supported Container Formats: Version 1, Version 2, Version 4 (Hardened), Version 5 (KMS+Provenance)")
    print("Default Authenticated Cipher: AES-256-GCM (NIST SP 800-38D) with RFC 5869 HKDF-SHA256")
    print("Experimental Research Cipher: AHCC Invertible S-Box & Cauchy Diffusion (Quarantined)")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    path = args.checkpoint_path
    if not os.path.exists(path):
        print(f"Error: File '{path}' does not exist.", file=sys.stderr)
        return 1

    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic == SecureModelCheckpoint.MAGIC_V5:
            ver = f.read(1)[0]
            salt = f.read(SecureModelCheckpoint.SALT_SIZE_V5)
            art_id_len = f.read(1)[0]
            art_id = f.read(art_id_len)
            key_id_len = f.read(1)[0]
            key_id = f.read(key_id_len).decode("utf-8", errors="replace")
            wrapped_key_len = struct.unpack(">H", f.read(2))[0]
            wrapped_key = f.read(wrapped_key_len)
            manifest_len = struct.unpack(">I", f.read(4))[0]

            print("================================================================================")
            print(f"  AHCC Checkpoint Header Inspection: {os.path.basename(path)}")
            print("================================================================================")
            print(f"  Container Format:      Version 5 (Production KMS Envelope & Provenance)")
            print(f"  Magic Bytes:           {magic.decode('latin1', errors='replace')}")
            print(f"  Version:               {ver}")
            print(f"  File Size:             {file_size} bytes ({file_size / (1024*1024):.2f} MiB)")
            print(f"  Artifact ID:           {art_id.hex()} ({len(art_id)} bytes)")
            print(f"  Key ID / ARN:          {key_id or '[None / Direct Key]'}")
            print(f"  Envelope Wrapped DEK:  {wrapped_key_len} bytes {'[Present]' if wrapped_key_len > 0 else '[None]'}")
            print(f"  Encrypted Manifest:    {manifest_len} bytes")
            print(f"  Payload Chunks Offset: {f.tell()} bytes")
            print("================================================================================")
            return 0

        elif magic == SecureModelCheckpoint.MAGIC_V4:
            ver = f.read(1)[0]
            salt = f.read(SecureModelCheckpoint.SALT_SIZE_V4)
            art_id_len = f.read(1)[0]
            art_id = f.read(art_id_len)
            key_id_len = f.read(1)[0]
            key_id = f.read(key_id_len).decode("utf-8", errors="replace")
            manifest_len = struct.unpack(">I", f.read(4))[0]

            print("================================================================================")
            print(f"  AHCC Checkpoint Header Inspection: {os.path.basename(path)}")
            print("================================================================================")
            print(f"  Container Format:      Version 4 (Production Hardened)")
            print(f"  Magic Bytes:           {magic.decode('latin1', errors='replace')}")
            print(f"  Version:               {ver}")
            print(f"  File Size:             {file_size} bytes ({file_size / (1024*1024):.2f} MiB)")
            print(f"  Artifact ID:           {art_id.hex()} ({len(art_id)} bytes)")
            print(f"  Key ID:                {key_id or '[None]'}")
            print(f"  Encrypted Manifest:    {manifest_len} bytes")
            print("================================================================================")
            return 0

        elif magic == SecureModelCheckpoint.MAGIC_V2:
            print(f"AHCC Checkpoint: Legacy Version 2 Format ({file_size} bytes).")
            return 0
        elif magic == SecureModelCheckpoint.MAGIC_V1:
            print(f"AHCC Checkpoint: Legacy Version 1 Format ({file_size} bytes).")
            return 0
        else:
            print(f"Error: Unknown magic header: {magic!r}. Not an AHCC checkpoint.", file=sys.stderr)
            return 1


def cmd_verify_distributed(args: argparse.Namespace) -> int:
    checkpoint_dir = args.checkpoint_dir
    if not os.path.isdir(checkpoint_dir):
        print(f"Error: Directory '{checkpoint_dir}' does not exist.", file=sys.stderr)
        return 1

    pubkey_bytes: Optional[bytes] = None
    if args.public_key:
        if not os.path.exists(args.public_key):
            print(f"Error: Public key file '{args.public_key}' does not exist.", file=sys.stderr)
            return 1
        with open(args.public_key, "rb") as f:
            pubkey_bytes = f.read()

    ledger: Optional[FreshnessLedger] = None
    if args.ledger:
        ledger = FreshnessLedger(args.ledger)

    try:
        manifest = verify_sharded_checkpoint(
            checkpoint_dir=checkpoint_dir,
            expected_signer_public_key=pubkey_bytes,
            ledger=ledger,
        )
        print("================================================================================")
        print("  DISTRIBUTED CHECKPOINT INVENTORY VERIFICATION: PASS")
        print("================================================================================")
        print(f"  Run ID:        {manifest.get('run_id')}")
        print(f"  Global Step:   {manifest.get('global_step')}")
        print(f"  Epoch:         {manifest.get('epoch')}")
        print(f"  World Size:    {manifest.get('world_size')}")
        print(f"  Shards Found:  {len(manifest.get('shards', []))} / {manifest.get('world_size')}")
        print(f"  Digital Sig:   {'VERIFIED [OK]' if pubkey_bytes else 'PRESENT (Unchecked)' if 'signature' in manifest else 'NONE'}")
        if ledger:
            print(f"  Anti-Rollback: VERIFIED (Monotonic step {manifest.get('global_step')}) [OK]")
        print("================================================================================")
        return 0
    except (DistributedCheckpointError, RollbackError, ProvenanceError) as exc:
        print(f"VERIFICATION FAILURE: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="ahcc",
        description="AHCC Secure AI Model Checkpoint Management and Auditing Tool",
    )
    subparsers = parser.add_subparsers(dest="command")

    # version
    subparsers.add_parser("version", help="Show AHCC version and supported formats")

    # inspect
    inspect_parser = subparsers.add_parser("inspect", help="Inspect container header without keys")
    inspect_parser.add_argument("checkpoint_path", help="Path to .ahcc checkpoint file")

    # verify-distributed
    dist_parser = subparsers.add_parser(
        "verify-distributed", help="Verify distributed sharded checkpoint inventory"
    )
    dist_parser.add_argument("checkpoint_dir", help="Path to distributed checkpoint directory")
    dist_parser.add_argument("--public-key", help="Path to Ed25519 32-byte public key file")
    dist_parser.add_argument("--ledger", help="Path to freshness ledger JSON file")

    args = parser.parse_args()
    if args.command == "version":
        return cmd_version(args)
    elif args.command == "inspect":
        return cmd_inspect(args)
    elif args.command == "verify-distributed":
        return cmd_verify_distributed(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
