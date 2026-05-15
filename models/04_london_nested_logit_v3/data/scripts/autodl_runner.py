"""Automated SSH+SCP runner for AutoDL experiments.

Uses paramiko to:
  1. SCP a bundle tarball to AutoDL
  2. SSH execute commands (extract, train)
  3. SCP result JSON(s) back

Reads SSH credentials from environment variables:
    AUTODL_HOST = connect.nmb2.seetacloud.com
    AUTODL_PORT = 15859
    AUTODL_USER = root
    AUTODL_PASSWORD = <password>

Or pass via CLI flags. Avoid committing passwords.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

import paramiko


REMOTE_BASE = "/root/autodl-tmp/v3_nested_smoke"
REMOTE_V3 = f"{REMOTE_BASE}/04_london_nested_logit_v3"


def connect(host, port, user, password):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=int(port), username=user, password=password,
                   timeout=30, banner_timeout=30, auth_timeout=30)
    return client


def scp_to(client, local_path: str, remote_path: str):
    sftp = client.open_sftp()
    print(f"[scp →] {local_path} → {remote_path}")
    sftp.put(local_path, remote_path)
    sftp.close()


def scp_from(client, remote_path: str, local_path: str):
    sftp = client.open_sftp()
    print(f"[scp ←] {remote_path} → {local_path}")
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    sftp.get(remote_path, local_path)
    sftp.close()


def run_cmd(client, command: str, tail_lines: int = 30):
    """Run a command (interactive shell to source ~/.bashrc for conda PATH)."""
    full = f"bash -lc '{command}'"
    print(f"[ssh] {command[:120]}{'...' if len(command) > 120 else ''}")
    stdin, stdout, stderr = client.exec_command(full, get_pty=True)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    if tail_lines:
        lines = (out + err).splitlines()
        for line in lines[-tail_lines:]:
            # Encode-safe print: strip chars that can't be encoded on Windows GBK terminal
            safe = line.encode("ascii", errors="replace").decode("ascii")
            print(f"   {safe}")
    rc = stdout.channel.recv_exit_status()
    return rc, out, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("AUTODL_HOST", "connect.nmb2.seetacloud.com"))
    ap.add_argument("--port", default=os.environ.get("AUTODL_PORT", "15859"))
    ap.add_argument("--user", default=os.environ.get("AUTODL_USER", "root"))
    ap.add_argument("--password", default=os.environ.get("AUTODL_PASSWORD"))
    ap.add_argument("--bundle", required=True, help="local path to bundle tar.gz")
    ap.add_argument("--remote-bundle-name", default=None, help="(default: basename)")
    ap.add_argument("--commands", nargs="+", required=True, help="commands to run on AutoDL after extracting bundle")
    ap.add_argument("--pull", nargs="*", default=[], help="remote paths (relative to v3 root) to scp back to local v3 evaluation_outputs/")
    ap.add_argument("--tail", type=int, default=30)
    args = ap.parse_args()

    if not args.password:
        sys.exit("AUTODL_PASSWORD env var or --password required.")

    bundle = Path(args.bundle).resolve()
    if not bundle.exists():
        sys.exit(f"Bundle not found: {bundle}")
    remote_bundle_name = args.remote_bundle_name or bundle.name

    print(f"=== Connecting to {args.user}@{args.host}:{args.port} ===")
    client = connect(args.host, args.port, args.user, args.password)
    try:
        # 1. SCP bundle up
        remote_bundle = f"{REMOTE_BASE}/{remote_bundle_name}"
        scp_to(client, str(bundle), remote_bundle)

        # 2. Extract
        rc, _, _ = run_cmd(client, f"cd {REMOTE_V3} && tar xzf ../{remote_bundle_name}", tail_lines=5)
        if rc != 0:
            sys.exit(f"tar extract failed: rc={rc}")

        # 3. Run user commands sequentially
        for cmd in args.commands:
            t0 = time.time()
            rc, out, err = run_cmd(client, f"cd {REMOTE_V3} && {cmd}", tail_lines=args.tail)
            elapsed = time.time() - t0
            print(f"   (took {elapsed:.0f}s, rc={rc})")
            if rc != 0:
                print(f"!! command failed: {cmd}")
                print(err[-2000:] if err else "")
                break

        # 4. Pull results back
        # bundle path is .../<v3_root>/data/autodl_bundle/file → go up 3 to v3 root
        local_v3 = bundle.parent.parent.parent
        for rel in args.pull:
            remote = f"{REMOTE_V3}/{rel}"
            local = local_v3 / rel
            try:
                scp_from(client, remote, str(local))
            except Exception as e:
                print(f"!! pull failed for {remote}: {e}")
    finally:
        client.close()
        print("=== disconnected ===")


if __name__ == "__main__":
    main()
