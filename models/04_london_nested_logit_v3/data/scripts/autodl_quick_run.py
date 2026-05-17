"""Quick AutoDL runner: scp a single file + run command(s) + pull results.

Doesn't require packaging a full bundle. Assumes the AutoDL instance already
has the v3 repo extracted at /root/autodl-tmp/v3_nested_smoke/04_london_nested_logit_v3/.

Usage:
    $env:AUTODL_PASSWORD = '<password>'
    python data/scripts/autodl_quick_run.py \
        --upload experiments/paper_a/train_cervero_shen.py:experiments/paper_a/train_cervero_shen.py \
        --commands "<cmd1>" "<cmd2>" \
        --pull evaluation_outputs/paper_a/v3l_boosting_4stage_seed0.json
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

REMOTE_V3 = "/root/autodl-tmp/v3_nested_smoke/04_london_nested_logit_v3"
V3_ROOT = Path(__file__).resolve().parents[2]


def connect(host, port, user, password):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=int(port), username=user, password=password,
                   timeout=30, banner_timeout=30, auth_timeout=30)
    return client


def scp_to(client, local_path: str, remote_path: str):
    sftp = client.open_sftp()
    print(f"[scp ->] {local_path}  ->  {remote_path}")
    sftp.put(local_path, remote_path)
    sftp.close()


def scp_from(client, remote_path: str, local_path: str):
    sftp = client.open_sftp()
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"[scp <-] {remote_path}  ->  {local_path}")
    sftp.get(remote_path, local_path)
    sftp.close()


def run_cmd(client, command: str, tail_lines: int = 40):
    full = f"bash -lc '{command}'"
    print(f"[ssh] {command[:200]}{'...' if len(command) > 200 else ''}")
    stdin, stdout, stderr = client.exec_command(full, get_pty=True)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    if tail_lines:
        lines = (out + err).splitlines()
        for line in lines[-tail_lines:]:
            safe = line.encode("ascii", errors="replace").decode("ascii")
            print(f"   {safe}")
    return stdout.channel.recv_exit_status()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("AUTODL_HOST", "connect.nmb2.seetacloud.com"))
    ap.add_argument("--port", default=os.environ.get("AUTODL_PORT", "15859"))
    ap.add_argument("--user", default=os.environ.get("AUTODL_USER", "root"))
    ap.add_argument("--password", default=os.environ.get("AUTODL_PASSWORD"))
    ap.add_argument("--upload", nargs="*", default=[],
                    help="local:remote pairs (remote relative to v3 root). "
                         "If only one path given, used for both.")
    ap.add_argument("--commands", nargs="+", required=True,
                    help="commands to run in v3 root (sequential)")
    ap.add_argument("--pull", nargs="*", default=[],
                    help="remote paths (relative to v3 root) to scp back to local v3 root")
    ap.add_argument("--tail", type=int, default=60)
    args = ap.parse_args()

    if not args.password:
        sys.exit("AUTODL_PASSWORD env var or --password required.")

    print(f"=== Connecting to {args.user}@{args.host}:{args.port} ===")
    client = connect(args.host, args.port, args.user, args.password)
    try:
        for spec in args.upload:
            if ":" in spec:
                lp, rp = spec.split(":", 1)
            else:
                lp = rp = spec
            local = (V3_ROOT / lp).resolve()
            remote = f"{REMOTE_V3}/{rp}"
            scp_to(client, str(local), remote)

        for cmd in args.commands:
            t0 = time.time()
            rc = run_cmd(client, f"cd {REMOTE_V3} && {cmd}", tail_lines=args.tail)
            elapsed = time.time() - t0
            print(f"   (took {elapsed:.0f}s, rc={rc})")
            if rc != 0:
                print(f"!! command failed: {cmd}")
                break

        for rel in args.pull:
            remote = f"{REMOTE_V3}/{rel}"
            local = V3_ROOT / rel
            try:
                scp_from(client, remote, str(local))
            except Exception as e:
                print(f"!! pull failed for {remote}: {e}")
    finally:
        client.close()
        print("=== disconnected ===")


if __name__ == "__main__":
    main()
