# -*- coding: utf-8 -*-
import paramiko, sys
import os
HOST, PORT, USER = "connect.nmb1.seetacloud.com", 36257, "root"
PW = os.environ["AUTODL_SSH_PW"]  # 密码走环境变量, 不进 git
REMOTE = "/root/autodl-tmp/v4_beijing"
PY = "/root/miniconda3/bin/python"
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, PORT, USER, PW, timeout=30)
def run(cmd):
    _, o, e = c.exec_command(cmd, timeout=60)
    return (o.read().decode("utf-8","replace") + e.read().decode("utf-8","replace")).strip()

action = sys.argv[1] if len(sys.argv) > 1 else "launch"
if action == "launch":
    sc = sys.argv[2] if len(sys.argv) > 2 else "1500"
    cmd = (f"cd {REMOTE} && mkdir -p logs && rm -rf experiments/__pycache__ && "
           f"nohup setsid {PY} experiments/autodl_stageb_estimate.py "
           f"--pt evaluation_outputs/v4_full_anchored_s0.pt --device cuda "
           f"--smoke-cells {sc} --steps 3 --msa-iter 150 --gmres-iter 30 "
           f"--out evaluation_outputs/_measure_s0.pt > logs/measure.log 2>&1 < /dev/null & echo LAUNCHED $!")
    print(run(cmd))
elif action == "full":
    sc = sys.argv[2] if len(sys.argv) > 2 else "2000"
    script = ("#!/bin/bash\ncd " + REMOTE + "\n"
              "for s in 0 1 2; do\n"
              "  echo \"==== seed $s ====\"\n"
              "  " + PY + " experiments/autodl_stageb_estimate.py "
              "--pt evaluation_outputs/v4_full_anchored_s${s}.pt --device cuda "
              "--smoke-cells " + sc + " --steps 40 --lr 0.02 --lambda-tt 1.0 "
              "--msa-iter 200 --gmres-iter 40 --out evaluation_outputs/v4_mech_beta_s${s}.pt\n"
              "done\necho ALL_DONE\n")
    sf = c.open_sftp()
    with sf.open(REMOTE + "/run_stageb.sh", "w") as f:
        f.write(script)
    sf.close()
    print(run(f"cd {REMOTE} && mkdir -p logs && rm -rf experiments/__pycache__ && "
              f"nohup setsid bash run_stageb.sh > logs/full.log 2>&1 < /dev/null & echo LAUNCHED $!"))
elif action == "pollfull":
    print("=== full.log tail ===")
    print(run(f"tail -n 16 {REMOTE}/logs/full.log 2>/dev/null"))
    print("=== gpu mem ===")
    print(run("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader"))
    print("=== outputs ===")
    print(run(f"ls -la {REMOTE}/evaluation_outputs/v4_mech_beta_s*.pt 2>/dev/null"))
    print("=== alive? ===")
    print(run("pgrep -af run_stageb | head -1"))
elif action == "poll":
    print("=== log tail ===")
    print(run(f"tail -n 18 {REMOTE}/logs/measure.log 2>/dev/null"))
    print("=== gpu mem ===")
    print(run("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader"))
    print("=== alive? ===")
    print(run("pgrep -af autodl_stageb_estimate | head -1"))
c.close()
