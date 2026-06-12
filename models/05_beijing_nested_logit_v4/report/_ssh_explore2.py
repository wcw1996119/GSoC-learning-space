# -*- coding: utf-8 -*-
import paramiko
import os
HOST, PORT, USER = "connect.nmb1.seetacloud.com", 36257, "root"
PW = os.environ["AUTODL_SSH_PW"]  # 密码走环境变量, 不进 git
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, PORT, USER, PW, timeout=30)
def run(cmd):
    _, o, e = c.exec_command(cmd, timeout=60)
    return (o.read().decode("utf-8", "replace") + e.read().decode("utf-8", "replace")).strip()
print("=== python candidates ===")
print(run("ls /root/miniconda3/bin/python* 2>/dev/null; which conda; ls /root/miniconda3/envs 2>/dev/null"))
print("=== torch check ===")
print(run("/root/miniconda3/bin/python -c 'import torch;print(torch.__version__, torch.cuda.is_available())' 2>&1 | tail -2"))
print("=== project tree (v4_beijing) ===")
print(run("ls /root/autodl-tmp/v4_beijing; echo '--- data/processed ---'; ls /root/autodl-tmp/v4_beijing/data/processed | grep -E 'modes|aux|grid|road_network|edges'; echo '--- experiments ---'; ls /root/autodl-tmp/v4_beijing/experiments 2>/dev/null | head; echo '--- evaluation_outputs ---'; ls /root/autodl-tmp/v4_beijing/evaluation_outputs 2>/dev/null | head"))
c.close()
