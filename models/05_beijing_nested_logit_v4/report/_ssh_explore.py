# -*- coding: utf-8 -*-
import paramiko
import os
HOST, PORT, USER = "connect.nmb1.seetacloud.com", 36257, "root"
PW = os.environ["AUTODL_SSH_PW"]  # 密码走环境变量, 不进 git
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, PORT, USER, PW, timeout=30)
def run(cmd):
    _, o, e = c.exec_command(cmd, timeout=60)
    out = o.read().decode("utf-8", "replace"); err = e.read().decode("utf-8", "replace")
    return (out + err).strip()
print("=== GPU ==="); print(run("nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader"))
print("=== python/torch ==="); print(run("python -c 'import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'"))
print("=== find project (v4_full_anchored .pt) ==="); print(run("find /root -maxdepth 5 -name 'v4_full_anchored_s0.pt' 2>/dev/null | head"))
print("=== find edges npz ==="); print(run("find /root -maxdepth 5 -name 'beijing_edges.npz' 2>/dev/null | head"))
print("=== find road_network npz ==="); print(run("find /root -maxdepth 5 -name 'beijing_road_network.npz' 2>/dev/null | head"))
print("=== disk ==="); print(run("df -h /root/autodl-tmp 2>/dev/null | tail -1; df -h /root | tail -1"))
c.close()
