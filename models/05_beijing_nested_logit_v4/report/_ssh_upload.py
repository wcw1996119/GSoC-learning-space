# -*- coding: utf-8 -*-
import paramiko, os
import os
HOST, PORT, USER = "connect.nmb1.seetacloud.com", 36257, "root"
PW = os.environ["AUTODL_SSH_PW"]  # 密码走环境变量, 不进 git
ROOT = r"D:\GIT\mesa Gsoc\GSoC-learning-space\models\05_beijing_nested_logit_v4"
REMOTE = "/root/autodl-tmp/v4_beijing"
files = [
    (f"{ROOT}/experiments/diff_equilibrium.py", f"{REMOTE}/experiments/diff_equilibrium.py"),
]
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, PORT, USER, PW, timeout=30)
sf = c.open_sftp()
for lp, rp in files:
    if not os.path.exists(lp):
        print("MISSING LOCAL:", lp); continue
    sf.put(lp, rp)
    print(f"uploaded {os.path.basename(lp)}  ({os.path.getsize(lp):,} B) -> {rp}")
sf.close(); c.close()
print("done.")
