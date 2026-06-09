import paramiko
try:
    c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('connect.nmb1.seetacloud.com',47881,'root','u/FxLryslS8E',timeout=25,banner_timeout=25,auth_timeout=25)
    def run(cmd):
        i,o,e=c.exec_command(f'bash -lc "{cmd}"'); return o.read().decode('utf-8','replace')+e.read().decode('utf-8','replace')
    print("连上了 ✓")
    print("gpu:", run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader").strip())
    print("bundle数据还在?", run("ls /root/autodl-tmp/v4_beijing/data/processed/beijing_edges.npz /root/autodl-tmp/v4_beijing/experiments/train_beijing.py 2>/dev/null | wc -l").strip(), "(2=都在)")
    print("已有结果pt:", run("ls -1 /root/autodl-tmp/v4_beijing/evaluation_outputs/*.pt 2>/dev/null | wc -l").strip())
    c.close()
except Exception as ex:
    print("连接失败:", repr(ex)[:160])
