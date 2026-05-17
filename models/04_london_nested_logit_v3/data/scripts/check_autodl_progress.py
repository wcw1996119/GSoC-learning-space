"""One-shot AutoDL inspection: process, GPU, recent log."""
import paramiko, sys, warnings
warnings.filterwarnings("ignore")

PASSWORD = sys.argv[1] if len(sys.argv) > 1 else None
if not PASSWORD:
    sys.exit("pass password as argv[1]")

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("connect.nmb2.seetacloud.com", port=15859, username="root",
          password=PASSWORD, timeout=30, banner_timeout=30)

cmd = (
    "ps aux | grep train_cervero | grep -v grep ; "
    "echo '---GPU---' ; "
    "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv ; "
    "echo '---PROCS---' ; "
    "ps -ef | grep python | grep -v grep | head -10"
)
stdin, stdout, stderr = c.exec_command(f"bash -lc \"{cmd}\"")
print(stdout.read().decode(errors="replace"))
err = stderr.read().decode(errors="replace")
if err.strip():
    print("STDERR:", err)
c.close()
