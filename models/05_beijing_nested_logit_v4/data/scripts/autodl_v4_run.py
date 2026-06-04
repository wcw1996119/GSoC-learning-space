"""v4 北京 AutoDL: 上传 bundle + 解包 + 装依赖 + GPU smoke / 启动训练。

凭据走环境变量 (不写死/不 commit):
  AUTODL_HOST AUTODL_PORT AUTODL_USER AUTODL_PASSWORD
用法:
  python autodl_v4_run.py upload          # scp + 解包 + pip
  python autodl_v4_run.py smoke           # 5ep GPU 验证 (同步)
  python autodl_v4_run.py train           # 启动 full seed0 后台 (nohup)
  python autodl_v4_run.py pull <rel...>   # 拉结果回本地 evaluation_outputs/
"""
import os, sys, time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import paramiko

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "data" / "v4_beijing_bundle.tar.gz"
RBASE = "/root/autodl-tmp"
RDIR = f"{RBASE}/v4_beijing"


def connect():
    h = os.environ["AUTODL_HOST"]; p = int(os.environ.get("AUTODL_PORT", "22"))
    u = os.environ.get("AUTODL_USER", "root"); pw = os.environ["AUTODL_PASSWORD"]
    c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"=== connect {u}@{h}:{p} ===")
    c.connect(hostname=h, port=p, username=u, password=pw,
              timeout=30, banner_timeout=30, auth_timeout=30)
    return c


def run(c, cmd, tail=40):
    print(f"[ssh] {cmd[:140]}")
    _, out, err = c.exec_command(f"bash -lc '{cmd}'", get_pty=True)
    txt = out.read().decode("utf-8", "replace") + err.read().decode("utf-8", "replace")
    for ln in txt.splitlines()[-tail:]:
        print("   " + ln.encode("ascii", "replace").decode("ascii"))
    return out.channel.recv_exit_status()


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else "upload"
    c = connect()
    try:
        if action == "upload":
            run(c, f"mkdir -p {RBASE}", tail=2)
            sftp = c.open_sftp()
            rb = f"{RBASE}/{BUNDLE.name}"
            print(f"[scp ->] {BUNDLE.name} ({BUNDLE.stat().st_size/1e6:.0f}MB) ...")
            t0 = time.time(); sftp.put(str(BUNDLE), rb); sftp.close()
            print(f"   uploaded in {time.time()-t0:.0f}s")
            run(c, f"cd {RBASE} && tar xzf {BUNDLE.name} && ls v4_beijing/experiments v4_beijing/data/processed", tail=20)
            run(c, "pip install -q torch numpy scipy 2>&1 | tail -3", tail=5)
        elif action == "smoke":
            rc = run(c, f"cd {RDIR} && python experiments/train_beijing.py "
                        f"--epochs 5 --device cuda --seed 0 --use-nn --use-consideration "
                        f"--gnn-mode residual --out evaluation_outputs/v4_gpu_smoke.pt", tail=30)
            print(f"smoke rc={rc}")
        elif action == "train":
            script = (
                "#!/bin/bash\n"
                f"cd {RDIR}\n"
                "mkdir -p evaluation_outputs logs\n"
                "for s in 0 1 2; do\n"
                "  python experiments/train_beijing.py --epochs 300 --device cuda --seed $s "
                "--use-nn --use-consideration --gnn-mode residual --residual-scale-init 0.1 "
                "--out evaluation_outputs/v4_full_s$s.pt\n"
                "done\n"
                "python experiments/train_beijing.py --epochs 300 --device cuda --seed 0 "
                "--out evaluation_outputs/v4_rum_s0.pt\n"
                "python experiments/train_beijing.py --epochs 300 --device cuda --seed 0 "
                "--use-nn --use-consideration --use-soc-mixture --gnn-mode residual "
                "--out evaluation_outputs/v4_socfull_s0.pt\n"
                "echo ALL_DONE\n"
            )
            import io
            sftp = c.open_sftp(); sftp.putfo(io.StringIO(script), f"{RDIR}/run_batch.sh"); sftp.close()
            # setsid 脱离会话, 防 SSH 断开 SIGHUP; 不用 pty
            launch = (f"cd {RDIR} && mkdir -p logs evaluation_outputs && "
                      f"setsid bash run_batch.sh </dev/null > logs/v4_batch.log 2>&1 & echo launched PID $!")
            _, o, e = c.exec_command(f"bash -lc '{launch}'")
            print("   " + o.read().decode("utf-8", "replace").strip())
        elif action == "socfull":
            # 重传改后的代码 (省显存版 forward), expandable_segments 抗碎片, setsid 后台
            sftp = c.open_sftp()
            for f in ("experiments/beijing_model.py", "experiments/train_beijing.py"):
                sftp.put(str(ROOT / f), f"{RDIR}/{f}")
            sftp.close()
            launch = (f"cd {RDIR} && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
                      f"setsid python experiments/train_beijing.py --epochs 300 --device cuda --seed 0 "
                      f"--use-nn --use-consideration --use-soc-mixture --gnn-mode residual "
                      f"--lr 0.03 --origin-chunks 8 "
                      f"--out evaluation_outputs/v4_socfull_s0.pt </dev/null > logs/v4_socfull.log 2>&1 & echo launched PID $!")
            _, o, e = c.exec_command(f"bash -lc '{launch}'")
            print("   " + o.read().decode("utf-8", "replace").strip())
        elif action == "socstatus":
            run(c, f"cd {RDIR} && tail -6 logs/v4_socfull.log 2>/dev/null; "
                   f"echo ===gpu===; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader", tail=12)
        elif action == "status":
            run(c, f"cd {RDIR} && echo ===batchlog===; tail -14 logs/v4_batch.log 2>/dev/null; "
                   f"echo ===donept===; ls -1 evaluation_outputs/*.pt 2>/dev/null; "
                   f"echo ===gpu===; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader", tail=30)
        elif action == "pull":
            sftp = c.open_sftp()
            for rel in sys.argv[2:]:
                loc = ROOT / rel; loc.parent.mkdir(parents=True, exist_ok=True)
                try:
                    sftp.get(f"{RDIR}/{rel}", str(loc)); print(f"[scp <-] {rel}")
                except Exception as e:
                    print(f"!! pull {rel} 失败: {e}")
            sftp.close()
    finally:
        c.close(); print("=== disconnected ===")


if __name__ == "__main__":
    main()
