"""收入矩实验 全自动 (targeted update, 不重传701MB):
  1) 校验远程实例+bundle在
  2) scp 改动的 2 文件 (train_beijing.py + cfps income npz)
  3) nohup 启 3-seed (income-time 锚), 启动后25s验证进程活着 (吸取 setsid 教训)
  4) 每5分钟轮询 -> pull v4_stgnn_inc_s{0,1,2}.pt
凭据走 env。进度写 data/auto_run_inc.log。
"""
import os, sys, time, io, re
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import paramiko

ROOT = Path(__file__).resolve().parents[2]
RBASE = "/root/autodl-tmp"; RDIR = f"{RBASE}/v4_beijing"
LOG = ROOT / "data" / "auto_run_inc.log"
PTS = [f"evaluation_outputs/v4_stgnn_inc_s{s}.pt" for s in (0, 1, 2)]
UPLOAD = [("experiments/train_beijing.py", "experiments/train_beijing.py"),
          ("data/processed/beijing_cfps_income_moment.npz", "data/processed/beijing_cfps_income_moment.npz")]


def log(m):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f: f.write(line + "\n")


def connect():
    c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(os.environ["AUTODL_HOST"], int(os.environ.get("AUTODL_PORT", "22")),
              os.environ.get("AUTODL_USER", "root"), os.environ["AUTODL_PASSWORD"],
              timeout=30, banner_timeout=30, auth_timeout=30)
    return c


def sh(cmd, tries=3):
    last = ""
    for _ in range(tries):
        try:
            c = connect(); _, o, e = c.exec_command(f"bash -lc '{cmd}'", get_pty=True)
            txt = o.read().decode("utf-8", "replace") + e.read().decode("utf-8", "replace")
            c.close(); return txt
        except Exception as ex:
            last = str(ex); time.sleep(15)
    log(f"  ssh 失败: {last}"); return ""


def main():
    open(LOG, "w").close(); log("=== income 实验 auto_run 启动 ===")
    try:
        # 1) 校验
        chk = sh(f"ls {RDIR}/data/processed/beijing_edges.npz {RDIR}/experiments/train_beijing.py 2>/dev/null | wc -l")
        if "2" not in chk:
            log("✗ 远程 bundle 不在 (实例释放了?) -> 需重新 full upload"); return
        log("1) 远程 bundle 在 ✓")
        # 2) scp 改动
        log("2) scp 改动 2 文件...")
        c = connect(); sftp = c.open_sftp()
        for loc, rem in UPLOAD:
            sftp.put(str(ROOT / loc), f"{RDIR}/{rem}"); log(f"   -> {rem}")
        sftp.close(); c.close()
        # 3) nohup 启动 + 验证活着
        log("3) nohup 启动 3-seed + 验证...")
        lines = ["#!/bin/bash", f"cd {RDIR}",
                 "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                 "mkdir -p evaluation_outputs logs"]
        for s in (0, 1, 2):
            lines.append(
                f"python experiments/train_beijing.py --epochs 300 --device cuda --seed {s} "
                f"--use-nn --use-dynamic --road-graph --use-consideration --use-soc-mixture --typed-mass-occ "
                f"--anchor-transit --anchor-share --anchor-mode-dist --anchor-income-time "
                f"--gnn-mode residual --origin-chunks 16 --lr 0.02 --out {PTS[s]}")
        lines.append("echo ALL_DONE")
        c = connect(); sftp = c.open_sftp(); sftp.putfo(io.StringIO("\n".join(lines) + "\n"), f"{RDIR}/run_inc.sh"); sftp.close()
        c.exec_command(f"bash -lc 'cd {RDIR} && nohup bash run_inc.sh > logs/v4_inc.log 2>&1 < /dev/null & echo started'")
        c.close(); time.sleep(25)
        st = sh("ps aux | grep '[t]rain_beijing' | wc -l; echo ---; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader")
        log(f"   25s后: {st.strip()}")
        if st.strip().split('\n')[0].strip() == "0":
            log("✗ 进程没活下来 (nohup 也失败?) -> 停, 睡醒看 log"); return
        log("   ✓ 进程活着, GPU 在吃")
        # 4) 轮询
        log("4) 轮询 (每5分钟)...")
        t0 = time.time()
        while time.time() - t0 < 20 * 3600:
            time.sleep(300)
            out = sh(f"ls -1 {RDIR}/evaluation_outputs/v4_stgnn_inc_s*.pt 2>/dev/null | wc -l; echo ===; "
                     f"grep -c ALL_DONE {RDIR}/logs/v4_inc.log 2>/dev/null; echo ===; "
                     f"grep 'best CPC' {RDIR}/logs/v4_inc.log 2>/dev/null | tail -1")
            parts = out.split("===")
            npt = int(re.search(r"\d+", parts[0]).group()) if parts and re.search(r"\d+", parts[0]) else 0
            done = int(re.search(r"\d+", parts[1]).group()) if len(parts) > 1 and re.search(r"\d+", parts[1]) else 0
            log(f"   [{(time.time()-t0)/3600:.1f}h] inc_pt={npt}/3 done={done} | {parts[2].strip() if len(parts)>2 else ''}")
            if done >= 1 and npt >= 3:
                log("   ✓ 完成"); break
        # 5) pull + 汇总
        c = connect(); sftp = c.open_sftp()
        for rel in PTS:
            loc = ROOT / rel; loc.parent.mkdir(parents=True, exist_ok=True)
            try: sftp.get(f"{RDIR}/{rel}", str(loc)); log(f"   <- {rel}")
            except Exception as e: log(f"   !! pull {rel}: {e}")
        sftp.close(); c.close()
        out = sh(f"grep 'best CPC' {RDIR}/logs/v4_inc.log 2>/dev/null")
        log("汇总:"); [log("   " + l) for l in out.splitlines()]
        log("=== income 实验完成 ✓ ===")
    except Exception as ex:
        import traceback; log("✗ 异常: " + repr(ex)); log(traceback.format_exc())


if __name__ == "__main__":
    main()
