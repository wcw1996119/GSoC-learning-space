"""全自动 AutoDL 流程 (一条后台命令跑完, 中途零人工):
  1) 等上传+解包完成 (轮询远程新 npz)
  2) GPU smoke 门控 (崩/OOM -> 中止, 不浪费)
  3) 启 3-seed 训练 (setsid 后台, 断连不影响)
  4) 每 5 分钟轮询直到 ALL_DONE
  5) pull 3 个 .pt 回本地
  6) 从日志汇总每 seed best CPC
凭据走 env: AUTODL_HOST/PORT/USER/PASSWORD。进度写 data/auto_run.log。
"""
import os, sys, time, io, re
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import paramiko

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "data" / "v4_beijing_bundle.tar.gz"
RBASE = "/root/autodl-tmp"; RDIR = f"{RBASE}/v4_beijing"
LOG = ROOT / "data" / "auto_run.log"
SEEDS = [0, 1, 2]
PTS = [f"evaluation_outputs/v4_stgnn_s{s}.pt" for s in SEEDS]


def log(m):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def connect():
    c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(os.environ["AUTODL_HOST"], int(os.environ.get("AUTODL_PORT", "22")),
              os.environ.get("AUTODL_USER", "root"), os.environ["AUTODL_PASSWORD"],
              timeout=30, banner_timeout=30, auth_timeout=30)
    return c


def sh(cmd, tries=3):
    """新连接执行 (避免长连接掉线), 返回 stdout+stderr。"""
    last = ""
    for _ in range(tries):
        try:
            c = connect()
            _, o, e = c.exec_command(f"bash -lc '{cmd}'", get_pty=True)
            txt = o.read().decode("utf-8", "replace") + e.read().decode("utf-8", "replace")
            c.close()
            return txt
        except Exception as ex:
            last = str(ex); time.sleep(15)
    log(f"  ssh 反复失败: {last}"); return ""


def wait_upload(timeout_s=3600):
    log("1) 等上传+解包 (轮询远程 beijing_road_graph.npz)...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        out = sh(f"ls -1 {RDIR}/data/processed/beijing_road_graph.npz "
                 f"{RDIR}/data/processed/beijing_dynamic_pop.npz "
                 f"{RDIR}/data/processed/beijing_dynamic_cong.npz "
                 f"{RDIR}/data/processed/beijing_micro_moments.npz 2>/dev/null | wc -l")
        n = int(re.search(r"\d+", out).group()) if re.search(r"\d+", out) else 0
        if n >= 4:
            log("   上传解包完成 (4 个新 npz 就位)"); return True
        time.sleep(30)
    log("   ✗ 等上传超时"); return False


def smoke():
    log("2) GPU smoke (5轮全栈, 门控)...")
    out = sh(f"cd {RDIR} && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
             f"timeout 2400 python experiments/train_beijing.py --epochs 5 --device cuda --seed 0 "
             f"--use-nn --use-dynamic --road-graph --use-consideration --use-soc-mixture --typed-mass-occ "
             f"--anchor-transit --anchor-share --anchor-mode-dist --gnn-mode residual "
             f"--origin-chunks 16 --lr 0.02 --out evaluation_outputs/v4_gpu_smoke.pt 2>&1 | tail -25")
    for ln in out.splitlines():
        log("   smoke| " + ln.encode("ascii", "replace").decode("ascii"))
    if "out of memory" in out.lower() or "OOM" in out:
        log("   ✗ smoke OOM -> 中止 (调大 origin-chunks 再来)"); return False
    cpcs = re.findall(r"CPC\s+([0-9.]+)", out)
    sec = re.search(r"\((\d+)s\)", out)
    if "[OK]" in out and cpcs and float(cpcs[-1]) > 0.4:
        if sec:
            per = int(sec.group(1)) / 5.0
            log(f"   ✓ smoke OK, 末轮 CPC {cpcs[-1]}, ~{per:.0f}s/轮 -> 估 3-seed×300轮 ≈ {per*300*3/3600:.1f}h")
        else:
            log(f"   ✓ smoke OK, 末轮 CPC {cpcs[-1]}")
        return True
    log("   ✗ smoke 异常 (未 [OK] 或 CPC 崩) -> 中止"); return False


def launch_train():
    log("3) 启 3-seed 训练 (setsid 后台)...")
    lines = ["#!/bin/bash", f"cd {RDIR}",
             "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
             "mkdir -p evaluation_outputs logs"]
    for s in SEEDS:
        lines.append(
            f"python experiments/train_beijing.py --epochs 300 --device cuda --seed {s} "
            f"--use-nn --use-dynamic --road-graph --use-consideration --use-soc-mixture --typed-mass-occ "
            f"--anchor-transit --anchor-share --anchor-mode-dist --gnn-mode residual "
            f"--origin-chunks 16 --lr 0.02 --out {PTS[s]}")
    lines.append("echo ALL_DONE")
    script = "\n".join(lines) + "\n"
    c = connect(); sftp = c.open_sftp(); sftp.putfo(io.StringIO(script), f"{RDIR}/run_batch.sh"); sftp.close()
    _, o, _ = c.exec_command(
        f"bash -lc 'cd {RDIR} && setsid bash run_batch.sh </dev/null > logs/v4_batch.log 2>&1 & echo launched $!'")
    log("   " + o.read().decode("utf-8", "replace").strip()); c.close()


def poll(timeout_h=20):
    log("4) 轮询训练 (每5分钟)...")
    t0 = time.time()
    while time.time() - t0 < timeout_h * 3600:
        time.sleep(300)
        out = sh(f"cd {RDIR} && tail -3 logs/v4_batch.log 2>/dev/null; echo ===; "
                 f"ls -1 evaluation_outputs/v4_stgnn_s*.pt 2>/dev/null | wc -l; echo ===; "
                 f"grep -c ALL_DONE logs/v4_batch.log 2>/dev/null")
        npt = 0; done = 0
        parts = out.split("===")
        if len(parts) >= 3:
            try: npt = int(re.search(r"\d+", parts[1]).group())
            except Exception: npt = 0
            try: done = int(re.search(r"\d+", parts[2]).group())
            except Exception: done = 0
        cpc = re.findall(r"best CPC ([0-9.]+)", out)
        tail = parts[0].strip().splitlines()[-1:] if parts else []
        log(f"   [{(time.time()-t0)/3600:.1f}h] pt={npt}/3 done={done} | {tail}")
        if done >= 1 and npt >= 3:
            log("   ✓ 训练完成 (ALL_DONE + 3 pt)"); return True
    log("   ✗ 训练轮询超时"); return False


def pull_and_summary():
    log("5) pull 结果...")
    c = connect(); sftp = c.open_sftp()
    for rel in PTS:
        loc = ROOT / rel; loc.parent.mkdir(parents=True, exist_ok=True)
        try: sftp.get(f"{RDIR}/{rel}", str(loc)); log(f"   <- {rel}")
        except Exception as e: log(f"   !! pull {rel} 失败: {e}")
    sftp.close()
    # 从远程日志抓每 seed best CPC
    out = sh(f"grep 'best CPC' {RDIR}/logs/v4_batch.log 2>/dev/null")
    log("6) 汇总:")
    for ln in out.splitlines():
        log("   " + ln.encode("ascii", "replace").decode("ascii"))
    c.close()


def main():
    open(LOG, "w").close()
    log("=== auto_run 启动 ===")
    try:
        if not wait_upload(): return
        if not smoke():
            log("smoke 未过, 不启训练。睡醒看 auto_run.log"); return
        launch_train()
        if not poll(): return
        pull_and_summary()
        log("=== 全部完成 ✓ 睡醒看 evaluation_outputs/v4_stgnn_s{0,1,2}.pt ===")
    except Exception as ex:
        import traceback; log("✗ 异常: " + repr(ex)); log(traceback.format_exc())


if __name__ == "__main__":
    main()
