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
        if action == "resume":
            # 断点续传: 从远程已有字节接着写 (笔记本断网后用)
            rb = f"{RBASE}/{BUNDLE.name}"
            sftp = c.open_sftp()
            try: rsize = sftp.stat(rb).st_size
            except Exception: rsize = 0
            lsize = BUNDLE.stat().st_size
            off = max(0, rsize - (1 << 20))   # 回退1MB覆盖可能的半截块
            print(f"[resume] 远程已有 {rsize/1e6:.0f}MB / {lsize/1e6:.0f}MB, 从 {off/1e6:.0f}MB 续传")
            if rsize >= lsize:
                print("  已完整, 跳过续传")
            else:
                import time as _t; t0 = _t.time()
                rf = sftp.open(rb, "r+b"); rf.seek(off)
                with open(BUNDLE, "rb") as f:
                    f.seek(off)
                    while True:
                        ch = f.read(1 << 20)
                        if not ch: break
                        rf.write(ch)
                rf.close(); print(f"  续传完成 {_t.time()-t0:.0f}s")
            sftp.close()
            run(c, f"cd {RBASE} && rm -rf v4_beijing && tar xzf {BUNDLE.name} && "
                   f"ls v4_beijing/data/processed | head", tail=12)
            run(c, "pip install -q torch numpy scipy 2>&1 | tail -2", tail=3)
        elif action == "upload":
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
                        f"--epochs 5 --device cuda --seed 0 --use-nn --use-dynamic --road-graph "
                        f"--use-consideration --use-soc-mixture --typed-mass-occ "
                        f"--anchor-transit --anchor-share --anchor-mode-dist "
                        f"--gnn-mode residual --origin-chunks 16 --lr 0.02 "
                        f"--out evaluation_outputs/v4_gpu_smoke.pt", tail=30)
            print(f"smoke rc={rc}")
        elif action == "train":
            script = (
                "#!/bin/bash\n"
                f"cd {RDIR}\n"
                "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
                "mkdir -p evaluation_outputs logs\n"
                # 升级版: 双路ST-GNN(动态腿) + 路网图 + mode双锚 + mode×距离矩 + typed-mass occ + 筛选 + soc
                "for s in 0 1 2; do\n"
                "  python experiments/train_beijing.py --epochs 300 --device cuda --seed $s "
                "--use-nn --use-dynamic --road-graph "
                "--use-consideration --use-soc-mixture --typed-mass-occ "
                "--anchor-transit --anchor-share --anchor-mode-dist --anchor-income-time "
                "--gnn-mode residual --origin-chunks 16 --lr 0.02 "
                "--out evaluation_outputs/v4_stgnn_inc_s$s.pt\n"
                "done\n"
                "echo ALL_DONE\n"
            )
            import io
            sftp = c.open_sftp(); sftp.putfo(io.StringIO(script), f"{RDIR}/run_batch.sh"); sftp.close()
            # setsid 脱离会话, 防 SSH 断开 SIGHUP; 不用 pty
            launch = (f"cd {RDIR} && mkdir -p logs evaluation_outputs && "
                      f"setsid bash run_batch.sh </dev/null > logs/v4_batch.log 2>&1 & echo launched PID $!")
            _, o, e = c.exec_command(f"bash -lc '{launch}'")
            print("   " + o.read().decode("utf-8", "replace").strip())
        elif action == "trainB":
            # 架构 B: GNN吸引力 A_j + 异质RUM; 第一版 OD-NLL 无锚, 3 seed
            script = (
                "#!/bin/bash\n"
                f"cd {RDIR}\n"
                "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
                "mkdir -p evaluation_outputs logs\n"
                "for s in 0 1 2; do\n"
                "  python -u experiments/train_beijing_B.py --epochs 300 --device cuda --seed $s "
                "--use-nn --use-dynamic --road-graph "
                "--use-consideration --use-soc-mixture "
                "--origin-chunks 16 --lr 0.02 "
                "--out evaluation_outputs/v4_B2_s$s.pt\n"        # B2: 占职业吸引力 + α=1
                "done\n"
                "echo ALL_DONE\n"
            )
            import io
            sftp = c.open_sftp()
            for f in ("experiments/beijing_model_B.py", "experiments/train_beijing_B.py"):
                sftp.put(str(ROOT / f), f"{RDIR}/{f}")   # 覆盖成最新 B 代码(占职业吸引力+α=1)
            sftp.putfo(io.StringIO(script), f"{RDIR}/run_B.sh"); sftp.close()
            # 先杀旧 run (bracket 正则防 pkill 匹配到自己 cmdline 而自杀)
            _, ko, _ = c.exec_command("bash -lc \"pkill -9 -f 'train_beijing[_]B'; pkill -9 -f 'run_B[.]sh'; true\"")
            ko.channel.recv_exit_status()
            import time as _t; _t.sleep(5)
            launch = (f"cd {RDIR} && mkdir -p logs evaluation_outputs && "
                      f"setsid bash run_B.sh </dev/null > logs/v4_B_batch.log 2>&1 & echo launched PID $!")
            _, o, e = c.exec_command(f"bash -lc '{launch}'")
            print("   launched: " + o.read().decode("utf-8", "replace").strip())
        elif action == "uploadedu":
            sftp = c.open_sftp()
            sftp.put(str(ROOT / "data/processed/beijing_edu_hukou.npz"), f"{RDIR}/data/processed/beijing_edu_hukou.npz")
            sftp.put(str(ROOT / "experiments/train_beijing.py"), f"{RDIR}/experiments/train_beijing.py")
            sftp.close(); print("   uploaded beijing_edu_hukou.npz + train_beijing.py")
        elif action == "smokeedu":
            EDUFLAGS = ("--use-nn --use-dynamic --road-graph --use-consideration --use-soc-mixture --typed-mass-occ "
                        "--anchor-transit --anchor-share --anchor-mode-dist --anchor-income-time --use-education "
                        "--gnn-mode residual --origin-chunks 16 --lr 0.02")
            rc = run(c, f"cd {RDIR} && python experiments/train_beijing.py --epochs 5 --device cuda --seed 0 "
                        f"{EDUFLAGS} --out evaluation_outputs/v4_edu_smoke.pt", tail=35)
            print(f"smokeedu rc={rc}")
        elif action == "trainedu":
            EDUFLAGS = ("--use-nn --use-dynamic --road-graph --use-consideration --use-soc-mixture --typed-mass-occ "
                        "--anchor-transit --anchor-share --anchor-mode-dist --anchor-income-time --use-education "
                        "--gnn-mode residual --origin-chunks 16 --lr 0.02")
            script = ("#!/bin/bash\n" f"cd {RDIR}\n"
                      "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
                      "mkdir -p evaluation_outputs logs\n"
                      "for s in 0 1 2; do\n"
                      f"  python -u experiments/train_beijing.py --epochs 300 --device cuda --seed $s {EDUFLAGS} "
                      "--out evaluation_outputs/v4_edu_s$s.pt\n"
                      "done\necho ALL_DONE\n")
            import io
            sftp = c.open_sftp(); sftp.putfo(io.StringIO(script), f"{RDIR}/run_edu.sh"); sftp.close()
            launch = (f"cd {RDIR} && pkill -9 -f 'train_beijing[.]py'; sleep 3; mkdir -p logs evaluation_outputs && "
                      f"setsid bash run_edu.sh </dev/null > logs/v4_edu_batch.log 2>&1 & echo launched PID $!")
            _, o, e = c.exec_command(f"bash -lc '{launch}'")
            print("   launched edu: " + o.read().decode("utf-8", "replace").strip())
        elif action == "statusedu":
            run(c, f"cd {RDIR} && echo ===log===; tail -16 logs/v4_edu_batch.log 2>/dev/null; "
                   f"echo ===pt===; ls -1 evaluation_outputs/v4_edu_*.pt 2>/dev/null; "
                   f"echo ===gpu===; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader", tail=30)
        elif action == "statusB":
            run(c, f"cd {RDIR} && echo ===Blog===; tail -16 logs/v4_B_batch.log 2>/dev/null; "
                   f"echo ===donept===; ls -1 evaluation_outputs/v4_B*.pt 2>/dev/null; "
                   f"echo ===gpu===; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader", tail=30)
        elif action == "sweepw":
            # w_nn 塌陷诊断: edu 配置不变, 只扫 --income-time-weight {2,1,0.5,0.2}, 150ep seed0
            # w=2.0 = 复现现状(150ep), 看 w_nn 是否随权重降而复活 + CPC 回不回升 -> 区分 A(NN活该归零)/B(锚太重误伤)
            sftp = c.open_sftp()
            for f in ("experiments/beijing_model.py", "experiments/train_beijing.py"):
                sftp.put(str(ROOT / f), f"{RDIR}/{f}")
            EDUFLAGS = ("--use-nn --use-dynamic --road-graph --use-consideration --use-soc-mixture --typed-mass-occ "
                        "--anchor-transit --anchor-share --anchor-mode-dist --anchor-income-time --use-education "
                        "--gnn-mode residual --origin-chunks 16 --lr 0.02")
            script = ("#!/bin/bash\n" f"cd {RDIR}\n"
                      "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
                      "mkdir -p evaluation_outputs logs\n"
                      "for w in 2.0 1.0 0.5 0.2; do\n"
                      "  tag=$(echo $w | tr -d '.')\n"
                      f"  echo \"=== income-time-weight=$w ===\"\n"
                      f"  python -u experiments/train_beijing.py --epochs 150 --device cuda --seed 0 {EDUFLAGS} "
                      "--income-time-weight $w --out evaluation_outputs/v4_eduw_$tag.pt\n"
                      "done\necho ALL_DONE\n")
            import io
            sftp.putfo(io.StringIO(script), f"{RDIR}/run_sweepw.sh"); sftp.close()
            launch = (f"cd {RDIR} && pkill -9 -f 'train_beijing[.]py'; sleep 3; mkdir -p logs evaluation_outputs && "
                      f"setsid bash run_sweepw.sh </dev/null > logs/v4_sweepw.log 2>&1 & echo launched PID $!")
            _, o, e = c.exec_command(f"bash -lc '{launch}'")
            print("   launched sweepw: " + o.read().decode("utf-8", "replace").strip())
        elif action == "sweepwstatus":
            run(c, f"cd {RDIR} && echo ===log===; tail -20 logs/v4_sweepw.log 2>/dev/null; "
                   f"echo ===pt===; ls -1 evaluation_outputs/v4_eduw_*.pt 2>/dev/null; "
                   f"echo ===gpu===; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader", tail=34)
        elif action == "ablmatch":
            # L1 职业匹配门 ablation: 等当前 sweep 跑完(不杀,排队), 再跑 Arm A(门开,sanity 复现 edu) + Arm B(门关,保留 L2/L3)
            # A vs B 同代码同权重同 seed = 干净隔离"职业匹配筛选门单独贡献多少 CPC"
            sftp = c.open_sftp()
            for f in ("experiments/beijing_model.py", "experiments/train_beijing.py"):
                sftp.put(str(ROOT / f), f"{RDIR}/{f}")
            EDUFLAGS = ("--use-nn --use-dynamic --road-graph --use-consideration --use-soc-mixture --typed-mass-occ "
                        "--anchor-transit --anchor-share --anchor-mode-dist --anchor-income-time --use-education "
                        "--gnn-mode residual --origin-chunks 16 --lr 0.02 --income-time-weight 2.0")
            script = ("#!/bin/bash\n" f"cd {RDIR}\n"
                      "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
                      "mkdir -p evaluation_outputs logs\n"
                      "echo '=== wait for running sweep (no kill, share GPU) ==='\n"
                      "while pgrep -f 'train_beijing[.]py' >/dev/null; do sleep 120; done\n"
                      "echo '=== Arm A: L1 match-gate ON (sanity ~v4_edu) ==='\n"
                      f"python -u experiments/train_beijing.py --epochs 150 --device cuda --seed 0 {EDUFLAGS} "
                      "--out evaluation_outputs/v4_ablmatch_on.pt\n"
                      "echo '=== Arm B: L1 match-gate OFF (keep L2 cost/L3 time) ==='\n"
                      f"python -u experiments/train_beijing.py --epochs 150 --device cuda --seed 0 {EDUFLAGS} "
                      "--no-match-filter --out evaluation_outputs/v4_ablmatch_off.pt\n"
                      "echo ABL_DONE\n")
            import io
            sftp.putfo(io.StringIO(script), f"{RDIR}/run_ablmatch.sh"); sftp.close()
            # ⚠ 不 pkill (会杀掉正在跑的 sweep); setsid 后台, 脚本自己 while-wait 排队
            launch = (f"cd {RDIR} && mkdir -p logs evaluation_outputs && "
                      f"setsid bash run_ablmatch.sh </dev/null > logs/v4_ablmatch.log 2>&1 & echo launched PID $!")
            _, o, e = c.exec_command(f"bash -lc '{launch}'")
            print("   queued ablmatch (等 sweep 结束自动开跑): " + o.read().decode("utf-8", "replace").strip())
        elif action == "ablmatchstatus":
            run(c, f"cd {RDIR} && echo ===log===; tail -20 logs/v4_ablmatch.log 2>/dev/null; "
                   f"echo ===pt===; ls -1 evaluation_outputs/v4_ablmatch_*.pt 2>/dev/null; "
                   f"echo ===gpu===; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader", tail=34)
        elif action == "extrapol":
            # nn-monotonic 外推验证: 等前面 train/ablation 跑完, 对 v4_full_anchored 3 seed(NN alive w_nn~0.09)
            # 跑总规疏解反事实三路 = full(自由NN) / freeze-nn / nn-monotonic, 看跨 seed CV
            # 复现 doc 04 的 236%/3.9% + 补缺的 nn-monotonic 行(回答"NN活着+单调约束能不能两全外推")
            sftp = c.open_sftp()
            for f in ("experiments/scenario_beijing_dispersal.py", "experiments/scenario_beijing_A.py",
                      "experiments/beijing_model.py", "experiments/train_beijing.py"):
                sftp.put(str(ROOT / f), f"{RDIR}/{f}")
            PTS = ("evaluation_outputs/v4_full_anchored_s0.pt evaluation_outputs/v4_full_anchored_s1.pt "
                   "evaluation_outputs/v4_full_anchored_s2.pt")
            script = ("#!/bin/bash\n" f"cd {RDIR}\n"
                      "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
                      "echo '=== wait for running train/ablation (share GPU, no kill) ==='\n"
                      "while pgrep -f 'train_beijing[.]py' >/dev/null; do sleep 120; done\n"
                      "echo '=== A: full model (free NN) -> expect high CV (236% disaster) ==='\n"
                      f"python -u experiments/scenario_beijing_dispersal.py --pts {PTS} --chunks 16\n"
                      "echo '=== B: freeze-nn (RUM-only) -> expect low CV (~4%) ==='\n"
                      f"python -u experiments/scenario_beijing_dispersal.py --pts {PTS} --chunks 16 --freeze-nn\n"
                      "echo '=== C: nn-monotonic (NN alive + monotone job-response) -> the question ==='\n"
                      f"python -u experiments/scenario_beijing_dispersal.py --pts {PTS} --chunks 16 --nn-monotonic\n"
                      "echo EXTRAPOL_DONE\n")
            import io
            sftp.putfo(io.StringIO(script), f"{RDIR}/run_extrapol.sh"); sftp.close()
            launch = (f"cd {RDIR} && mkdir -p logs && "
                      f"setsid bash run_extrapol.sh </dev/null > logs/v4_extrapol.log 2>&1 & echo launched PID $!")
            _, o, e = c.exec_command(f"bash -lc '{launch}'")
            print("   queued extrapol (waits for ablation): " + o.read().decode("utf-8", "replace").strip())
        elif action == "extrapolstatus":
            run(c, f"cd {RDIR} && tail -40 logs/v4_extrapol.log 2>/dev/null; "
                   f"echo ===gpu===; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader", tail=46)
        elif action == "frozenmask":
            # 职业当真筛选: typed-mass 引力保留 + 外生硬职业 mask(非补偿考虑集). 两臂同 150ep seed0:
            #   A baseline = 现状(typed-mass + 软门)  B frozen = typed-mass + 硬 mask + 关软门
            # B - A > 0 = 硬 mask 加了软门/连续 log 给不了的非补偿筛选信息
            sftp = c.open_sftp()
            for f in ("experiments/beijing_model.py", "experiments/train_beijing.py"):
                sftp.put(str(ROOT / f), f"{RDIR}/{f}")
            sftp.put(str(ROOT / "data/processed/beijing_occ_mask.npz"), f"{RDIR}/data/processed/beijing_occ_mask.npz")
            EDUFLAGS = ("--use-nn --use-dynamic --road-graph --use-consideration --use-soc-mixture --typed-mass-occ "
                        "--anchor-transit --anchor-share --anchor-mode-dist --anchor-income-time --use-education "
                        "--gnn-mode residual --origin-chunks 16 --lr 0.02 --income-time-weight 2.0")
            script = ("#!/bin/bash\nset -e\n" f"cd {RDIR}\n"
                      "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
                      "mkdir -p evaluation_outputs logs\n"
                      "while pgrep -f 'train_beijing[.]py' >/dev/null; do sleep 120; done\n"
                      "echo '=== SMOKE: frozen mask 3ep (fail-fast wiring check) ==='\n"
                      f"python -u experiments/train_beijing.py --epochs 3 --device cuda --seed 0 {EDUFLAGS} "
                      "--use-frozen-occ-mask --no-match-filter --out evaluation_outputs/v4_frozen_smoke.pt\n"
                      "echo '=== A: baseline (typed-mass + soft match-gate) ==='\n"
                      f"python -u experiments/train_beijing.py --epochs 150 --device cuda --seed 0 {EDUFLAGS} "
                      "--out evaluation_outputs/v4_frozen_baseline.pt\n"
                      "echo '=== B: typed-mass + frozen hard occ-mask (soft gate off) ==='\n"
                      f"python -u experiments/train_beijing.py --epochs 150 --device cuda --seed 0 {EDUFLAGS} "
                      "--use-frozen-occ-mask --no-match-filter --out evaluation_outputs/v4_frozen_mask.pt\n"
                      "echo FROZEN_DONE\n")
            import io
            sftp.putfo(io.StringIO(script), f"{RDIR}/run_frozen.sh"); sftp.close()
            launch = (f"cd {RDIR} && mkdir -p logs evaluation_outputs && "
                      f"setsid bash run_frozen.sh </dev/null > logs/v4_frozen.log 2>&1 & echo launched PID $!")
            _, o, e = c.exec_command(f"bash -lc '{launch}'")
            print("   launched frozenmask: " + o.read().decode("utf-8", "replace").strip())
        elif action == "frozenmaskstatus":
            run(c, f"cd {RDIR} && echo ===log===; tail -18 logs/v4_frozen.log 2>/dev/null; "
                   f"echo ===pt===; ls -1 evaluation_outputs/v4_frozen_*.pt 2>/dev/null; "
                   f"echo ===gpu===; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader", tail=30)
        elif action == "softlex":
            # 伦敦 winning 设计移植: soft-lexicographic 职业门(τ_s≥floor + k_s≥20, 防塌陷防糊, 自动学每职业筛多少)
            #   A baseline = typed-mass + 自由阈值软门(现状, 会塌陷)  B soft-lex = typed-mass + soft-lex 门
            # B - A > 0 + soft-lex 学出 differential 筛除率 = 北京职业筛选在"对的设计"下能成立(之前两次是失败模式)
            sftp = c.open_sftp()
            for f in ("experiments/beijing_model.py", "experiments/train_beijing.py"):
                sftp.put(str(ROOT / f), f"{RDIR}/{f}")
            EDUFLAGS = ("--use-nn --use-dynamic --road-graph --use-consideration --use-soc-mixture --typed-mass-occ "
                        "--anchor-transit --anchor-share --anchor-mode-dist --anchor-income-time --use-education "
                        "--gnn-mode residual --origin-chunks 16 --lr 0.02 --income-time-weight 2.0")
            script = ("#!/bin/bash\nset -e\n" f"cd {RDIR}\n"
                      "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
                      "mkdir -p evaluation_outputs logs\n"
                      "while pgrep -f 'train_beijing[.]py' >/dev/null; do sleep 120; done\n"
                      "echo '=== SMOKE: soft-lex 3ep (fail-fast wiring check) ==='\n"
                      f"python -u experiments/train_beijing.py --epochs 3 --device cuda --seed 0 {EDUFLAGS} "
                      "--soft-lex-match --out evaluation_outputs/v4_softlex_smoke.pt\n"
                      "echo '=== A: baseline (typed-mass + free-threshold soft gate) ==='\n"
                      f"python -u experiments/train_beijing.py --epochs 150 --device cuda --seed 0 {EDUFLAGS} "
                      "--out evaluation_outputs/v4_softlex_baseline.pt\n"
                      "echo '=== B: typed-mass + soft-lexicographic match gate (floor+clamp) ==='\n"
                      f"python -u experiments/train_beijing.py --epochs 150 --device cuda --seed 0 {EDUFLAGS} "
                      "--soft-lex-match --out evaluation_outputs/v4_softlex.pt\n"
                      "echo SOFTLEX_DONE\n")
            import io
            sftp.putfo(io.StringIO(script), f"{RDIR}/run_softlex.sh"); sftp.close()
            launch = (f"cd {RDIR} && mkdir -p logs evaluation_outputs && "
                      f"setsid bash run_softlex.sh </dev/null > logs/v4_softlex.log 2>&1 & echo launched PID $!")
            _, o, e = c.exec_command(f"bash -lc '{launch}'")
            print("   launched softlex: " + o.read().decode("utf-8", "replace").strip())
        elif action == "softlexstatus":
            run(c, f"cd {RDIR} && echo ===log===; tail -22 logs/v4_softlex.log 2>/dev/null; "
                   f"echo ===pt===; ls -1 evaluation_outputs/v4_softlex*.pt 2>/dev/null; "
                   f"echo ===gpu===; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader", tail=34)
        elif action == "softlex3seed":
            # soft-lex 3-seed 确认: seed0 已有(v4_softlex), 补 seed1/2; 确认 differential(制造筛多/白领不筛)+CPC中性 稳
            sftp = c.open_sftp()
            for f in ("experiments/beijing_model.py", "experiments/train_beijing.py"):
                sftp.put(str(ROOT / f), f"{RDIR}/{f}")
            EDUFLAGS = ("--use-nn --use-dynamic --road-graph --use-consideration --use-soc-mixture --typed-mass-occ "
                        "--anchor-transit --anchor-share --anchor-mode-dist --anchor-income-time --use-education "
                        "--gnn-mode residual --origin-chunks 16 --lr 0.02 --income-time-weight 2.0 --soft-lex-match")
            script = ("#!/bin/bash\nset -e\n" f"cd {RDIR}\n"
                      "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
                      "mkdir -p evaluation_outputs logs\n"
                      "while pgrep -f 'train_beijing[.]py' >/dev/null; do sleep 120; done\n"
                      "for s in 1 2; do\n"
                      f"  echo \"=== soft-lex seed $s ===\"\n"
                      f"  python -u experiments/train_beijing.py --epochs 150 --device cuda --seed $s {EDUFLAGS} "
                      "--out evaluation_outputs/v4_softlex_s$s.pt\n"
                      "done\necho SOFTLEX3_DONE\n")
            import io
            sftp.putfo(io.StringIO(script), f"{RDIR}/run_softlex3.sh"); sftp.close()
            launch = (f"cd {RDIR} && mkdir -p logs evaluation_outputs && "
                      f"setsid bash run_softlex3.sh </dev/null > logs/v4_softlex3.log 2>&1 & echo launched PID $!")
            _, o, e = c.exec_command(f"bash -lc '{launch}'")
            print("   launched softlex3seed: " + o.read().decode("utf-8", "replace").strip())
        elif action == "softlex3status":
            run(c, f"cd {RDIR} && echo ===log===; tail -16 logs/v4_softlex3.log 2>/dev/null; "
                   f"echo ===pt===; ls -1 evaluation_outputs/v4_softlex_s*.pt 2>/dev/null; "
                   f"echo ===gpu===; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader", tail=28)
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
