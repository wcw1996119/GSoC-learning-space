# AutoDL 全量训练说明 (北京 v4)

⚠ GOVERNANCE: bundle 含北京 OD 派生数据, 上传 = 出库 (2026-07 红线), 需用户授权。

## 环境
```
pip install torch numpy scipy
# 推荐 32GB 显存 (vGPU/A100); 24GB 可试 K=3 无 soc, soc 混合(21类)可能 OOM
```

## 解包
```
tar -xzf v4_beijing_bundle.tar.gz && cd v4_beijing
```

## 全量训练 (16907 格 / 30.4M 候选对 / 4 时段)

**baseline (纯 RUM, 不带 NN/筛选)** — 对照组:
```
python experiments/train_beijing.py --epochs 300 --device cuda --seed 0 \
    --out evaluation_outputs/v4_rum_s0.pt
```

**完整模型 (NN residual + 词典筛选 + tier 混合, 不含 soc)** — 主 spec:
```
python experiments/train_beijing.py --epochs 300 --device cuda --seed 0 \
    --use-nn --use-consideration --gnn-mode residual --residual-scale-init 0.1 \
    --out evaluation_outputs/v4_full_s0.pt
```

**完整 + soc 混合 (21 类, 显存大)** — 职业可识别性诊断:
```
python experiments/train_beijing.py --epochs 300 --device cuda --seed 0 \
    --use-nn --use-consideration --use-soc-mixture --gnn-mode residual \
    --out evaluation_outputs/v4_socfull_s0.pt
```

**3-seed**: 上面每条改 `--seed 0/1/2`, 取 CPC mean±std。

## 关注指标 (拉回本地分析)
- CPC (4 时段 + 流量加权总), 跟 smoke(2000格)0.68 不可比 (全图更难)
- 参数符号: λ∈(0,1], γ_M>0, ν_D<0, β_t<0 (步行最痛), w_nn, T_max 档
- ⚠ delta_match 预期平 (区级职业广播, 撞墙第4次) — 不调参压它
- NN 贡献: 对比 baseline vs full 的 CPC drop

## 注意
- 历史伦敦 CPC 0.56 不可直接比 (T=4 时段语义 + 16907 格不同)
- 本地 Windows CPU 稀疏 GNN 会 segfault, 只能 smoke; 全量必须 GPU
