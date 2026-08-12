# 探气上甑动态覆盖控制台

这是论文实验配套的静态系统界面。页面播放 V12 真实 MuJoCo held-out rollout，而不是在浏览器中伪造动作或重新运行策略。

## 运行

从仓库根目录启动 HTTP 服务：

```bash
cd /path/to/rl_robot
python3 -m http.server 8765 --bind 127.0.0.1
```

本机打开 `http://127.0.0.1:8765/system_ui/`。远程服务器可使用 SSH 本地端口转发：

```bash
ssh -N -L 8765:127.0.0.1:8765 <server>
```

## 数据与权重

- `data/` 包含网页直接读取的 `trajectory.csv` 和 `summary.json`。
- `assets/` 包含论文轨迹对比图。
- 网页运行和播放不需要 PyTorch checkpoint，也不需要 GPU。
- checkpoint 仅在重新执行 MuJoCo 策略、生成新 rollout 时需要。
- 一条回放是 seed 100 的定性案例，不替代论文中的跨训练种子、环境种子聚合统计。

## 重新生成静态回放

若本地已有 V12 checkpoint，先生成原始 rollout，再导出网页所需字段：

```bash
.venv/bin/python system_ui/generate_ablation_replays.py --cuda-visible-devices 5
.venv/bin/python system_ui/export_static_data.py
```

`generate_ablation_replays.py` 默认读取本地 `runs/` 下的 checkpoint；这些大文件不进入 Git 历史。`export_static_data.py` 会把网页实际使用的字段写入 `system_ui/data/`。

## 当前范围

- 完整方法、Horizon-2，以及 No attention、No carry、No prediction、No service reward、No residual 消融均可回放。
- 末端位置、活动目标、阶段、覆盖率和延迟逐步来自真实 rollout。
- ROS、MoveIt! 和真实机械臂接口只表示集成边界，当前不会发送实机控制命令。
