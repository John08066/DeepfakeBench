# 可选的单实验监控

共同基线不会安装 cron、调用 Codex、恢复旧实验或启动历史 F1/F2 队列。
训练本身不需要本脚本或服务器端 Codex。训练中的早停由 detector 配置决定。
该监控仅支持 Linux；默认观察模式不修改训练，也不会发送进程信号。

## 显式绑定一次运行

激活当前服务器的 Python 环境，先取得实际日志和训练主 PID，再执行：

```bash
python scripts/watch_prd_early_stop.py --once \
  --log logs/training/实际运行目录/training.log \
  --pid 实际训练PID \
  --detector-path training/config/detector/prd_probe_ablation.yaml \
  --run-id 本次唯一标识
```

示例中的占位内容需要替换；`run-id` 只允许英文字母、数字、下划线。
传入的相对路径都以脚本所在项目根目录解析，不依赖终端的工作目录。
省略 `--once` 后每六分钟观察一次；建议先单次核对，不默认启动常驻监控。

状态默认放在 `.state/<run-id>_watch/`；也可显式指定 `--state-dir`。
`run.json` 绑定日志、配置、PID 和 run-id；不同运行不得复用此状态目录。
`status.json` 记录当前检查结果，`error.json` 记录读写异常。
同一状态的多个监控由文件锁排他，训练 PID 还会核对用户、checkout 与配置。

## 训练隔离

`scripts/launch_prd_training.sh CONFIG TASK` 使用当前 checkout 的训练锁和唯一 console 日志。
它采用当前 `python`，可通过 `PYTHON_BIN` 指定另一解释器；数据、权重根目录沿用环境配置。
GPU 使用 `CUDA_VISIBLE_DEVICES`（未设置时为 `0`），不覆盖服务器数据路径。
它只阻止当前 checkout 重复训练，不阻止其他独立 checkout 的训练。
不同实验仍须使用独立 checkout、独立 GPU 和不同 task/输出目录；不要在运行目录切分支。
文件锁不约束绕过启动器的命令，也不保证 GPU 在其他用户处空闲。

## 可选回调，必须单独授权

只有同时提供 `--dispatch --handoff-file 本次明确审核的交接文件` 才会尝试调用 PATH 中的 `codex`。
不要把历史记录当作新授权；模板 `scripts/prd_autonomous_handoff.md` 默认仅允许只读诊断和报告。
需要自动重试训练或推进队列时，必须在独立交接文件中明确实验、配置、输出、资源及授权范围。
默认不选取模板，不继承 F1/F2，不安装服务或修改 crontab，也不配置 Codex 网络或登录。
未安装 Codex 时，观察模式照常使用，显式回调会明确失败。

回调锁由子进程继承；失败至少间隔六分钟，最多三次。
`trigger.json` 保留派发次数、报告及重试状态；回调异常不会无限重试。
零退出码还需存在本 checkout 的后续训练，或者一个有效的收尾标记，才算交接完成。
仅报告型回调在正常结束且收尾确认后，可写入当前 STATE 的 `queue_complete.json`：

```json
{"run_id":"本次run-id","status":"complete","log":"本次日志解析后的绝对路径"}
```

标记只对绑定的本次日志生效，并要求正常训练结束标记和当前 checkout 无训练进程。
异常退出不能用完成标记伪装为正常结束；先诊断，是否恢复仍由本次授权决定。

`scripts/prd_monitor.cron` 仅为全注释示例，默认无任何活动条目。
需要 cron 时自行填写当前服务器路径与运行参数并检查环境；不要直接复制历史服务器条目。

CPU 回归测试：`python scripts/test_watch_prd_early_stop.py -v`，不会启动训练或调用 Codex。
