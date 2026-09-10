# PRD 监控恢复与交接

用户 crontab 已安装 `scripts/prd_monitor.cron`：每六分钟执行一次检查，
不依赖 SSH、VS Code 或 tmux 存活。系统 cron 与项目挂载恢复后，下一个检查点继续。
本次未通过重启整机做验收，不能保证停电期间运行或挂载/cron 故障时可用。

## 状态与重试

- `.state/prd_r2_abs_diff_20260910_watch/status.json`：当前R2的最后检查时间、进程和事件原因；旧STATE只保留归档。
- 同目录 `trigger.json`：交接次数、状态、报告路径、下次允许重试时间。
- `dispatching` 中断后重新核对现场；失败至少间隔六分钟，总计最多三次。
- `codex_returned` 表示 Codex 返回零退出码；正常交接还要求现场存在后续训练，但完整迭代验收仍须查看回调报告。
- `exhausted` 保留失败记录，不无限消耗额度；`blocked_active_training` 禁止重复派发。
- `error.json` 保存读取/状态异常；历史错误不代表本次检查也失败，应比较时间。
- `logs/RealTime/prd_monitor_recovery.log` 为定时任务输出；交接输出单独按时间命名。

进程消失且没有正常结束标记时，自动生成实验摘要，再调用受限 Codex 仅诊断并写报告，
不把异常退出当作早停，不擅自从下一实验开始。正常结束/三轮连续下降仍使用原交接流程。

## 防重与边界

监控和 Codex 子进程共享排他文件锁；重复监控拿不到锁就退出。
重试前扫描同用户 `train.py` 进程，发现已有训练则暂缓交接。
交接指令要求核对已有成果及前次输出，避免重做部分完成的动作。
正式训练统一使用scripts/launch_prd_training.sh的独占训练锁。
该锁不约束绕过启动器的人工命令，也不是通用 exactly-once 事务。

切换实验时，正常交接需一起更新脚本中的 LOG、STATE、TRAINING_PID 与配置匹配条件，
不重启仍在持锁的监控；由 cron 下一检查点接管。
当前 Codex PATH 指向已验证的扩展版本；扩展删除/升级后若路径不可用，交接明确失败，需更新此配置。

验收命令：`python3 scripts/test_watch_prd_early_stop.py -v`。
单次只观察：`python3 scripts/watch_prd_early_stop.py --once`。
实际授权交接：增加 `--dispatch`；已安装 cron 使用此模式。

Codex 调用沿用正常审批和沙箱，未开启权限绕过。
OpenAI Docs 核对来源：https://learn.chatgpt.com/docs/non-interactive-mode
