# PRD实验详细记录

> 本文件记录运行事实；机制结论、分工和下一步见[PRD实验总表.md](PRD实验总表.md)。核验快照：2026-10-03 13:53 Asia/Shanghai。种子与BS仅为配置和复现实验追踪字段，不作为新研究贡献。

## 当前服务器与占用

| 服务器/GPU | 登录及运行环境 | 分支 / 运行核验时HEAD | 当前状态 |
|---|---|---|---|
| 4090-48G / GPU0 | SSH已验证；conda `PRD`；训练/测试数据在本地 `/home/zhaoting.ding/local_datasets` | `prd-research-4090` / `f570bed` | F1 orig运行，PID2528606；epoch22，epoch21末AUC 0.79045014、连续下降2；43480MiB、100% |
| 4090-48G / GPU1 | 同服务器独立工作树；conda `PRD`；本地数据，NAS日志 | `codex/prd4090-gpu1-20260929` / `7810b2e` | T1运行，PID1155273；epoch9，epoch8末AUC 0.84476423、连续下降1；41008MiB、100% |
| 203-1 / GPU0 | SSH别名 `203-1-新`；conda `prd-common`；V100 32GB；数据 `/datasets/Deepfake` | `prd-203-1-seed3407` / `98ba9c0` | 10-03 14:12 SSH实查无训练/队列/tmux，GPU 0MiB；13:22 CUDA初始化及64×64前后向已通过。仅计划CPU诊断；GPU恢复根因未知 |

路径索引：

- `W0` = `/home/zhaoting.ding/disk/zhaoting.ding/DeepfakeBench`。
- `W1` = `/home/zhaoting.ding/prd-worktrees/gpu1-20260929`；`W1/logs`链接到`W0/logs/gpu1_20260929`。
- `W203` = `/root/DeepfakeBench-prd-common`；`L203` = `/root/autodl-tmp/prd-common/outputs/logs`。
- 4090资源授权截至2026-10-07 00:00 Asia/Shanghai；本次未延长。203直连origin已验证；4090直连20秒超时，本次采用PC已认证Git与SSH bundle中转。未来服务器需具备同一私有仓库的访问授权，不能假定自动已有凭据。

| 当前运行 | task / 代码提交 | 开始时间 | 队列 / 证据 |
|---|---|---|---|
| A01 / P06：4090 GPU0 F1 orig，seed42，BS32/64 | `prd_f1_orig_seed42_gpu0_20260929` / `f570bed` | 10-01 10:50:26 epoch0 | 队列PID1615052；`W0/.state/prd_gpu0_queue_resumed_20260929/runner.log`；`W0/logs/<task>/console.log` |
| A02 / P06：4090 GPU1 T1，seed42，BS32/64 | `prd_t1_seed42_gpu1_20260929` / `7810b2e` | 10-02 14:21:10 epoch0 | 队列PID1155225；`W1/.state/prd_gpu1_queue_resumed_20261002/runner.log`；`W1/logs/<task>/console.log` |

2026-10-03 13:52已撤销四个未启动项：GPU0 `prd_f2_residual_seed42_gpu0_20260929`；GPU1 `prd_t2_blur_seed42_gpu1_20260929`、`prd_t3_identity_seed42_gpu1_20260929`、`prd_r2_abs_diff_seed42_gpu1_20260929`。各`.state/<task>/cancellation.json`记录`cancelled_before_start`和`training_started=false`，没有completion或console。原循环先完成当前组归档/Git，再执行下一任务`mkdir`（非-p）；`set -e`使它在已存在的取消状态目录处退出，未调用下一launcher。两卡当前训练保持运行，取消门槛与控制路径已实际核验，真实队列退出待当前组收尾验收。预期队列exit1不表示当前训练失败，禁止删除取消标记、复活旧队列或将取消项计为完成。证据：各树`.state/prd_dispatch_hold_20261003/decision.json`。曾尝试仅SIGSTOP父队列但随后仍S，未确认持续生效；实际控制不依赖该信号。

监控仅继续当前A01/A02收尾：GPU0 `prd` 下一次2026-10-03 15:30；GPU1 `26-09-29-17-30-prd4090-gpu1` 下一次17:15，均北京时间单次预约。完成归档、结果提交并更新共享表后停止各自监控；不再要求旧有限队列所有取消项产生完成文件。

## 结果口径

- 以下AUC依次为CDFv2、DFDCP、DFDC及三集均值，均来自同一平均最佳checkpoint；不是各数据集各自最佳值拼接。训练集为FF++c23。目标三集参与checkpoint和早停选择，属于开发评估。
- epoch按原代码零基编号；best列为epoch/global step。早停为连续三次逐轮下降，回升即清零；配置nEpochs=30的循环包含epoch0至30。
- 历史T1与后续组的测试BS、训练结束预算及模型输入宽度存在差别，不能直接解释为单变量机制因果效应。T1=R1=F3仅是同一基线的三个位置；重试不增加机制实验数量。

## 4090历史运行结果

### 早期批次（seed1024）

| 记录ID / 机制 | BS训练/测试 | 状态 | best | CDFv2 | DFDCP | DFDC | 平均 |
|---|---|---|---|---:|---:|---:|---:|
| H01 BASELINE | 32/16 | 旧运行结束；最后完整epoch19，非本次早停 | 5/4938 | 0.89166540 | 0.89987072 | 0.85470935 | 0.88208182 |
| H02 T1=R1=F3 | 32/16 | 用户结束于epoch15评测；非早停 | 5/4938 | 0.90832015 | 0.89655072 | 0.86160275 | 0.88882454 |
| H03 T2 blur | 32/16 | 异常中断；最后完整epoch 0 | 1/1346 | 0.88469157 | 0.88175436 | 0.84919338 | 0.87187977 |
| H04 T2 blur | 32/16 | 完成；早停epoch6 | 5/4938 | 0.90226601 | 0.90286437 | 0.85146178 | 0.88553072 |
| H05 T3 identity | 32/16 | 完成；早停epoch15 | 14/13020 | 0.90254265 | 0.88002049 | 0.86638508 | 0.88298274 |
| H06 R2 abs_diff | 32/16 | 完成；早停epoch19 | 5/4938 | 0.90176126 | 0.88755704 | 0.84786158 | 0.87905996 |
| H07 R3 cosine | 32/16 | 异常中断；最后完整epoch 1 | 2/2244 | 0.90986268 | 0.84510217 | 0.82986268 | 0.86160918 |
| H08 R3 cosine | 32/16 | 异常中断；最后完整epoch 0 | 0/898 | 0.87505320 | 0.84092965 | 0.81898938 | 0.84499074 |
| H09 R3 cosine | 32/16 | 异常中断；最后完整epoch 无 | — | — | — | — | — |
| H10 R3 cosine | 32/64 | 异常中断；最后完整epoch 10 | 2/2693 | 0.91401253 | 0.87060377 | 0.83409644 | 0.87290425 |
| H11 F1 orig | 32/64 | 完成；早停epoch19 | 12/11224 | 0.89236845 | 0.89278156 | 0.83667936 | 0.87394312 |
| H12 F2 residual | 32/64 | 完成；早停epoch7 | 4/4489 | 0.91399475 | 0.89292355 | 0.85123818 | 0.88605216 |

### 后续完整对照批次（seed2026）

| 记录ID / 机制 | BS训练/测试 | 状态 | best | CDFv2 | DFDCP | DFDC | 平均 |
|---|---|---|---|---:|---:|---:|---:|
| C01 T1=R1=F3 | 32/64 | 完成；早停epoch4 | 2/2244 | 0.90962631 | 0.89386922 | 0.83582218 | 0.87977257 |
| C02 T2 blur | 32/64 | 完成；早停epoch5 | 2/2693 | 0.90531712 | 0.88338358 | 0.85375440 | 0.88081836 |
| C03 T3 identity | 32/64 | 完成；早停epoch4 | 1/1795 | 0.91367335 | 0.88175411 | 0.83044898 | 0.87529214 |
| C04 F1 orig | 32/64 | 完成；早停epoch29 | 1/1346 | 0.90219608 | 0.89407548 | 0.83329957 | 0.87652371 |
| C05 F2 residual | 32/64 | 完成；早停epoch17 | 9/8530 | 0.88887762 | 0.88556745 | 0.85107919 | 0.87517476 |
| C06 R2 abs_diff | 32/64 | 完成；最大epoch30 | 10/9428 | 0.92543325 | 0.88264527 | 0.86427628 | 0.89078493 |

H03为T2第一次异常尝试，H04为同一T2的重试。H07–H10是同一R3的四次尝试，全部未完成；H09连完整epoch也没有，其余AUC仅为中途checkpoint，不能写为最终完成结果。H05训练正常早停，但旧外层启动器曾返回127；训练摘要证据与外层状态需区分。

### 运行定位及代码证据

历史运行的训练日志为`<工作树>/logs/training/csy/<run>/training.log`，源权重为同目录`test/avg/ckpt_best.pth`，元数据为`run_metadata.json`；摘要为`<工作树>/scripts/experiment_summaries/<run>_实验摘要.md`。代码提交取运行元数据，历史Git说明曾改写，原运行hash不得回填成新hash。

| 记录ID | 工作树 / run | 代码提交 | 首条日志 → 最后日志（非一律代表成功完成） |
|---|---|---|---|
| H01 | W0 / `lora_2026-08-31-22-24-08` | `未核验` | 2026-08-31 22:24:09 → 2026-09-03 01:00:34 |
| H02 | W0 / `lora_prd_sd15_signed_clip_2026-09-07-09-14-33` | `d35990e789ecfc8d404a44bde5e1470f00449a27` | 2026-09-07 09:14:33 → 2026-09-08 23:27:07 |
| H03 | W0 / `lora_prd_t2_blur_signed_clip_2026-09-09-00-04-18` | `1dc53a2801e0558f06badaaed45570e7fa79804b` | 2026-09-09 00:04:18 → 2026-09-09 01:41:44 |
| H04 | W0 / `lora_prd_t2_blur_signed_clip_restart_20260909_2026-09-09-09-45-04` | `542a45069619e89e882a535aadba129211b0ad38` | 2026-09-09 09:45:04 → 2026-09-09 18:18:13 |
| H05 | W0 / `lora_prd_t3_identity_signed_clip_20260909_2026-09-09-18-29-10` | `1d18e98463d2e804ebc5dadb3c30191ee909b17a` | 2026-09-09 18:29:10 → 2026-09-10 14:32:21 |
| H06 | W0 / `lora_prd_r2_abs_diff_20260910_2026-09-10-14-43-55` | `94ca37e13cabde05d72bb6ef3a881e55430f5f5e` | 2026-09-10 14:43:55 → 2026-09-12 15:27:03 |
| H07 | W0 / `lora_prd_r3_cosine_20260912_2026-09-12-15-37-48` | `c670de8aaaeba5c2b38931b3fe65507030ebd91e` | 2026-09-12 15:37:48 → 2026-09-12 21:03:38 |
| H08 | W0 / `lora_prd_r3_cosine_restart_20260914_2026-09-14-08-58-57` | `8fa77d35b97b3a0dff69bf2007413839ccec4cb9` | 2026-09-14 08:58:57 → 2026-09-14 11:48:49 |
| H09 | W0 / `lora_prd_r3_cosine_retry_hard_20260914_2026-09-14-16-47-39` | `a89bcec50b94b37fd1807d455ecd90defd80e798` | 2026-09-14 16:47:39 → 2026-09-14 17:21:15 |
| H10 | W0 / `lora_prd_r3_testbs64_20260914_2026-09-14-17-25-11` | `50ccde65f22edac19a999b6b3979e1a6f62b2eb7` | 2026-09-14 17:25:11 → 2026-09-15 20:08:02 |
| H11 | W0 / `lora_prd_f1_orig_20260918_2026-09-18-17-09-38` | `1249d8a557fdd7b0d042de9a3e642474266a8b61` | 2026-09-18 17:09:38 → 2026-09-20 14:14:12 |
| H12 | W0 / `lora_prd_f2_residual_20260920_2026-09-20-14-30-49` | `f38e7364e5ca058522f38f1730621c68146e368d` | 2026-09-20 14:30:49 → 2026-09-21 08:32:21 |
| C01 | W0 / `lora_prd_t1_seed2026_20260924_2026-09-24-22-58-34` | `cbc65995a72760bd9a03dc460902a7e14ea6adc5` | 2026-09-24 22:58:34 → 2026-09-25 09:52:16 |
| C02 | W0 / `lora_prd_t2_blur_seed2026_20260925_2026-09-25-15-17-37` | `10362e68001abca291aacbbc2910dbde24d41a98` | 2026-09-25 15:17:37 → 2026-09-25 22:23:25 |
| C03 | W0 / `lora_prd_t3_identity_seed2026_20260926_2026-09-26-14-18-30` | `413291f33c586726b16bd6c0006ff567f96f8077` | 2026-09-26 14:18:30 → 2026-09-26 20:07:50 |
| C04 | W0 / `lora_prd_f1_orig_seed2026_20260926_2026-09-26-20-07-58` | `76c7ae039fd7ad22bebf9e798a96ca770f3cd8b2` | 2026-09-26 20:07:58 → 2026-09-29 16:15:37 |
| C05 | W0 / `lora_prd_f2_residual_seed2026_20260926_2026-09-29-16-36-48` | `b8462f709116ecbddba93b8e0f85927e9f42534e` | 2026-09-29 16:36:48 → 2026-10-01 10:50:15 |
| C06 | W1 / `lora_prd_r2_abs_diff_seed2026_gpu1_20260929_2026-09-29-13-02-20` | `e8bd3ba9719a4e1efec4379c2bb353c9d9bbd4b7` | 2026-09-29 13:02:20 → 2026-10-02 13:56:02 |

### 完成归档与中文Git记录

本次核对了下列completion内容、结果提交与摘要；SHA256为归档证据记录值，未在本轮对全部4090大权重重新计算。早期H01–H12的独立归档hash本轮未复核；摘要提交已查Git历史，见下表，不把摘要提交等同归档hash验证；H02摘要明确另有`W0/checkpoints/prd_t1_final/ckpt_best.pth`。

| 记录ID | task / 权威completion | 归档目录 | SHA256 | 中文结果提交 |
|---|---|---|---|
| C01 | `prd_t1_seed2026_20260924` / `.state/prd_t1_seed2026_20260924/completion.json` | `W0/logs/checkpoint_backups/prd_t1_seed2026_20260924` | `b52c5455e5c8d8d6d411be6c96dd35600c6346cd9282b560744f389a8323b900` | 697c174d58bcec0a6fd972e10c67c832670ec320 记录 F2 与 T1 seed=2026 实验结果 |
| C02 | `prd_t2_blur_seed2026_20260925` / `.state/prd_t2_blur_seed2026_20260925/completion.json` | `W0/logs/checkpoint_backups/prd_t2_blur_seed2026_20260925` | `6b4817d8e1464503c181c9820b4f8c2db03873522ae7b3bede2a3fd4d3743223` | 757dd4e13e1b945066672642d96e462f5de911db 记录 T2 高斯模糊 seed=2026 实验结果 |
| C03 | `prd_t3_identity_seed2026_20260926` / `.state/prd_t3_identity_seed2026_20260926/completion.json` | `W0/logs/checkpoint_backups/prd_t3_identity_seed2026_20260926` | `f212364e0be96d43c826c50e12db6b8c244b8d2a8c34cfe7171fe26e0005818e` | 76c7ae039fd7ad22bebf9e798a96ca770f3cd8b2 记录 T3 恒等探针 seed=2026 实验结果 |
| C04 | `prd_f1_orig_seed2026_20260926` / `.state/prd_f1_orig_seed2026_20260926/completion.recovered.json` | `W0/logs/checkpoint_backups/prd_f1_orig_seed2026_20260926` | `7a46f4fa311733aea80ca862ab3c81c8638128d6c3a5b03c9609e4f068d8e505` | b8462f709116ecbddba93b8e0f85927e9f42534e 恢复 F1 seed2026 归档并修复 GPU0 队列交接 |
| C05 | `prd_f2_residual_seed2026_20260926` / `.state/prd_f2_residual_seed2026_20260926/completion.json` | `W0/logs/checkpoint_backups/prd_f2_residual_seed2026_20260926` | `5adae2e62bcd0d9681828c225b5571bdf72744d7763efa33f3e4b3fc1ee4f9ed` | f570beda28c9d63edce29d91769960ce7e3c53ac 记录 F2 残差特征 seed=2026 实验结果 |
| C06 | `prd_r2_abs_diff_seed2026_gpu1_20260929` / `.state/prd_r2_abs_diff_seed2026_gpu1_20260929/completion.recovered.json` | `W1/logs/checkpoint_backups/prd_r2_abs_diff_seed2026_gpu1_20260929` | `d9780077bc0eff3387e8bb88af80ec2d17d7660627d23c3f9c2cd1719046f86e` | 7810b2eac1861b22f81d9d400f5027b90ef1dfe2 恢复 GPU1 R2 seed2026 归档并接续 seed42 队列 |

C04（F1）训练退出0，但运行中Bash被原地重写导致外层127；权威文件为`completion.recovered.json`，原失败文件保留。C06（R2）训练退出0且达到epoch30，首次归档因缺少checkpoint_backups父目录失败；恢复归档后以`completion.recovered.json`为准。不得据旧失败记录重跑。

同一T1 seed2026权重的测试BS16/64诊断已完成，AUC差（16−64）分别为CDFv2 `+0.000001515751`、DFDCP `−0.000004318525`、DFDC `−0.000000381785`。证据为`W0/scripts/experiment_summaries/prd_t1_seed2026_eval_bs_comparison.json`。该诊断不是新增训练，不证明不同训练BS等价。

## 203-1运行记录

| 记录ID / 机制 | 配置 | 状态 | best / stop | CDFv2 | DFDCP | DFDC | 平均 |
|---|---|---|---|---:|---:|---:|---:|
| D01 T1 | seed3407，BS16/64 | 完成；TRAINING_EXIT_CODE=0 | epoch3 / 早停epoch11 | 0.83825408 | 0.88201386 | 0.85481217 | 0.85836004 |

- D01 task：`prd_t1_seed3407_bs16_203_retry_20260930_a1`；2026-09-30 16:25:08启动，2026-10-03 10:16:24完成归档状态。
- run：`lora_prd_t1_seed3407_bs16_203_retry_20260930_a1_2026-09-30-16-25-34-344158`；代码提交`62cc7521bf9ff3db794b9ad0fb279f1a5979b3d2`，结果提交`3b281d46a260886ad348e0fdc823f2a420bf029f`（中文结果提交，摘要已核验）。
- 权威状态：`W203/.state/prd_t1_seed3407_bs16_203_retry_20260930_a1/completion.json`。源权重：`L203/training/<run>/test/avg/ckpt_best.pth`。归档：`L203/checkpoint_backups/prd_t1_seed3407_bs16_203_retry_20260930_a1`；包含权重、manifest、摘要、元数据和source_status。
- SHA256：`bfd35a301ae126871053666b106a81a753521af79d5d08a6f29c73b401d5bf6f`，字节数1401384250；本轮源/归档hash一致。

| 同机制的历史尝试 / 排队项 | 状态与证据 |
|---|---|
| 2026-09-27 seed3407预检 | `.state/prd_seed3407_queue_20260927/completion.json`为`blocked_preflight_oom`，正式run为空、队列未安装；记录提交`7200cc0`。不计正式训练结果 |
| 2026-09-27旧BS16队列 | `.state/prd_seed3407_bs16_queue_20260927`记录13:01:08启动；仅据此不能认定完成，后续转tmux运行 |
| 2026-09-27旧T1 tmux运行 | task=`prd_t1_seed3407_bs16_203_tmux_20260927`，09-27 13:12:40启动，09-30 10:57:54退出1；DataLoader worker Bus error。失败尝试，不能与D01最终结果混合 |
| T2 blur三次重试 | task前缀`prd_t2_blur_seed3407_bs16_203_retry_20260930_`，a1于10-03 10:16:26、a2于10:17:56、a3于10:19:05启动；均在epoch0前`No CUDA GPUs are available`并退出1。无completion、无结果提交，属同一T2的三次失败 |
| T3 identity | 本轮无状态目录及队列启动事件，未启动；不能记为失败或完成 |

203与4090同时改变了训练BS、seed、硬件及环境，不能当作只改变随机种子的严格对照。13:22小矩阵CUDA前后向实际通过，仅证明该时刻CUDA可用，不证明正式batch或未来稳定性。13:25准备的`run_prd_203_continue_20261003.sh`仍是未提交文件；14:12复核无新队列状态或训练进程，不能把准备当启动。14:14按用户新方向新增`W203/.state/prd_seed3407_bs16_continue_20261003/cancellation.json`及T2/T3 `continue_20261003_a1`状态目录的同名取消标记（`cancelled_before_start`）。旧脚本`set -e`及任务`mkdir`非-p门槛阻止进入launcher；脚本原文保留、无kill、旧失败证据不变；不是训练失败。脚本SHA256=`3b12b3d196c3c20bbf99ffd7dc38d7e9df2b793f0f5345d941d3275b92b5ed3e`。旧监控已删除，未宣称新监控已安装。

### P04-D01 认领协议：域/标签混杂审计与统一视频清单

- 状态：**首次数值计算未完成；修正辅助split解析后复核**；认领2026-10-03 14:14、首次执行14:20；负责人/对话：203 / `01a0e12a-41bd-7b81-999d-fd62332d03cb`。认领提交`5324e24`已push并在203读回。服务器203，仅CPU，不占GPU。共享表查重版本`d32c9b3`；N01/N02候选属4090，本项仅覆盖数据分布混杂和统一清单，不替代表示诊断。
- 科学假设与否定标准：原始三目标域的真假比例可使域先验`P(fake|domain)`产生描述性池化AUC>0.55；若≤0.55不支持实质比例混杂。使用同一JSON索引的域×标签等量对照应给出AUC=0.5、互信息=0。不是外推分类器性能，不证明PRD实际上利用域偏差，也不能解释逐集AUC。
- 输入：现有`/datasets/Deepfake/DeepfakeBench/config/dataset_json`的CDFv2/DFDCP/DFDC测试划分及当前detector标签映射；记录JSON SHA256、原始video/frame计数、空记录、重复帧路径与跨split精确video ID交集。不同视频/域的真实人物或伪造源谱系不能仅由文件名认定独立，将单列限制。
- 固定协议：先统计所有有有效帧的视频；输出每域×二元标签`k=min(32,六格最小视频数)`视频；按`SHA256(P04-D01/v1|dataset|label|video_id)`升序选取，每视频排序中位帧。无随机seed搜索、无目标AUC调参。每行含domain、label、video_id、frame及JSON hash；同一视频多帧不作为独立样本。域分数的AUC仅按视频计数精确计算，同时报告原始分布和选中分布。
- 代码/配置：服务器基线`98ba9c0`；诊断代码`4a36bc13474a74f6b80b50a1766e090ff463c132`，`scripts/p04_d01.py` SHA256=`1a276be0f9a3b14beb96c845bdcbe8ff0a5bb8020ba57085693e379928836cde`，语法检查通过；未push训练代码。二元标签读取`training/config/test_config.yaml`，detector中的三目标域及test划分不变；video_id按loader的`label_string + '_' + 原始video_key`定义。输出`W203/.state/p04_d01_20261003/`，不可覆盖。输入结构不符、任一域缺标签或SHA变化即停，不静默换数据。
- 预算与停止：一次CPU诊断，单进程、墙钟≤300秒、地址空间≤2GiB、GPU=0、新训练=0；完成清单/统计/hash即停止。203后续GPU训练时限尚未明确，本项不依赖该预算，不扩展训练矩阵。无需长任务定时唤醒。
- 首次执行按结构检查停止：DFDC的`val`是含label/frames的单记录，不是video_id→record映射，导致TypeError；无最终统计或manifest。三域主`test`结构正常。保留首目录和`failure.json`；修复只把该辅助val标为“结构无效，重叠未知”，不更换test、样本规则或判据；CDFv2/DFDCP已发现val与test至少有同ID，需完整计数确认。修复复核使用新目录`.state/p04_d01_20261003_a2`，初次数秒加复核合计仍≤300秒。不是新机制实验或训练重试。

## 较早未形成正式结果的启动记录

4090日志还存在2026-08-31的18:00:47、21:05:24、21:09:24、21:14:38启动记录，及BS16/BS8/BS4探索（22:35:16、22:35:52、22:36:26）；本轮未取得正常完成与归档证据，不能计为已完成新机制实验。2026-09-07 09:12:17及09:13:35的T1尝试无完整epoch；正式历史T1为H02的09:14:33运行。保留原日志，不把启动目录数量当实验数量。

历史摘要提交索引（提交内容已读回；H03及H07–H10是异常记录提交）：

| 记录 | 摘要提交 | 记录 | 摘要提交 |
|---|---|---|---|
| H01/H02 | `303132f` | H03 | `542a450` |
| H04 | `66f9f85` | H05 | `c670de8` |
| H06 | `9e5ed49` | H07 | `241ca22` |
| H08 | `32c38db` | H09 | `f71e885` |
| H10 | `5f08c39` | H11 | `619554f` |
| H12 | `697c174` | — | — |

## 共用Git同步方法

仓库为私有仓库`https://github.com/John08066/DeepfakeBench.git`，共享分支为`codex/prd-progress`。该分支的两表是唯一事实表；main及训练分支根目录的同名文件仅为入口，不维护第二份结果。各服务器训练代码和Git索引保持独立。Git不会自动让一个分支的文件出现在所有其他分支，因此入口与读取流程必须一起使用。

每次训练前，在有仓库访问权限的机器执行：

```bash
git fetch origin codex/prd-progress &&
git show FETCH_HEAD:PRD实验总表.md &&
git show FETCH_HEAD:PRD实验详细记录.md
```

认领与状态更新使用独立文档工作树，不能在运行训练的工作树切换分支。首次可用`git worktree add --detach ../prd-progress-docs FETCH_HEAD`；复用时先确认该文档工作树无未提交修改，再fetch并快进到远端。仅编辑自己负责的行，提交与推送示例：

```bash
git -C ../prd-progress-docs add -- PRD实验总表.md PRD实验详细记录.md
git -C ../prd-progress-docs commit -m "登记服务器与PRD机制实验认领"
git -C ../prd-progress-docs push origin HEAD:refs/heads/codex/prd-progress
```

只有push成功且读回确认本次认领仍有效，才启动对应任务。被非快进拒绝时重新fetch、读表判重，保留双方行后再提交；禁止force push。不要覆盖整表来解决冲突。同一机制的重复验证必须写明必要性，不能换seed后绕过占用检查。

4090直连origin在本次20秒检查中超时。可由PC读取/push同一远端分支，制作仅文档及必要祖先的Git bundle，通过SSH送到服务器，再`git fetch <bundle> refs/heads/codex/prd-progress:refs/remotes/origin/codex/prd-progress`。必须核对PC远端SHA与服务器ref一致，并读出两表；不能把传了文件当成已push，也不能把缓存当成训练前最新状态。服务器更新经PC中转回共享分支，同样以远端push成功为启动前提。不复制GitHub token、SSH密钥或密码。

新服务器登记模板：服务器号/GPU、负责人、SSH别名、工作树/分支/代码提交、环境及数据路径、最近SSH和CUDA核验时间、当前问题ID/task/状态、开始或结束时间、结果/归档/提交证据。SSH可达、nvidia-smi可读、PyTorch能训练分别记录，不混为“已登录可训练”。不允许把未知状态写为空闲；占用认领仍需现场核验。

## 旧PROJECT_REPORT.md迁移

旧文件是通用DeepfakeBench架构/操作介绍，不是PRD实验时间线。本次以服务器原始日志、元数据、摘要、completion、权重归档及Git重建上述历史；旧文件的有用入口保留如下，不沿用其旧Python环境和Xception示例作为PRD配置。

| 用途 | 入口 | 使用边界 |
|---|---|---|
| 数据预处理与索引 | `preprocessing/preprocess.py`、`rearrange.py`、`dataset2lmdb_test.py` | 人脸帧→JSON索引→可选LMDB；不要擅自重建/覆盖现用数据 |
| 训练与评测 | `training/train.py`、`training/test.py`、`training/config/` | 具体参数以各run配置副本/metadata为准，不以旧报告示例替代 |
| PRD模型与训练协议 | `training/detectors/lora_detector.py`、`training/trainer/`、`training/early_stopping.py` | 查探针、残差、dropout、分类头及早停真实实现 |
| 数据/指标/分析 | `training/dataset/`、`training/metrics/`、`analysis/`、`training/tsne.py` | 验证数据协议与同checkpoint指标；可视化不是机制证明 |
| 原项目说明与历史 | `README.md`；旧`PROJECT_REPORT.md`在Git历史中 | 原文件初始记录`88a831a`；迁移后从现用分支删除，历史仍可git show读取 |

两表维护全局进度；各run原摘要、日志、权重和失败证据仍保留，它们是证据而非第三份手写全局报告。该迁移不删除训练数据或模型。

## 文档接入验收（2026-10-03）

| 位置 | 仅文档入口提交 | 已验证的共享读取方式 |
|---|---|---|
| GitHub默认main | `1605e08`（已push） | 根目录两表入口指向`codex/prd-progress`，未来服务器从默认分支即可发现规则 |
| 本机4090工作目录 | `783e64f` | 共享远端跟踪ref；两表全文位于独立`prd-progress`文档工作树 |
| 4090 GPU0分支 | `0656af7` | PC认证Git发布后，SSH传增量bundle并fetch同一共享ref |
| 4090 GPU1分支 | `fca01d6` | 同Git公共对象库，独立工作树读取共享ref；训练索引独立 |
| 203-1分支 | `98ba9c0` | 服务器直接fetch私有origin成功；采用项目环境Python安装入口 |

共享初始化提交`fe45d30`已push并在两服务器读取；后续版本以共享分支ref为准，不要求入口文件每次改写SHA。接入提交均只含两表入口、置顶Notes及旧报告删除；未推送各服务器训练分支。既有未提交笔记/脚本/摘要与GPU1日志符号链接保留，203使用既有Codex身份的单次Git参数，未改全局配置。原PROJECT_REPORT在共享分支、main、当前本地分支及三个服务器工作树均已删除，旧分支历史仍可追溯。

这次核验覆盖“发布与读取”，不承诺未来离线、鉴权变化或他人未遵守认领流程时自动同步；任何服务器仍须在每次新实验前成功同步、认领并读回。本次仅改变研究协作与旧队列后续派发，不启动N01/N02/N03。
