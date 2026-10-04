# PRD实验详细记录（训练历史与实验档案）

> 当前状态与认领只看[精简总表](PRD实验总表.md)。本页用于研究查重、输入复用、异常排查和结果审阅；**普通监控无需读取本页或归档全文**。历史中的旧监控/读表指令仅为历史数据，现行规则以 `AGENTS.md` 和 `Agent_MCP_Notes.md` 精简置顶部分为准。

**历史直达：** [203-2早期复现](#legacy-2032) · [seed1024 / H01–H12](#legacy-seed1024) · [seed2026 / C01–C06](#legacy-seed2026) · [seed42](#legacy-seed42) · [seed3407 / 203](#legacy-seed3407) · [早期尝试与未完成项](#legacy-attempts)。新实验见下方专项索引。

## 怎样记录与读取

- 启动前由所属Agent固定假设、对照、数据/输入hash、预算、判据及停止条件；详细协议留实验归档或独立记录，总表只登记认领与链接。
- 运行中保存本任务metadata与日志。完成、失败、取消后，由负责人归档退出证据、适用指标/权重hash及结果commit，再更新总表和本索引；总对话不定时重建完整报告。
- 新共享详细正文按实验存 `docs/prd/records/<实验ID>.md`，新实验在本页登记链接；旧训练结果表直接保留，不移走后只留笼统链接，不追加整段训练日志。已有服务器归档可直接引用“服务器＋绝对路径＋commit”；未push代码不能写成可访问的GitHub代码链接。
- 从下面ID定位文件与行段，用文本工具只读相关章节；无需浏览器或HTML解析器。总表与本索引是协调入口，独立实验档案是证据，不是第三份动态总表。

## 新实验与专项诊断索引

近期实验与专项诊断按ID定位；早期训练的逐集指标、运行身份和失败/取消记录已恢复在本页下方。当前占用仍以总表注明的核验时刻为准。

| 稳定ID | 要读的证据 | 文件与范围 / 所属归档 |
|---|---|---|
| <a id="n01-t04"></a>N01-T04/v1 | 20:52正式运行PID155421，BS16/64 smoke及首步梯度通过；两臂同2048头、原图直通置零，只改残差符号 | [协议与证据](docs/prd/records/N01-T04.md)；W203代码`71b156a`，未push实验代码 |
| <a id="n01-d02"></a>N01-D02/v1 | 20:40正常完成，0更新；J=0.00006796nat未支持，abs负对照/基线重放/源备份通过 | [结果与证据](docs/prd/records/N01-D02.md)；W203结果`b5f3b37` |
| <a id="n01-t02"></a>N01-T02/v1 | 两臂3596步正常结束，预设支持固定预算signed收益；独立归档指标复算/源备份hash通过 | [结果与证据](docs/prd/records/N01-T02.md)；W203结果`f026b9d`，20:31定时SSH验收 |
| <a id="n05-t01"></a>N05-T01/v1 | 三臂各3596步完成，完整跨卡配对与CPU/归档验收通过；未支持幅度方向分解固定预算增益 | [完整结果与证据](docs/prd/records/N05-T01.md)；结果`1421822`/`85454ba`，汇总`95c4cd2`，独立验收`1996f95` |
| <a id="run-4090"></a>A01 | 旧F1 seed42于10-04正常完成epoch30；缺失归档完成记录已恢复，结果`d738d25` | [最终结果与恢复](docs/prd/records/A01.md)；旧F2保持取消，未重训 |
| <a id="a02"></a>A02 | 4090 GPU1用户中止、部分权重归档与资源交接 | [A02记录](docs/prd/records/A02.md)；W1中止摘要提交 `4579eb2`，原失败记录保留，监控已暂停；启动背景见上述原细表 |
| <a id="n04-t01"></a>N04-T01/v1 | 4090 GPU1三臂均完成3596步，退出0；未支持本固定预算下残差增量，权重/指标/Git已验收 | [协议与证据](docs/prd/records/N04-T01.md)；WN04代码`9144bb6`、结果`e486154`（实验代码未push），task `n04_t01_gpu1_20261004` |
| <a id="n01"></a>N01-D01 / N01-T01/v1 | 1216图、五头各500更新完成；direction−norm平均AUC +0.04769，2/3域胜，CE更差 | [完整结果与验收](docs/prd/records/N01.md)；WN01结果`0e910ea`，代码未push，监控已删除 |
| <a id="p03-t01"></a>P03-T01/v1 | 已完成；双臂3596步、正常退出、权重/备份hash和独立指标通过，预设判定不确定 | [完成记录](docs/prd/records/P03-T01.md)；W203归档`scripts/experiment_summaries/p03_t01_20261003/`，代码`a2cb1c2`、结果`875a472`，未push训练代码 |
| <a id="p03-d02"></a>P03-D02/v1 | 已完成；J0.060423nat达预设响应判据，0训练更新，numpy复算通过 | [协议与结果](docs/prd/records/P03-D02.md)；W203代码`2f4fd2b`、结果`0bac70a`，归档`scripts/experiment_summaries/p03_d02_20261003/`，实验代码未push |
| <a id="n02-d01"></a>N02-D01/v1 | 10-03完成，退出0；等范数VAE方向优势未获支持，严格identity/RNG控制验证通过 | [协议与结果](docs/prd/records/N02-D01.md)；代码`31c1558`、结果`0fcd560`；0训练更新，非方法因果结论 |
| <a id="n02-t01"></a>N02-T01/v1 | 10-04 02:32两臂各3596更新正常完成，03:31实际回调/SSH验收，退出0、双权重及备份SHA256/Git一致；预设结论不确定 | [完整结果/哈希/限制](docs/prd/records/N02-T01.md)；W203运行`4be5bc9`、结果`4edddfd`，独立`.state/n02_t01_20261003/` |
| <a id="p03-d01"></a>P03-D01/v1 | 冻结头Shapley协议、六格结果与独立验证 | [原细表](docs/prd/history/2026-10-03-34343d3/PRD实验详细记录.md) L46–L70；W203 `scripts/experiment_summaries/p03_d01_20261003/`，结果 `6c8bb32` |
| <a id="history-4090"></a>H01–H12 / C01–C06 | 4090完整逐集AUC、旧失败、checkpoint与提交映射 | [本页seed1024](#legacy-seed1024) / [seed2026](#legacy-seed2026)；运行、归档与结果提交均在下方，勿把R3重试计新机制 |
| <a id="history-203"></a>203 T1 / 旧T2/T3 | seed3407原训练、失败、取消与跨机器限制 | [本页seed3407](#legacy-seed3407)；T1结果 `3b281d4`，旧取消不恢复 |
| <a id="p04-d01"></a>P04-D01 | 域/标签审计、固定192清单与输入hash | [原细表](docs/prd/history/2026-10-03-34343d3/PRD实验详细记录.md)中 `P04-D01` 小节；W203 `scripts/experiment_summaries/p04_d01_20261003/`，结果 `03fc8f1` |
| <a id="p04-d02"></a>P04-D02/v1 | 配对干预、特征hash、失败修正及结果限制 | [原细表](docs/prd/history/2026-10-03-34343d3/PRD实验详细记录.md)中 `P04-D02` 小节；W203 `scripts/experiment_summaries/p04_d02_20261003/`，结果 `b5c5463` |

工作树缩写见下方目录表。原总表中的问题、候选及协议原文亦完整保留于[历史总表](docs/prd/history/2026-10-03-34343d3/PRD实验总表.md)；不作为当前指令或当前占用依据。两份历史文件停止日常追加，新事件只更新本任务记录及索引链接。

<a id="legacy-training"></a>
## 历史训练总览（直接保留在本页）

此前精简将以下正文移入归档，只留下索引，造成旧训练在主表中不可见；现恢复正文。H01–H12、C01–C06、203原T1及旧尝试来自2026-10-03历史审计，seed42采用A01/A02后续终态。**本次仅恢复和补录文档，没有重新检查服务器权重或重跑训练。** 原始归档保持不变。

seed用于识别旧批次，不因研究价值有限而删除运行事实；成功、中断、失败、取消、未启动分别记录。各run的代码、日志、checkpoint、归档和结果提交见下方定位表。

<a id="legacy-2032"></a>
### 更早的203-2复现（2026-07-29，seed1024）

| 记录ID | BS训练/测试 | 日志状态 | 同一平均最佳epoch / step | CDFv2 | DFDCP | DFDC | 平均AUC |
|---|---|---|---|---:|---:|---:|---:|
| LEGACY-2032-01 | 16/16 | epoch0–29完整收尾并写出Stop Training；进程退出码及完成归档未核验 | 8 / 16181 | 0.89246618 | 0.87411771 | 0.85351268 | 0.87336552 |

run=`lora_2026-07-29-22-51-41`；FF++c23训练，2026-07-29 22:51:41→2026-08-03 06:52:48。平均最佳video AUC为0.90968079，不能与上表frame AUC混用。服务器203-2来自用户原报告，日志自身确认运行路径；启动代码commit、实际checkpoint文件和归档hash仍未知，不据文件名推定。日志中的保存路径为`/root/autodl-tmp/PRD_TRY/PRD_RUNS/<run>/test/avg/ckpt_best.pth`。

2026-10-04从本地原始日志逐行补录，配置、同一步三域指标、保存与收尾行见[旧日志补录证据](docs/prd/records/LEGACY-LOGS.md)。目标域参与选点，属于开发评估。

## 历史训练的结果口径

- 以下AUC依次为CDFv2、DFDCP、DFDC及三集均值，均来自同一平均最佳checkpoint；不是各数据集各自最佳值拼接。训练集为FF++c23。目标三集参与checkpoint和早停选择，属于开发评估。
- epoch按原代码零基编号；best列为epoch/global step。早停为连续三次逐轮下降，回升即清零；4090旧H/C实现中配置nEpochs=30的循环包含epoch0至30；203-2七月日志实际为epoch0至29，不能混用。
- 历史T1与后续组的测试BS、训练结束预算及模型输入宽度存在差别，不能直接解释为单变量机制因果效应。T1=R1=F3仅是同一基线的三个位置；重试不增加机制实验数量。

## 4090历史运行结果

<a id="legacy-seed1024"></a>
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

<a id="legacy-seed2026"></a>
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

2026-10-03历史审计核对了下列completion内容、结果提交与摘要；SHA256为归档证据记录值，未在该次历史审计对全部4090大权重重新计算。早期H01–H12的独立归档hash该次历史审计未复核；摘要提交已查Git历史，见下表，不把摘要提交等同归档hash验证；H02摘要明确另有`W0/checkpoints/prd_t1_final/ckpt_best.pth`。

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

<a id="legacy-seed3407"></a>
## 203-1运行记录（seed3407）

| 记录ID / 机制 | 配置 | 状态 | best / stop | CDFv2 | DFDCP | DFDC | 平均 |
|---|---|---|---|---:|---:|---:|---:|
| D01 T1 | seed3407，BS16/64 | 完成；TRAINING_EXIT_CODE=0 | epoch3 / 早停epoch11 | 0.83825408 | 0.88201386 | 0.85481217 | 0.85836004 |

- D01 task：`prd_t1_seed3407_bs16_203_retry_20260930_a1`；2026-09-30 16:25:08启动，2026-10-03 10:16:24完成归档状态。
- run：`lora_prd_t1_seed3407_bs16_203_retry_20260930_a1_2026-09-30-16-25-34-344158`；代码提交`62cc7521bf9ff3db794b9ad0fb279f1a5979b3d2`，结果提交`3b281d46a260886ad348e0fdc823f2a420bf029f`（中文结果提交，摘要已核验）。
- 权威状态：`W203/.state/prd_t1_seed3407_bs16_203_retry_20260930_a1/completion.json`。源权重：`L203/training/<run>/test/avg/ckpt_best.pth`。归档：`L203/checkpoint_backups/prd_t1_seed3407_bs16_203_retry_20260930_a1`；包含权重、manifest、摘要、元数据和source_status。
- SHA256：`bfd35a301ae126871053666b106a81a753521af79d5d08a6f29c73b401d5bf6f`，字节数1401384250；该次历史审计源/归档hash一致。

| 同机制的历史尝试 / 排队项 | 状态与证据 |
|---|---|
| 2026-09-27 seed3407预检 | `.state/prd_seed3407_queue_20260927/completion.json`为`blocked_preflight_oom`，正式run为空、队列未安装；记录提交`7200cc0`。不计正式训练结果 |
| 2026-09-27旧BS16队列 | `.state/prd_seed3407_bs16_queue_20260927`记录13:01:08启动；用户于13:10:45主动取消，旧输出保留且不计正式结果；取消证据为同目录`cancelled_by_user.json`，后续从预训练重新开始tmux运行（总对话09-27收件报告） |
| 2026-09-27旧T1 tmux运行 | task=`prd_t1_seed3407_bs16_203_tmux_20260927`，09-27 13:12:40启动，09-30 10:57:54退出1；DataLoader worker Bus error。失败尝试，不能与D01最终结果混合 |
| T2 blur三次重试 | task前缀`prd_t2_blur_seed3407_bs16_203_retry_20260930_`，a1于10-03 10:16:26、a2于10:17:56、a3于10:19:05启动；均在epoch0前`No CUDA GPUs are available`并退出1。无completion、无结果提交，属同一T2的三次失败 |
| T3 identity | 该次历史审计无状态目录及队列启动事件，未启动；不能记为失败或完成 |

09-27总对话已收件的运行身份补充：旧screen T1 run=`lora_prd_t1_seed3407_bs16_203_20260927_2026-09-27-13-01-16-242102`，启动代码`08707b086c13c644023587798bb746ed4cad0154`；重新开始的tmux T1 run=`lora_prd_t1_seed3407_bs16_203_tmux_20260927_2026-09-27-13-12-52-619520`，启动代码`44bbf93f639514bcc1b1a19d3736d9a1ef893f22`。二者与09-30最终成功的D01是不同尝试，不能覆盖或混记成绩。

203与4090同时改变了训练BS、seed、硬件及环境，不能当作只改变随机种子的严格对照。13:22小矩阵CUDA前后向实际通过，仅证明该时刻CUDA可用，不证明正式batch或未来稳定性。13:25准备的`run_prd_203_continue_20261003.sh`仍是未提交文件；14:12复核无新队列状态或训练进程，不能把准备当启动。14:14按用户新方向新增`W203/.state/prd_seed3407_bs16_continue_20261003/cancellation.json`及T2/T3 `continue_20261003_a1`状态目录的同名取消标记（`cancelled_before_start`）。旧脚本`set -e`及任务`mkdir`非-p门槛阻止进入launcher；脚本原文保留、无kill、旧失败证据不变；不是训练失败。脚本SHA256=`3b12b3d196c3c20bbf99ffd7dc38d7e9df2b793f0f5345d941d3275b92b5ed3e`。旧监控已删除，未宣称新监控已安装。

<a id="legacy-seed42"></a>
## 4090后续批次（seed42，含用户中止和未启动项）

| 记录ID / 机制 | BS训练/测试 | 最终状态 | best epoch / step | CDFv2 | DFDCP | DFDC | 平均AUC |
|---|---|---|---|---:|---:|---:|---:|
| A01 F1 orig | 32/64 | 10-04完成epoch30、退出0；缺失归档已恢复，非早停 | 8 / 8081 | 0.90693026 | 0.86149680 | 0.83289162 | 0.86710623 |
| A02 T1 | 32/64 | 10-03用户主动中止；最后完整epoch9，epoch10评估中退出143 | 3 / 3142 | 0.89001793 | 0.88787989 | 0.85626940 | 0.87805574 |

A02数字仅为中止前部分checkpoint观察，不能当作完成实验的最终成绩。两项均由目标三域参与选点，不能与固定预算新实验直接合并排名。

| 记录ID | 工作树 / run | 运行代码 / 结果提交 | 训练日志与权威状态 |
|---|---|---|---|
| A01 | W0 / `lora_prd_f1_orig_seed42_gpu0_20260929_2026-10-01-10-50-24` | `f570bed` / `d738d258cb16e5c7cfb0cead1a75095c63eb3eb7` | `W0/logs/training/csy/<run>/training.log`；`.state/prd_f1_orig_seed42_gpu0_20260929/completion.json` |
| A02 | W1 / `lora_prd_t1_seed42_gpu1_20260929_2026-10-02-14-21-07` | `7810b2eac1861b22f81d9d400f5027b90ef1dfe2` / `4579eb2047b3c477698b143bb89c29a0c06ae2e8` | `W1/logs/training/csy/<run>/training.log`；`.state/prd_t1_seed42_gpu1_20260929/termination.user_requested.json` |

- A01归档：`W0/logs/checkpoint_backups/prd_f1_orig_seed42_gpu0_20260929/ckpt_best.pth`，已有SHA256 `b073f0f084aff30a05c443b9534b9ccb0580a8aa670928b86f98153c9349ec64`；[完整完成与恢复记录](docs/prd/records/A01.md)。
- A02归档：`W1/logs/checkpoint_backups/prd_t1_seed42_gpu1_20260929_user_stopped_20261003/`，已有权重SHA256 `2f23925ae4404098a0229cc82cc125224d42ecf670cec3cf51d8da0c5238c7c0`；[完整用户中止与部分归档](docs/prd/records/A02.md)。旧failed文件保留，但终态解释为用户取消。
- GPU0 F2 seed42、GPU1 T2/T3/R2 seed42均为取消后续队列，未启动，不应报成训练失败，也不因补回历史而恢复。

| 取消前预定任务 | 状态 / 证据 |
|---|---|
| `prd_f2_residual_seed42_gpu0_20260929` | `cancelled_before_start`；W0 `.state/<task>/cancellation.json` |
| `prd_t2_blur_seed42_gpu1_20260929` | `cancelled_before_start`；W1 `.state/<task>/cancellation.json` |
| `prd_t3_identity_seed42_gpu1_20260929` | `cancelled_before_start`；W1 `.state/<task>/cancellation.json` |
| `prd_r2_abs_diff_seed42_gpu1_20260929` | `cancelled_before_start`；W1 `.state/<task>/cancellation.json` |

<a id="legacy-attempts"></a>
## 早期启动、batch尝试与缺失证据

| 记录 / 日志目录 | 已知事实 | 证据边界 |
|---|---|---|
| 4090 08-31 18:00:47、21:05:24、21:09:24、21:14:38启动目录 | 2026-10-03历史审计确认目录/日志存在 | 未取得正常完成、指标和归档证据；不填为成功或确定故障 |
| LEGACY-BS16-01：`lora_bs16_2026-08-31-22-35-16` | seed1024、BS16/16；本次从旧console补得完整epoch0，随后进入epoch1 | 同checkpoint三域AUC 0.86243397 / 0.83797757 / 0.82136700，平均0.84059285；仅中途结果，文件末尾在评估进度条，退出原因未知 |
| 4090 BS8 / BS4：08-31 22:35:52 / 22:36:26 | 历史审计记录的batch探索启动 | 本地未找到相应完整日志；保留待查，不合并成新机制或猜测为OOM |
| T1：09-07 09:12:17、09:13:35 | 无完整epoch；正式运行是H02的09:14:33 | 失败原因、退出码未核验；不将目录数当作完成实验数 |

BS16补录源及H01的BS32 console交叉定位见[旧日志补录证据](docs/prd/records/LEGACY-LOGS.md)。未核验项保留原始服务器定位，不能用聊天估计补成最终指标。

### 历史摘要提交索引

以下提交由2026-10-03历史审计读回；H03及H07–H10是异常记录提交，不是新完成实验。

| 记录 | 摘要提交 | 记录 | 摘要提交 |
|---|---|---|---|
| H01/H02 | `303132f` | H03 | `542a450` |
| H04 | `66f9f85` | H05 | `c670de8` |
| H06 | `9e5ed49` | H07 | `241ca22` |
| H08 | `32c38db` | H09 | `f71e885` |
| H10 | `5f08c39` | H11 | `619554f` |
| H12 | `697c174` | — | — |

历史路径补充：`L203=/root/autodl-tmp/prd-common/outputs/logs`；`W1/logs`链接到`W0/logs/gpu1_20260929`。其他工作树见下方目录表。

## 工作树目录（历史定位，非当前占用）

| 缩写 | 服务器绝对目录 | 分支 |
|---|---|---|
| W0 | `/home/zhaoting.ding/disk/zhaoting.ding/DeepfakeBench` | `prd-research-4090` |
| W1 | `/home/zhaoting.ding/prd-worktrees/gpu1-20260929` | `codex/prd4090-gpu1-20260929` |
| W203 | `/root/DeepfakeBench-prd-common`（SSH `203-1-新`） | `prd-203-1-seed3407` |
| WN01 | `/home/zhaoting.ding/prd-worktrees/gpu0-n01-20261003` | `codex/prd4090-n01-20261003` |
| WN04 | `/home/zhaoting.ding/prd-worktrees/gpu1-n04-20261004` | `codex/prd4090-n04-20261004` |
| WN05-0/1 | `/home/zhaoting.ding/prd-worktrees/gpu{0,1}-n05-20261004` | `codex/prd4090-n05-gpu{0,1}-20261004` |
| WN06-0/1 | `/home/zhaoting.ding/prd-worktrees/gpu{0,1}-n06-20261004` | `codex/prd4090-n06-gpu{0,1}-20261004` |

203负责人对话：`01a0e12a-41bd-7b81-999d-fd62332d03cb`。本表只解释历史记录中的目录缩写；当前任务与核验时间见总表。

## Git同步与并发写入

新认领、恢复上下文或发布状态前，仅同步并读取总表：

```bash
git fetch origin refs/heads/codex/prd-progress:refs/remotes/origin/codex/prd-progress &&
git show refs/remotes/origin/codex/prd-progress:PRD实验总表.md
```

认领/状态发布用独立文档工作树，禁止在运行训练的工作树切分支。复用文档工作树前检查未提交修改并同步远端；只提交本次总表行、索引和本实验记录。push被拒绝后重新同步、检查研究/资源冲突并保留双方更新，不force，不仅机械解决文本冲突后启动。远端读回自己的认领仍有效，且现场资源/锁核验后方可开始。

4090直连失败时可由PC认证Git与SSH bundle中转，核对同一远端SHA；203显式fetch上面的目标ref，避免只更新FETCH_HEAD而读到旧origin引用。共享不可用时保留本地运行证据、标记待发布，不重启在途任务。凭据不入文档。

## 迁移完整性

- 原总表 SHA256：`6343b580f08b61fa32791bccb0090d520b3dd686eb42cbeccb5f49f4691e8477`。
- 原细表 SHA256：`6e622fdf0213d65be7a87d240d0a84e18929b5af5f313476ad90e2d41e4f24b1`。
- 此次仅改协作文档；不改变现有实验、取消标记、队列或监控。文档发布不等于已向其他运行中的对话投递消息。

- [N06-T01：残差训练配对与梯度干预](docs/prd/records/N06-T01.md)：4090两卡，复用N05 direction，持续机制研究授权于2026-10-04更新。

- [N07-T01：原图与联合分支的训练轨迹和源域验证](docs/prd/records/N07-T01.md)：用户解除等待，4090双卡配对学习曲线，非seed/超参扫描。
