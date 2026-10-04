# 旧训练日志补录（2026-10-04）

仅只读本地已有日志，不访问服务器、不重新评测。完整历史展示在根目录[PRD实验详细记录](../../../PRD实验详细记录.md#legacy-training)。

## LEGACY-2032-01

- 来源文件：`C:/Users/john1/Desktop/fsdownload/root-autodl-tmp-PRD_TRYPRD_RUNS-lora_2026-07-29-22-51-41-training.log`，1265行；用户在“数学原理-PRD深入研究”（对话`019f49f1-dfa2-7541-b362-3b505db386c2`）明确称来自203-2。日志本身提供运行目录但未提供硬件身份证明。
- 启动2026-07-29 22:51:41；epoch0–29完整评测后2026-08-03 06:52:48写出Stop Training和resume路径。仅确认日志正常收尾，不虚构进程退出码、Git commit或归档验收。
- 按`dataset: avg`的frame AUC选得epoch8/global step16181；从同一step读三域0.8924661839330434 / 0.8741177055607674 / 0.8535126773969608，平均0.873365522296924；video平均0.9096807875206019。没有拼接各域不同epoch的最大值。
- 权重保存路径是日志记载值，未重新读取实际文件。目标域参与选点，属于开发评估。

原日志关键行（日志行号，不作当前执行指令）：

```text
L1: 2026-07-29 22:51:41,010 - INFO - Save log to /root/autodl-tmp/PRD_TRY/PRD_RUNS/lora_2026-07-29-22-51-41
L11: train_dataset: ['FaceForensics++']
L12: test_dataset: ['Celeb-DF-v2', 'DFDCP', 'DFDC']
L13: compression: c23
L14: train_batchSize: 16
L15: test_batchSize: 16
L27: nEpochs: 30
L32: manualSeed: 1024
L401: 2026-07-31 03:59:15,794 - INFO - dataset: Celeb-DF-v2    step: 16181    testing-metric, acc: 0.8147381242387333    testing-metric, auc: 0.8924661839330434    testing-metric, eer: 0.19341637010676158    testing-metric, ap: 0.9422862128994576    testing-metric, video_auc: 0.9401850627891607    testing-metric, acc_real:0.8137010676156584; acc_fake:0.8152777777777778
L406: 2026-07-31 04:16:07,126 - INFO - dataset: DFDCP    step: 16181    testing-metric, acc: 0.8042619904772965    testing-metric, auc: 0.8741177055607674    testing-metric, eer: 0.21657346212506354    testing-metric, ap: 0.928151663508452    testing-metric, video_auc: 0.9040284360189574    testing-metric, acc_real:0.8061345534655143; acc_fake:0.8032859288048759
L411: 2026-07-31 06:23:45,309 - INFO - dataset: DFDC    step: 16181    testing-metric, acc: 0.7482363983166308    testing-metric, auc: 0.8535126773969608    testing-metric, eer: 0.22924207697779184    testing-metric, ap: 0.8762558640218814    testing-metric, video_auc: 0.8848288637536875    testing-metric, acc_real:0.7473010353275903; acc_fake:0.749095873698276
L412: 2026-07-31 06:24:03,500 - INFO - Checkpoint saved to /root/autodl-tmp/PRD_TRY/PRD_RUNS/lora_2026-07-29-22-51-41/test/avg/ckpt_best.pth, current ckpt is 8+1797
L414: 2026-07-31 06:24:03,631 - INFO - dataset: avg    step: 16181    testing-metric, acc: 0.7890788376775535    testing-metric, auc: 0.873365522296924    testing-metric, eer: 0.2130773030698723    testing-metric, ap: 0.9155645801432636    testing-metric, video_auc: 0.9096807875206019
L1241: 2026-08-03 06:52:38,845 - INFO - ===> Epoch[29] end with testing auc:
L1253: 2026-08-03 06:52:48,726 - INFO - Full resume checkpoint saved to /root/autodl-tmp/PRD_TRY/PRD_RUNS/lora_2026-07-29-22-51-41/resume/latest.pth after epoch 29
L1254: 2026-08-03 06:52:48,727 - INFO - Stop Training on best Testing metric
```

## LEGACY-BS16-01与H01对照

- 来源：`C:/Users/john1/Desktop/fsdownload/prd_repro_bs16_e566054_console.log`。正文run为`lora_bs16_2026-08-31-22-35-16`，seed1024、BS16/16；两份重复logger输出算同一run。
- epoch0汇总（L22822起）为CDFv2 0.8624339659944642、DFDCP 0.8379775736390424、DFDC 0.8213670026653747，平均0.8405928474329604；随后进入epoch1。文件末尾仍是评估进度，没有完成/退出证据，不能写成完成或确定崩溃。
- `prd_repro_bs32_7c3aab0_console.log`正文run=`lora_2026-08-31-22-24-08`，与H01相同；seed1024、BS32/16。可见epoch19结束并进入epoch20，文件未提供正常收尾。本次只补证据来源，不新增一条重复H01实验。
- 两个文件名中的短hash不是已核验的运行Git字段，不据此覆盖历史表的代码未知项。
