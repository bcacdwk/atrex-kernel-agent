# Schema 验证记录

两条 L20D 实验按用户要求于 **2026-11-24 02:03 UTC 主动停止**，不再继续长跑。
实测源码为 `043316b82959983ea962dc3a680db8c433e7e207`，使用独立非 editable 四包安装及 Claude/Qwen；本次收尾仅整理文档和文件。

| 验证层 | 已核结果与边界 |
| --- | --- |
| 发行与默认模式 | 实际 sdist → wheel → 独立安装；803 个产品文件、545 个 Python 模块、143 份 Schema 材料闭包一致。发行回归、公开 CLI/host 的 ON/OFF、capture/restore、原生资源和实际 bwrap/helper 检查通过。安装验证时完整 prefix 为 2623 项，缺失、额外、变更及 cache 均为 0；本次未重跑代码测试。 |
| 已有 kernel + Schema：vector | 初始 pair 经 optimize/keep 后实现，首个四轮窗口结束后再次 optimize，并把六项新行动交给 coder。D5 产生闭包实现、通过新独立资格及全三种形状正确性；正式 ABBA 比较慢于 champion，未晋升。D6 实际读取正式负值后复测；同源正式验收复用原拒绝结果，并明确判为无新进展。D7 在用户停机时中断。 |
| 无 SOTA：GEMM | fresh design 发布 Schema v1；D1、D3 新独立资格通过，正式全十种形状、四轮 ABBA/f3 验收通过并原生晋升。D2 的资格失败反馈被 D3 消费并修正。D4 耗尽单轮预算后自动收口；首轮 optimize 发布执行方案确有变化的 Schema v2，D5 从工作草稿接续，未继承旧 Schema 资格。用户停机时，v2 的新实现尚未完成独立资格及正式验收。 |
| 停机 | 两个 owner 各收到一次身份核验后的 SIGTERM，15 个原 owner/后代全部退出，未使用 KILL；真实退出回执为 exit_code=2、trace interrupted。最后在途 GEMM Job 经 GET 确认 cancelled、cleanup_pending=false；原生 unknown/SystemExit 等记录未改写。 |

两条新实验约于 2026-11-23 20:24 UTC 启动，运行约 **5 小时 39 分钟**。
已观察到候选拒绝、单轮预算结束后的自动接续，以及新 Schema 发布后的 coder 交接；**未完成原计划的 72 小时稳定性验收**。
Schema v2 的资格和正式验收未完成，不能把启动实现阶段视为实现成功。
开发测量中的正值未必在正式比较复现，本次不据此宣称稳定提速。

`candidate_ready`、独立 Schema 资格、完整 native Eval 与 champion 晋升分别取证。
资格是模型对忠实实现的判断，不是形式证明；历史失败、unknown、未完成测量及未知消费均按原状态保留。
旧根的资格或晋升只属于其原 source/root/发行，未移植为新根证据；测试数和 native token 数不累计为供应商账单。

停机证据、完整历史归档及恢复索引统一保存在本机 `~/archives/aka-cleanup-20261124/`：

- `stop/STOP.md`、`stop/final-stop.safe.json`：进程身份、信号、退出回执及最后 Job 状态。
- `history/`、`tmp-history/`：原实验目录和已确认临时目录的压缩档、成员清单及校验记录。
- `finalization/`：保留的发行、调研和最终整理索引；完整发行报告与运行回执仍可从历史归档按原路径查找。

实验原目录在完整归档校验后移除；这不改变其原失败、不确定性或计费边界。
使用方式与恢复约束见 [Schema workflow](schema-workflow.md)。
KernelSchema 的运行/构建依赖已移除；来源与许可边界见 [SOURCES.json](../src/aka/workflow/schema/assets/SOURCES.json)、[Schema NOTICE](../src/aka/workflow/schema/NOTICE.md) 和 [L20D NOTICE](../examples/schema/l20d/NOTICE.md)。
