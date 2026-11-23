# Schema 实施与运行验证记录

记录截至 2026-11-23。两条真实原链曾在 `c2e635e` 完成独立资格与正式 acceptance：vector 的 v5/Episode13 和 GEMM 的 v2/Episode6。随后 GEMM Episode8 因缺少原幂等 key 在保护门退出；vector 也于 16:40 UTC 因真实 `EDQUOT` 异常退出，finally 中的退出回执写入失败，`owner.json` 的 running 是旧值。已知 Eval84 为 cancelled、cleanup=false，没有新的成功测量；旧失败、unknown 与费用均保留。

native Resources 修复已通过独立四包安装验证，尚未部署到 live。后继 auxiliary 提交保护与本次 fresh preparation 事实修复仅有源码 CPU 证据：新上下文可核验并优先采用本 run 完整成功 V0 的原始 owned Job，旧 snapshot fallback、冻结 context 与显式 Task 审阅权限保持。合并发行的独立安装及两条完整新场景仍待验证；fresh C 尚未实测，连续 72 小时有效稳定未完成，CPU 或安装通过不能代替模型/GPU 证据。

配对起点修复只影响未来新 E；旧已冻结 GEMM a11/v2 的 seed、intent 和费用保持。新 Schema 仍须重新资格，合法 fail/unknown draft 可作实现起点，不能继承资格。

GEMM 原 Episode5 的 Coder 超时后自然异常退出：远端 Eval 已 cancelled，原取消确认失败使 native logical record 和两条 Schema permit 保持 unknown。此次修复新增精确原绑定的公开取消结算：只对原 Job 做一次禁止重定向、无 artifact 下载的 GET；显式清理或身份冲突继续拒绝。合法确认保留全部原 physical attempt、请求、费用和原 envelope，只追加查询证据并结清 logical record 与对应 permits，不产生新测量、资格或晋升。正常返回、startup 和公开 `resume_after_repair(campaign=original)` 共用该路径。

取消结算修复已随 `c2e635e` 通过四包独立安装及 fresh admission，原 GEMM 已经公开恢复到下一 Episode。恢复保留旧 physical unknown、归档和费用，仅结清已核实的 logical record 与 permits；恢复本身没有产生测量或晋升。后续 Episode6 另行完成资格与正式验证。源码、安装与恢复证据分别见 N 的 `final-audit-cancelled-eval-confirmation.md`、`release-cancelled-eval-recovery.md` 和 `resume-gemm-cancelled-eval-live.md`。

## 目标与证据边界

已有同目标 SOTA kernel + Schema 时，持续 optimize(E) → implement(D)；无 SOTA 时，design(C) → implement(D) → optimize(E) → implement(D)。C/E 负责设计，D 忠实实现冻结设计，让 kernel 与 Schema 随证据共同演进。关闭 Schema 保持传统 AKA。正式实验使用核实的 Claude/Qwen、projected/bwrap、LegacyExecutor 与 L20D，不测 extract。

本记录只保留问题、修复和关键证据。详细日志、各次失败原件、原生回执、费用与 hash 均在外置根：

- `R=/home/shaohanyong.shy/aka-schema-l20d-20261122`：早期诊断和两条旧 503 unknown。
- `N=/home/shaohanyong.shy/aka-schema-l20d-a94-validation`：后续发行、两条真实原根及本轮审查/CPU 材料。

压缩仅涉及本文，不删除或改写上述失败原件。CPU fake 不能证明真实模型效果、GPU 性能或长期稳定；不同批次与快照的测试数、调用数和 token 有重叠，不相加作为唯一用例数或供应商账单。部分历史 usage 不完整，未知费用仍保留未知。

## 已提交实现与发行验证

五轮实现覆盖语言/材料、Planner、原生 Episode 控制、恢复与预算、公开装配及独立安装；CPU 原生链跨过 36 Episode / 9 窗。第 5 轮广域 304 项、定点 30 项和独立安装 25 项通过，最后公开 Task 投影修复另在 source/installed 各补一项。此前大组没有在最后局部修正后全部重跑。

`035e932` 已移除 KernelSchema 子模块依赖，迁移语言、规范和所选材料为包内数据，并清理 48 项历史讨论/交接文档。固定来源与授权边界保留在 [SOURCES.json](../src/aka/workflow/schema/assets/SOURCES.json) 和 [NOTICE.md](../src/aka/workflow/schema/NOTICE.md)；未把上游无根许可证声明解释为新的许可。运行与构建无需上游 checkout。

提交后曾发现 `progress.py` 漏包，改为动态收集 Schema 根目录业务模块并增加缺漏回归。后继四包均走实际 sdist → wheel → 独立非 editable 安装，并核对源、包和安装内容。到 `789c40f` 的闭包为 799 文件、541 Python 模块、143 Schema 材料。此前 `e6f4866` 安装版 187 项及独立安装 57 项、`a21a952` 相关证据仍保留，未冒称在 789 全量重跑。

`789c40f` 的实际最终 prefix 验证包含进程 helper 的正常退出、非零退出、信号、超时和子树回收，以及三种公开装配和原根接续检查；两条真实新 Coder 已执行工具，helper 均带 `-B`。每阶段完整 3069 项 prefix 内容/权限/链接清单一致，未靠删除缓存或改 seal 获得通过。详见 N 的 `qualify-and-switch-789.md`。

## 真实运行已经证明什么

| 路径 | 已保留事实 | 当前限制 |
| --- | --- | --- |
| 有 SOTA 的 vector | 原根持续 optimize → D → E → D，v5/Episode13 的 source `83fc180f` 完成独立六义务 PASS；正式 Eval69 完整 3 shapes/4 runs PASS，6.8233 μs 对 6.8781 μs，原生改善约 0.803%，accepted=true。随后健康接续 c2e 并继续 D，后续自然 pivot 不算新晋升。 | 旧负比较、不完整测量与 unknown 保留，不累加历史改善。尚未完成 72 小时观察。 |
| 无 SOTA 的 GEMM | 旧发行完成 design v1 → D → E，真实 E 根据旧资格反馈发布 v2。c2e 恢复后的 v2/Episode6 source `66832ab2` 完成独立六义务 PASS；正式 Eval54 完整 10 shapes PASS，462.5123 μs 对 7752.9556 μs，accepted=true。下一 D 已读取反馈，随后自然 pivot 继续。 | fresh C 未在 c2e 重跑；原 v1 fail、旧 unknown 与费用保留。后续基础设施 inconclusive 和单边 Eval PASS 不算 kernel FAIL 或新晋升，长期有效重复改进仍待观察。 |

2026-11-23 05:59:45 UTC 的保留快照确认两条 789 owner 身份正确、零自动重启：vector 已至 v4 的第三窗口，GEMM 仍在 v1 的首个 D 窗口；这不是实时状态声明。两条 789 接续均先由原 owner 停止并真实结清，再经公开 `driver_factory` / `resume_after_repair` 各启动一次；root/config/Task/prepared、旧调用和预算保持。vector 继承原 Episode 余额，GEMM 只为下一 Episode 分配原合同额度，未重置已用预算。该 789 安装不包含下述 `20dcc9e` 修复，不能把新发行的只读接续验证写成已经向 live Coder 送达或完成新版本纠错。

## 真实问题与已发行处理

| 问题 | 修复与证据 | 保留边界 |
| --- | --- | --- |
| FUSE 发布/严格文件校验、材料身份和业务代码混绑、业务模块漏包。 | 原子 noreplace 发布、材料清单限定到材料/prompts/NOTICE、动态打包及独立安装闭包检查；真实 FUSE 与安装路径验证。 | 初次失败及短暂时间戳漂移原件保留，没有取消硬链接或不可变文件校验。 |
| V0 准入漏计、已准备 Task 投影为空、终态 attempt 反馈漏读、恢复输入漂移。 | 原生绑定族共享准入、合并完整公开 Task、消费原 terminal evidence、冻结请求和幂等结算；补原生/公开入口反例。 | 只对有据旧执行记一次 historical-unadmitted；不造旧准入、不退款，不把 worker 私有依赖放进 Task。 |
| Dev 默认超时超后端上限；物理 unknown 下相同完整请求可能被另一请求再次提交；候选源拒绝原因丢失。 | 默认 Dev 1800 秒、精确原 owner 未提交结算、完整摘要锁与原持久记录检查、有界可修源反馈。 | 保护限于相同 namespace/完整请求；独立 Eval 可继续。旧 unknown 不据缺 JobRef 或控制面 failed 自动结清。 |
| V0 提交 503；SDK 幂等 key 未贯穿原生 launcher。 | 持久保存原 key/body/route/timeout 并贯穿 SDK；仅显式 verified gateway 的同一活调用可有界重放，逐次准入计费。 | R 中两条无原 key 的 503 尝试仍 unknown，列表未找到不等于未提交；新成功运行不结算旧费用。 |
| host 变量误入 worker，网关拒绝并含模型鉴权配置。 | host/worker 在身份冻结前分离，worker_environment 默认空，保留显式工具 env；后继真实 V0 的目标、输入、依赖 pin 与 evaluator 身份已核实。 | 旧拒绝请求、key 与消费保留；不能据接口断言服务未记录原请求。 |
| GEMM 多次 thinking-only 截断；固定 budget 与 effort 同发被 Qwen 拒绝。 | 先落完整草稿再 check；writer-only 可选预算；固定模式明确禁当前模型 effort。真实后继 enabled/8192、max32000、无 effort 被接受并使用工具，最终发布 v1。 | 默认/其他角色不变。旧 thinking-only、HTTP400 和有界重试的原输出及消耗保留，不把离线 wire 检查当 provider 接受。 |
| 资格输出格式无效、进展漏认完整 ABBA、D 将较慢实现直接回退。 | 有界且计量的格式修正；完整 Task/source/原计划进展绑定；明确 D 完成目标是忠实实现，完整 Eval 后可 candidate_ready。 | 合法 fail/unknown 不重试洗判；资格、正确性、性能与晋升分开，缺回执不推断未申请。 |
| a21 helper 未继承 `-B`，封存 prefix 新增缓存。 | 原缺陷在真实 namespace 重现，789 helper 增加 `-B`；最终安装及两 live 工具调用后的完整 prefix 检查无新增缓存。 | 旧 cache/seal 原件保留，没有热补旧安装或删除痕迹。 |

早期诊断曾误输出含签名依赖 URL 的嵌套字段，worker 边界问题也已向用户披露；本轮集成探查再次发生同类 stdout 偏差并已报告。原值不入库，失败日志在外置根保留。后续采用类型检查和最小字段输出；本文不声称所有历史 stdout 或网络检查全程零偏差。

## 本轮已提交闭环修复与证据

| 问题 | 已提交行为 | 证据与状态 |
| --- | --- | --- |
| 原 GEMM 的具体资格 fail 没有可靠进入下一 D/E。 | 原生 archive 中的原判定按 Task/design/实际 sealed source/context 精确投影；D 只取当前 seed，E 明标历史 reviewed source。给出有界原因、原义务、可核 hash/行号和内联引用；旧 row 不改。 | 红/绿 CPU 原生 seal → review → archive → settlement → 新 prompt/E 链通过。真实旧 GEMM 只读视图保留 Hardware/Tile_pipeline 两项 fail 及引用；加入 legacy 标记后的 D/E 长度为 5500/7814 字符，未向 live 注入。 |
| C/E 不知道执行框架和可信架构/工具链，旧 Contract 的事实错误又被整体冻结。 | 新 context 标记 selected framework 和事实来源；可信 TaskConstraintReview/RetainedJobEvidence 经现有 embedding 接入。只有 Architecture、Stack.Language、Stack.Toolchain 可按精确 proof 形成新版本。纯事实纠错不重置 hypothesis、draft、progress、窗口或预算。 | 实际 Planner/native owner/checker/publication 与公开 factory 同根接续 CPU 链通过；模型自授权、错 Task/source/route/policy、越叶更改均拒。旧 v1/fail/cost 保持。详见 [工作流边界](schema-workflow.md#frozen-execution-facts-and-contract-corrections)。 |
| 独审发现旧根首次新 D 的 context 未持久继承，下一 D 会再次丢反馈；真实 nvcc banner 又使 context 构造失败。 | 先用当前 D 实际 binding 筛反馈，再保存并复用冻结 context；受限解析标准编译器版本，未知 banner 保守 unknown。统一反馈选择器和 publication context 原件读取，单一可选原件入口。 | 独立红反例保留，修后 legacy D → 新 D → 下一 D、磁盘恢复和真实 native provider 格式的 CPU 回归通过；重放不重新 discovery。没有新增状态机或历史账本。 |
| vector 历史 Device 为 `AKA L20D`，实际 target 为 `L20D`，阻断合法后继纠错。 | 仅可信 host 明确声明的旧锚点及其逐字同标签谱系可保留该显示值；核完整 Task、target、原 operation/parent/字节，canonical Device 仍为 L20D。new C 不获 alias 许可，Device 不可改。 | 实际 vector v4 沿已审阅 v3 锚点、以及 GEMM v1，各用自己的 retained Job/native owner 只读通过 successor/proof/checker/Contract 门。只在内存形成候选，0 publication；旧 Task/design/fail 不变。 |
| 同事实的新证据 digest 可能使 keep 循环，或把旧判定移到新 context。 | 同事实 keep 保 parent 原 context 给 D/reviewer；事实冲突需有据新版本。旧 source/context 摘要不适用于新 source/context，旧请求恢复只读原字节。 | same-facts keep、新事实 correction、旧摘要错绑定、导入谱系拒绝及 legacy keep → D 的定向 CPU 链通过。 |

最终源码独审通过后，`20dcc9eefda95fcb1bbdec1e2bdcdc5acf434c7a` 的四包均实际走 sdist → wheel → 新持久 prefix 非 editable 安装。独立遍历核对 **802 文件、544 Python 模块、143 Schema 材料**，源、sdist、wheel、安装内容及 commit Git blob 一致；未复制旧环境或回退源码。

最终安装验证使用新 prefix、独立 cwd 和 `-I -B`，各范围分别为：

- 五模块专项 **93 passed**，覆盖执行事实、纠错授权、context 继承/replay、资格反馈、历史 Device 和公开同根接续。
- 独立端到端链及默认/公开 host/CLI Schema off **4 passed**；核新 E 纠错 → D seal/review fail → 下一 E keep/下一 D 的具体反馈和 context，并拒绝同事实不同 digest 的错误 Coder 绑定。
- off / on-None / on-8192 **3 种公开装配**通过，含 strict capture/restore 和原生 Agent/Executor 共享；terminal process 为 synthetic。
- 实际安装版 `proc_view → bwrap` helper 的正常退出、非零退出、SIGINT、SIGTERM、等待后取消 **5 场景**通过；实际 `run_bounded` 的 **1.5 秒超时场景**另行通过，均核输出和清理。

上述各阶段完整 **3072 项** prefix 内容/权限/链接 seal 保持一致，AKA cache/pyc 为 0；没有删除缓存、热改安装或更改基准 seal。三组安装 guard 的违规计数均为 0，这不代表整个历史会话的网络或保密保证。不同范围及此前源码批次不累加，未重跑全仓、旧 36 Episode 或全部历史套件；此前红/绿及失败原件保留。

20dcc 阶段的后继准备曾以最终安装版对两条实际原根完成只读 strict capture/restore、公开装配及 successor/proof/checker/Contract 门，0 publication、0 原件写入。随后 GEMM 的真实新 E 读取旧 fail 反馈、发表三叶事实纠错 v2，新 D 输入也已核对；首批 E/D 后完整 3072 项 prefix seal 一致。当时尚待的新 source 资格和晋升已在上述 c2e 阶段完成；历史交接见 N 的 `validate-gemm-schema-closure-live-handoff.md`。

细节以 N 下五份外置报告为准：`implement-qualification-feedback-delivery.md`、`implement-planner-contract-correction.md`、`integrate-schema-execution-authority.md`、`fix-and-simplify-schema-closure.md`、`fix-legacy-schema-device-label.md`。后两份更新了早期接口和阻塞结论；最终独审见 `final-audit-schema-authority.md`，实际提交/发行及安装版后继准备分别见 `commit-release-schema-closure.md`、`prepare-schema-closure-successors.md`。没有把各阶段增量相加为统一代码或费用总数。

后续先完成合并发行的干净安装，再按 fresh 准入门为两条完整 Task 启动新场景；旧根不恢复，旧资格不移植到新 digest。不能写 state 提前 `close_window`，不能宽松旧资源锁或热补已封存 prefix；正常窗口仍由原生控制器推进。历史两次正式晋升不抹去旧 fail、unknown 和费用，也不代表 72 小时目标达成。
