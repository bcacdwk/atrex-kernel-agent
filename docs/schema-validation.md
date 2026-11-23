# Schema 实施与运行验证记录

记录截至 2026-11-23。五轮实现、清理和后续已发行修复均已提交；vector 与 GEMM 两个原根已各一次同根接续到 `789c40f`，保留快照确认实际 owner 在运行。本轮资格反馈、执行事实纠错及历史 Device 标签兼容已通过最终源码独审，仍为**未提交候选，待 commit、发行及实测**，尚未用于 live。有效重复进展及连续 72 小时观察仍未完成。

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
| 有 SOTA 的 vector | 原根发布 v2；首个 D 完成资格与正式 acceptance，候选 7.2518 μs、基线 7.5445 μs，原生聚合改善约 4.04%。同 v2 四 Episode 窗口结束后自动进入 E，发布精确 parent 的 v3，再进入 D；后续保留快照已至 v4。 | 首个 optimize → D 窗口 → E → 新版本 → D 已真实走通。后续资格 pass 而 accepted=false 的候选保留为 draft，champion 不变；探索 ABBA 和跨记录 unknown 不能冒充正式提升。尚未完成 72 小时观察。 |
| 无 SOTA 的 GEMM | 经多次 writer 失败后发布 design v1，原控制器进入多个 D。首个新源全 Task Eval 为 10 shapes、完整/PASS、193.3558 μs。Episode2 实际提交 candidate_ready，但独立资格为 fail，绑定其精确 sealed source，未晋升；789 后继继续 D。 | 尚未证明无 SOTA 的完整 design → D → E → D 有效重复链。原 v1 的 Hardware/Tile_pipeline fail 与费用保持；Eval PASS、正常退出和 candidate_ready 均不等于 Schema 符合性。 |

2026-11-23 05:59:45 UTC 的保留快照确认两条 789 owner 身份正确、零自动重启：vector 已至 v4 的第三窗口，GEMM 仍在 v1 的首个 D 窗口；这不是实时状态声明。两条 789 接续均先由原 owner 停止并真实结清，再经公开 `driver_factory` / `resume_after_repair` 各启动一次；root/config/Task/prepared、旧调用和预算保持。vector 继承原 Episode 余额，GEMM 只为下一 Episode 分配原合同额度，未重置已用预算。运行中的 789 不包含下述未提交修复，不能把只读原件验证写成已经向 live Coder 送达或完成新版本纠错。

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

## 本轮闭环候选：修复与证据

| 问题 | 最终候选行为 | 证据与状态 |
| --- | --- | --- |
| 原 GEMM 的具体资格 fail 没有可靠进入下一 D/E。 | 原生 archive 中的原判定按 Task/design/实际 sealed source/context 精确投影；D 只取当前 seed，E 明标历史 reviewed source。给出有界原因、原义务、可核 hash/行号和内联引用；旧 row 不改。 | 红/绿 CPU 原生 seal → review → archive → settlement → 新 prompt/E 链通过。真实旧 GEMM 只读视图保留 Hardware/Tile_pipeline 两项 fail 及引用；加入 legacy 标记后的 D/E 长度为 5500/7814 字符，未向 live 注入。 |
| C/E 不知道执行框架和可信架构/工具链，旧 Contract 的事实错误又被整体冻结。 | 新 context 标记 selected framework 和事实来源；可信 TaskConstraintReview/RetainedJobEvidence 经现有 embedding 接入。只有 Architecture、Stack.Language、Stack.Toolchain 可按精确 proof 形成新版本。纯事实纠错不重置 hypothesis、draft、progress、窗口或预算。 | 实际 Planner/native owner/checker/publication 与公开 factory 同根接续 CPU 链通过；模型自授权、错 Task/source/route/policy、越叶更改均拒。旧 v1/fail/cost 保持。详见 [工作流边界](schema-workflow.md#frozen-execution-facts-and-contract-corrections)。 |
| 独审发现旧根首次新 D 的 context 未持久继承，下一 D 会再次丢反馈；真实 nvcc banner 又使 context 构造失败。 | 先用当前 D 实际 binding 筛反馈，再保存并复用冻结 context；受限解析标准编译器版本，未知 banner 保守 unknown。统一反馈选择器和 publication context 原件读取，单一可选原件入口。 | 独立红反例保留，修后 legacy D → 新 D → 下一 D、磁盘恢复和真实 native provider 格式的 CPU 回归通过；重放不重新 discovery。没有新增状态机或历史账本。 |
| vector 历史 Device 为 `AKA L20D`，实际 target 为 `L20D`，阻断合法后继纠错。 | 仅可信 host 明确声明的旧锚点及其逐字同标签谱系可保留该显示值；核完整 Task、target、原 operation/parent/字节，canonical Device 仍为 L20D。new C 不获 alias 许可，Device 不可改。 | 实际 vector v4 沿已审阅 v3 锚点、以及 GEMM v1，各用自己的 retained Job/native owner 只读通过 successor/proof/checker/Contract 门。只在内存形成候选，0 publication；旧 Task/design/fail 不变。 |
| 同事实的新证据 digest 可能使 keep 循环，或把旧判定移到新 context。 | 同事实 keep 保 parent 原 context 给 D/reviewer；事实冲突需有据新版本。旧 source/context 摘要不适用于新 source/context，旧请求恢复只读原字节。 | same-facts keep、新事实 correction、旧摘要错绑定、导入谱系拒绝及 legacy keep → D 的定向 CPU 链通过。 |

最终独审在稳定的最终源码上重跑五模块，**93 passed**；另新写独立端到端链与 Schema off 检查，**2 passed**。后者直接核新 E 纠错 → D seal/review fail → 下一 E keep/下一 D 的具体反馈和 context，并拒绝同事实不同 digest 的错误 Coder 绑定。两条实际原根只读门再次通过，0 publication、0原件写入。不同范围和前作者重叠批次不累加；此前红/绿及失败原件保留，没有全仓、36 Episode 或新安装版套件全量重跑。CPU fake model/measurement 和本地服务的两批 guard 均记录外部 connect 为 0，不是整个历史会话的网络或保密保证。源码 AST、diff 格式和冻结材料清单检查通过，新模块在动态发行输入内；尚未 build/install 新四包，也未验证新发行的 live capture/restore。

细节以 N 下五份外置报告为准：`implement-qualification-feedback-delivery.md`、`implement-planner-contract-correction.md`、`integrate-schema-execution-authority.md`、`fix-and-simplify-schema-closure.md`、`fix-legacy-schema-device-label.md`。后两份更新了早期接口和阻塞结论；最终独审及重跑证据见 `final-audit-schema-authority.md`。没有把各阶段增量相加为统一代码或费用总数。

最终源码独审在指定源码、CPU 和只读原件范围内未发现阻挡合并的剩余实质问题；本轮仍待 commit、发行验收及实测。真实后继须在原 owner 正常结清及授权的新 operation 边界接入可信审阅/原件，由新 E 纠正事实并处理原 pipeline 设计问题，再由 D 产生新 sealed source，完成全 Task 测试、独立资格和原生测量。三叶事实纠错不豁免原 Tile_pipeline fail，CPU 或只读门通过不代表线上修复、晋升或 72 小时目标达成。
