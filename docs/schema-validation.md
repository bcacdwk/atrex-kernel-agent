# Schema 实施与运行验证记录

当前阶段：五轮实现及清理已提交，真实实验已推进到 `246ba5c`。已有 kernel 的链路完成 optimize、实现与同卡性能对比，首轮因 Schema 审查输出格式无效而未晋升；无 SOTA GEMM 仍在 design。尚未形成完整 D/E 闭环或多日稳定证据。本文件保留简要问题与修复证据，原始日志外置。

## 正式目标

- L20D 已有同源 SOTA kernel + Schema：optimize → implement 持续迭代。
- L20D 无 SOTA：design → implement → optimize → implement 持续迭代。
- 使用 AKA_experiment 核实的 Claude/Qwen 配置，正式实验不测 extract。完成代码检查、清理和 commit 后开始，先验证有效轮次，再观察连续多日运行；未观察足够时长不宣称多日稳定。

## 已发现与处理

| 问题 | 处理与当前证据 |
| --- | --- |
| 旧方案按 Codex/private/Gateway 设计，与实际实验配置不同。 | 核实为 Claude、`qwen3.8-max[1m]`、bwrap/projected、LegacyExecutor；实现沿该路径，配置值及凭据从私有实验配置取得。 |
| GEMM 的现有 Schema 属于 Blackwell，不能配 L20D 晋升 kernel。 | 排除该配对，另核对已有同 GPU 来源；无 SOTA benchmark 本身仍可使用。 |
| 材料清单无差别包含业务模块，业务编辑会作废材料身份。 | 清单收窄到语言、材料、prompts、NOTICE；业务编辑不再改变 release，相关回归通过。 |
| 只按 session_id 去重可能漏计真实新调用。 | 使用持久 operation + attempt 标识；同次缓存复读幂等，同会话标识的新调用分别计量。 |
| 尚无设计符合性检查会让候选永久 unknown。 | 绑定关键义务与封存源码，保留原生检查；必要项复用现有 Agent 做计量的只读审阅。结果是检查/模型判断，不能冒称形式证明。 |
| 短演示额度或同图停滞会误结束长期优化。 | 明确持续模式无默认短总上限；单调用和窗口仍有界，同图停滞交回 Planner。显式用户有限预算仍有效。 |
| 物理作业终态不一定代表原 Experiment 已结算。 | E/finish 前复核原 owner 和 Experiment；未知阻断新增副作用，恢复查原操作，不能重开任务掩盖未决状态。 |
| 共享临时测试环境中的 pytest 在运行间消失。 | 改用本任务专用 CPU 环境，统一由单个代理管理依赖；既有源代码不受影响，相关组合在专用环境复验。 |
| 参考实验的测量并发数来自外壳，原生入口未提供对应配置。 | 补显式可选配置并检查真实Legacy服务的传递；未配置保持原默认，正式实验设置1。 |
| Planner 的 next_actions 未传 Coder，且优化输入缺少实际源码/未晋升测量内容。 | 已将动作落到实现上下文，并提供受限可读源及绑定当前候选的事实摘要；独立回归通过。 |
| lost-return 恢复时新诊断改变了原请求，导致 intent 冲突。 | 已持久冻结每次请求，恢复同操作不重造输入；独立回归通过。 |
| finished 根可能跳过身份检查，active time 恢复可能重复扣账。 | 已补纯读取身份核验与已计耗时增量结算；独立回归通过。 |
| Episode 内重试直到结束才同步 token，显式有限根预算可能多发请求。 | 已将原生已知消耗同步到下一次准入前；7-token反例回归通过，仍不承诺精确控制供应商内部计费。 |
| 原生准备只返回增量，真实已就绪Task的增量为空，Planner会收到空Task。 | 改为合并原生已批准公开文件与authoring增量；真实provider路径源码及最终安装包回归通过，worker私有依赖没有进入公共视图。 |
| 实验外壳专用的评测器环境变量在独立AKA不生效，空Task需求还会覆盖env。 | 外置启动配置改用原生requirements文件，并明确patch problem.worker_requirements；两数据集的normal/ABBA实际参数均核到严格wheel SHA pin。真实远端安装仍待正式实验验证。 |

## 代码级证据

- 第 1 轮：语言/材料 77 项 CPU 检查通过；原生接口与环境/off组合 45 项通过，另旧 Episode/seed/off 53 项通过。
- 第 2 轮：Planner/材料/语言联合 111 项通过；控制器/原生组合 65 项通过，另恢复定点检查通过。CPU 原生链已跨过 36 Episode / 9 窗，编号与账目保留。
- 第 3 轮：阶段、控制器、Planner、来源、输入、材料、语言与原生/off相关大组 270 项通过，另 raw 异文恢复保护通过；公开输入保留来源与规格迁移差异。
- 第 4 轮：最新四包与工作区逐字节一致，143项材料完整；独立安装环境14项入口/功能/off检查通过。当前未出现KS运行、源checkout或网络依赖。
- 第 5 轮：304项广域、30项定点与25项独立安装检查通过；最后公开Task投影修复后补source/installed各1项定点验证，并再次核对四包源码及143项材料。此前组合未在最后局部修正后全部重跑，不能把数量相加宣称更大的完整套件。
- 真实私有配置投影检查：两例公共Task/硬件及公开文档未包含被核查的秘密原值，worker私有快照保留必要的严格依赖pin。未调用模型/GPU。
- 清理了48项历史讨论/交接/过时文档，保留正式指南、必要出处与本记录；提交前仅修尾部空行并重生成清单，语言/材料77项复验通过。
- 不同测试组合存在重叠，不把上述数量相加作为唯一用例总数。CPU 假后端结果不证明真实模型、GPU性能或多日稳定性。

## 提交后诊断

- 外置运行根：`/home/shaohanyong.shy/aka-schema-l20d-20261122`，安装035e932对应四包；Claude/Qwen、两套AKA自有输入、严格评测器wheel pin及共享原生GPU准入已核查。未修改或接管既有实验。
- 2026-11-22 12:26 UTC，vector诊断首次启动在材料快照发布时退出：FUSE硬链接删除别名后仍返回nlink=2，触发不可变文件校验。创建的owner/worker均已退出，零模型、零GPU调用，未生成Task/V0/run-state；原失败前置目录和日志保留。
- 已复现该文件系统行为；产品改为优先使用原子noreplace重命名，保留所有原有校验。相关CPU 130项通过，真实FUSE资源快照、同内容重放、内容冲突、真硬链接拒绝及并发检查通过。SIGKILL后即时恢复仍可能因该挂载的短暂时间戳漂移被严格拒绝，原状态保留，经独立稳定性/内容核验后同请求恢复成功；没有弱化校验。修复提交与续跑结果后补。
- 私有诊断曾因将JSON字符串误当集合而可能输出敏感配置字符；已停止该输出方式并采用类型断言/安全字段白名单，未写入源码仓库。记录不保留原值。
- `51e8549` 显式继承安装后，2026-11-22 12:48 UTC vector新V0在L20D完成：3形状、6正确性case PASS，当前几何均值7.648663838微秒；原job已停止且cleanup清零，实际评测器版本/哈希匹配。随后开始真实Claude/Qwen optimize调用。
- 准入漏点已确认并修复：SupervisorRuntime/SharedMeasurements持有scope安装前绑定的Legacy副本，V0派发未计入根。绑定族现共享可关闭scope attachment；退出后旧引用保持拒绝，实际派发前执行准入。最终新增8项、控制器/native 47项、公共入口/off 5项检查通过。
- 准备反馈现读取原terminal attempt的真实assessment，并绑定原准备源码；原结果的comparability unknown/environment null保持。显式修复可依据唯一原Experiment/physical attempt/Job回执记录一次historical-unadmitted消费更正，不造旧admission，不赠送已用额度；重复幂等，证据不全不猜。
- 本任务已于12:52 UTC受控中断首个optimize，所有本任务进程退出，无新D/GPU；原调用保留unknown/KeyboardInterrupt与267.184544秒，V0原owner终态。后继版本需核原owner证据并明确resume_after_repair，旧快照/已用预算不清零。

后续补充真实轮次、正确性/性能、停止或恢复原因与观察时长；凭据和原始大日志不入库。

### 真实 E/D 与 Dev 边界

- b3508c0 显式恢复后，原中断调用按证据结算、原V0漏计仅更正一次，checkpoint未改/V0未重测。optimize正常发布设计v2，真实Coder收到固定root/dispatch/设计身份与Planner动作。
- vector新候选完成3形状、18 correctness cases及每形状20 performance samples；当前测量7.1733055796微秒，但comparability unknown，不能称正式提升。eval与disassemble原作业均已停止/清理完成；尚未资格晋升或结束Episode。
- Dev请求继承3600秒而后端上限1800，在本地构造提交参数前失败，却保留unknown并阻断后续。修复操作默认限额及结构化not_submitted事实；原owner可凭精确旧记录/停止/调用顺序证据显式结算单attempt，Schema核同原permit，不退款，不把无JobRef等同未提交。新15项、既有native组合38项及控制器等60项检查通过（组合重叠）。
- 无SOTA GEMM的新V0已通过10形状、60正确性case，目标L20D和评测器身份吻合，准入计量正常。其design两次仅thinking耗尽32000输出且无候选；原调用已受控停止并留存成本，准备同Qwen/同任务/同验收、仅Planner effort改high的独立配置试验。旧根不改，新试验调用/派发额度扣除旧根已证实消费。

### 审查协议与有效进展

- `246ba5c` 显式继承安装后，旧本地未提交记录依证据结算，派发和调用消费保留，原 V0 未重测。真实 Dev 使用1800秒上限成功，Eval仍保留3600秒。
- vector 的完整同分配 ABBA 原生 gate PASS：候选7.1325415954微秒、原基线7.5162039397微秒。后续 correctness-only 也通过。跨记录环境身份仍 unknown；这些结果尚不等于正式晋升。
- 首轮生产策略审查接受，但 Schema reviewer 的3条 change_evidence 使用嵌套对象，未满足 source/start/end/claim 协议。原输出保留，规范结果为 unknown，未晋升；控制器自然进入第二 Episode。不能把原始未验证的 pass 转成有效结论。
- 进展消费者仅读聚合结果、忽略 terminal attempt，并误把完整原生 ABBA 当作部分测量，导致漏认真实进展。修复复用原生记录投影及完整 Task/源码/计划绑定；区分首次有效测量、同次 ABBA 和须已知环境的跨记录比较，unknown 不冒充停滞或进展。相关136项通过，补充 Coder 确定性接线后最终82项通过（组合重叠）。原始未晋升源码只读取出及完整 ABBA 请求构造已验证，未新增 GPU 调用；旧 unknown 审查结果保持，运行安装尚未升级。
- GEMM 的 high 独立试验真实请求已核实，V0通过10形状、60 correctness cases；截至15:47 UTC仍未产出设计，已两次 thinking-only 达到32000输出。调低 effort 尚未证明解决问题；仍需有效设计与后续 D/E 实测。
- qualification 现复用原生调用/留存路径，对格式无效的完整输出最多追加配置允许的修正次数；合法 fail/unknown、调用失败及未知 owner 不触发格式重试。reviewer 补了扁平 change_evidence 示例，旧终态不改。相关92项和最终64项 CPU 组合通过（重叠），涵盖计量、预算、丢返回恢复、输入身份及保留文件损坏。
- GEMM high 至15:54又发生第三次 thinking-only 截断，仍无候选。max/high 输入相同，约26.7万字节中85%是规范，截断思考未进入后续请求。design 提示仅补充“简单完整初版尽早落稿→check→按诊断完善”，保留所有义务和验收门；与 reviewer 统一更新 release，真实效果待新版本试验。旧根的封存调用不能悄悄换提示。
- 独立 CPU probe 走完“坏格式→合法 pass→原生资格门→归档→进展消费”，1项通过；保留旧 diagnostics 不会否决修正后的合法结果，两个 attempt 独立准入和计量。此证据不改变真实旧 Episode 的 unknown。
- 旧运行已受控停止、原调用经 owner API 结算，消费未变；原进程全部退出，相关8个历史 GPU jobs 均终态且清理完成。`fd94dd2` 显式安装后，实际 import 检查发现新 progress.py 漏出发行包：构建脚本的 Schema 模块白名单未更新。文件一致性检查只覆盖已打包项，未能发现遗漏；失败发生在新 Task/模型/GPU 启动前。
- 构建改为只自动收集 Schema 根目录的业务 Python 模块，冻结材料清单和分包边界不变；补了实际漏包与未来新增模块两项回归，相关80项 CPU 检查通过。四包实际 sdist→wheel 全量核对798源文件、540 Python 模块无遗漏/重复；独立非 editable 安装在禁用源码读取和 PYTHONPATH 后导入及 continuous 组合通过，全部798安装文件一致，依赖检查通过。共享实测环境后继安装仍需单独核验。

### 持续配置与提交故障

- `f205518` 共享安装核验通过。安装环境的进展/审查/材料组合先通过53项，随后 FUSE 上新写模拟提示的负例触发文件变化保护；失败原样保留，同安装字节在 `/tmp` 定点及剩余37项通过，未改保护逻辑。Dev/准备23项及实际文件、组合、依赖固定、安装继承检查通过。
- 16:20 UTC，两个明确的新 continuous 根各启动一次：vector max/max 使用原配对，GEMM high/max 无 SOTA；保留单调用、窗口与重试边界，总额度无短诊断上限。旧消费和原根保留。两者在新 V0 提交时均收到 HTTP503/tengine HTML，0模型调用、各1派发/1物理尝试，原进程已退出；无 JobRef，远端状态保持 unknown，尚无新 V0 或有效持续轮次。
- 新旧成功/失败请求的任务参数、硬件、超时、依赖、网关设置及执行器版本已核对；仅等内容 requirements 路径及线程/工具标识变化。服务列表暂未找到对应请求，不能据此把 unknown 改成未提交。
- 发现 SDK 在提交副本中生成幂等 key，而原生 inline Eval 没有持久保留。修复仅针对 recorded prepared Eval/run/ABBA：执行前用现有不可变 artifact 保存 key 与原请求、超时、环境绑定，并通过 SDK 既有 raw submit 传递。14项故障和67项既有 CPU 回归通过；旧未知尝试不能补造 key，也没有新增自动重发。发行包验证和实测仍待完成。
- 一次子代理诊断误输出含签名依赖地址的嵌套字段，已停止该输出方式并告知用户；后续只输出明确的叶子字段、哈希及等值结果，原值未入库。
- `64866b3` 独立发行检查发现真实 V0 链还有漏接：Legacy 非 SSH 分支未将已持久化 key 注入 launcher payload。仅扩展既有 prepared_payload 接线并新增完整 V0→launcher→SDK raw submit 回归；源码82项、独立非 editable 安装83项通过，均0 skipped。原失败安装用例也原样通过，798文件在 source/sdist/wheel/installed 间一致，无真实网络调用。共享运行环境仍为 f205518，旧503状态未改。
- 同一活 Legacy 调用的有限恢复已实现：需显式 verified URL 和 idempotent_replay 开关，固定原 key/body/route/AK，最多额外两次且每轮原准入扣账，保留唯一截止时间及失败原件；同原作业终态证据才能结清关联 unknown。默认关闭，不增加跨进程恢复或通用 scoped replay。最终120项源码检查通过（38新+82既有），独立反例与定点审查通过；安装版及真实新调用仍待验证。
- 独审修复了五项初稿缺口：顶层禁止重放策略、原错误字段完整留存、第二次存档失败不丢已知 JobRef、损坏回执禁止重试、慢准入/启动/存档不延长截止时间。实际所选服务0.14.37自身文档支持同请求同 key 重放；普通列表明确隐藏归档作业，因此旧两条无 key 的 unknown 仍需服务关联证据，不能由本修复解锁。
