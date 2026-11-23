"""Select the AKA Schema controller over the unchanged native Light owners."""
from aka.workflow.light import plugin as light
from aka.workflow.legacy.plugin import WORKFLOW

name = "schema"
provide = ("workflow",)
inject = light.inject
identity_packages = (*light.identity_packages, "aka.workflow.schema")
Config = {"type": "object", "properties": {
    "schema": {"type": "object"}, "native": {"type": "object"}, "startup": {"type": "boolean"}},
    "required": ["schema", "native"], "additionalProperties": False}
Defaults = {"startup": True}


def validate_config(config):
    from .launch import SchemaConfig
    SchemaConfig.load(config["schema"])
    light.validate_config(config["native"])


def apply(ctx, config):
    from pathlib import Path
    import hashlib
    from aka.core.filesystem import read_regular_bytes
    from aka.resources.plugins import PluginRegistry
    from aka.workflow.legacy.agent_host import process_policy
    from .launch import SchemaConfig
    from .language import CheckPolicy, check_design, collect_design
    from .materials import select_materials
    from .planner import Planner, PlannerConfig
    from .run_state import RunLimits
    from .workflow import SchemaWorkflow, SotaStart, SuppliedStart

    selected = SchemaConfig.load(config["schema"])
    materials = PluginRegistry(plugin_dir="", resources=select_materials(wiki_ids=selected.wiki_ids),
                               agent_skills=(), plugins=())
    # Planner inputs and native Coder capabilities have different owners.
    # Borrow the same public Resources selection as ordinary Light; never put
    # executable tools/Skills into the Planner's retained materials snapshot.
    resources = PluginRegistry()
    planner_options = dict(selected.planner)
    policy = CheckPolicy(**planner_options.pop("check_policy", {}))
    initial = None
    if selected.initial is not None:
        supplied = selected.initial
        path = Path(supplied["schema"])
        files = collect_design(path.parent if path.is_file() and path.name == "root.yaml" else path)
        check_design(files, policy=policy)
        evidence = {f"{index:03d}-{Path(path).name}": read_regular_bytes(Path(path), limit=8 * 1024 * 1024)
                    for index, path in enumerate(supplied["evidence"])}
        initial = SuppliedStart(files, dict(supplied["source_files"]), supplied["hardware"], evidence, policy)
    sota = None
    if selected.sota is not None:
        from .extract_sources import SourceDescriptor
        sota = SotaStart(SourceDescriptor.from_dict(selected.sota["descriptor"]),
            selected.sota["target_gpu"], selected.sota["target_hardware"])

    def driver_factory(request):
        from aka.resources.wiki.providers import selected_provider
        # Boot may apply contributing modules after Workflow. Resolve the full
        # native selection at the same request boundary as ordinary Light.
        if selected_provider() is not None:
            raise ValueError("Driven Light requires explicit resources without a dynamic Wiki provider")
        workflow.resources = PluginRegistry()
        settings = request.settings
        limits = dict(selected.limits)
        total_fields = ("max_windows", "max_episodes", "max_agent_calls", "max_dispatches", "max_agent_active_seconds", "max_tokens")
        if not selected.continuous and not any(limits.get(key) is not None for key in total_fields):
            if settings.max_iters <= 0:
                raise ValueError("Finite Schema requires a positive native max_iters or explicit total allowance")
            limits["max_episodes"] = settings.max_iters
        limits = RunLimits(continuous=selected.continuous, **limits)
        if settings.setup_timeout > limits.per_call_active_seconds:
            raise ValueError("Native setup_timeout exceeds the Schema single-call allowance")
        planner_config = PlannerConfig(agent_cli=settings.agent_cli, check_policy=policy,
            **{"timeout_seconds": limits.per_call_active_seconds, **planner_options})
        if planner_config.timeout_seconds > limits.per_call_active_seconds:
            raise ValueError("Planner timeout exceeds the Schema single-call allowance")
        identity = hashlib.sha256(str(request.workspace.absolute()).encode()).hexdigest()
        retained = request.workspace.parent / ".aka-schema-inputs" / identity
        snapshot = materials.export_snapshot(retained / "materials")
        planner = Planner(ctx.agent, snapshot, config=planner_config)
        return SchemaWorkflow(request.workspace / ".atrex_long_horizon" / "schema",
            planner=planner, executor=ctx.executor, workspace=request.workspace,
            limits=limits, initial=initial, sota=sota)

    ctx.contribute("agent_process_policies", "workflow", process_policy(ctx.execution_context.environment))
    workflow = light.create_workflow(ctx, config["native"], driver_factory=driver_factory, resources=resources)
    ctx.effect(workflow.close, label="close Schema native Workflow and private runtimes")
    ctx.provide(WORKFLOW, workflow)
    if config["startup"]:
        from aka.workflow.legacy.startup import WorkflowStartup
        ctx.provide("startup", WorkflowStartup(workflow, configuration=config["native"], observer=ctx.workflow_observer,
            evaluation_defaults=ctx.evaluation_defaults, execution_environment=ctx.execution_context.environment))
