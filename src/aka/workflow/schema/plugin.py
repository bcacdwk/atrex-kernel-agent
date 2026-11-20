"""Explicit optional Schema composition. Importing this module enables nothing."""
from aka.workflow.legacy.plugin import WORKFLOW

name = "schema"
provide = ("workflow",)
inject = ("agent", "source", "candidate", "executor", "execution_context", "evaluator_factory",
          "evaluation_defaults", "experiment_factory", "measurement_parallelism", "workflow_observer",
          "experiment_episode_evidence", "evaluation_episode_interpreter")
identity_packages = ("aka.workflow.schema", "aka.workflow.light", "aka.workflow.legacy", "aka.contracts")
Config = {"type": "object", "properties": {
    "selection": {"type": "string", "minLength": 1}, "selection_sha256": {"type": "string"},
    "startup": {"type": "boolean"}}, "required": ["selection", "selection_sha256"], "additionalProperties": False}
Defaults = {"startup": True}


def apply(ctx, config):
    from .provider import NativeModules, SchemaSelection, create_workflow
    selected = SchemaSelection.load(config["selection"], expected_sha256=config["selection_sha256"])
    modules = NativeModules(**{name: getattr(ctx, name) for name in inject})
    workflow = create_workflow(selected, modules)
    ctx.effect(workflow.close, label="close Schema workflow and its native runtimes")
    ctx.provide(WORKFLOW, workflow)
    if config["startup"]:
        from aka.workflow.legacy.startup import WorkflowStartup
        ctx.provide("startup", WorkflowStartup(workflow, observer=ctx.workflow_observer,
                    evaluation_defaults=ctx.evaluation_defaults,
                    execution_environment=ctx.execution_context.environment))
