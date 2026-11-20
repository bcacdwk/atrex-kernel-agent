"""Select A2's existing ProblemProvider only in a Schema composition."""
from aka.task.problem.plugin import PROBLEM
from .agent_plugin import Config, Defaults

name = "problem-plugin"
provide = ("problem",)
identity_packages = ("aka.workflow.schema", "aka.task.problem")


def apply(ctx, config):
    from .provider import SchemaSelection
    selection = SchemaSelection.load(config["selection"], expected_sha256=config["selection_sha256"])
    ctx.provide(PROBLEM, selection.problem_provider())
