"""On-only replacement of the original Agent provider, using its native owner."""
from aka.agent.plugin import AGENT

name = "schema-agent"
provide = ("agent",)
identity_packages = ("aka.workflow.schema", "aka.agent")
Config = {"type": "object", "properties": {"selection": {"type": "string", "minLength": 1},
          "selection_sha256": {"type": "string"}}, "required": ["selection", "selection_sha256"],
          "additionalProperties": False}
Defaults = {}


def apply(ctx, config):
    from .provider import SchemaSelection
    from .session import native_agent
    selection = SchemaSelection.load(config["selection"], expected_sha256=config["selection_sha256"])
    agent = native_agent(selection.claude_profile())
    ctx.effect(agent.close, label="close Schema restricted native Agent")
    ctx.provide(AGENT, agent)
