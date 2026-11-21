"""Apply the normal Light plugin with a Schema-aware borrowed Agent."""
from copy import deepcopy

from aka.workflow.light import plugin as light

name = "standard-plugin"
provide, inject = light.provide, light.inject
identity_packages = (*light.identity_packages, "aka.workflow.schema")
Config = deepcopy(light.Config)
Config["properties"].update({
    "schema_context": {"type": "object"},
    "owner_account": {"type": "string", "minLength": 1},
    "owner_limits": {"type": "object", "properties": {
        "model_calls": {"type": "integer", "minimum": 1},
        "evaluator_jobs": {"type": "integer", "minimum": 1}},
        "required": ["model_calls", "evaluator_jobs"], "additionalProperties": False},
    "owner_workspace": {"type": "string", "minLength": 1},
    "owner_run_id": {"type": "string", "minLength": 1},
})
Config["required"] = ["schema_context"]
Defaults = light.Defaults
_OPTIONAL = {"schema_context", "owner_account", "owner_limits", "owner_workspace", "owner_run_id"}


def validate_config(config):
    from .standard import context_prompt
    context_prompt(config["schema_context"])
    light.validate_config({k: v for k, v in config.items() if k not in _OPTIONAL})


def apply(ctx, config):
    from .standard import SchemaAgent
    budget = None
    if config.get("owner_account"):
        from .standard_budget import StandardBudget, BudgetExecutor
        budget = StandardBudget(config["owner_account"], workspace=config["owner_workspace"],
            identity=config["schema_context"]["identity"], limits=config.get("owner_limits"),
            run_id=config.get("owner_run_id"))
        ctx.effect(budget.close, label="release unused slots in original Schema parent account")

    class Context:
        agent = SchemaAgent(ctx.agent, config["schema_context"], budget)
        executor = BudgetExecutor(ctx.executor, budget) if budget is not None else ctx.executor

        def __getattr__(self, name):
            return getattr(ctx, name)

        def provide(self, key, value):
            if getattr(key, "name", key) == "workflow":
                self.workflow = TrackedWorkflow(value, config["schema_context"]["identity"], budget)
                value = self.workflow
            elif key == "startup":
                value._workflow = self.workflow
            return ctx.provide(key, value)

    light.apply(Context(), {k: v for k, v in config.items() if k not in _OPTIONAL})


class TrackedWorkflow:
    def __init__(self, native, identity, budget):
        self.native, self.identity, self.budget = native, identity, budget

    def __getattr__(self, name):
        return getattr(self.native, name)

    def campaign(self, request):
        return self._campaign(request, self.native._load_problem(request))

    def _campaign(self, request, problem):
        campaign = self.native._campaign(request, problem)
        native_run = campaign.run
        def run():
            import json
            from pathlib import Path
            outcome = {"schema_identity": self.identity}
            try:
                reason = native_run()
                outcome.update(status="completed", stop_reason=reason)
                return reason
            except BaseException as error:
                outcome.update(status="failed", stop_reason=type(error).__name__)
                raise
            finally:
                path = Path(request.workspace) / "schema-native-result.json"
                if self.budget is not None:
                    outcome["budget_receipts"] = str(self.budget.directory)
                if path.parent.is_dir():
                    path.write_text(json.dumps(outcome, indent=2) + "\n")
        campaign.run = run
        return campaign
