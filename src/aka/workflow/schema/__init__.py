"""AKA-owned Schema capabilities; importing AKA does not select this workflow."""

__all__ = ("SchemaConfig", "SchemaWorkflow", "RunLimits", "Planner", "PlannerConfig",
           "PlannerRequest", "ExistingStart", "SuppliedStart", "SotaStart", "extract")


def __getattr__(name):
    from importlib import import_module
    modules = {"SchemaConfig": "launch", "SchemaWorkflow": "workflow", "RunLimits": "run_state",
               "Planner": "planner", "PlannerConfig": "planner", "PlannerRequest": "planner",
               "ExistingStart": "workflow", "SuppliedStart": "workflow", "SotaStart": "workflow", "extract": "workflow"}
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module("." + modules[name], __name__), name)
    globals()[name] = value
    return value
