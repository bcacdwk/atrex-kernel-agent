"""Row-softmax Task mapped to AKA's existing explicit Problem roles.

No legacy layout marker detection, parser copy, public evaluator or switch.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile

from aka.contracts.content import PublicInputContent, WorkspaceFile
from aka.contracts.problem import AgentProblem, EvaluationInput, InitialSource, LoadedProblem
from aka.task.problem import snapshot
from kernelschema.artifacts import Bundle
from kernelschema.contracts import ProtocolError, jcs, loads, sha256
from kernelschema.evaluation import RowSoftmaxTask, stage

ADAPTER = "kernelschema-row-softmax-v1"
BINDING = "schema-task.json"


def evaluation_bundle(evaluation: EvaluationInput):
    if evaluation.adapter != ADAPTER:
        raise ProtocolError("Schema evaluator requires its explicit Task adapter")
    selected = {item.path: item for item in evaluation.files}
    if len(selected) != len(evaluation.files) or BINDING not in selected:
        raise ProtocolError("missing or duplicate Schema Task content")
    item = selected[BINDING]
    binding = loads(item.source.read_bytes() if item.source else item.text.encode())
    bundle = Bundle(evaluation.directory / "bundle")
    task = RowSoftmaxTask(bundle, binding["task"])
    task.require_evaluation_scope(binding["scope"])
    bundle.closure([binding["task"], binding["scope"]])
    return bundle, binding


@dataclass
class SchemaProblemContent:
    problem: LoadedProblem
    workspace: Path
    generated_digest: str = ""

    @property
    def evaluation(self):
        return self.problem.evaluation

    @property
    def private_reference_dir(self):
        return None  # legacy private-shapes sentinel; EvaluationInput owns privacy.

    def operator_files(self):
        return self.problem.agent.files

    def prepare_public(self):
        return PublicInputContent(True, self.operator_files())

    def authoring_assets(self):
        return ()

    def validate_generated(self, generated):
        raise ProtocolError("Schema Task contract cannot be generated or rewritten")

    generated_file = validate_generated

    def record_generated(self):
        raise ProtocolError("Schema Task has no generated contract")

    def assert_private(self):
        for item in self.operator_files():
            path = self.workspace / item.path
            if path.is_symlink() or not path.is_file() or path.read_bytes() != item.source.read_bytes():
                raise ProtocolError("public Task material changed: " + item.path)
        # Host must additionally apply the role filesystem allowlist. This check
        # catches accidental bootstrap copies, not arbitrary model-side attacks.
        for name in ("bundle", BINDING, "evaluator", "inputs.json", "frozen_inputs.json"):
            if (self.workspace / name).exists():
                raise ProtocolError("private evaluator leaked into Agent workspace: " + name)


@dataclass(frozen=True)
class SchemaProblemProvider:
    task: dict
    scope: dict
    materials: dict | None = None
    role: str = "coder"

    def load(self, directory, *, optimization_mode=None, snapshot_directory=None,
             expose_evaluator_shapes=True):
        if snapshot_directory is None:
            raise ProtocolError("Schema Task requires an explicit private snapshot_directory")
        origin = Path(directory).absolute()
        configuration = dict(task=self.task, scope=self.scope, materials=self.materials, role=self.role)
        mode = "schema:" + (optimization_mode or "leaderboard")
        if Path(snapshot_directory).exists():
            result = snapshot.restore(Path(snapshot_directory), origin, mode, source_configuration=configuration)
            evaluation_bundle(result.evaluation)
            return result
        bundle = Bundle(origin)
        adapter = RowSoftmaxTask(bundle, self.task)
        adapter.require_evaluation_scope(self.scope)
        task = adapter.task
        roots = [self.task, self.scope] + ([self.materials] if self.materials else [])
        closure = bundle.closure(roots)
        public = {"contract.yaml": task["contract"], "acceptance.md": task["acceptance"]}
        private = bundle.closure([task["evaluator"]])
        forbidden_hashes = {sha256(value.data) for value in private.values()}
        if self.materials:
            for entry in bundle.document(self.materials, "materials")["entries"]:
                if self.role in entry["roles"]:
                    reference = entry["file"]
                    if reference["path"] in private or reference["sha256"] in forbidden_hashes:
                        raise ProtocolError("Materials cannot expose private evaluator bytes")
                    # IDs are not filesystem names; ordinal names avoid traversal.
                    public[f"materials/{len(public) - 2}.txt"] = reference
        with tempfile.TemporaryDirectory(prefix="aka-schema-task-") as raw:
            temp = Path(raw)
            # Verify complete installed evaluator and baseline bytes before V0
            # materialization; contract identity alone is insufficient.
            stage(bundle, self.task, task["baseline_source"], temp / "validation", scope=self.scope)
            for path, content in closure.items():
                dest = temp / "evaluation/bundle" / path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(content.data)
                dest.chmod(content.mode)
            binding_path = temp / "evaluation" / BINDING
            binding_path.write_bytes(jcs(dict(task=self.task, scope=self.scope)))
            public_files = []
            for name, reference in public.items():
                dest = temp / "public" / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(bundle.read(reference))
                public_files.append(WorkspaceFile(name, source=dest))
            baseline = temp / "initial/kernel.py"
            baseline.parent.mkdir()
            baseline.write_bytes(bundle.source(task["baseline_source"]))
            baseline.chmod(0o644)
            evaluation_dir = temp / "evaluation"
            evaluation = EvaluationInput(ADAPTER, evaluation_dir,
                tuple(WorkspaceFile(path.relative_to(evaluation_dir).as_posix(), source=path)
                      for path in sorted(evaluation_dir.rglob("*")) if path.is_file()),
                worker_requirements=WorkspaceFile("worker-requirements.txt", text=""))
            contract = temp / "public/contract.yaml"
            problem = LoadedProblem(task["task_id"], evaluation_dir, contract, None, contract, "provided",
                AgentProblem(tuple(public_files)),
                InitialSource("kernel", baseline, (WorkspaceFile("kernel.py", source=baseline),)), evaluation)
            result = snapshot.capture(problem, Path(snapshot_directory), origin, mode, source_configuration=configuration)
        evaluation_bundle(result.evaluation)
        return result

    def content(self, problem, workspace, *, generated_digest=""):
        evaluation_bundle(problem.evaluation)
        return SchemaProblemContent(problem, Path(workspace), generated_digest)

    def runtime_files(self, problem):
        evaluation_bundle(problem.evaluation)
        return ()  # No evaluator Python enters any model role through bootstrap.

    def snapshot(self, problem):
        return snapshot.inspect(problem.snapshot_directory)[1]

    def export_snapshot(self, problem, destination):
        return snapshot.copy_snapshot(self.snapshot(problem), destination)[1]

    def validate_snapshot(self, value, *, destination=None):
        result = snapshot.inspect(value.directory, identity=value.identity)[0]
        evaluation_bundle(result.evaluation)
        if destination is not None and Path(destination).exists():
            snapshot.inspect(Path(destination), identity=value.identity)
        return result

    def restore_snapshot(self, value, destination):
        result = snapshot.copy_snapshot(value, destination)[0]
        evaluation_bundle(result.evaluation)
        return result
