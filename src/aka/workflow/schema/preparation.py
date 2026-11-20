"""Exact C across native episodes; no model, evaluator, budget or planner loop.

Use one instance as both A1's lifecycle and preparation callback. Creation is
explicit and only allowed before native work; reopening never guesses a seed.
The caller holds the original native workflow lock for every method here.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

from aka.contracts.workspace import RevisionContent
from aka.workflow.light.long_horizon.store import CampaignStore
from aka.workflow.light.orchestration.version_records import version_record_path
from kernelschema.api import RunContext
from kernelschema.artifacts import Publication, fsync_directory, read_regular, source_manifest
from kernelschema.contracts import ProtocolError, UnsupportedError, digest, loads, sha256

from .source import SourceBinding, bind_source, freeze_native_source, read_native_source, selected_seed


NAMESPACE = "aka-source-lifecycle"
FORMAT = "aka.schema.source-lifecycle.v1"
# Native harness/public files are not C. Their bytes are bound separately.
PUBLIC_PYTHON = {"reference.py", "input.py", "valid.py", "test_kernel.py", "profile_driver.py"}
DIAGNOSTICS = {"scratch", "plans", "profiles"}
RUNTIME = {".git", "tools", "skills", "memory", "input"}
SOURCE_SUFFIXES = {".py", ".pyi", ".cu", ".cuh", ".c", ".cc", ".cpp", ".h", ".hpp", ".so"}


def _real_directory(path):
    path = Path(path).absolute()
    if path.resolve(strict=True) != path or not path.is_dir():
        raise ProtocolError("native source directory is missing or traverses a symlink")
    return path


def _context(workspace):
    """Single-file closure guard, not a second Source codec.

    Scratch stays diagnostic. Installed native tools and public context stay
    outside C; ordinary auxiliary source files cannot silently enter it.
    """
    workspace = _real_directory(workspace)
    for parent, directories, names in os.walk(workspace, followlinks=False):
        relative = Path(parent).relative_to(workspace)
        directories[:] = [name for name in directories
                          if not (relative == Path(".") and
                                  (name in RUNTIME | DIAGNOSTICS or name.startswith(".atrex")))]
        for name in (*directories, *names):
            path = Path(parent) / name
            local = path.relative_to(workspace).as_posix()
            if path.suffix in SOURCE_SUFFIXES and local != "kernel.py" and local not in PUBLIC_PYTHON:
                raise ProtocolError("single_file has undeclared auxiliary source: " + local)
    result = {}
    for name in sorted(PUBLIC_PYTHON | {"solution.json"}):
        if (workspace / name).exists() or (workspace / name).is_symlink():
            data, mode = read_regular(workspace, name)
            result[name] = {"sha256": sha256(data), "mode": mode}
    return result


class SeedLifecycle:
    """Persistent source owner, delegating every fee admission to the sole owner.

    ``run_episode`` is A1's existing single-e call, not a new business loop.
    ``install_result`` accepts A3's already authorized native result exactly
    once. It neither seals/evaluates a candidate nor absorbs a usage receipt.
    """

    def __init__(self, run: RunContext, engine, owner):
        self.run, self.engine, self.owner = run, engine, owner
        self.campaign = engine.base_campaign
        self.candidate = self.campaign.candidate_workspace
        self.workspace = Path(engine.workspace).absolute()
        if self.workspace.resolve() != self.workspace:
            raise ProtocolError("native lifecycle workspace cannot traverse symlinks")

    def _binding(self):
        q = self.run.request
        if digest(q) != self.run.request_sha256:
            raise ProtocolError("fixed request changed")
        if q["operation"] != "implement" or q["mode"] != "code":
            raise UnsupportedError("native seed episodes support implement/code only")
        c = self.campaign
        if c.framework_baseline != "never":
            raise UnsupportedError("Schema no-tools preparation requires never; restricted V1 is not qualified")
        if c.module_options.file_system != "single_file":
            raise UnsupportedError("Schema preparation supports single_file only")
        from .task import ADAPTER, evaluation_bundle
        if c.problem.evaluation.adapter != ADAPTER:
            raise UnsupportedError("Schema preparation requires the explicit row-softmax Task adapter")
        _, native_task = evaluation_bundle(c.problem.evaluation)
        if native_task["task"] != q["task"] or native_task["scope"] != q["scope"]:
            raise ProtocolError("native evaluator Task/scope differs from the fixed request")
        if c.framework.lower() != "triton" or c.convert_after:
            raise UnsupportedError("fixed row-softmax Schema requires Triton without native stack conversion")
        seed = selected_seed(self.run.bundle, q["task"], q["seed"])
        task = self.run.bundle.document(q["task"], "task")
        initial = c.problem.initial_source
        actual = {f.path: f.source.read_bytes() if f.source else f.text.encode("utf-8") for f in initial.files}
        if initial.kind != "kernel" or actual != {"kernel.py": self.run.bundle.source(task["baseline_source"])}:
            raise ProtocolError("native V0 InitialSource must equal Task.baseline_source")
        return {"format": FORMAT, "native_workspace": str(self.workspace),
                "request": self.run.request_ref, "task": q["task"], "schema": q["schema"],
                "rules": q["rules"], "scope": q["scope"], "profile": q["backend"],
                "seed": bind_source(self.run.bundle, seed, self.candidate).as_dict(),
                "baseline": bind_source(self.run.bundle, task["baseline_source"], self.candidate).as_dict(),
                "native": {"framework_baseline": c.framework_baseline, "framework": c.framework,
                           "optimization_mode": c.optimization_mode, "file_system": c.module_options.file_system,
                           "platform": c.platform, "arch": c.arch, "convert_after": c.convert_after,
                           "agent_cli": c.agent_cli,
                           "v0_reviewers": list(c._framework_baseline_correctness_reviewers()),
                           "max_version": self.engine.max_version, "max_episodes": self.engine.max_episodes,
                           "token_budget": self.engine.token_budget}}

    @classmethod
    def create(cls, run, engine, owner):
        self = cls(run, engine, owner)
        path = run.root / f".runtime/state/{NAMESPACE}.json"
        if path.exists() or path.is_symlink():
            raise ProtocolError("source lifecycle already exists; reopen it")
        native, active = CampaignStore.checkpoint(self.workspace)
        if (native.episodes or active is not None or self.candidate.current_revision(self.workspace)
                or (self.workspace / ".atrex_long_horizon").exists()):
            raise ProtocolError("cannot invent a seed for existing native work")
        binding = self._binding()
        run.save_state(NAMESPACE, {"binding": binding, "v0": None, "episodes": {},
            "latest": {"episode": 0, "binding": binding["seed"]}, "fence": None})
        return self

    @classmethod
    def reopen(cls, run, engine, owner):
        self = cls(run, engine, owner)
        self._load()
        return self

    def _load(self, *, allow_fenced=False):
        state = self.run.load_state(NAMESPACE)  # Missing/corrupt is never create.
        if (not isinstance(state, dict) or set(state) != {"binding", "v0", "episodes", "latest", "fence"}
                or state["binding"] != self._binding()):
            raise ProtocolError("source lifecycle binding changed or is corrupt")
        if state["fence"] and not allow_fenced:
            raise ProtocolError("source lifecycle fenced: " + state["fence"]["reason"])
        entries = state["episodes"]
        if not isinstance(entries, dict) or set(entries) != {str(i) for i in range(1, len(entries) + 1)}:
            raise ProtocolError("source episode history is missing or corrupt")
        previous = {"episode": 0, "binding": state["binding"]["seed"]}
        for key, entry in sorted(entries.items(), key=lambda item: int(item[0])):
            if (entry.get("phase") not in {"intent", "ready", "admitted"}
                    or entry.get("intent", {}).get("episode") != int(key)
                    or entry.get("seed") != previous["binding"]
                    or int(key) != previous["episode"] + 1):
                raise ProtocolError("source episode lineage is inconsistent")
            if entry.get("terminal") is not None:
                if entry["phase"] != "admitted":
                    raise ProtocolError("terminal source lacks admission")
                previous = {"episode": int(key), "binding": entry["current"]}
        if state["latest"] != previous or entries and state["v0"] is None:
            raise ProtocolError("latest C or settled V0 source record is inconsistent")
        for value in [state["latest"]["binding"], *[e["current"] for e in state["episodes"].values()]]:
            if bind_source(self.run.bundle, value["source"], self.candidate).as_dict() != value:
                raise ProtocolError("retained source binding is corrupt")
        return state

    def _save(self, state):
        self.run.save_state(NAMESPACE, state)

    def _fence(self, state, reason, *, episode=None):
        state["fence"] = {"reason": reason, "episode": episode}
        self._save(state)
        raise ProtocolError(reason)

    def _read(self, workspace, *, expected=None, revision=None):
        if revision is None:
            _context(workspace)
        return read_native_source(workspace, self.candidate, expected=expected, revision=revision)

    def _freeze(self, workspace, *, expected=None, revision=None, publication_id=None):
        self._read(workspace, expected=expected, revision=revision)
        pub = Publication(self.run.root, publication_id)
        binding = freeze_native_source(pub, workspace, self.candidate, expected=expected, revision=revision)
        pub.commit([binding.source])
        return binding

    def _v0(self, state):
        path = version_record_path(self.workspace, 0, prefer_summary=self.campaign.generate_memory)
        raw, _ = read_regular(path.parent, path.name)
        record = json.loads(raw)  # Native records retain their original float codec.
        revision = record.get("git_commit_hash")
        if not revision:
            raise ProtocolError("V0 measurement lacks its original committed source")
        source = self._read(self.workspace, revision=revision,
                            expected=state["binding"]["baseline"]["source"])
        value = {"path": str(path), "sha256": sha256(raw), "revision": revision,
                 "source": state["binding"]["baseline"],
                 "native_identity": self.candidate.source_identity(source)}
        if state["v0"] is not None and state["v0"] != value:
            raise ProtocolError("settled V0 evidence changed or disappeared")
        if state["v0"] is None:
            state["v0"] = value
            self._save(state)

    def admit(self, *, phase, episode, workspace, active):
        state = self._load()
        if Path(workspace).absolute() != self.workspace:
            raise ProtocolError("native lifecycle workspace changed")
        if state["v0"] is not None or phase == "episode":
            self._v0(state)
        entry = state["episodes"].get(str(episode))
        if active is not None:
            if entry is None and active["phase"] != "preparing":
                raise ProtocolError("active draft has no original seed intent")
            if entry is not None:
                self._active(entry, active)
                if any(r["phase"] != "installed" for r in entry["results"].values()):
                    raise ProtocolError("reconcile original native result installation before another invocation")
                if entry["phase"] == "admitted":
                    self._draft(entry)
                elif active["phase"] != "preparing":
                    raise ProtocolError("native episode advanced before seed admission")
        return self.owner.admit(phase=phase, episode=episode, workspace=workspace, active=active)

    def _active(self, entry, active):
        expected = entry["intent"]
        for key in ("episode", "base_commit", "episode_branch", "worktree"):
            if active.get(key) != expected[key]:
                raise ProtocolError("native active episode differs from source intent: " + key)

    def _draft(self, entry):
        draft = _real_directory(entry["draft"])
        state = loads(read_regular(draft.parent, "state.json")[0])
        if not isinstance(state, dict) or not state.get("kernel_digest"):
            raise ProtocolError("admitted EpisodeWorkspace state is missing or corrupt")
        if _context(draft) != entry["draft_context"]:
            raise ProtocolError("admitted draft public source context changed")
        self._read(draft)  # Validate complete bytes/mode without installing them.
        return draft, state

    def __call__(self, *, worktree, active):
        state = self._load()
        key = str(worktree.episode)
        entry = state["episodes"].get(key)
        if entry is not None:
            self._active(entry, active)
            if entry["phase"] == "admitted":
                self._draft(entry)
                return  # An admitted draft is authoritative, even with preparing active.
        else:
            if state["latest"]["episode"] != worktree.episode - 1:
                raise ProtocolError("next episode lacks exact previous terminal C")
            if self.candidate.current_revision(self.workspace) != worktree.base_commit:
                raise ProtocolError("seed ancestry differs from current native HEAD")
            incumbent = self._freeze(self.workspace, revision=worktree.base_commit)
            context = _context(worktree.path)
            entry = {"intent": {k: active[k] for k in
                     ("episode", "base_commit", "episode_branch", "worktree")},
                     "phase": "intent", "seed": state["latest"]["binding"],
                     "current": state["latest"]["binding"], "incumbent": incumbent.as_dict(),
                     "worktree_context": context, "draft": None,
                     "solution_metadata": {"present": "solution.json" in context,
                                           "identity": context.get("solution.json"), "required": False},
                     "draft_context": None, "results": {}, "terminal": None,
                     "terminal_publication_id": None}
            state["episodes"][key] = entry
            self._save(state)  # Intent before touching the native source.
        if _context(worktree.path) != entry["worktree_context"]:
            raise ProtocolError("preparing worktree source context changed")
        self._install(worktree.path, entry["seed"])
        entry["phase"] = "ready"
        self._save(state)

    def _install(self, workspace, binding):
        data = self.run.bundle.source(binding["source"])
        self.candidate.install_source(workspace, RevisionContent({"kernel.py": data}))
        path = Path(workspace) / "kernel.py"
        # Native installation owns paths/bytes. This owner adds crash durability.
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            os.fchmod(descriptor, 0o644)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        fsync_directory(Path(workspace))
        captured = self._read(workspace, expected=binding["source"])
        if self.candidate.source_identity(captured) != binding["native_identity"]:
            raise ProtocolError("installed source failed exact native roundtrip")

    def before_draft(self, *, worktree, active):
        entry = self._load()["episodes"][str(worktree.episode)]
        self._active(entry, active)
        if entry["phase"] == "admitted":
            _, draft_state = self._draft(entry)
            source = self.candidate.read_source(worktree.path)
            if self.candidate.source_identity(source) != draft_state["kernel_digest"]:
                raise ProtocolError("controller source drift would overwrite an admitted draft")
        elif entry["phase"] != "ready":
            raise ProtocolError("draft requires complete seed-ready source")

    def draft_ready(self, *, worktree, active, draft):
        state = self._load()
        entry = state["episodes"][str(worktree.episode)]
        self._active(entry, active)
        draft = _real_directory(draft)
        if entry["phase"] == "admitted":
            if str(draft) != entry["draft"]:
                raise ProtocolError("registered native draft locator changed")
            self._draft(entry)
            return
        if entry["phase"] != "ready":
            raise ProtocolError("native draft registered before seed ready")
        self._read(draft, expected=entry["seed"]["source"])
        entry.update(phase="admitted", draft=str(draft), draft_context=_context(draft))
        self._draft(entry)
        self._save(state)  # Durable admission before first native invocation.

    def current_source(self, episode=None):
        """Last complete stable C, also readable while a damaged draft is fenced."""
        state = self._load(allow_fenced=True)
        value = state["latest"]["binding"] if episode is None else state["episodes"][str(episode)]["current"]
        return SourceBinding(**value)

    def install_result(self, episode, result, *, authority):
        """Accept one A3 result under its original reservation/intent.

        A5 settles usage and handles seals/evaluations/audits. Replaying this
        operation returns its original binding without overwriting later work.
        An interrupted write with unknown draft bytes fences and retains C.
        """
        from .authority import NativeCallAuthority
        from .session import NativeRoleResult
        if (not isinstance(result, NativeRoleResult) or not isinstance(authority, NativeCallAuthority)
                or authority.context != self.run):
            raise ProtocolError("source result requires its original native authority and result")
        state = self._load()
        entry = state["episodes"].get(str(episode))
        if entry is None or entry["phase"] != "admitted":
            raise ProtocolError("result installation requires an active admitted draft")
        operation = authority.reservation["operation_id"]
        intents = [i for i in self.run.intents() if i["operation_id"] == operation]
        if len(intents) != 1:
            raise ProtocolError("result has no original native invocation intent")
        intent = intents[0]
        locator = intent["native_locator"] or {}
        if (intent["episode"] != episode or intent["role"] != "coder"
                or intent["reservation"] != authority.reservation
                or locator.get("session_id") != result.invocation.session_id
                or locator.get("assignment_sha256") != result.assignment_sha256
                or locator.get("profile") != result.profile):
            raise ProtocolError("result differs from its original native invocation")
        native = result.native_source()
        target_sha256 = digest(source_manifest(native.files["kernel.py"]))
        previous = entry["results"].get(operation)
        if previous is not None:
            if previous["source_sha256"] != target_sha256 or previous["intent_sha256"] != digest(intent):
                raise ProtocolError("native operation returned conflicting complete source")
            if previous["phase"] == "installed":
                return SourceBinding(**previous["target"])
        if entry["terminal"] is not None:
            raise ProtocolError("terminal draft cannot accept a new native result")
        if previous is None and intent["source"] != entry["current"]["source"]:
            raise ProtocolError("native result was authorized for another current C")
        draft, _ = self._draft(entry)
        actual = self._read(draft)
        allowed = {entry["current"]["kernel_sha256"]}
        if previous is not None:
            allowed.add(sha256(native.files["kernel.py"]))
        if sha256(actual.files["kernel.py"]) not in allowed:
            self._fence(state, "draft changed during native result installation; retained last complete C",
                        episode=episode)
        if previous is None:
            previous = {"phase": "publishing", "target": None,
                        "source_sha256": target_sha256, "publication_id": uuid4().hex,
                        "intent_sha256": digest(intent)}
            entry["results"][operation] = previous
            self._save(state)
        pub = Publication(self.run.root, previous["publication_id"])
        binding = result.freeze_source(pub, self.candidate).as_dict()
        previous.update(phase="installing", target=binding)
        self._save(state)
        # Use A3's actual RevisionContent at the original installation boundary.
        if native.files != {"kernel.py": self.run.bundle.source(binding["source"])}:
            raise ProtocolError("native result changed during acceptance")
        self._install(draft, binding)
        entry["current"] = binding
        entry["results"][operation]["phase"] = "installed"
        self._save(state)
        return SourceBinding(**binding)

    def settled(self, *, episode, workspace, attempt):
        state = self._load()
        entry = state["episodes"].get(str(episode))
        if entry is None or entry["phase"] != "admitted":
            raise ProtocolError("terminal episode has no admitted source binding")
        terminal = {"attempt_sha256": sha256(json.dumps(attempt, sort_keys=True, allow_nan=False,
                     separators=(",", ":")).encode()), "head": self.candidate.current_revision(self.workspace)}
        if entry["terminal"] is None:
            try:
                draft, _ = self._draft(entry)
                actual = self._read(draft)
                capture = {"publication_id": uuid4().hex,
                           "sha256": digest(source_manifest(actual.files["kernel.py"]))}
                if entry["terminal_publication_id"] is None:
                    entry["terminal_publication_id"] = capture
                    self._save(state)
                else:
                    capture = entry["terminal_publication_id"]
                binding = self._freeze(draft, expected=capture,
                                       publication_id=capture["publication_id"]).as_dict()
            except (OSError, ValueError) as error:
                self._fence(state, "terminal draft source unavailable: " + str(error), episode=episode)
            entry["current"] = binding
            entry["terminal"] = terminal
            state["latest"] = {"episode": episode, "binding": binding}
            self._save(state)
        elif entry["terminal"]["attempt_sha256"] != terminal["attempt_sha256"]:
            raise ProtocolError("native terminal changed after C publication")
        self.owner.settled(episode=episode, workspace=workspace, attempt=attempt)

    def run_episode(self, episode):
        from .bridge import run_episode
        self._load()
        return run_episode(self.engine, episode, lifecycle=self, prepare_episode=self)
