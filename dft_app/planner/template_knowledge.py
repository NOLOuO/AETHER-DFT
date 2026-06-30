from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from rapidfuzz import fuzz

from dft_app.remote.config import RemoteClusterConfig
from dft_app.remote.ssh_remote_runner import SSHRemoteRunner


_KEY_INCAR_TAGS = [
    "ENCUT",
    "ISPIN",
    "MAGMOM",
    "ISMEAR",
    "SIGMA",
    "EDIFF",
    "IBRION",
    "POTIM",
    "NSW",
    "ISIF",
    "EDIFFG",
    "ICHAIN",
    "IOPT",
    "LCLIMB",
    "IMAGES",
    "SPRING",
    "IVDW",
]


@dataclass
class TemplateRecord:
    relative_path: str
    name: str
    task_family: str
    elements: list[str] = field(default_factory=list)
    element_counts: list[int] = field(default_factory=list)
    kpoints_mesh: str | None = None
    incar_tags: dict[str, str] = field(default_factory=dict)
    signals: list[str] = field(default_factory=list)
    child_directories: list[str] = field(default_factory=list)
    source: str = "unknown"
    score: float = 0.0

    def to_prompt_dict(self) -> dict[str, Any]:
        payload = {
            "relative_path": self.relative_path,
            "task_family": self.task_family,
            "elements": self.elements,
            "element_counts": self.element_counts,
            "kpoints_mesh": self.kpoints_mesh,
            "incar_tags": self.incar_tags,
            "signals": self.signals,
        }
        if self.child_directories:
            payload["child_directories"] = self.child_directories
        return payload


class PlannerTemplateKnowledge:
    """Retrieve local task-card style template summaries for planner prompting."""

    def __init__(
        self,
        project_root: Path | None = None,
        remote_config: RemoteClusterConfig | None = None,
    ):
        self.project_root = project_root or Path(__file__).resolve().parents[2]
        self.remote_config = remote_config
        cache_path = os.getenv("SEMI_DFT_PLANNER_TEMPLATE_CACHE_PATH")
        self.cache_path = (
            Path(cache_path)
            if cache_path
            else self.project_root / "cache" / "planner_template_cache.json"
        )
        self.remote_template_dir = os.getenv("SEMI_DFT_PLANNER_TEMPLATE_REMOTE_DIR", "")
        self.cache_max_age_seconds = int(
            os.getenv("SEMI_DFT_PLANNER_TEMPLATE_CACHE_MAX_AGE", "43200")
        )

    def build_context(
        self,
        *,
        prompt: str,
        material_name: str | None = None,
        max_templates: int = 3,
    ) -> dict[str, Any]:
        index_payload, source, message = self._load_or_refresh_index()
        directories = index_payload.get("directories") or []
        details = index_payload.get("details") or {}
        ranked_directories = sorted(
            directories,
            key=lambda item: self._score_directory(
                item,
                prompt=prompt,
                material_name=material_name,
            ),
            reverse=True,
        )

        selected_records: list[TemplateRecord] = []
        dirty = False
        for entry in ranked_directories[: max_templates * 2]:
            relative_path = str(entry.get("relative_path") or "").strip()
            if not relative_path:
                continue

            detail_payload = details.get(relative_path)
            needs_refresh = detail_payload is None or self._needs_detail_refresh(detail_payload)
            if needs_refresh and source != "unavailable":
                detail_payload = self._fetch_detail(relative_path)
                if detail_payload is not None:
                    details[relative_path] = detail_payload
                    dirty = True

            if detail_payload is None:
                detail_payload = {
                    "relative_path": relative_path,
                    "name": str(entry.get("name") or Path(relative_path).name),
                    "task_family": self._classify_template(
                        relative_path=relative_path,
                        incar_tags={},
                        elements=[],
                    ),
                    "signals": self._signals_from_path(relative_path),
                }

            record = self._to_record(detail_payload)
            record.source = source
            record.score = self._score_record(
                record,
                prompt=prompt,
                material_name=material_name,
            )
            selected_records.append(record)
            if len(selected_records) >= max_templates:
                break

        if dirty:
            index_payload["details"] = details
            self._write_cache(index_payload)

        selected_records.sort(key=lambda item: item.score, reverse=True)
        return {
            "enabled": bool(selected_records),
            "source": source,
            "message": message,
            "remote_template_dir": self.remote_template_dir,
            "known_rules": [
                "若模板信号出现 VTST、ICHAIN=2、IBRION=3、POTIM=0，应优先视为过渡态搜索而非普通 relax。",
                "若模板包含 Pt slab 且 KPOINTS 接近 Gamma 2x2x1、ISIF=2、ISPIN=2，通常是表面或吸附体系模板。",
                "若模板不含金属表面元素且以 C/H/Br 等小分子为主，通常更接近孤立分子或有机片段优化模板。",
            ],
            "templates": [record.to_prompt_dict() for record in selected_records],
        }

    def _load_or_refresh_index(self) -> tuple[dict[str, Any], str, str]:
        cache_payload = self._read_cache()
        if cache_payload is not None and self._cache_is_fresh(cache_payload):
            return cache_payload, "cache", "已使用本地模板缓存。"

        remote_payload = self._fetch_directory_index()
        if remote_payload is not None:
            merged = remote_payload
            if cache_payload is not None:
                merged["details"] = {
                    **(cache_payload.get("details") or {}),
                    **(remote_payload.get("details") or {}),
                }
            self._write_cache(merged)
            return merged, "remote", "已从远程模板目录刷新索引。"

        if cache_payload is not None:
            return cache_payload, "stale_cache", "远程模板目录暂不可用，已回退到旧缓存。"

        return {"directories": [], "details": {}}, "unavailable", "未找到可用模板索引。"

    def _fetch_directory_index(self) -> dict[str, Any] | None:
        command = (
            f"base={self._quote(self.remote_template_dir)}; "
            "find \"$base\" -mindepth 1 -maxdepth 2 -type d | sort | "
            "while read d; do "
            "[ -f \"$d/INCAR\" ] || [ -f \"$d/KPOINTS\" ] || [ -f \"$d/POSCAR\" ] || continue; "
            "rel=${d#\"$base\"/}; "
            "if [ \"$rel\" != \"$d\" ]; then printf '%s\\n' \"$rel\"; fi; "
            "done"
        )
        output = self._run_remote_command(command, timeout=120)
        if output is None:
            return None

        directories = []
        for line in output.splitlines():
            relative_path = line.strip()
            if not relative_path:
                continue
            directories.append(
                {
                    "relative_path": relative_path,
                    "name": Path(relative_path).name,
                    "signals": self._signals_from_path(relative_path),
                }
            )

        if not directories:
            return None

        return {
            "generated_at": int(time.time()),
            "directories": directories,
            "details": {},
        }

    def _fetch_detail(self, relative_path: str) -> dict[str, Any] | None:
        remote_path = str(PurePosixPath(self.remote_template_dir) / PurePosixPath(relative_path))
        command = (
            f"dir={self._quote(remote_path)}; "
            "printf '%s\\n' '__PATH__'; printf '%s\\n' \"$dir\"; "
            "printf '%s\\n' '__INCAR__'; [ -f \"$dir/INCAR\" ] && head -n 80 \"$dir/INCAR\"; "
            "printf '%s\\n' '__KPOINTS__'; [ -f \"$dir/KPOINTS\" ] && cat \"$dir/KPOINTS\"; "
            "printf '%s\\n' '__POSCAR__'; [ -f \"$dir/POSCAR\" ] && head -n 16 \"$dir/POSCAR\"; "
            "printf '%s\\n' '__CHILDREN__'; find \"$dir\" -maxdepth 1 -mindepth 1 -type d | sort | "
            "head -n 12 | sed \"s#^$dir/##\"; "
            "printf '%s\\n' '__END__'"
        )
        output = self._run_remote_command(command, timeout=120)
        if output is None:
            return None

        sections = self._parse_sections(output)
        incar_tags = self._extract_incar_tags(sections.get("INCAR", ""))
        elements, counts = self._extract_poscar_signature(sections.get("POSCAR", ""))
        detail = {
            "relative_path": relative_path,
            "name": Path(relative_path).name,
            "task_family": self._classify_template(
                relative_path=relative_path,
                incar_tags=incar_tags,
                elements=elements,
            ),
            "elements": elements,
            "element_counts": counts,
            "kpoints_mesh": self._extract_kpoints_mesh(sections.get("KPOINTS", "")),
            "incar_tags": incar_tags,
            "signals": self._build_signals(
                relative_path=relative_path,
                incar_tags=incar_tags,
                elements=elements,
            ),
            "child_directories": [
                line.strip()
                for line in sections.get("CHILDREN", "").splitlines()
                if line.strip()
            ],
        }
        return detail

    def _run_remote_command(self, remote_command: str, timeout: int) -> str | None:
        if self.remote_config is None and os.getenv(
            "SEMI_DFT_PLANNER_TEMPLATE_ENABLE_REMOTE", ""
        ).strip().lower() not in {"1", "true", "yes", "on"}:
            return None
        try:
            config = self.remote_config or RemoteClusterConfig.from_env()
        except Exception:
            return None

        runner = SSHRemoteRunner(config)
        try:
            backend = runner._select_backend(config)
            tools_error = runner._ensure_local_tools(config, backend)
            if tools_error is not None:
                return None
            result = runner._run_remote_command(
                config,
                remote_command,
                timeout=timeout,
                backend=backend,
            )
        except Exception:
            return None

        if result.returncode != 0:
            return None
        return (result.stdout or "").strip()

    @staticmethod
    def _parse_sections(text: str) -> dict[str, str]:
        matches = list(re.finditer(r"__([A-Z]+)__", text))
        if not matches:
            return {}

        sections: dict[str, str] = {}
        for index, match in enumerate(matches):
            key = match.group(1)
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            sections[key] = text[start:end].strip()
        return sections

    @staticmethod
    def _extract_incar_tags(text: str) -> dict[str, str]:
        tags: dict[str, str] = {}
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            normalized = key.strip().upper()
            if normalized in _KEY_INCAR_TAGS:
                tags[normalized] = value.split("#", 1)[0].strip()
        return tags

    @staticmethod
    def _extract_kpoints_mesh(text: str) -> str | None:
        for line in text.splitlines():
            stripped = line.strip()
            if re.fullmatch(r"\d+\s+\d+\s+\d+", stripped):
                return stripped
        return None

    @staticmethod
    def _extract_poscar_signature(text: str) -> tuple[list[str], list[int]]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for index in range(len(lines) - 1):
            first = lines[index].split()
            second = lines[index + 1].split()
            if not first or not second:
                continue
            if all(re.fullmatch(r"[A-Z][a-z]?", token) for token in first) and all(
                re.fullmatch(r"\d+", token) for token in second
            ):
                return first, [int(token) for token in second]
        return [], []

    def _classify_template(
        self,
        *,
        relative_path: str,
        incar_tags: dict[str, str],
        elements: list[str],
    ) -> str:
        upper_path = relative_path.upper()
        if (
            "TS" in upper_path
            or incar_tags.get("ICHAIN") == "2"
            or incar_tags.get("IBRION") == "3"
        ):
            return "transition_state_search"
        if "PT" in {element.upper() for element in elements}:
            if len(elements) == 1:
                return "clean_slab_relax"
            return "surface_adsorption_relax"
        if elements and all(element.upper() in {"H", "C", "N", "O", "S", "BR", "CL"} for element in elements):
            return "isolated_molecule_relax"
        return "general_vasp_template"

    def _build_signals(
        self,
        *,
        relative_path: str,
        incar_tags: dict[str, str],
        elements: list[str],
    ) -> list[str]:
        signals = self._signals_from_path(relative_path)
        elements_upper = {element.upper() for element in elements}
        if incar_tags.get("ICHAIN") == "2":
            signals.append("VTST dimer")
        if incar_tags.get("IBRION") == "3":
            signals.append("VTST ionic mode")
        if incar_tags.get("ISIF") == "2":
            signals.append("fixed cell relaxation")
        if incar_tags.get("ISPIN") == "2":
            signals.append("spin polarized")
        if "PT" in elements_upper:
            signals.append("Pt slab/system")
        if "BR" in elements_upper:
            signals.append("contains bromine")
        if elements_upper and elements_upper <= {"H", "C", "N", "O", "S", "BR", "CL"}:
            signals.append("organic or molecular template")
        deduped = []
        for item in signals:
            if item not in deduped:
                deduped.append(item)
        return deduped

    @staticmethod
    def _signals_from_path(relative_path: str) -> list[str]:
        upper_path = relative_path.upper()
        signals: list[str] = []
        if "TS" in upper_path:
            signals.append("path suggests transition state")
        if "START" in upper_path:
            signals.append("path suggests initial adsorption geometry")
        if "TEST" in upper_path:
            signals.append("path suggests exploratory template")
        return signals

    def _score_directory(
        self,
        entry: dict[str, Any],
        *,
        prompt: str,
        material_name: str | None,
    ) -> float:
        relative_path = str(entry.get("relative_path") or "")
        candidate_text = f"{relative_path} {entry.get('name', '')}".lower()
        prompt_text = f"{prompt} {material_name or ''}".lower()
        score = fuzz.partial_ratio(prompt_text, candidate_text)

        prompt_tokens = set(self._tokenize(prompt_text))
        path_tokens = set(self._tokenize(candidate_text))
        score += 12 * len(prompt_tokens & path_tokens)

        if self._looks_like_ts_request(prompt_text) and "TS" in relative_path.upper():
            score += 80
        if self._looks_like_adsorption_request(prompt_text):
            if "START" in relative_path.upper():
                score += 45
            if any(token in path_tokens for token in {"pt", "br"}):
                score += 18
        return float(score)

    def _score_record(
        self,
        record: TemplateRecord,
        *,
        prompt: str,
        material_name: str | None,
    ) -> float:
        prompt_text = f"{prompt} {material_name or ''}".lower()
        base = self._score_directory(
            {
                "relative_path": record.relative_path,
                "name": record.name,
            },
            prompt=prompt,
            material_name=material_name,
        )

        if self._looks_like_ts_request(prompt_text) and record.task_family == "transition_state_search":
            base += 100
        if self._looks_like_adsorption_request(prompt_text) and record.task_family == "surface_adsorption_relax":
            base += 50
        if "Pt slab/system" in record.signals and ("pt" in prompt_text or "111" in prompt_text):
            base += 30
        if "organic or molecular template" in record.signals and "molecule" in prompt_text:
            base += 15
        return float(base)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token for token in re.split(r"[^a-zA-Z0-9]+", text.lower()) if token]

    @staticmethod
    def _looks_like_ts_request(prompt_text: str) -> bool:
        keywords = ["过渡态", "transition state", "ts ", "ts-", "dimer", "vtst", "爬山"]
        return any(keyword in prompt_text for keyword in keywords)

    @staticmethod
    def _looks_like_adsorption_request(prompt_text: str) -> bool:
        keywords = ["吸附", "adsorption", "adsorb", "slab", "surface", "表面", "pt(111)", "pt111"]
        return any(keyword in prompt_text for keyword in keywords)

    def _read_cache(self) -> dict[str, Any] | None:
        if not self.cache_path.exists():
            return None
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write_cache(self, payload: dict[str, Any]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(payload)
        payload["generated_at"] = int(time.time())
        self.cache_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _cache_is_fresh(self, payload: dict[str, Any]) -> bool:
        generated_at = int(payload.get("generated_at") or 0)
        if generated_at <= 0:
            return False
        return (time.time() - generated_at) <= self.cache_max_age_seconds

    @staticmethod
    def _needs_detail_refresh(payload: dict[str, Any]) -> bool:
        if not payload:
            return True
        if payload.get("kpoints_mesh") in {None, ""}:
            return True
        if not payload.get("incar_tags"):
            return True
        return False

    @staticmethod
    def _to_record(payload: dict[str, Any]) -> TemplateRecord:
        return TemplateRecord(
            relative_path=str(payload.get("relative_path") or ""),
            name=str(payload.get("name") or ""),
            task_family=str(payload.get("task_family") or "general_vasp_template"),
            elements=[str(item) for item in (payload.get("elements") or [])],
            element_counts=[int(item) for item in (payload.get("element_counts") or [])],
            kpoints_mesh=(
                str(payload.get("kpoints_mesh"))
                if payload.get("kpoints_mesh") is not None
                else None
            ),
            incar_tags={
                str(key): str(value)
                for key, value in (payload.get("incar_tags") or {}).items()
            },
            signals=[str(item) for item in (payload.get("signals") or [])],
            child_directories=[str(item) for item in (payload.get("child_directories") or [])],
        )

    @staticmethod
    def _quote(value: str) -> str:
        escaped = value.replace("'", "'\"'\"'")
        return f"'{escaped}'"
