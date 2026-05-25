from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .adsorption_models import CandidateManifest, CandidateSelection


class ConfirmedCandidateHandoff:
    def load_manifest(self, manifest_path: Path) -> dict[str, Any]:
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def materialize_selection(
        self,
        *,
        manifest_path: Path,
        candidate_id: str,
        output_dir: Path,
    ) -> CandidateSelection:
        manifest = self.load_manifest(manifest_path)
        candidate = next(
            (item for item in manifest.get("candidates", []) if item.get("candidate_id") == candidate_id),
            None,
        )
        if candidate is None:
            raise ValueError(f"candidate_id 不存在: {candidate_id}")

        output_dir.mkdir(parents=True, exist_ok=True)
        source_poscar = Path(candidate["exported_files"]["poscar_path"])
        source_cif = Path(candidate["exported_files"]["cif_path"])
        source_summary = Path(candidate["exported_files"]["summary_path"])

        selected_poscar = output_dir / "POSCAR.selected"
        selected_cif = output_dir / "structure.selected.cif"
        selected_summary = output_dir / "selection_summary.json"
        shutil.copyfile(source_poscar, selected_poscar)
        shutil.copyfile(source_cif, selected_cif)
        shutil.copyfile(source_summary, selected_summary)

        selection = CandidateSelection(
            manifest_path=str(manifest_path),
            candidate_id=candidate_id,
            selected_poscar_path=str(selected_poscar),
            selected_cif_path=str(selected_cif),
            selected_summary_path=str(selected_summary),
            metadata={
                "material_name": manifest.get("material_name"),
                "source_prompt": manifest.get("source_prompt"),
                "candidate": candidate,
            },
        )
        selection_path = output_dir / "selection.json"
        selection_path.write_text(
            json.dumps(selection.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return selection
