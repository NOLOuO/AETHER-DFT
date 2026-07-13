from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aether_dft.campaign_dossier import build_campaign_dossier


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an auditable AETHER-DFT campaign dossier.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--run-root", action="append", required=True, dest="run_roots")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = build_campaign_dossier(
        project=args.project,
        session_id=args.session_id,
        run_roots=args.run_roots,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
