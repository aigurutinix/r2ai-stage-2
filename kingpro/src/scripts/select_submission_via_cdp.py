"""Select exactly one measured submission for the public leaderboard via CDP.

Dry-run is the default. Execute requires an exact local ZIP SHA and writes a
permanent one-shot receipt before clicking the unique row action.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from submit_via_cdp import Cdp, DEFAULT_TARGET, choose_target, sha256, targets


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_DIR = ROOT / ".leaderboard-selection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-id", type=int, required=True)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--port", type=int, default=9222)
    parser.add_argument("--target-fragment", default=DEFAULT_TARGET)
    parser.add_argument("--target-id")
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-sha256")
    return parser.parse_args()


def inspect(cdp: Cdp, submission_id: int, filename: str) -> dict:
    expression = f"""
    (() => {{
      const sid={submission_id};
      const filename={json.dumps(filename)};
      const rows=[...document.querySelectorAll('tr.submission_row')];
      const matches=rows.filter(row => {{
        const cells=[...row.querySelectorAll(':scope > td')];
        const rect=row.getBoundingClientRect();
        const style=getComputedStyle(row);
        return Number((cells[0]?.innerText||'').trim())===sid &&
               (cells[1]?.innerText||'').trim()===filename &&
               rect.width>0 && rect.height>0 && style.display!=='none' &&
               style.visibility!=='hidden';
      }});
      if(matches.length!==1) return {{rowCount:matches.length}};
      const row=matches[0];
      const add=[...row.querySelectorAll('span[data-tooltip]')]
        .filter(node => node.getAttribute('data-tooltip')==='Thêm vào Bảng xếp hạng');
      const remove=[...row.querySelectorAll('span[data-tooltip]')]
        .filter(node => /(?:Xóa|Gỡ).*(?:Bảng xếp hạng)/i.test(node.getAttribute('data-tooltip')||''));
      const selected=[...row.querySelectorAll('span[data-tooltip]')]
        .filter(node => node.getAttribute('data-tooltip')==='Trên Bảng xếp hạng');
      return {{
        rowCount:1,
        text:row.innerText,
        finished:(row.innerText||'').includes('Finished'),
        addActionCount:add.length,
        removeActionCount:remove.length,
        selectedActionCount:selected.length,
        onLeaderboard:(remove.length===1 || selected.length===1) && add.length===0
      }};
    }})()
    """
    value = cdp.evaluate(expression)
    return value if isinstance(value, dict) else {"rowCount": 0}


def write_lock(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise SystemExit(f"REFUSED: selection receipt already exists: {path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    archive = args.file.resolve()
    if not archive.is_file() or archive.suffix.lower() != ".zip":
        raise SystemExit(f"REFUSED: invalid local ZIP: {archive}")
    archive_sha = sha256(archive)
    lock = args.state_dir.resolve() / f"submission-{args.submission_id}.json"
    pages = targets(args.port, args.target_fragment)
    target = choose_target(pages, args.target_id)
    cdp = Cdp(target["webSocketDebuggerUrl"])
    try:
        before = inspect(cdp, args.submission_id, archive.name)
        reasons = []
        if before.get("rowCount") != 1:
            reasons.append(f"expected one exact row, found {before.get('rowCount')}")
        if not before.get("finished"):
            reasons.append("submission is not visibly Finished")
        if before.get("onLeaderboard"):
            reasons.append("submission is already selected")
        if before.get("addActionCount") != 1:
            reasons.append(f"expected one add action, found {before.get('addActionCount')}")
        if lock.exists():
            reasons.append(f"one-shot selection receipt exists: {lock}")
        report = {
            "mode": "execute" if args.execute else "dry-run",
            "submissionId": args.submission_id,
            "file": str(archive),
            "sha256": archive_sha,
            "targetId": target["id"],
            "before": before,
            "eligible": not reasons,
            "refusalReasons": reasons,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not args.execute:
            return 0
        if reasons:
            raise SystemExit("REFUSED: execute preconditions failed")
        if not args.confirm_sha256 or args.confirm_sha256.upper() != archive_sha:
            raise SystemExit("REFUSED: exact --confirm-sha256 is mandatory")
        receipt = {
            "state": "armed",
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "submissionId": args.submission_id,
            "filename": archive.name,
            "sha256": archive_sha,
            "targetId": target["id"],
        }
        write_lock(lock, receipt)
        clicked = cdp.evaluate(
            f"""
            (() => {{
              const sid={args.submission_id};
              const filename={json.dumps(archive.name)};
              const rows=[...document.querySelectorAll('tr.submission_row')].filter(row => {{
                const cells=[...row.querySelectorAll(':scope > td')];
                const rect=row.getBoundingClientRect();
                const style=getComputedStyle(row);
                return Number((cells[0]?.innerText||'').trim())===sid &&
                       (cells[1]?.innerText||'').trim()===filename &&
                       rect.width>0 && rect.height>0 && style.display!=='none' &&
                       style.visibility!=='hidden';
              }});
              if(rows.length!==1) return {{ok:false,reason:'non-unique-visible-row',count:rows.length}};
              const row=rows[0];
              const actions=[...row.querySelectorAll('span[data-tooltip]')]
                .filter(node => node.getAttribute('data-tooltip')==='Thêm vào Bảng xếp hạng');
              if(actions.length!==1) return {{ok:false,reason:'non-unique-action',count:actions.length}};
              actions[0].click();
              return {{ok:true}};
            }})()
            """
        )
        receipt["state"] = "selection_clicked_once"
        receipt["clickedAt"] = datetime.now(timezone.utc).isoformat()
        lock.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if not isinstance(clicked, dict) or not clicked.get("ok"):
            raise SystemExit(f"REFUSED_AFTER_LOCK: selection click failed: {clicked!r}")
        time.sleep(3)
        after = inspect(cdp, args.submission_id, archive.name)
        receipt["after"] = after
        receipt["state"] = "confirmed" if after.get("onLeaderboard") else "clicked_unconfirmed"
        lock.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "result": "SELECTED_ONCE" if after.get("onLeaderboard") else "CLICKED_ONCE_UNCONFIRMED_DO_NOT_RETRY",
                    "receipt": str(lock),
                    "after": after,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if after.get("onLeaderboard") else 2
    finally:
        cdp.close()


if __name__ == "__main__":
    raise SystemExit(main())
