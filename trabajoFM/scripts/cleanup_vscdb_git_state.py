"""Reset stuck SCM / diff-editor state in a VS Code workspace database.

Usage (close VS Code first!):
    python cleanup_vscdb_git_state.py              # dry run — list keys
    python cleanup_vscdb_git_state.py --apply      # delete the keys
    python cleanup_vscdb_git_state.py --restore    # restore DB from backup
"""
import argparse, os, shutil, sqlite3, pathlib, sys

parser = argparse.ArgumentParser(
    description="Reset stuck SCM/diff state in VS Code workspace DB",
)
group = parser.add_mutually_exclusive_group()
group.add_argument("--apply", action="store_true",
                   help="Actually delete matching keys (default is dry run)")
group.add_argument("--restore", action="store_true",
                   help="Restore state.vscdb from the latest backup")
args = parser.parse_args()

db = (pathlib.Path(os.environ["APPDATA"])
      / "Code" / "User" / "workspaceStorage"
      / "abdb0dbeffcaf70adb922f3da23b79c6" / "state.vscdb")
bak = db.with_name("state.vscdb.before-git-clean.bak")

# ── Restore mode ──────────────────────────────────────────
if args.restore:
    if not bak.exists():
        raise SystemExit(f"No backup found at: {bak}")
    shutil.copy2(bak, db)
    print(f"Restored {db}\n    from {bak}")
    sys.exit(0)

# ── Cleanup mode (dry-run / apply) ───────────────────────
if not db.exists():
    raise SystemExit(f"DB not found: {db}")

shutil.copy2(db, bak)
print(f"Backup created: {bak}")

con = sqlite3.connect(str(db))
cur = con.cursor()

# Only target SCM/diff-view keys — not Copilot chat or PR extension state
exact_keys = [
    "vscode.git",
    "scm.graphView.referencesFilter",
    "scm.history",
    "scm.viewState2",
    "scm:view:visibleRepositories",
    "workbench.scm.views.state",
    "workbench.view.scm.numberOfVisibleViews",
]

placeholders = ",".join("?" for _ in exact_keys)
cur.execute(f"SELECT key FROM ItemTable WHERE key IN ({placeholders})", exact_keys)
rows = cur.fetchall()

print(f"Found {len(rows)} matching keys:")
for (k,) in rows:
    print(f"  - {k}")

if args.apply:
    cur.execute(f"DELETE FROM ItemTable WHERE key IN ({placeholders})", exact_keys)
    con.commit()
    print(f"\nDeleted {cur.rowcount} keys.")
else:
    print("\nDry run only. Re-run with --apply to delete.")

con.close()
