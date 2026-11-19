def delete_workbook(wb_id: str) -> dict:
    """Permanently remove one workbook, its data and its own stored files."""
    workbook = require_workbook(wb_id)
    root = Path(workbook["storage_root"])
    with db.transaction() as conn:
        conn.execute(
            "DELETE FROM cells WHERE sheet_id IN"
            " (SELECT id FROM sheets WHERE workbook_id=?)", (wb_id,)
        )
        conn.execute("DELETE FROM sheets WHERE workbook_id=?", (wb_id,))
        for table in ("batches", "versions", "exports", "edits_log", "analyses"):
            conn.execute(f"DELETE FROM {table} WHERE workbook_id=?", (wb_id,))
        conn.execute("DELETE FROM workbooks WHERE id=?", (wb_id,))
    for sub in config.SUBDIRS:
        shutil.rmtree(root / sub / wb_id, ignore_errors=True)
    return {"id": wb_id, "name": workbook["name"]}
@bp.delete("/workbooks/<wb_id>")
@api
def delete_workbook(wb_id):
    return {"deleted": workbooks.delete_workbook(wb_id)}
function fileMenuButton(wb) {
  return '<button class="ws-small-btn" type="button" data-file-menu="' + esc(wb.id) +
    '" aria-haspopup="menu" aria-expanded="false">Options</button>';
}

function closeFileMenu() {
  var menu = $("ws-file-menu");
  if (menu) menu.remove();
  qsa("[data-file-menu]").forEach(function (button) {
    button.setAttribute("aria-expanded", "false");
  });
}

function openFileMenu(anchor, id) {
  var wb = findWorkbook(id);
  if (!wb) return;
  closeFileMenu();
  var archived = !!wb.archived;
  var entries = [
    ["open", "Open in Excel editor"],
    ["analyse", "Analyse again"],
    ["report", "View analysis report"],
    ["download", "Download original file"],
    ["rename", "Rename workbook"],
    [archived ? "restore" : "archive", archived ? "Restore from archive" : "Archive (hide, keep data)"],
    ["delete", "Delete permanently", "danger"],
  ];
  var menu = document.createElement("div");
  menu.id = "ws-file-menu";
  menu.className = "ws-file-menu";
  menu.setAttribute("role", "menu");
  menu.setAttribute("aria-label", "File options for " + wb.name);
  menu.innerHTML = '<div class="ws-file-menu-title">' + esc(wb.name) + "</div>" +
    entries.map(function (entry) {
      return '<button type="button" role="menuitem" class="' + (entry[2] || "") +
        '" data-file-action="' + entry[0] + '" data-id="' + esc(wb.id) + '">' + entry[1] + "</button>";
    }).join("");
  $("ws-pages").appendChild(menu);
  var anchorEl = anchor.tagName === "TR" ? (anchor.querySelector("[data-file-menu]") || anchor) : anchor;
  var rect = anchorEl.getBoundingClientRect();
  var width = 234;
  var left = Math.min(Math.max(8, rect.right - width), window.innerWidth - width - 8);
  var height = menu.offsetHeight || 280;
  var top = rect.bottom + 6;
  if (top + height > window.innerHeight - 8) top = Math.max(8, rect.top - height - 6);
  menu.style.left = left + "px";
  menu.style.top = top + "px";
  anchorEl.setAttribute("aria-expanded", "true");
  var first = menu.querySelector("button");
  if (first) setTimeout(function () {
    try { first.focus({ preventScroll: true }); } catch (error) { /* ignore */ }
  }, 20);
}
async function deleteWorkbookById(id) {
  var wb = findWorkbook(id);
  if (!wb) return;
  if (S.wb && S.wb.id === id) {
    await flushSave();
    if (S.queue.length) { toast("Pending changes must be saved before deleting.", "error"); return; }
  }
  var confirmed = await confirmDialog({
    title: "Delete this workbook permanently?",
    text: "\"" + wb.name + "\" and its versions, analysis history and exported copies will be removed from this computer. This cannot be undone. Other workbooks are not touched. Use Archive instead if you only want to hide it.",
    button: "Delete permanently",
    danger: true,
  });
  if (!confirmed) return;
  await api("/workbooks/" + encodeURIComponent(id), { method: "DELETE" });
  if (S.wb && S.wb.id === id) await closeWorkbook(false);
  if (S.reportWbId === id) { S.reportWbId = null; S.reportAnalysis = null; }
  await refreshLibrary();
  await setPage("files");
  toast("Deleted \"" + wb.name + "\". Other workbooks were not changed.");
}
if (action === "delete") { await deleteWorkbookById(id); return; }
// Home recent files
return '<tr data-file-menu="' + esc(wb.id) + '"><td class="ws-name-cell" title="' + esc(wb.name) + '">' + esc(wb.name) + "</td>" +
  "<td>" + esc(typeLabel(wb)) + "</td>" +
  "<td>" + esc(when(wb.updated_at)) + "</td>" +
  '<td class="ws-num">' + esc(wb.record_count || 0) + "</td>" +
  "<td>" + analysisPill(wb) + "</td>" +
  '<td class="ws-actions-cell">' + fileMenuButton(wb) + "</td></tr>";

// Home search results
return '<tr data-file-menu="' + esc(wb.id) + '"><td class="ws-name-cell">' + esc(wb.name) + "</td><td>" + esc(typeLabel(wb)) + "</td><td>" + esc(when(wb.updated_at)) + "</td>" +
  '<td class="ws-actions-cell">' + fileMenuButton(wb) + "</td></tr>";

// Upload page recent imports
return '<tr data-file-menu="' + esc(wb.id) + '"><td class="ws-name-cell">' + esc(wb.name) + "</td><td>" + esc(typeLabel(wb)) + "</td><td>" + esc(when(wb.created_at)) + "</td>" +
  '<td class="ws-num">' + esc(wb.record_count || 0) + "</td><td>" + analysisPill(wb) + "</td>" +
  '<td class="ws-actions-cell">' + fileMenuButton(wb) + "</td></tr>";

// My files page
return '<tr data-file-menu="' + esc(wb.id) + '"><td class="ws-name-cell" title="' + esc(wb.name) + '">' + esc(wb.name) + "</td>" +
  "<td>" + esc(typeLabel(wb)) + "</td>" +
  "<td>" + esc(when(wb.created_at)) + "</td>" +
  "<td>" + esc(when(wb.updated_at)) + "</td>" +
  '<td class="ws-num">' + esc(wb.record_count || 0) + "</td>" +
  '<td class="ws-num">' + esc(wb.sheet_count || 0) + "</td>" +
  "<td>" + analysisPill(wb) + "</td>" +
  '<td class="ws-actions-cell">' + fileMenuButton(wb) + "</td></tr>";
  $("ws-pages").addEventListener("click", function (event) {
  var menuButton = event.target.closest("[data-file-menu]");
  if (menuButton) {
    event.preventDefault();
    var trigger = menuButton.tagName === "TR"
      ? (menuButton.querySelector("[data-file-menu]") || menuButton) : menuButton;
    if (trigger.getAttribute("aria-expanded") === "true") closeFileMenu();
    else openFileMenu(trigger, trigger.dataset.fileMenu);
    return;
  }
  var fileButton = event.target.closest("[data-file-action]");
  if (fileButton) {
    event.preventDefault();
    if (fileButton.closest("#ws-file-menu")) closeFileMenu();
    handleFileAction(fileButton.dataset.fileAction, fileButton.dataset.id).catch(reportError);
    return;
  }
  // ...rest of the existing listener...
  document.addEventListener("click", function (event) {
  if (!event.target || !event.target.closest) return;
  if (event.target.closest("#ws-file-menu") || event.target.closest("[data-file-menu]")) return;
  closeFileMenu();
});
document.addEventListener("keydown", function (event) {
  if (event.key === "Escape") closeFileMenu();
});
window.addEventListener("resize", closeFileMenu);
$("ws-pages").addEventListener("scroll", closeFileMenu);
.ws-table tbody tr[data-file-menu] { cursor: pointer; }
.ws-table tbody tr[data-file-menu]:hover { background: var(--hover-bg); }

.ws-file-menu {
  position: fixed; z-index: 90; min-width: 234px; padding: 6px;
  border-radius: 12px; border: 1px solid var(--line); background: var(--panel);
  box-shadow: 0 22px 48px -14px rgba(0, 0, 0, .5);
  display: flex; flex-direction: column;
}
.ws-file-menu-title {
  padding: 7px 10px 8px; margin-bottom: 4px; border-bottom: 1px solid var(--line);
  color: var(--dim); font-size: 10.5px; letter-spacing: .5px; text-transform: uppercase;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.ws-file-menu button {
  width: 100%; padding: 8px 10px; border: 0; border-radius: 8px; background: transparent;
  color: var(--text); font-size: 12.6px; text-align: left;
}
.ws-file-menu button:hover,
.ws-file-menu button:focus-visible { background: var(--hover-bg); outline: none; }
.ws-file-menu button.danger { color: #dc2626; }
.ws-file-menu button.danger:hover { background: rgba(220, 38, 38, .12); }
def test_delete_workbook_removes_only_that_workbook(self):
    other = self.client.post(
        f"{PREFIX}/workbooks", json={"name": "Keep this workbook"}
    ).get_json()["workbook"]
    folder = _TMP / "storage" / "workbooks" / self.wb_id
    backups = _TMP / "storage" / "backups" / self.wb_id
    self.assertTrue(folder.is_dir())
    self.assertTrue(backups.is_dir())

    deleted = self.client.delete(f"{PREFIX}/workbooks/{self.wb_id}")
    self.assertEqual(deleted.status_code, 200, deleted.get_data(as_text=True))
    self.assertEqual(deleted.get_json()["deleted"]["id"], self.wb_id)
    self.assertEqual(self.client.get(f"{PREFIX}/workbooks/{self.wb_id}").status_code, 404)
    self.assertFalse(folder.exists())
    self.assertFalse(backups.exists())

    listing = self.client.get(f"{PREFIX}/workbooks").get_json()["workbooks"]
    ids = [item["id"] for item in listing]
    self.assertNotIn(self.wb_id, ids)
    self.assertIn(other["id"], ids)
    self.assertTrue((_TMP / "storage" / "workbooks" / other["id"]).is_dir())

    missing = self.client.delete(f"{PREFIX}/workbooks/missing")
    self.assertEqual(missing.status_code, 404)
    python app.py