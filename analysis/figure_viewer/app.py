from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path
from queue import Queue, Empty
from typing import Callable, Dict, List, Sequence

import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

from analysis.figure_viewer.catalog import CatalogScanProgress, DEFAULT_RESULTS_DEPTH, discover_figure_records
from analysis.figure_viewer.layout import (
    FIELD_LABELS,
    HIERARCHY_FIELDS,
    BrowserNode,
    SlotSelection,
    browser_children,
    browser_node_label,
    build_results_index,
    comparison_label,
    comparison_signature,
    field_options,
    resolve_selection,
    selection_from_record,
    selection_with_field,
)
from analysis.figure_viewer.models import FigureRecord
from analysis.figure_viewer.rendering import image_for_preview
from analysis.figure_viewer.storage import NoteStore


IMAGE_NOTE_SCOPE = "image"
COMPARISON_NOTE_SCOPE = "comparison"


def _truncate(text: str, limit: int = 96) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 1)].rstrip() + "..."


def _set_text(widget: tk.Text, text: str) -> None:
    widget.configure(state="normal")
    widget.delete("1.0", tk.END)
    widget.insert("1.0", text)
    widget.configure(state="disabled")


def _note_line(note: dict) -> str:
    timestamp = str(note.get("created_at") or "").strip()
    text = str(note.get("note_text") or "").strip().replace("\n", " ")
    if timestamp and text:
        return f"{timestamp} | {text}"
    return timestamp or text or "Note"


def _default_notes_db_path(repo_root: Path) -> Path:
    return repo_root / ".figure_viewer" / "notes.sqlite"


def _relative_path_text(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _selection_path_text(selection: SlotSelection | None) -> str:
    if selection is None:
        return ""
    values = [getattr(selection, field) for field in HIERARCHY_FIELDS if getattr(selection, field)]
    return " / ".join(values)


def _record_summary(record: FigureRecord | None) -> str:
    if record is None:
        return "No figure selected"
    return record.display_label or record.title or record.preview_path.name


def _record_details(record: FigureRecord | None, repo_root: Path) -> str:
    if record is None:
        return "Select a figure leaf in the explorer or choose a path in a slot."
    payload = record.as_context()
    payload["source_paths"] = [str(path) for path in record.source_paths]
    payload["source_kinds"] = list(record.source_kinds)
    lines: List[str] = []
    for key in [
        "figure_key",
        "comparison_key",
        "comparison_label",
        "display_label",
        "title",
        "pipeline",
        "preset",
        "split",
        "basis",
        "family",
        "cohort",
        "scope",
        "compartment",
        "variant",
        "source_root",
        "manifest_path",
        "preview_path",
    ]:
        value = payload.get(key, "")
        if value:
            lines.append(f"{key}: {value}")
    if payload.get("source_kinds"):
        lines.append(f"source_kinds: {', '.join(payload['source_kinds'])}")
    if payload.get("source_paths"):
        lines.append("source_paths:")
        lines.extend(f"  - {item}" for item in payload["source_paths"])
    if payload.get("metadata"):
        lines.append("metadata:")
        lines.extend("  - {}: {}".format(key, value) for key, value in sorted(payload["metadata"].items()))
    lines.append("")
    lines.append(f"relative_path: {_relative_path_text(record.preview_path, repo_root)}")
    return "\n".join(lines)


def _figure_choice_label(record: FigureRecord, index: int, seen: set[str]) -> str:
    base = record.display_label or record.title or record.preview_path.name
    label = f"{index + 1}. {_truncate(base, 96)}"
    if label in seen:
        label = f"{label} [{record.preview_path.name}]"
    if label in seen:
        label = f"{label} [{_truncate(record.preview_path.as_posix(), 60)}]"
    return label


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent: tk.Widget, *, height: int | None = None) -> None:
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0, background="#f7f7f7", height=height)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.inner = ttk.Frame(self.canvas)
        self._window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

    def _on_inner_configure(self, _event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self._window_id, width=event.width)


class StartupProgressWindow:
    def __init__(self, parent: tk.Tk) -> None:
        self.parent = parent
        self.window = tk.Toplevel(parent)
        self.window.withdraw()
        self.window.title("Loading Figure Viewer")
        self.window.resizable(False, False)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self.hide)
        self.window.columnconfigure(0, weight=1)

        frame = ttk.Frame(self.window, padding=18)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)

        self.phase_var = tk.StringVar(value="Scanning results/")
        self.detail_var = tk.StringVar(value="Preparing startup scan...")
        self.count_var = tk.StringVar(value="0 / 0 scanned")
        self.remaining_var = tk.StringVar(value="Waiting for file counts...")
        self._mode = "indeterminate"

        ttk.Label(frame, textvariable=self.phase_var, font=("TkDefaultFont", 11, "bold"), anchor="w").grid(row=0, column=0, sticky="ew")
        ttk.Label(frame, textvariable=self.detail_var, wraplength=380, justify="left").grid(row=1, column=0, sticky="ew", pady=(6, 10))
        self.progress = ttk.Progressbar(frame, mode="indeterminate", maximum=100)
        self.progress.grid(row=2, column=0, sticky="ew")
        ttk.Label(frame, textvariable=self.count_var, anchor="w").grid(row=3, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(frame, textvariable=self.remaining_var, anchor="w", foreground="#555555").grid(row=4, column=0, sticky="ew", pady=(2, 0))
        self._center()

    def _center(self) -> None:
        self.window.update_idletasks()
        width = 460
        height = 180
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x = max((screen_width - width) // 2, 0)
        y = max((screen_height - height) // 3, 0)
        self.window.geometry(f"{width}x{height}+{x}+{y}")

    def show(self) -> None:
        if not self.window.winfo_exists():
            return
        self.window.deiconify()
        self.window.lift()
        self._center()

    def begin_scan(self, *, phase: str = "Scanning results/", detail: str = "Preparing startup scan...") -> None:
        if not self.window.winfo_exists():
            return
        self.show()
        self.phase_var.set(phase)
        self.detail_var.set(detail)
        self.count_var.set("0 / 0 scanned")
        self.remaining_var.set("Waiting for file counts...")
        self._mode = "indeterminate"
        self.progress.configure(mode="indeterminate", value=0)
        self.progress.start(12)

    def update_progress(self, progress: CatalogScanProgress) -> None:
        if not self.window.winfo_exists():
            return
        self.show()
        self.phase_var.set(progress.phase or "Scanning results/")
        self.detail_var.set(progress.message or progress.phase or "Scanning...")
        if progress.total > 0:
            if self._mode != "determinate":
                self.progress.stop()
                self.progress.configure(mode="determinate")
                self._mode = "determinate"
            current = min(max(progress.current, 0), progress.total)
            remaining = max(progress.total - current, 0)
            percent = int(round((current / progress.total) * 100)) if progress.total else 0
            self.progress.configure(maximum=max(progress.total, 1), value=current)
            self.count_var.set(f"{current:,} / {progress.total:,} scanned ({percent}%)")
            self.remaining_var.set(f"{remaining:,} remaining")
        else:
            if self._mode != "indeterminate":
                self.progress.configure(mode="indeterminate", value=0)
                self.progress.start(12)
                self._mode = "indeterminate"
            self.count_var.set("0 / 0 scanned")
            self.remaining_var.set(progress.message or "Waiting for file counts...")
        self.window.update_idletasks()

    def hide(self) -> None:
        if not self.window.winfo_exists():
            return
        try:
            self.progress.stop()
        except tk.TclError:
            pass
        self.window.withdraw()

    def destroy(self) -> None:
        if not self.window.winfo_exists():
            return
        try:
            self.progress.stop()
        except tk.TclError:
            pass
        self.window.destroy()


class ExplorerPanel(ttk.Frame):
    def __init__(self, app: "FigureViewerApp", parent: tk.Widget) -> None:
        super().__init__(parent, padding=10)
        self.app = app
        self.browser_root: BrowserNode | None = None
        self.selected_node: BrowserNode | None = None
        self.selected_record: FigureRecord | None = None
        self._tree_to_node: Dict[str, BrowserNode] = {}
        self._populated: set[str] = set()

        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Results Explorer", font=("TkDefaultFont", 11, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(header, textvariable=self.app.catalog_status_var, foreground="#555555").grid(row=1, column=0, sticky="w", pady=(2, 0))

        button_row = ttk.Frame(header)
        button_row.grid(row=0, column=1, rowspan=2, sticky="e")
        ttk.Button(button_row, text="Refresh", command=self.app.refresh_catalog).grid(row=0, column=0, padx=(0, 8))
        self.load_button = ttk.Button(button_row, text="Add to Slot", command=self.add_selected_to_slot, state="disabled")
        self.load_button.grid(row=0, column=1)

        ttk.Separator(self, orient="horizontal").grid(row=1, column=0, sticky="ew", pady=(10, 10))

        self.explorer_paned = ttk.Panedwindow(self, orient="vertical")
        self.explorer_paned.grid(row=2, column=0, sticky="nsew")

        tree_container = ttk.Frame(self.explorer_paned)
        tree_container.columnconfigure(0, weight=1)
        tree_container.rowconfigure(0, weight=1)
        self.explorer_paned.add(tree_container, weight=3)

        tree_frame = ttk.Frame(tree_container)
        tree_frame.grid(row=0, column=0, sticky="nsew")
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)

        self.tree = ttk.Treeview(tree_frame, show="tree", selectmode="browse", height=20)
        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll.grid(row=0, column=1, sticky="ns")
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.tree.bind("<<TreeviewOpen>>", self._on_tree_open)
        self.tree.bind("<Double-1>", lambda _event: self.add_selected_to_slot())

        details_container = ttk.Frame(self.explorer_paned)
        details_container.columnconfigure(0, weight=1)
        details_container.rowconfigure(0, weight=1)
        self.explorer_paned.add(details_container, weight=2)

        details = ttk.LabelFrame(details_container, text="Selected Figure", padding=8)
        details.grid(row=0, column=0, sticky="nsew")
        details.columnconfigure(0, weight=1)
        details.rowconfigure(2, weight=1)
        self.explorer_summary_var = tk.StringVar(value="Select a figure leaf to inspect it.")
        self.explorer_path_var = tk.StringVar(value="")
        ttk.Label(details, textvariable=self.explorer_summary_var, wraplength=360, justify="left").grid(row=0, column=0, sticky="ew")
        ttk.Label(details, textvariable=self.explorer_path_var, wraplength=360, justify="left", foreground="#555555").grid(row=1, column=0, sticky="ew", pady=(2, 6))
        self.details_text = tk.Text(details, height=12, wrap="word", state="disabled", background="#fcfcfc", relief="flat")
        self.details_text.grid(row=2, column=0, sticky="nsew")
        details_scroll = ttk.Scrollbar(details, orient="vertical", command=self.details_text.yview)
        self.details_text.configure(yscrollcommand=details_scroll.set)
        details_scroll.grid(row=2, column=1, sticky="ns")

    def set_catalog(self, browser_root: BrowserNode | None, figure_count: int) -> None:
        self.browser_root = browser_root
        self._tree_to_node.clear()
        self._populated.clear()
        self.tree.delete(*self.tree.get_children(""))

        if browser_root is None:
            root_id = self.tree.insert("", "end", text="results")
            self.selected_node = None
            self.selected_record = None
            self.explorer_summary_var.set("No figures found under results/")
            self.explorer_path_var.set("")
            _set_text(self.details_text, "No figures were indexed. Try refreshing after results are generated.")
            self.load_button.configure(state="disabled")
            self._tree_to_node[root_id] = BrowserNode(name="results", path_parts=("results",), figure_count=0)
            self.tree.selection_set(root_id)
            return

        root_text = f"results ({browser_root.figure_count})"
        root_id = self.tree.insert("", "end", text=root_text, open=True)
        self._tree_to_node[root_id] = browser_root
        self.tree.selection_set(root_id)
        self.tree.focus(root_id)
        self._populate_item(root_id)
        self._refresh_selection_details(root_id)
        self.app.set_status(f"Indexed {figure_count} figures under results/")

    def _populate_item(self, item_id: str) -> None:
        if item_id in self._populated:
            return
        node = self._tree_to_node.get(item_id)
        if node is None:
            return
        children = browser_children(node)
        if not children:
            self._populated.add(item_id)
            return
        for child_id in self.tree.get_children(item_id):
            self.tree.delete(child_id)
        for child in children:
            child_id = self.tree.insert(item_id, "end", text=browser_node_label(child))
            self._tree_to_node[child_id] = child
            if child.children:
                self.tree.insert(child_id, "end", text="")
        self._populated.add(item_id)

    def _on_tree_open(self, _event: tk.Event) -> None:
        selected = self.tree.focus()
        if selected:
            self._populate_item(selected)

    def _on_tree_select(self, _event: tk.Event) -> None:
        selected = self.tree.selection()
        if not selected:
            self.selected_node = None
            self.selected_record = None
            self.explorer_summary_var.set("Select a figure leaf to inspect it.")
            self.explorer_path_var.set("")
            _set_text(self.details_text, "")
            self.load_button.configure(state="disabled")
            return
        self._refresh_selection_details(selected[0])

    def _refresh_selection_details(self, item_id: str) -> None:
        node = self._tree_to_node.get(item_id)
        self.selected_node = node
        self.selected_record = node.record if node and node.record is not None else None
        if node is None:
            self.explorer_summary_var.set("Select a figure leaf to inspect it.")
            self.explorer_path_var.set("")
            _set_text(self.details_text, "")
            self.load_button.configure(state="disabled")
            return
        path_text = Path(*node.path_parts).as_posix()
        self.explorer_path_var.set(path_text)
        if node.record is None:
            self.explorer_summary_var.set(f"{browser_node_label(node)}")
            _set_text(
                self.details_text,
                f"{path_text}\n\n{node.figure_count} figures beneath this folder.\nExpand it to browse deeper.",
            )
            self.load_button.configure(state="disabled")
            return
        record = node.record
        self.explorer_summary_var.set(_record_summary(record))
        _set_text(self.details_text, _record_details(record, self.app.repo_root))
        self.load_button.configure(state="normal")

    def add_selected_to_slot(self) -> None:
        record = self.selected_record
        if record is None:
            self.app.set_status("Select a leaf figure in the explorer first.")
            return
        self.app.load_record_into_slot(record)

    def load_selected_into_active_slot(self) -> None:
        self.add_selected_to_slot()


class SlotView:
    def __init__(self, app: "FigureViewerApp", parent: tk.Widget, index: int) -> None:
        self.app = app
        self.index = index
        self.selection = SlotSelection()
        self.record: FigureRecord | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.base_image: Image.Image | None = None
        self.zoom_percent = 100
        self._ui_guard = False
        self._figure_label_to_record: Dict[str, FigureRecord] = {}
        self.field_vars: Dict[str, tk.StringVar] = {field: tk.StringVar(value="") for field in HIERARCHY_FIELDS}
        self.field_widgets: Dict[str, ttk.Combobox] = {}
        self.figure_var = tk.StringVar(value="")
        self.title_var = tk.StringVar(value="No figure selected")
        self.selection_var = tk.StringVar(value="Choose a pipeline to start.")
        self.source_var = tk.StringVar(value="")
        self.notes_title_var = tk.StringVar(value="Image Notes")
        self.notes_body_var = tk.StringVar(value="")
        self.zoom_var = tk.StringVar(value="100%")
        self.frame = ttk.LabelFrame(parent, text=self._header_text(), padding=10)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(2, weight=1)

        self._build_header()
        self._build_controls()
        self._build_preview()
        self._build_notes()
        self._bind_activity(self.frame)
        self._bind_activity(self.preview_canvas)
        self._bind_activity(self.notes_entry)
        self._bind_activity(self.notes_text)
        self.sync_to_records()

    def _bind_activity(self, widget: tk.Widget) -> None:
        widget.bind("<Button-1>", lambda _event: self.app.set_active_slot(self.index), add="+")
        widget.bind("<FocusIn>", lambda _event: self.app.set_active_slot(self.index), add="+")

    def _build_header(self) -> None:
        header = ttk.Frame(self.frame)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, textvariable=self.title_var, wraplength=640, justify="left", font=("TkDefaultFont", 11, "bold")).grid(row=0, column=0, sticky="w")
        button_row = ttk.Frame(header)
        button_row.grid(row=0, column=1, sticky="e")
        ttk.Button(button_row, text="Reset", command=self.reset).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(button_row, text="Remove", command=lambda: self.app.remove_slot(self)).grid(row=0, column=1)
        ttk.Label(self.frame, textvariable=self.selection_var, wraplength=640, justify="left", foreground="#555555").grid(row=1, column=0, sticky="ew", pady=(4, 0))
        ttk.Label(self.frame, textvariable=self.source_var, wraplength=640, justify="left", foreground="#555555").grid(row=2, column=0, sticky="ew", pady=(2, 8))

    def _build_controls(self) -> None:
        controls = ttk.Frame(self.frame)
        controls.grid(row=3, column=0, sticky="ew")
        for column in range(4):
            controls.columnconfigure(column, weight=1)

        for index, field_name in enumerate(HIERARCHY_FIELDS):
            row = 0 if index < 4 else 2
            column = index % 4
            ttk.Label(controls, text=FIELD_LABELS[field_name]).grid(row=row, column=column, sticky="w")
            combo = ttk.Combobox(controls, textvariable=self.field_vars[field_name], state="disabled", width=20)
            combo.grid(row=row + 1, column=column, sticky="ew", padx=(0, 8), pady=(0, 8))
            combo.bind("<<ComboboxSelected>>", lambda _event, field_name=field_name: self._on_field_selected(field_name))
            combo.bind("<FocusIn>", lambda _event: self.app.set_active_slot(self.index), add="+")
            self.field_widgets[field_name] = combo

        ttk.Label(controls, text="Figure").grid(row=2, column=3, sticky="w")
        self.figure_combo = ttk.Combobox(controls, textvariable=self.figure_var, state="disabled", width=28)
        self.figure_combo.grid(row=3, column=3, sticky="ew", padx=(0, 8), pady=(0, 8))
        self.figure_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_figure_selected())
        self.figure_combo.bind("<FocusIn>", lambda _event: self.app.set_active_slot(self.index), add="+")

    def _build_preview(self) -> None:
        preview_frame = ttk.Frame(self.frame)
        preview_frame.grid(row=4, column=0, sticky="nsew", pady=(0, 10))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(1, weight=1)
        zoom_row = ttk.Frame(preview_frame)
        zoom_row.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        zoom_row.columnconfigure(4, weight=1)
        ttk.Label(zoom_row, text="Zoom").grid(row=0, column=0, sticky="w")
        ttk.Button(zoom_row, text="−", width=3, command=lambda: self.adjust_zoom(-10)).grid(row=0, column=1, padx=(8, 4))
        ttk.Button(zoom_row, text="+", width=3, command=lambda: self.adjust_zoom(10)).grid(row=0, column=2, padx=(0, 8))
        ttk.Button(zoom_row, text="Reset", command=self.reset_zoom).grid(row=0, column=3)
        ttk.Label(zoom_row, textvariable=self.zoom_var, foreground="#555555").grid(row=0, column=4, sticky="w", padx=(10, 0))
        self.preview_canvas = tk.Canvas(preview_frame, background="#fafafa", highlightthickness=0, width=620, height=320)
        yscroll = ttk.Scrollbar(preview_frame, orient="vertical", command=self.preview_canvas.yview)
        xscroll = ttk.Scrollbar(preview_frame, orient="horizontal", command=self.preview_canvas.xview)
        self.preview_canvas.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        self.preview_canvas.grid(row=1, column=0, sticky="nsew")
        yscroll.grid(row=1, column=1, sticky="ns")
        xscroll.grid(row=2, column=0, sticky="ew")
        self.preview_canvas.bind("<Control-MouseWheel>", self._on_preview_zoom_wheel, add="+")
        self.preview_canvas.bind("<Control-Button-4>", lambda _event: self.adjust_zoom(10), add="+")
        self.preview_canvas.bind("<Control-Button-5>", lambda _event: self.adjust_zoom(-10), add="+")

    def _build_notes(self) -> None:
        self.notes_frame = ttk.LabelFrame(self.frame, text=self.notes_title_var.get(), padding=8)
        self.notes_title_var.trace_add("write", lambda *_: self.notes_frame.configure(text=self.notes_title_var.get()))
        self.notes_frame.grid(row=5, column=0, sticky="ew")
        self.notes_frame.columnconfigure(0, weight=1)
        self.notes_frame.rowconfigure(1, weight=1)
        self.notes_text = tk.Text(self.notes_frame, height=7, wrap="word", state="disabled", background="#fcfcfc", relief="flat")
        self.notes_text.grid(row=0, column=0, sticky="ew")
        notes_scroll = ttk.Scrollbar(self.notes_frame, orient="vertical", command=self.notes_text.yview)
        self.notes_text.configure(yscrollcommand=notes_scroll.set)
        notes_scroll.grid(row=0, column=1, sticky="ns")
        entry_frame = ttk.Frame(self.notes_frame)
        entry_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        entry_frame.columnconfigure(0, weight=1)
        self.notes_entry = tk.Text(entry_frame, height=3, wrap="word")
        self.notes_entry.grid(row=0, column=0, sticky="ew")
        entry_scroll = ttk.Scrollbar(entry_frame, orient="vertical", command=self.notes_entry.yview)
        self.notes_entry.configure(yscrollcommand=entry_scroll.set)
        entry_scroll.grid(row=0, column=1, sticky="ns")
        button_row = ttk.Frame(entry_frame)
        button_row.grid(row=1, column=0, columnspan=2, sticky="e", pady=(6, 0))
        self.add_note_button = ttk.Button(button_row, text="Add Image Note", command=self.add_image_note, state="disabled")
        self.add_note_button.grid(row=0, column=0)

    def _header_text(self) -> str:
        bits = [f"Slot {self.index + 1}"]
        if self.app.active_slot_index == self.index:
            bits.append("active")
        if self.record is not None:
            bits.append(_truncate(_record_summary(self.record), 60))
        elif self.selection.initialized or any(getattr(self.selection, field) for field in HIERARCHY_FIELDS):
            path_text = _selection_path_text(self.selection)
            if path_text:
                bits.append(_truncate(path_text, 60))
        else:
            bits.append("empty")
        return " - ".join(bits)

    def refresh_header(self) -> None:
        self.frame.configure(text=self._header_text())

    def _set_canvas_message(self, message: str) -> None:
        self.base_image = None
        self.preview_canvas.delete("all")
        self.preview_canvas.create_text(16, 16, anchor="nw", fill="#777777", text=message)
        self.preview_canvas.configure(scrollregion=self.preview_canvas.bbox("all"))
        self.photo = None

    def _render_preview_image(self) -> None:
        if self.base_image is None:
            self.photo = None
            return
        scale = max(self.zoom_percent, 1) / 100.0
        width = max(int(round(self.base_image.width * scale)), 1)
        height = max(int(round(self.base_image.height * scale)), 1)
        display_image = self.base_image
        if (width, height) != self.base_image.size:
            try:
                resample = Image.Resampling.LANCZOS
            except AttributeError:  # pragma: no cover - Pillow compatibility
                resample = Image.LANCZOS
            display_image = self.base_image.resize((width, height), resample=resample)
        self.photo = ImageTk.PhotoImage(display_image)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.preview_canvas.configure(scrollregion=(0, 0, width, height))

    def _set_canvas_image(self, image) -> None:
        self.base_image = image
        self._render_preview_image()

    def set_zoom_percent(self, percent: int) -> None:
        self.zoom_percent = max(25, min(400, int(percent)))
        self.zoom_var.set(f"{self.zoom_percent}%")
        if self.base_image is not None:
            self._render_preview_image()

    def adjust_zoom(self, delta: int) -> None:
        self.set_zoom_percent(self.zoom_percent + delta)

    def reset_zoom(self) -> None:
        self.set_zoom_percent(100)

    def _on_preview_zoom_wheel(self, event: tk.Event) -> None:
        delta = 0
        if getattr(event, "num", None) == 4:
            delta = 10
        elif getattr(event, "num", None) == 5:
            delta = -10
        elif getattr(event, "delta", 0):
            delta = 10 if event.delta > 0 else -10
        if delta:
            self.adjust_zoom(delta)

    def _set_combo(self, combo: ttk.Combobox, var: tk.StringVar, values: Sequence[str], value: str, enabled: bool) -> None:
        self._ui_guard = True
        try:
            combo.configure(values=list(values), state="readonly" if enabled else "disabled")
            var.set(value)
        finally:
            self._ui_guard = False

    def _set_pristine_state(self, records: Sequence[FigureRecord]) -> None:
        pipeline_options = field_options(records, SlotSelection(), "pipeline")
        self._set_combo(self.field_widgets["pipeline"], self.field_vars["pipeline"], pipeline_options, "", bool(pipeline_options))
        for field_name in HIERARCHY_FIELDS[1:]:
            self._set_combo(self.field_widgets[field_name], self.field_vars[field_name], [], "", False)
        self._set_combo(self.figure_combo, self.figure_var, [], "", False)
        self.record = None
        self.title_var.set("No figure selected")
        self.selection_var.set("Choose a pipeline to start.")
        self.source_var.set("")
        self.notes_title_var.set("Image Notes")
        self._set_canvas_message("Choose a pipeline to start.")
        self._render_image_notes([])
        self.add_note_button.configure(state="disabled")
        self.refresh_header()

    def _render_loading_state(self) -> None:
        self._set_combo(self.figure_combo, self.figure_var, [], "", False)
        for field_name in HIERARCHY_FIELDS:
            self._set_combo(self.field_widgets[field_name], self.field_vars[field_name], [], "", False)
        self.record = None
        self.title_var.set("Scanning results/")
        self.selection_var.set("Waiting for the catalog to load in the background.")
        self.source_var.set("")
        self.notes_title_var.set("Image Notes")
        self._set_canvas_message("Waiting for the catalog to load in the background.")
        self._render_image_notes([])
        self.add_note_button.configure(state="disabled")
        self.refresh_header()

    def _render_empty_catalog_state(self) -> None:
        self._set_combo(self.figure_combo, self.figure_var, [], "", False)
        for field_name in HIERARCHY_FIELDS:
            self._set_combo(self.field_widgets[field_name], self.field_vars[field_name], [], "", False)
        self.record = None
        self.title_var.set("No figures found")
        self.selection_var.set("results/ did not yield any figures.")
        self.source_var.set("")
        self.notes_title_var.set("Image Notes")
        self._set_canvas_message("No figures were indexed under results/")
        self._render_image_notes([])
        self.add_note_button.configure(state="disabled")
        self.refresh_header()

    def _render_no_match_state(self, records: Sequence[FigureRecord], resolved: SlotSelection) -> None:
        self._render_path_controls(records, resolved)
        self._set_combo(self.figure_combo, self.figure_var, [], "", False)
        self._figure_label_to_record = {}
        self.record = None
        self.title_var.set("No figure matched the current path")
        path_text = _selection_path_text(resolved)
        self.selection_var.set(path_text or "Choose a pipeline to start.")
        self.source_var.set("")
        self.notes_title_var.set("Image Notes")
        self._set_canvas_message(path_text or "No figure matched the current path.")
        self._render_image_notes([])
        self.add_note_button.configure(state="disabled")
        self.refresh_header()

    def _render_candidate_state(self, candidate_records: Sequence[FigureRecord], resolved: SlotSelection) -> None:
        path_text = _selection_path_text(resolved)
        self.title_var.set("Multiple figures match the current path")
        self.selection_var.set(path_text or "Choose a figure from the dropdown.")
        self.source_var.set("")
        self.figure_var.set("")
        self._figure_label_to_record = {}
        labels: List[str] = []
        seen: set[str] = set()
        for index, record in enumerate(candidate_records):
            label = _figure_choice_label(record, index, seen)
            seen.add(label)
            labels.append(label)
            self._figure_label_to_record[label] = record
        self._set_combo(self.figure_combo, self.figure_var, labels, "", bool(labels))
        self._set_canvas_message(path_text or "Choose a figure from the dropdown.")
        self._render_image_notes([])
        self.add_note_button.configure(state="disabled")
        self.refresh_header()

    def _render_path_controls(self, records: Sequence[FigureRecord], resolved: SlotSelection) -> List[FigureRecord]:
        prefix_values: Dict[str, str] = {}
        for field_name in HIERARCHY_FIELDS:
            prefix_selection = SlotSelection(
                pipeline=prefix_values.get("pipeline", ""),
                preset=prefix_values.get("preset", ""),
                split=prefix_values.get("split", ""),
                basis=prefix_values.get("basis", ""),
                family=prefix_values.get("family", ""),
                cohort=prefix_values.get("cohort", ""),
                scope=prefix_values.get("scope", ""),
                initialized=bool(prefix_values),
            )
            options = field_options(records, prefix_selection, field_name)
            current = getattr(resolved, field_name)
            if current not in options:
                current = ""
            if current:
                prefix_values[field_name] = current
            self._set_combo(self.field_widgets[field_name], self.field_vars[field_name], options, current, bool(options))
        candidate_records = [record for record in records if all(not value or getattr(record, field) == value for field, value in prefix_values.items())]
        return candidate_records

    def _render_image_notes(self, notes: Sequence[dict]) -> None:
        if self.record is None:
            self.notes_title_var.set("Image Notes")
            _set_text(self.notes_text, "Select a figure to see image notes.")
            return
        self.notes_title_var.set(f"Image Notes - {_truncate(_record_summary(self.record), 48)} ({len(notes)})")
        if notes:
            body = "\n\n".join(_note_line(note) for note in notes)
        else:
            body = "No image notes yet for this figure."
        _set_text(self.notes_text, body)

    def _refresh_image_notes(self) -> None:
        if self.record is None:
            self._render_image_notes([])
            return
        notes = self.app.notes_store.list_notes(scope=IMAGE_NOTE_SCOPE, scope_key=self.record.figure_key)
        self._render_image_notes(notes)
        self.add_note_button.configure(state="normal")

    def sync_to_records(self, *, preserve_figure_key: str = "") -> None:
        records = self.app.records
        if not records:
            if self.app.catalog_loading:
                self._render_loading_state()
            else:
                self._render_empty_catalog_state()
            return

        if not self.selection.initialized and not any(getattr(self.selection, field) for field in HIERARCHY_FIELDS) and not self.selection.figure_key:
            self._set_pristine_state(records)
            self._refresh_image_notes()
            return

        resolved, candidate_records = resolve_selection(records, self.selection, preserve_figure_key=preserve_figure_key)
        self.selection = resolved
        self._render_path_controls(records, resolved)
        current_record = next((record for record in candidate_records if record.figure_key == resolved.figure_key), None)
        if current_record is None and len(candidate_records) == 1:
            current_record = candidate_records[0]
        self.record = current_record
        if current_record is None:
            if candidate_records:
                self._render_candidate_state(candidate_records, resolved)
            else:
                self._render_no_match_state(records, resolved)
            return

        self.title_var.set(_record_summary(current_record))
        self.selection_var.set(_selection_path_text(resolved) or "")
        self.source_var.set(_relative_path_text(current_record.preview_path, self.app.repo_root))
        self.figure_var.set("")
        self._figure_label_to_record = {}
        labels: List[str] = []
        seen: set[str] = set()
        for index, record in enumerate(candidate_records):
            label = _figure_choice_label(record, index, seen)
            seen.add(label)
            labels.append(label)
            self._figure_label_to_record[label] = record
        current_label = next((label for label, record in self._figure_label_to_record.items() if record.figure_key == current_record.figure_key), labels[0] if labels else "")
        self._set_combo(self.figure_combo, self.figure_var, labels, current_label, bool(labels))
        image = image_for_preview(current_record.preview_path)
        self._set_canvas_image(image)
        self._refresh_image_notes()
        self.refresh_header()

    def _on_field_selected(self, field_name: str) -> None:
        if self._ui_guard:
            return
        value = self.field_vars[field_name].get().strip()
        previous_key = self.record.figure_key if self.record is not None else ""
        self.app.set_active_slot(self.index)
        self.selection = selection_with_field(self.selection, field_name, value)
        self.sync_to_records(preserve_figure_key=previous_key)
        self.app.slot_did_change(self)

    def _on_figure_selected(self) -> None:
        if self._ui_guard:
            return
        label = self.figure_var.get().strip()
        record = self._figure_label_to_record.get(label)
        if record is None:
            return
        self.app.set_active_slot(self.index)
        self.selection = selection_from_record(record)
        self.sync_to_records(preserve_figure_key=record.figure_key)
        self.app.slot_did_change(self)

    def load_record(self, record: FigureRecord) -> None:
        self.selection = selection_from_record(record)
        self.sync_to_records(preserve_figure_key=record.figure_key)

    def reset(self) -> None:
        self.selection = SlotSelection()
        self.sync_to_records()
        self.app.slot_did_change(self)

    def add_image_note(self) -> None:
        if self.record is None:
            return
        note_text = self.notes_entry.get("1.0", tk.END).strip()
        if not note_text:
            self.app.set_status("Type a note before saving it.")
            return
        context = self.record.as_context()
        context.update(
            {
                "slot_index": self.index,
                "slot_title": self.frame.cget("text"),
                "selection_path": _selection_path_text(self.selection),
                "comparison_signature": comparison_signature(self.app.current_comparison_records()),
            }
        )
        self.app.notes_store.add_note(scope=IMAGE_NOTE_SCOPE, scope_key=self.record.figure_key, note_text=note_text, context=context)
        self.notes_entry.delete("1.0", tk.END)
        self._refresh_image_notes()
        self.app.set_status("Saved image note.")


class ComparisonNotesPanel:
    def __init__(self, app: "FigureViewerApp", parent: tk.Widget) -> None:
        self.app = app
        self.frame = ttk.LabelFrame(parent, text="Comparison Notes", padding=10)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(1, weight=1)
        self.title_var = tk.StringVar(value="No comparison selected")
        ttk.Label(self.frame, textvariable=self.title_var, wraplength=760, justify="left", font=("TkDefaultFont", 11, "bold")).grid(row=0, column=0, sticky="w")
        self.summary_text = tk.Text(self.frame, height=9, wrap="word", state="disabled", background="#fcfcfc", relief="flat")
        self.summary_text.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        summary_scroll = ttk.Scrollbar(self.frame, orient="vertical", command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=summary_scroll.set)
        summary_scroll.grid(row=1, column=1, sticky="ns", pady=(6, 0))
        entry_frame = ttk.Frame(self.frame)
        entry_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        entry_frame.columnconfigure(0, weight=1)
        self.entry_text = tk.Text(entry_frame, height=3, wrap="word")
        self.entry_text.grid(row=0, column=0, sticky="ew")
        entry_scroll = ttk.Scrollbar(entry_frame, orient="vertical", command=self.entry_text.yview)
        self.entry_text.configure(yscrollcommand=entry_scroll.set)
        entry_scroll.grid(row=0, column=1, sticky="ns")
        button_row = ttk.Frame(entry_frame)
        button_row.grid(row=1, column=0, columnspan=2, sticky="e", pady=(6, 0))
        self.add_button = ttk.Button(button_row, text="Add Comparison Note", command=self.add_note)
        self.add_button.grid(row=0, column=0)
        self.entry_text.bind("<FocusIn>", lambda _event: self.app.set_status("Editing comparison notes."), add="+")

    def refresh(self) -> None:
        records = self.app.current_comparison_records()
        if not records:
            if self.app.catalog_loading:
                self.title_var.set("Comparison Notes")
                _set_text(self.summary_text, "Waiting for the catalog to finish loading before comparisons are available.")
            else:
                self.title_var.set("No comparison selected")
                _set_text(self.summary_text, "Add slots and select figures to build a comparison.\n\nComparison notes are attached to the ordered set of slot figures.")
            self.add_button.configure(state="disabled")
            return

        label = comparison_label(records)
        signature = comparison_signature(records)
        notes = self.app.notes_store.list_notes(scope=COMPARISON_NOTE_SCOPE, scope_key=signature)
        lines: List[str] = [f"Comparison: {label}", f"Signature: {signature}", "", "Selected figures:"]
        for index, record in enumerate(records, start=1):
            lines.append(f"{index}. {_record_summary(record)}")
            lines.append(f"   {_relative_path_text(record.preview_path, self.app.repo_root)}")
        lines.append("")
        if notes:
            lines.append(f"Notes ({len(notes)}):")
            lines.extend(f"- {_note_line(note)}" for note in notes)
        else:
            lines.append("No comparison notes yet.")
        self.title_var.set(f"Comparison Notes - {label}")
        _set_text(self.summary_text, "\n".join(lines))
        self.add_button.configure(state="normal")

    def add_note(self) -> None:
        records = self.app.current_comparison_records()
        if not records:
            return
        note_text = self.entry_text.get("1.0", tk.END).strip()
        if not note_text:
            self.app.set_status("Type a comparison note before saving it.")
            return
        signature = comparison_signature(records)
        context = {
            "comparison_label": comparison_label(records),
            "comparison_signature": signature,
            "slot_count": len(records),
            "slot_records": [record.as_context() for record in records],
        }
        self.app.notes_store.add_note(scope=COMPARISON_NOTE_SCOPE, scope_key=signature, note_text=note_text, context=context)
        self.entry_text.delete("1.0", tk.END)
        self.refresh()
        self.app.set_status("Saved comparison note.")


class FigureViewerApp:
    def __init__(
        self,
        *,
        repo_root: Path,
        notes_db_path: Path | None = None,
        summary_depth_limit: int = DEFAULT_RESULTS_DEPTH,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.summary_depth_limit = summary_depth_limit
        self.notes_store = NoteStore(notes_db_path or _default_notes_db_path(self.repo_root))
        self.records: List[FigureRecord] = []
        self.browser_root: BrowserNode | None = None
        self.catalog_loading = True
        self.catalog_generation = 0
        self.closing = False
        self.active_slot_index: int | None = None
        self.slots: List[SlotView] = []
        self.slot_rows: List[List[SlotView]] = []
        self.slot_row_frames: List[ttk.Frame] = []
        self._row_column_extents: Dict[ttk.Frame, int] = {}
        self._slot_row_extent = -1
        self.startup_loading_window: StartupProgressWindow | None = None
        self._ui_queue: Queue[Callable[[], None]] = Queue()

        self.root = tk.Tk()
        self.slot_side_var = tk.StringVar(master=self.root, value="Right")
        self.root.title("Figure Viewer")
        self.root.geometry("1500x960")
        self.root.minsize(1220, 800)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.status_var = tk.StringVar(value="Launching figure viewer...")
        self.catalog_status_var = tk.StringVar(value="Scanning results/ in the background...")

        self._build_ui()
        self.startup_loading_window = StartupProgressWindow(self.root)
        self.startup_loading_window.begin_scan(detail="Preparing the startup scan...")
        self.root.after(50, self._drain_ui_queue)
        self.root.after_idle(self.refresh_catalog)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)

        outer = ttk.Panedwindow(self.root, orient="horizontal")
        outer.grid(row=0, column=0, sticky="nsew")

        self.explorer_panel = ExplorerPanel(self, outer)
        outer.add(self.explorer_panel, weight=1)

        right = ttk.Frame(outer, padding=10)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        outer.add(right, weight=3)

        toolbar = ttk.Frame(right)
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(0, weight=1)
        ttk.Label(toolbar, text="Comparison Workspace", font=("TkDefaultFont", 11, "bold")).grid(row=0, column=0, sticky="w")
        button_row = ttk.Frame(toolbar)
        button_row.grid(row=0, column=1, sticky="e")
        ttk.Label(button_row, text="Side").grid(row=0, column=0, padx=(0, 6))
        self.slot_side_combo = ttk.Combobox(
            button_row,
            textvariable=self.slot_side_var,
            values=["Right", "Bottom", "Left", "Top"],
            state="readonly",
            width=8,
        )
        self.slot_side_combo.grid(row=0, column=1, padx=(0, 8))
        ttk.Button(button_row, text="Add Slot", command=self.add_slot).grid(row=0, column=2, padx=(0, 8))
        ttk.Button(button_row, text="Refresh Catalog", command=self.refresh_catalog).grid(row=0, column=3)

        self.right_paned = ttk.Panedwindow(right, orient="vertical")
        self.right_paned.grid(row=1, column=0, sticky="nsew", pady=(10, 0))

        self.workspace_frame = ttk.Frame(self.right_paned)
        self.workspace_frame.columnconfigure(0, weight=1)
        self.workspace_frame.rowconfigure(0, weight=1)
        self.right_paned.add(self.workspace_frame, weight=4)

        self.slots_stack = ttk.Frame(self.workspace_frame)
        self.slots_stack.grid(row=0, column=0, sticky="nsew")
        self.slots_stack.columnconfigure(0, weight=1)
        self.slots_stack.rowconfigure(0, weight=1)

        self.slots_placeholder = ttk.Label(
            self.slots_stack,
            text="No slots yet. Add a slot to start comparing figures.",
            padding=18,
            anchor="center",
            justify="center",
        )
        self.slots_placeholder.grid(row=0, column=0, sticky="nsew")

        self.slots_container = ScrollableFrame(self.slots_stack)
        self.slots_container.grid(row=0, column=0, sticky="nsew")
        self.slots_container.inner.columnconfigure(0, weight=1)
        self.slots_placeholder.tkraise()

        self.comparison_panel = ComparisonNotesPanel(self, self.right_paned)
        self.right_paned.add(self.comparison_panel.frame, weight=1)

        status = ttk.Label(self.root, textvariable=self.status_var, anchor="w", padding=(10, 6))
        status.grid(row=1, column=0, sticky="ew")

    def run(self) -> int:
        self.root.mainloop()
        return 0

    def close(self) -> None:
        if self.closing:
            return
        self.closing = True
        try:
            if self.startup_loading_window is not None:
                self.startup_loading_window.destroy()
        finally:
            try:
                self.notes_store.close()
            finally:
                self.root.destroy()

    def set_status(self, message: str) -> None:
        if not self.closing:
            self.status_var.set(message)

    def _drain_ui_queue(self) -> None:
        if self.closing:
            return
        try:
            if not self.root.winfo_exists():
                return
        except tk.TclError:
            return
        while True:
            try:
                callback = self._ui_queue.get_nowait()
            except Empty:
                break
            try:
                callback()
            except Exception:
                pass
        if not self.closing:
            try:
                self.root.after(50, self._drain_ui_queue)
            except tk.TclError:
                pass

    def _schedule_ui(self, callback) -> None:
        if self.closing:
            return
        self._ui_queue.put(callback)

    def _show_startup_loading(self, *, detail: str | None = None) -> None:
        if self.startup_loading_window is None:
            return
        self.startup_loading_window.begin_scan(detail=detail or "Preparing the startup scan...")

    def _update_startup_progress(self, progress: CatalogScanProgress) -> None:
        if self.startup_loading_window is None:
            return
        self.startup_loading_window.update_progress(progress)
        if progress.total > 0:
            current = min(max(progress.current, 0), progress.total)
            remaining = max(progress.total - current, 0)
            self.catalog_status_var.set(
                f"{progress.phase}: {current:,}/{progress.total:,} scanned, {remaining:,} remaining"
            )
            self.set_status(self.catalog_status_var.get())
        else:
            self.catalog_status_var.set(progress.message or progress.phase or "Scanning results/")
            self.set_status(self.catalog_status_var.get())

    def _catalog_progress(self, generation: int, progress: CatalogScanProgress) -> None:
        if self.closing or generation != self.catalog_generation:
            return
        self._update_startup_progress(progress)

    def refresh_catalog(self) -> None:
        self.catalog_generation += 1
        generation = self.catalog_generation
        self.catalog_loading = True
        self._show_startup_loading(detail="Scanning results/ in the background...")
        self.catalog_status_var.set("Scanning results/ in the background...")
        self.set_status("Scanning results/ in the background...")

        def on_progress(progress: CatalogScanProgress) -> None:
            self._schedule_ui(lambda progress=progress, generation=generation: self._catalog_progress(generation, progress))

        def worker() -> None:
            try:
                records = discover_figure_records(
                    repo_root=self.repo_root,
                    include_review_figures=False,
                    summary_depth_limit=self.summary_depth_limit,
                    progress_callback=on_progress,
                )
                browser_root = build_results_index(records, self.repo_root)
            except Exception as exc:  # pragma: no cover - error path depends on local data
                self._schedule_ui(lambda exc=exc, generation=generation: self._catalog_failed(generation, exc))
                return
            self._schedule_ui(lambda records=records, browser_root=browser_root, generation=generation: self._catalog_loaded(generation, records, browser_root))

        threading.Thread(target=worker, daemon=True).start()

    def _catalog_failed(self, generation: int, exc: Exception) -> None:
        if self.closing or generation != self.catalog_generation:
            return
        self.catalog_loading = False
        if self.startup_loading_window is not None:
            self.startup_loading_window.hide()
        self.catalog_status_var.set("Failed to load results catalog.")
        self.set_status(f"Failed to load results catalog: {exc}")
        messagebox.showerror("Figure Viewer", f"Could not load the results catalog:\n\n{exc}")
        for slot in self.slots:
            slot.sync_to_records()
        self.comparison_panel.refresh()

    def _catalog_loaded(self, generation: int, records: Sequence[FigureRecord], browser_root: BrowserNode | None) -> None:
        if self.closing or generation != self.catalog_generation:
            return
        self.catalog_loading = False
        if self.startup_loading_window is not None:
            self.startup_loading_window.hide()
        self.records = list(records)
        self.browser_root = browser_root
        self.catalog_status_var.set(f"{len(records)} figures indexed under results/")
        self.set_status(f"Loaded {len(records)} figures from results/")
        self.explorer_panel.set_catalog(browser_root, len(records))
        for slot in self.slots:
            slot.sync_to_records(preserve_figure_key=slot.record.figure_key if slot.record is not None else "")
        self.comparison_panel.refresh()

    def _normalized_slot_side(self, side: str | None = None) -> str:
        value = (side if side is not None else self.slot_side_var.get() or "").strip().lower()
        if value not in {"top", "bottom", "left", "right"}:
            return "right"
        return value

    def _create_slot_row_frame(self) -> ttk.Frame:
        row_frame = ttk.Frame(self.slots_container.inner)
        row_frame.columnconfigure(0, weight=1)
        row_frame.rowconfigure(0, weight=1)
        return row_frame

    def _slot_location(self, slot: SlotView) -> tuple[int, int] | None:
        for row_index, row in enumerate(self.slot_rows):
            for column_index, candidate in enumerate(row):
                if candidate is slot:
                    return row_index, column_index
        return None

    def _rebuild_slot_layout(self) -> None:
        if not self.slot_rows:
            self.slots = []
            self._slot_row_extent = -1
            self.slots_placeholder.tkraise()
            return

        self.slots_container.tkraise()
        cleared_row_limit = max(self._slot_row_extent, len(self.slot_rows) - 1)
        for row_index in range(cleared_row_limit + 1):
            self.slots_container.inner.rowconfigure(row_index, weight=0)
        self.slots_container.inner.columnconfigure(0, weight=1)

        flattened: List[SlotView] = []
        for row_index, (row_frame, row) in enumerate(zip(self.slot_row_frames, self.slot_rows)):
            row_frame.grid(row=row_index, column=0, sticky="ew", pady=(0, 10 if row_index < len(self.slot_rows) - 1 else 0))
            row_frame.rowconfigure(0, weight=1)
            cleared_col_limit = max(self._row_column_extents.get(row_frame, -1), len(row) - 1)
            for column_index in range(cleared_col_limit + 1):
                row_frame.columnconfigure(column_index, weight=0)
            for column_index, slot in enumerate(row):
                row_frame.columnconfigure(column_index, weight=1)
                slot.frame.grid(row=0, column=column_index, sticky="nsew", padx=(0, 8 if column_index < len(row) - 1 else 0))
                flattened.append(slot)
            self._row_column_extents[row_frame] = max(len(row) - 1, 0)
            self.slots_container.inner.rowconfigure(row_index, weight=1)

        self._slot_row_extent = len(self.slot_rows) - 1
        self.slots = flattened
        for index, slot in enumerate(self.slots):
            slot.index = index
            slot.refresh_header()

    def add_slot(self, initial_record: FigureRecord | None = None) -> SlotView:
        side = self._normalized_slot_side()
        anchor = self.active_slot() or (self.slots[-1] if self.slots else None)
        anchor_location = self._slot_location(anchor) if anchor is not None else None

        if not self.slot_rows or anchor_location is None:
            row_frame = self._create_slot_row_frame()
            slot = SlotView(self, row_frame, len(self.slots))
            self.slot_rows.append([slot])
            self.slot_row_frames.append(row_frame)
        else:
            anchor_row_index, anchor_column_index = anchor_location
            if side == "top":
                row_frame = self._create_slot_row_frame()
                slot = SlotView(self, row_frame, len(self.slots))
                self.slot_rows.insert(anchor_row_index, [slot])
                self.slot_row_frames.insert(anchor_row_index, row_frame)
            elif side == "bottom":
                row_frame = self._create_slot_row_frame()
                slot = SlotView(self, row_frame, len(self.slots))
                self.slot_rows.insert(anchor_row_index + 1, [slot])
                self.slot_row_frames.insert(anchor_row_index + 1, row_frame)
            else:
                row_frame = self.slot_row_frames[anchor_row_index]
                row = self.slot_rows[anchor_row_index]
                insert_index = anchor_column_index if side == "left" else anchor_column_index + 1
                slot = SlotView(self, row_frame, len(self.slots))
                row.insert(insert_index, slot)

        self._rebuild_slot_layout()
        self.set_active_slot(slot.index)
        if initial_record is not None:
            slot.load_record(initial_record)
        self.comparison_panel.refresh()
        return slot

    def remove_slot(self, slot: SlotView) -> None:
        location = self._slot_location(slot)
        if location is None:
            return
        old_index = self.slots.index(slot)
        row_index, column_index = location
        row = self.slot_rows[row_index]
        row.pop(column_index)
        if not row:
            row_frame = self.slot_row_frames.pop(row_index)
            self.slot_rows.pop(row_index)
            self._row_column_extents.pop(row_frame, None)
            row_frame.destroy()
        else:
            slot.frame.destroy()

        self._rebuild_slot_layout()
        if not self.slots:
            self.active_slot_index = None
            self.slots_placeholder.tkraise()
        else:
            self.set_active_slot(min(old_index, len(self.slots) - 1))
        self.comparison_panel.refresh()
        self.set_status("Removed a slot.")

    def set_active_slot(self, index: int | None) -> None:
        if index is None or not self.slots:
            self.active_slot_index = None
        else:
            self.active_slot_index = max(0, min(index, len(self.slots) - 1))
        for slot in self.slots:
            slot.refresh_header()

    def active_slot(self) -> SlotView | None:
        if self.active_slot_index is None:
            return None
        if 0 <= self.active_slot_index < len(self.slots):
            return self.slots[self.active_slot_index]
        return None

    def _slot_for_addition(self) -> SlotView | None:
        slot = self.active_slot()
        if slot is not None and slot.record is None:
            return slot
        for candidate in self.slots:
            if candidate.record is None:
                return candidate
        return None

    def load_record_into_slot(self, record: FigureRecord) -> None:
        slot = self._slot_for_addition()
        if slot is None:
            slot = self.add_slot()
        slot.load_record(record)
        self.set_active_slot(slot.index)
        self.slot_did_change(slot)
        self.set_status(f"Added {record.title or record.preview_path.name} to Slot {slot.index + 1}.")

    def slot_did_change(self, _slot: SlotView) -> None:
        self.comparison_panel.refresh()

    def current_comparison_records(self) -> List[FigureRecord]:
        return [slot.record for slot in self.slots if slot.record is not None]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Native results-first figure viewer")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2], help="Repository root containing results/")
    parser.add_argument("--notes-db", type=Path, default=None, help="Path to the note database")
    parser.add_argument("--summary-depth-limit", type=int, default=DEFAULT_RESULTS_DEPTH, help="Maximum depth within figures/ to index summary outputs")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    repo_root = Path(args.repo_root).resolve()
    notes_db = Path(args.notes_db).resolve() if args.notes_db is not None else _default_notes_db_path(repo_root)
    try:
        app = FigureViewerApp(repo_root=repo_root, notes_db_path=notes_db, summary_depth_limit=args.summary_depth_limit)
    except tk.TclError as exc:  # pragma: no cover - depends on local desktop session
        print(f"Could not launch the native viewer: {exc}", file=sys.stderr)
        return 1
    return app.run()
