from __future__ import annotations

import json
import subprocess
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .demo_pipeline import build_demo
from .recipe_schema import default_recipe, load_recipe
from .tmux_runner import launch


class DemoBuilderGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Synthetic demo builder")
        self.geometry("760x560")
        self.recipe = default_recipe()
        self.output_dir = tk.StringVar(value=str(Path("/tmp/codex_demo")))
        self.status = tk.StringVar(value="Ready")
        self._build_widgets()

    def _build_widgets(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)
        ttk.Label(root, text="Codex demo builder", font=("TkDefaultFont", 16, "bold")).pack(anchor="w")
        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, pady=10)
        effects = ttk.Frame(notebook, padding=10)
        topology = ttk.Frame(notebook, padding=10)
        style = ttk.Frame(notebook, padding=10)
        notebook.add(effects, text="Effects")
        notebook.add(topology, text="Topology")
        notebook.add(style, text="Trace style")
        self.basal = tk.DoubleVar(value=2.0)
        self.apical = tk.DoubleVar(value=0.5)
        self.soma1 = tk.DoubleVar(value=3.0)
        self.soma2 = tk.DoubleVar(value=1.0 / 3.0)
        for row, (label, variable) in enumerate((("Basal scale", self.basal), ("Apical scale", self.apical), ("Soma 1 scale", self.soma1), ("Soma 2 scale", self.soma2))):
            ttk.Label(effects, text=label).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Entry(effects, textvariable=variable, width=12).grid(row=row, column=1, sticky="w", pady=4)
        ttk.Label(topology, text="bouton_1 -> soma_1\nbouton_2 -> independent\nbouton_3 -> soma_2").pack(anchor="w")
        self.noise = tk.DoubleVar(value=0.0)
        self.seed = tk.IntVar(value=12345)
        for row, (label, variable) in enumerate((("Noise", self.noise), ("Seed", self.seed))):
            ttk.Label(style, text=label).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Entry(style, textvariable=variable, width=12).grid(row=row, column=1, sticky="w", pady=4)
        controls = ttk.Frame(root)
        controls.pack(fill="x")
        ttk.Label(controls, text="Output").pack(side="left")
        ttk.Entry(controls, textvariable=self.output_dir, width=48).pack(side="left", padx=6)
        ttk.Button(controls, text="Browse", command=self._browse).pack(side="left")
        for label, command in (("Build", self._build), ("Validate", self._validate), ("Run both", self._run)):
            ttk.Button(controls, text=label, command=command).pack(side="left", padx=3)
        ttk.Label(root, textvariable=self.status).pack(anchor="w", pady=6)

    def _browse(self) -> None:
        selected = filedialog.askdirectory()
        if selected:
            self.output_dir.set(selected)

    def _current_recipe(self):
        recipe = json.loads(json.dumps(self.recipe))
        recipe["seed"] = int(self.seed.get())
        recipe["trace_style"]["noise"] = float(self.noise.get())
        recipe["dendrites"]["basal"]["activity_scale"] = float(self.basal.get())
        recipe["dendrites"]["basal"]["frequency_scale"] = float(self.basal.get())
        recipe["dendrites"]["apical"]["activity_scale"] = float(self.apical.get())
        recipe["dendrites"]["apical"]["frequency_scale"] = float(self.apical.get())
        recipe["somas"]["soma_1"]["activity_scale"] = float(self.soma1.get())
        recipe["somas"]["soma_1"]["frequency_scale"] = float(self.soma1.get())
        recipe["somas"]["soma_2"]["activity_scale"] = float(self.soma2.get())
        recipe["somas"]["soma_2"]["frequency_scale"] = float(self.soma2.get())
        return recipe

    def _build(self) -> None:
        try:
            build_demo(self._current_recipe(), Path(self.output_dir.get()))
            self.status.set("Demo built")
        except Exception as exc:
            messagebox.showerror("Build failed", str(exc))

    def _validate(self) -> None:
        command = ["python3", "-m", "analysis.demo_pipeline.demo_pipeline", "validate", "--output-dir", self.output_dir.get()]
        completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[2], check=False)
        self.status.set("Validation passed" if completed.returncode == 0 else "Validation failed")

    def _run(self) -> None:
        try:
            self._build()
            launch(Path(self.output_dir.get()), "dendrites")
            launch(Path(self.output_dir.get()), "soma_bouton")
            self.status.set("tmux sessions started")
        except Exception as exc:
            messagebox.showerror("Run failed", str(exc))


def main() -> int:
    DemoBuilderGUI().mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
