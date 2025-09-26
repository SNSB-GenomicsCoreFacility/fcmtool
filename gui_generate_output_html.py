#!/usr/bin/env python3
import os
import sys
import threading
import subprocess
import shutil
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

APP_TITLE = "Species HTML Report – GUI Wrapper"

def default_script_path():
    # Prefer generate_species_html.py next to this file; fall back to cwd
    here = Path(__file__).resolve().parent
    candidate = here / "generate_output_html.py"
    if candidate.is_file():
        return str(candidate)
    # Last resort: let user pick later
    return ""

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("720x520")
        self.minsize(680, 500)

        self.script_path_var = tk.StringVar(value=default_script_path())
        self.data_folder_var = tk.StringVar()
        self.cluster_csv_var = tk.StringVar()
        self.yaml_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True)

        # Row: path to generate_species_html.py
        self._path_row(frm, "Generator script (unchanged):", self.script_path_var,
                       browse_cmd=self._browse_script, row=0)

        ttk.Separator(frm).grid(row=1, column=0, columnspan=3, sticky="ew", **pad)

        # Inputs
        self._path_row(frm, "Data folder:", self.data_folder_var,
                       browse_cmd=self._browse_data_folder, row=2)

        self._path_row(frm, "Cluster CSV:", self.cluster_csv_var,
                       browse_cmd=self._browse_cluster_csv, row=3)

        self._path_row(frm, "Config YAML:", self.yaml_var,
                       browse_cmd=self._browse_yaml, row=4)

        ttk.Separator(frm).grid(row=5, column=0, columnspan=3, sticky="ew", **pad)

        # Output directory (destination for final HTML files)
        self._path_row(frm, "Output directory:", self.output_dir_var,
                       browse_cmd=self._browse_output_dir, row=6)

        # Run / Quit buttons
        btns = ttk.Frame(frm)
        btns.grid(row=7, column=0, columnspan=3, sticky="e", **pad)

        self.run_btn = ttk.Button(btns, text="Run", command=self._start_run)
        self.run_btn.grid(row=0, column=0, padx=(0, 6))

        self.quit_btn = ttk.Button(btns, text="Quit", command=self.destroy)
        self.quit_btn.grid(row=0, column=1)

        # Log box
        ttk.Label(frm, text="Log:").grid(row=8, column=0, sticky="w", **pad)
        self.log = tk.Text(frm, height=14, wrap="word")
        self.log.grid(row=9, column=0, columnspan=3, sticky="nsew", **pad)
        self.log.configure(state="disabled")

        # Make the lower area stretch
        frm.rowconfigure(9, weight=1)
        frm.columnconfigure(1, weight=1)

    def _path_row(self, parent, label, var, browse_cmd, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=1, sticky="ew", padx=6, pady=4)
        parent.columnconfigure(1, weight=1)
        ttk.Button(parent, text="Browse…", command=browse_cmd).grid(row=row, column=2, sticky="e")

    # Browse handlers
    def _browse_script(self):
        p = filedialog.askopenfilename(title="Select generate_output_html.py",
                                       filetypes=[("Python", "*.py"), ("All files", "*.*")])
        if p:
            self.script_path_var.set(p)

    def _browse_data_folder(self):
        p = filedialog.askdirectory(title="Select data folder")
        if p:
            self.data_folder_var.set(p)

    def _browse_cluster_csv(self):
        p = filedialog.askopenfilename(title="Select cluster CSV",
                                       filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if p:
            self.cluster_csv_var.set(p)

    def _browse_yaml(self):
        p = filedialog.askopenfilename(title="Select config YAML",
                                       filetypes=[("YAML", "*.yml *.yaml"), ("All files", "*.*")])
        if p:
            self.yaml_var.set(p)

    def _browse_output_dir(self):
        p = filedialog.askdirectory(title="Select output directory for HTML")
        if p:
            self.output_dir_var.set(p)

    # Logging
    def _log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")
        self.update_idletasks()

    # Validation
    def _validate(self):
        script = Path(self.script_path_var.get()).expanduser()
        data_dir = Path(self.data_folder_var.get()).expanduser()
        cluster = Path(self.cluster_csv_var.get()).expanduser()
        yaml_p = Path(self.yaml_var.get()).expanduser()
        out_dir = Path(self.output_dir_var.get()).expanduser()

        errs = []
        if not script.is_file():
            errs.append("Generator script not found.")
        if not data_dir.is_dir():
            errs.append("Data folder does not exist.")
        if not cluster.is_file():
            errs.append("Cluster CSV not found.")
        if not yaml_p.is_file():
            errs.append("Config YAML not found.")
        if not out_dir:
            errs.append("Please choose an output directory.")
        else:
            out_dir.mkdir(parents=True, exist_ok=True)

        if errs:
            messagebox.showerror("Validation error", "\n".join(errs))
            return None
        return script, data_dir, cluster, yaml_p, out_dir

    def _set_running(self, running: bool):
        self.run_btn.configure(state="disabled" if running else "normal")
        self.quit_btn.configure(state="disabled" if running else "normal")

    def _start_run(self):
        vals = self._validate()
        if not vals:
            return
        script, data_dir, cluster, yaml_p, out_dir = vals

        # Run in a separate thread to avoid freezing the UI
        t = threading.Thread(target=self._run_pipeline, args=(script, data_dir, cluster, yaml_p, out_dir), daemon=True)
        self._set_running(True)
        t.start()

    def _run_pipeline(self, script, data_dir, cluster, yaml_p, out_dir):
        try:
            self._log("Starting generator script…")
            cmd = [sys.executable, str(script), str(data_dir), str(cluster), str(yaml_p)]
            self._log(f"$ {' '.join(cmd)}")

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(script.parent)
            )

            if proc.stdout:
                for line in proc.stdout.splitlines():
                    self._log(line)
            if proc.stderr:
                for line in proc.stderr.splitlines():
                    self._log("[stderr] " + line)

            if proc.returncode != 0:
                messagebox.showerror("Run failed", f"The script exited with status {proc.returncode}. See log for details.")
                return

            # After success, copy generated HTMLs from <data_folder>/species_html/ to chosen output dir
            src_dir = Path(data_dir) / "species_html"
            if not src_dir.is_dir():
                # The original script always writes here; if missing, warn.
                messagebox.showwarning("No output found", f"Expected output directory not found:\n{src_dir}")
                return

            htmls = list(src_dir.glob("*.html"))
            if not htmls:
                messagebox.showwarning("Nothing to copy", f"No HTML files found in:\n{src_dir}")
                return

            self._log(f"Copying {len(htmls)} file(s) to: {out_dir}")
            for f in htmls:
                dest = out_dir / f.name
                shutil.copy2(f, dest)
                self._log(f"Copied: {f.name}")

            messagebox.showinfo("Done", f"Finished. Copied {len(htmls)} HTML file(s) to:\n{out_dir}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self._set_running(False)


if __name__ == "__main__":
    # On macOS, make sure Tk appears in front
    try:
        # Optional nicety; doesn't fail if unavailable
        from ctypes import cdll
        try:
            cdll.LoadLibrary("/System/Library/Frameworks/AppKit.framework/AppKit")
        except Exception:
            pass
    except Exception:
        pass

    app = App()
    app.mainloop()
