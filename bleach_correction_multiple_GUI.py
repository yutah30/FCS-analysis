"""
bleach_correction_GUI.py

LSM fluorescence bleaching correction GUI
- Select multiple .lsm files
- Perform photobleaching correction using double-exponential fitting
- Save corrected data (.txt), fit plots (fit + residuals, .pdf), and fit parameters (.csv)
"""

import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tifffile import imread
import pandas as pd
import czifile


# =========================================================
# Exponential models
# =========================================================
def multi_exp(t, *params):
    """
    Multi-exponential decay model.

    params = [A1, tau1, A2, tau2, ..., An, taun, C]
    """
    C = params[-1]
    n = (len(params) - 1) // 2

    y = np.zeros_like(t, dtype=float)
    for i in range(n):
        A = params[2 * i]
        tau = params[2 * i + 1]
        y += A * np.exp(-t / tau)

    return y + C

def correct_bleaching_multi_exponential(
    data, scan_time_ms,
    n_exp, base_name,
    output_dir, save_plot=True
):
    """
    Correct photobleaching using multi-exponential fitting (2–4 components).

    Parameters:
        data (ndarray): 2D array (time x space)
        n_exp (int): number of exponential components (2–4)

    Returns:
        ndarray: bleach-corrected data
    """
    if n_exp < 2 or n_exp > 4:
        raise ValueError("n_exp must be between 2 and 4.")

    frames, _ = data.shape
    t = np.arange(frames) * scan_time_ms

    mean_trace = np.nanmean(data, axis=1)

    valid = ~np.isnan(mean_trace)
    t_fit = t[valid]
    y_fit = mean_trace[valid]

    # ===== 初期値自動生成 =====
    total_drop = mean_trace[0] - mean_trace[-1]

    # 振幅：等分 or やや重み付き
    A_inits = np.linspace(0.5, 0.1, n_exp)
    A_inits = A_inits / A_inits.sum() * total_drop

    # 時定数：logスケールで分散
    tau_inits = np.logspace(
        np.log10(frames / 20000),
        np.log10(frames / 50),
        n_exp
    )

    C_init = mean_trace[-1]

    p0 = []
    for A, tau in zip(A_inits, tau_inits):
        p0.extend([A, tau])
    p0.append(C_init)

    # ===== フィッティング =====
    try:
        popt, pcov = curve_fit(
            multi_exp,
            t_fit,
            y_fit,
            p0=p0,
            maxfev=30000
        )
        perr = np.sqrt(np.abs(np.diag(pcov))) if pcov is not None else np.array([np.nan,np.nan,np.nan])
        fit_curve = multi_exp(t, *popt)
        residuals = mean_trace - fit_curve
        SSR = np.sum(residuals**2)
        MSE = np.mean(residuals**2)
        R2 = 1 - SSR / np.sum((mean_trace - np.mean(mean_trace))**2)
        print(f"✅ Fit success: {base_name}")

    except RuntimeError:
        print(f"⚠️ Fit failed for {base_name}. Using original data.")
        return data, None

    # ===== フィット結果 =====
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        fig, axs = plt.subplots(2, 1, figsize=(7, 6), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)

        # --- Upper: fit curve ---
        axs[0].plot(t*1e-3, mean_trace, 'b-', label='Measured')
        axs[0].plot(t*1e-3, fit_curve, 'r--', label=f"{n_exp}-exp fit")
        axs[0].set_ylabel("Mean Intensity")
        axs[0].legend()
        axs[0].grid(True)
        axs[0].set_title("Photobleaching Fit and Residuals")

        # --- Lower: residuals ---
        axs[1].plot(t*1e-3, residuals, color='gray')
        axs[1].axhline(0, color='black', linestyle='--', linewidth=0.8)
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Residuals")
        axs[1].grid(True)

        plt.tight_layout()
        pdf_path = os.path.join(output_dir, f"{base_name}_fit_bleaching.pdf")
        plt.savefig(pdf_path, dpi=300)
        plt.close(fig)
        print(f"📄 Saved fit+residuals PDF: {pdf_path}")

    # ===== 補正 =====
    correction = fit_curve[:, np.newaxis]
    correction0 = correction[0]
    sqrt_ratio = np.sqrt(correction / correction0)

    corrected_data = data / sqrt_ratio + correction0 * (1 - sqrt_ratio)

    # ===== 補正前後比較 =====
    if save_plot:
        corrected_trace = np.nanmean(corrected_data, axis=1)
        plt.figure(figsize=(7,4))
        plt.plot(t*1e-3, mean_trace, 'r', label="Before")
        plt.plot(t*1e-3, corrected_trace, 'g', label="After")
        plt.xlabel("Time (s)")
        plt.ylabel("Mean intensity")
        plt.title("Photobleaching Correction: Comparison")
        plt.legend()
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{base_name}_fit_comparison.pdf")
        plt.savefig(save_path, dpi=300)
        plt.close()
    # --- fit results dict ---
    param_names = []
    values = []
    stderrs = []

    for i in range(n_exp):
        param_names.extend([f"A{i+1}", f"tau{i+1}"])
        values.extend([popt[2*i], popt[2*i+1]])
        stderrs.extend([perr[2*i], perr[2*i+1]])

    param_names.append("C")
    values.append(popt[-1])
    stderrs.append(perr[-1])

    # metrics
    param_names.extend(["SSR", "MSE", "R2"])
    values.extend([SSR, MSE, R2])
    stderrs.extend([np.nan, np.nan, np.nan])

    fit_results = pd.DataFrame({
        "Parameter": param_names,
        "Value": values,
        "StdErr": stderrs
    })

    return corrected_data, fit_results

# =========================================================
# Extract scan_time_ms from filename
# =========================================================
def extract_scan_time_ms(filename):
    match = re.search(r"_(\d+(?:\.\d+)?)ms", filename)
    if match:
        return float(match.group(1))
    else:
        print(f"⚠️ ファイル名にスキャン時間が見つかりません: {filename}")
        return None


# =========================================================
# GUI class
# =========================================================
class BleachCorrectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("LSM Photobleaching Correction Tool")
        self.root.geometry("660x500")

        self.file_type_var = tk.StringVar(value="lsm")
        self.files = []
        self.output_dir = ""
        self.n_exp_var = tk.IntVar(value=2)  # 2,3,4 から選択
        self.use_scan_time_var = tk.BooleanVar(value=False)
        self.scan_time_var = tk.DoubleVar(value=0.763)
        self.build_gui()

    def build_gui(self):
        frm = tk.Frame(self.root, padx=12, pady=12)
        frm.pack(fill="both", expand=True)

        tk.Label(frm, text="① Select file type", font=("Arial", 11, "bold")).pack(anchor="w")
        setup_frame = tk.Frame(frm)
        setup_frame.pack(anchor="w", pady=6)
        tk.Radiobutton(setup_frame, text=".lsm", variable=self.file_type_var, value="lsm").pack(side="left", padx=6)
        tk.Radiobutton(setup_frame, text=".czi", variable=self.file_type_var, value="czi").pack(side="left", padx=6)
        tk.Radiobutton(setup_frame, text=".txt", variable=self.file_type_var, value="txt").pack(side="left", padx=6)
        tk.Radiobutton(setup_frame, text=".tif", variable=self.file_type_var, value="tif").pack(anchor="w") # New Radiobutton

        tk.Label(frm, text="\n② Select input files", font=("Arial", 11, "bold")).pack(anchor="w")
        tk.Button(frm, text="Select input files...", command=self.select_files).pack(anchor="w", pady=6)
        self.file_label = tk.Label(frm, text="No files selected", fg="gray")
        self.file_label.pack(anchor="w")

        tk.Label(frm, text="\n③ Select output folder", font=("Arial", 11, "bold")).pack(anchor="w")
        tk.Button(frm, text="Select output folder...", command=self.select_output_dir).pack(anchor="w", pady=6)
        self.output_label = tk.Label(frm, text="No folder selected", fg="gray")
        self.output_label.pack(anchor="w")

        tk.Label(frm, text="\n④ Select number of exponential decay", font=("Arial", 11, "bold")).pack(anchor="w")
        n_frame = tk.Frame(frm)
        n_frame.pack(anchor="w", pady=4)

        tk.Label(n_frame, text="n:").pack(side="left")

        # OptionMenu で 2,3,4 を選択
        n_options = (2, 3, 4)
        self.n_exp_menu = tk.OptionMenu(n_frame, self.n_exp_var, *n_options)
        self.n_exp_menu.config(width=4)
        self.n_exp_menu.pack(side="left", padx=4)
        
        tk.Checkbutton(
            frm,
            text="Use manual scan time [ms]",
            variable=self.use_scan_time_var,
            command=self.on_toggle_manual_scan_time
        ).pack(anchor="w", pady=4)
        self.scan_time_entry = tk.Entry(frm, textvariable=self.scan_time_var, width=10, state="disabled")
        self.scan_time_entry.pack(anchor="w", pady=2)
    
        tk.Button(frm, text="Run bleaching correction", bg="#4caf50", fg="white",
                  command=self.run_correction).pack(pady=15)

        self.status_label = tk.Label(frm, text="Ready", fg="blue")
        self.status_label.pack()

    def on_toggle_manual_scan_time(self):
        """手動スキャン時間の有効／無効切り替え。"""
        if self.use_scan_time_var.get():
            self.scan_time_entry.config(state="normal")
        else:
            self.scan_time_entry.config(state="disabled")
            
    def select_files(self):
            file_type = self.file_type_var.get()
            if file_type == "lsm":
                files = filedialog.askopenfilenames(
                    title="Select .lsm files",
                    filetypes=[
                        ("LSM files", "*.lsm"),
                        ("All files", "*.*")
                    ]
                )
            elif file_type == "czi": 
                files = filedialog.askopenfilenames(
                    title="Select .czi files",
                    filetypes=[
                        ("CZI files", "*.czi"),
                        ("All files", "*.*")
                    ]
                )
            elif file_type == "tif":
                files = filedialog.askopenfilenames(
                    title="Select .tif files",
                    filetypes=[
                        ("TIFF files", "*.tif"),
                        ("All files", "*.*")
                    ]
                )
            else:  # Assumed "txt"
                files = filedialog.askopenfilenames(
                    title="Select .txt files",
                    filetypes=[
                        ("Text files", "*.txt"),
                        ("All files", "*.*")
                    ]
                )
            if files:
                self.files = list(files)
                self.file_label.config(text=f"{len(files)} files selected", fg="green")
            else:
                self.file_label.config(text="No files selected", fg="gray")

    def select_output_dir(self):
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self.output_dir = folder
            self.output_label.config(text=folder, fg="green")
        else:
            self.output_label.config(text="No folder selected", fg="gray")

    def run_correction(self):
        if not self.files:
            messagebox.showwarning("No files", "Please select LSM files first.")
            return
        if not self.output_dir:
            messagebox.showwarning("No output folder", "Please select an output folder.")
            return     

        self.status_label.config(text="Processing...", fg="orange")
        self.root.update_idletasks()
        n_exp = int(self.n_exp_var.get())

        for filepath in self.files:
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            print(f"Processing {base_name} ...")

            # --- scan_time_ms ---
            if self.use_scan_time_var.get():
                scan_time_ms = float(self.scan_time_var.get())
            else:
                scan_time_ms = extract_scan_time_ms(base_name)

            if not scan_time_ms:
                messagebox.showwarning("No value of scan time", "Please input the value.")
                return

            try:
                if self.file_type_var.get() == "czi":
                    image = czifile.imread(filename=filepath)
                    data = np.squeeze(image)
                else:
                    image = imread(filepath).astype(np.float32)
                    data = np.squeeze(image)
                print(f"Original data shape: {data.shape}")
                # if data.ndim == 3:
                #     data = data.reshape(data.shape[0], -1)
                # elif data.ndim == 2:
                #     data = data
                # else:
                #     raise ValueError("Unsupported data shape")


                corrected, fit_results = correct_bleaching_multi_exponential(
                                            data, scan_time_ms,
                                            n_exp, base_name,
                                            self.output_dir, save_plot=True
                                        )

                # === Save corrected data ===
                out_path = os.path.join(self.output_dir, f"{base_name}_bleach_corrected.txt")
                np.savetxt(out_path, corrected, delimiter=",", fmt="%.6e")
                print(f"✅ Saved corrected data: {out_path}")

                # === Save fit parameters & metrics ===
                if fit_results is not None:
                    df = fit_results
                    csv_path = os.path.join(self.output_dir, f"{base_name}_bleach_fit_params.csv")
                    df.to_csv(csv_path, index=False)
                    print(f"✅ Saved fit parameters: {csv_path}")

            except Exception as e:
                print(f"❌ Error processing {base_name}: {e}")

        self.status_label.config(text="Completed ✅", fg="green")
        messagebox.showinfo("Done", "All bleaching corrections are complete!")


# =========================================================
# Run GUI
# =========================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = BleachCorrectionGUI(root)
    root.mainloop()
