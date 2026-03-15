import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import cupy as cp
from tifffile import imread  # 必要に応じて他の読み込み関数に変更可


# =========================================================
# GPU-based multi-tau autocorrelation function
# =========================================================
def autocorr_fcs_multipletau_gpu(timeseries_cpu, scan_time_ms, m=8):
    timeseries = cp.asarray(timeseries_cpu)
    meansignal = cp.nanmean(timeseries)
    deltatimeseries = timeseries - meansignal
    maxtau = len(timeseries) - 1

    fcorr, tcorr, sigmas = [], [], []

    # --- short lags ---
    for i in range(1, m):
        a = deltatimeseries[:-i]
        b = deltatimeseries[i:]
        valid = ~cp.isnan(a + b)
        a_valid, b_valid = a[valid], b[valid]

        if len(a_valid) == 0:
            fcorr.append(cp.nan)
            sigmas.append(cp.nan)
            tcorr.append(i)
            continue

        fcorri = cp.sum(a_valid * b_valid)
        sigmaisq = cp.sum((a_valid ** 2) * (b_valid ** 2))
        valid_range = len(a_valid)

        sigmai = (1 / (cp.sqrt(valid_range) * meansignal ** 2)) * \
                 cp.sqrt((1 / valid_range) * (sigmaisq - (fcorri ** 2) / valid_range))

        fcorr.append(fcorri / valid_range)
        sigmas.append(sigmai)
        tcorr.append(i)

    # --- long lags (multi-tau) ---
    tlag = 1
    current = timeseries.copy()
    while current.shape[0] > m:
        if current.shape[0] % 2 == 1:
            current = current[:-1]
        current = cp.nanmean(current.reshape(-1, 2), axis=1)
        deltatimeseries = current - meansignal
        maxtau = len(current) - 1
        tlag *= 2

        maxindex = min(maxtau, m - 1)
        for j in range(m // 2, maxindex):
            a = deltatimeseries[:-j]
            b = deltatimeseries[j:]
            valid = ~cp.isnan(a + b)
            a_valid, b_valid = a[valid], b[valid]
            if len(a_valid) == 0:
                continue

            fcorri = cp.sum(a_valid * b_valid)
            sigmaisq = cp.sum((a_valid ** 2) * (b_valid ** 2))
            valid_range = len(a_valid)

            sigmai = (1 / (cp.sqrt(valid_range) * meansignal ** 2)) * \
                     cp.sqrt((1 / valid_range) * (sigmaisq - (fcorri ** 2) / valid_range))

            fcorr.append(fcorri / valid_range)
            sigmas.append(sigmai)
            tcorr.append(j * tlag)

    fcorr = cp.asnumpy(cp.array(fcorr)) / (cp.asnumpy(meansignal) ** 2)
    sigmas = cp.asnumpy(cp.array(sigmas))
    tcorr = cp.asnumpy(cp.array(tcorr)) * scan_time_ms

    if len(sigmas) >= m // 2:
        sigmas[-(m // 2) + 1:] = sigmas[-(m // 2)]

    return tcorr, fcorr, sigmas

# =========================================================
# FCS correlation calculation per spatial position
# =========================================================
def scanningFCS_gpu(data, scan_time_ms, m=8):
    num_positions = data.shape[1]
    correlations, sigmas_list = [], []
    taus = None

    for x in range(num_positions):
        trace = data[:, x]
        if np.isnan(trace).any() or np.all(trace == 0):
            continue
        tau, g, sigma = autocorr_fcs_multipletau_gpu(trace, scan_time_ms, m=m)
        correlations.append(g)
        taus = tau
        sigmas_list.append(sigma)

    return np.array(correlations), taus, np.array(sigmas_list)

# =========================================================
# Multiple-tau spatial lags generator
# =========================================================
def generate_multiple_tau_lags(max_lag: int, m: int = 16) -> np.ndarray:
    """
    Generate symmetric multiple-tau spatial lags Δx.
    """
    lags = [0]
    tau = 1
    while tau <= max_lag:
        for i in range(1, m):
            lag = i * tau
            if lag > max_lag:
                break
            lags.append(lag)
        tau *= 2
    # 対称（±dx）
    lags = sorted(set([-l for l in lags if l != 0] + lags))
    return np.array(lags, dtype=int)

# =========================================================
# GPU-based spatiotemporal autocorrelation calculation
# =========================================================
def spatiotemporal_autocorr(
    data: np.ndarray,
    max_tau: int = 100,
    max_dx: int = None,
    scan_time_ms: float = 1.0,
    m: int = 16
):
    """
    走査型FCSデータから時空間自己相関関数 G(Δx, τ) をGPUで計算。

    Parameters
    ----------
    data : np.ndarray
        形状 (frames, pixels) の2次元データ
    max_tau : int
        最大ラグ時間（フレーム単位）
    max_dx : int
        最大空間ラグ（ピクセル単位）
    scan_time_ms : float
        スキャン1ラインあたりの時間 [ms]
    m : int
        multi-tauのブロックサイズ
    """
    frames, pixels = data.shape
    data_gpu = cp.asarray(data, dtype=cp.float32)
    mean_intensity = cp.nanmean(data_gpu)
    delta = data_gpu - mean_intensity

    if max_dx is None:
        max_dx = pixels // 2

    dx_list = generate_multiple_tau_lags(max_dx, m=m)
    tau_list = []
    G2D_list = []

    tau = 1
    while tau <= max_tau:
        G_tau = cp.zeros(len(dx_list), dtype=cp.float32)
        line1 = delta[:-tau, :]
        line2 = delta[tau:, :]

        for i, dx in enumerate(dx_list):
            if dx < 0:
                a = line1[:, :dx]
                b = line2[:, -dx:]
            elif dx > 0:
                a = line1[:, dx:]
                b = line2[:, :-dx]
            else:
                a, b = line1, line2

            if a.size == 0:
                G_tau[i] = cp.nan
                continue

            vals = cp.nanmean(a * b, axis=1)
            G_tau[i] = cp.nanmean(vals) / (mean_intensity**2)

        G2D_list.append(cp.asnumpy(G_tau))
        tau_list.append(tau)
        tau = tau + 1 if tau < m else tau * 2

    G2D = np.vstack(G2D_list)
    tau_list = np.array(tau_list, dtype=np.float32) * scan_time_ms  # [ms]
    dx_list = np.array(dx_list, dtype=np.int32)

    return G2D, tau_list, dx_list

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
class AutocorrelationMultipleTauGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Autocorrelation calculation - Multiple-tau - Tool")
        self.root.geometry("660x500")

        self.file_type_var = tk.StringVar(value="lsm")
        self.corr_dims_var = tk.StringVar(value="temporal")
        self.use_scan_time_var = tk.BooleanVar(value=False)
        self.scan_time_var = tk.DoubleVar(value=1.0)
        self.files = []
        self.output_dir = ""

        self.build_gui()

    def build_gui(self):
        frm = tk.Frame(self.root, padx=12, pady=12)
        frm.pack(fill="both", expand=True)

        tk.Label(frm, text="① Select file type", font=("Arial", 11, "bold")).pack(anchor="w")
        setup_frame = tk.Frame(frm)
        setup_frame.pack(anchor="w", pady=6)
        tk.Radiobutton(setup_frame, text=".lsm", variable=self.file_type_var, value="lsm").pack(side="left", padx=6)
        tk.Radiobutton(setup_frame, text=".txt", variable=self.file_type_var, value="txt").pack(side="left", padx=6)

        tk.Label(frm, text="\n② Select input files", font=("Arial", 11, "bold")).pack(anchor="w")
        tk.Button(frm, text="Select input files...", command=self.select_files).pack(anchor="w", pady=6)
        self.file_label = tk.Label(frm, text="No files selected", fg="gray")
        self.file_label.pack(anchor="w")

        tk.Label(frm, text="\n③ Select output folder", font=("Arial", 11, "bold")).pack(anchor="w")
        tk.Button(frm, text="Select output folder...", command=self.select_output_dir).pack(anchor="w", pady=6)
        self.output_label = tk.Label(frm, text="No folder selected", fg="gray")
        self.output_label.pack(anchor="w")

        tk.Label(frm, text="\n④ Select ACF calculation type", font=("Arial", 11, "bold")).pack(anchor="w")
        setup_frame = tk.Frame(frm)
        setup_frame.pack(anchor="w", pady=6)
        tk.Radiobutton(setup_frame, text="temporal", variable=self.corr_dims_var, value="temporal").pack(side="left", padx=6)
        tk.Radiobutton(setup_frame, text="spatiotemporal", variable=self.corr_dims_var, value="spatiotemporal").pack(side="left", padx=6)

        # --- 手動スキャン時間
        tk.Checkbutton(
            frm,
            text="Use manual scan time [ms]",
            variable=self.use_scan_time_var,
            command=self.on_toggle_manual_scan_time
        ).pack(anchor="w", pady=4)
        self.scan_time_entry = tk.Entry(frm, textvariable=self.scan_time_var, width=10, state="disabled")
        self.scan_time_entry.pack(anchor="w", pady=2)

        tk.Button(frm, text="Run autocorrelation calculation",
                  bg="#4caf50", fg="white",
                  command=self.run_calculation).pack(pady=15)

        self.status_label = tk.Label(frm, text="Ready", fg="blue")
        self.status_label.pack()

    def on_toggle_manual_scan_time(self):
        """手動スキャン時間の有効／無効切り替え。"""
        if self.use_scan_time_var.get():
            self.scan_time_entry.config(state="normal")
        else:
            self.scan_time_entry.config(state="disabled")

    def select_files(self):
        if self.file_type_var.get() == "lsm":
            files = filedialog.askopenfilenames(
                title="Select .lsm files",
                filetypes=[("LSM files", "*.lsm"), ("All files", "*.*")]
            )
        else:
            files = filedialog.askopenfilenames(
                title="Select .txt files",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
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

    def run_calculation(self):
        if not self.files:
            messagebox.showwarning("No files", "Please select input files first.")
            return
        if not self.output_dir:
            messagebox.showwarning("No output folder", "Please select an output folder.")
            return

        self.status_label.config(text="Processing...", fg="orange")
        self.root.update_idletasks()
        error_files = []

        for filepath in self.files:
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            print(f"Processing: {base_name}")

            # --- scan_time_ms ---
            if self.use_scan_time_var.get():
                scan_time_ms = float(self.scan_time_var.get())
            else:
                scan_time_ms = extract_scan_time_ms(base_name)

            if scan_time_ms is None:
                messagebox.showwarning(
                    "Scan time not found",
                    f"Scan time could not be extracted from filename:\n{base_name}\n\n"
                    "Enable 'Use manual scan time [ms]' or rename the file (e.g., *_1.0ms)."
                )
                self.status_label.config(text="Aborted", fg="red")
                return
            
            try:
                if self.file_type_var.get() == "lsm":
                    data = imread(filepath).astype(np.float32)
                else:
                    data = np.loadtxt(filepath, delimiter=',').astype(np.float32)

                if self.corr_dims_var.get() == "temporal":                   
                    correlations, taus, sigmas = scanningFCS_gpu(data, scan_time_ms)
                    # === Save ===
                    np.savetxt(os.path.join(self.output_dir, f"{base_name}_correlation.csv"),
                            correlations, delimiter=",", fmt="%.6e")
                    np.savetxt(os.path.join(self.output_dir, f"{base_name}_tau.csv"),
                            taus, delimiter=",", fmt="%.6e")
                    np.savetxt(os.path.join(self.output_dir, f"{base_name}_sigma.csv"),
                            sigmas, delimiter=",", fmt="%.6e")
                    print(f"Temporal ACF calculation completed\n✅ Saved results for {base_name}")
                else:  # spatiotemporal
                    G2D, taus, dxs = spatiotemporal_autocorr(
                        data,
                        max_tau=10000,
                        max_dx=None,
                        scan_time_ms=scan_time_ms,
                        m=8
                    )
                    # === Save ===
                    np.savetxt(os.path.join(self.output_dir, f"{base_name}_spatiotemporal_correlations.txt"),
                            G2D, delimiter=",", fmt="%.6e")
                    np.savetxt(os.path.join(self.output_dir, f"{base_name}_spatiotemporal_taus.txt"),
                            taus, delimiter=",", fmt="%.6e")
                    np.savetxt(os.path.join(self.output_dir, f"{base_name}_spatiotemporal_dxs.txt"),
                            dxs, delimiter=",", fmt="%.6e")
                    print(f"Spatiotemporal ACF calculation completed\n✅ Saved results for {base_name}")

            except Exception as e:
                print(f"❌ Error processing {base_name}: {e}")
                error_files.append((base_name, str(e)))
                continue

        # 完了メッセージを少しだけリッチに
        if error_files:
            self.status_label.config(text="Completed with errors ⚠️", fg="orange")
            msg = "Some files could not be processed:\n\n"
            for name, err in error_files:
                msg += f"- {name}: {err}\n"
            messagebox.showwarning("Completed with errors", msg)
        else:
            self.status_label.config(text="Completed ✅", fg="green")
            messagebox.showinfo(
                "Done",
                f"All autocorrelation calculations are complete!\n\n"
                f"Files processed: {len(self.files)}\n"
                f"Mode: {self.corr_dims_var.get()}\n"
                f"Scan time [ms]: {'manual ' + str(self.scan_time_var.get()) if self.use_scan_time_var.get() else 'from filename'}"
            )
# =========================================================
# Run GUI
# =========================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = AutocorrelationMultipleTauGUI(root)
    root.mainloop()
