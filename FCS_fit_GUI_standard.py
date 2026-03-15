#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dynamic FCS fitting GUI
 - ACF matrix (positions x lags) を読み込み
 - ヒートマップ上でポジション選択（単一 or 範囲平均）
 - 右側で ACF とフィットカーブ＆残差を表示
 - モデル一覧はテンプレートから自動生成
 - get_model_config はテンプレートに基づき動的にモデル関数と param_names, p0, bounds を生成
 - GUI 上で各パラメータの初期値 p0, 下限 / 上限 bounds を編集可能
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import SpanSelector
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit


# ===========================
# 基本モデル関数（building blocks）
# ===========================

def triplet_factor(tau, T, tau_T):
    """Triplet モジュール: 1 + T/(1-T) * exp(-tau / tau_T)"""
    return 1.0 + (T / (1.0 - T)) * np.exp(-tau / tau_T)

def single_exp(tau, A1, tau_exp1):
    return A1 * np.exp(-tau / tau_exp1)

# --- 1 コンポーネント拡散（N は別に扱う）---

def G_comp_2D(tau, tau_D):
    """2D（軸方向無視） G(τ) 成分（正規化前）"""
    return (1.0 + tau / tau_D) ** -1.0

def G_comp_3D(tau, tau_D, S):
    """3D Gaussian detection PSF G(τ) 成分（正規化前）"""
    return (1.0 + tau / tau_D) ** -1.0 * (1.0 + tau / (S ** 2 * tau_D)) ** -0.5


# ===========================
# 動的モデルテンプレート
# ===========================

# ベースモデルの役割だけ定義（次で "2D"/"3D" を掛け合わせる）
# name は GUI での表示と同じコア部分（(2D)/(3D) を除く）
MODEL_TEMPLATES = {
    # 1 comp
    "1D":                dict(mode="1comp", rxn = False, triplet=False, exp=False, offset=False),
    "1D+offset":         dict(mode="1comp", rxn = False, triplet=False, exp=False, offset=True),
    "1D+exp+offset":     dict(mode="1comp", rxn = False, triplet=False, exp=True,  offset=True),
    "1D+bind+offset":    dict(mode="binding", rxn = False, triplet=False, exp=False, offset=True),
    "1D+bind+exp+offset":dict(mode="binding", rxn = False, triplet=False, exp=True, offset=True),
    "1D+rxn+offset":     dict(mode="1comp", rxn = True, triplet=False, exp=False, offset=True),
    "1D+rxn+exp+offset": dict(mode="1comp", rxn = True, triplet=False, exp=True, offset=True),

    "1T1D":              dict(mode="1comp", rxn = False, triplet=True,  exp=False, offset=False),
    "1T1D+offset":       dict(mode="1comp", rxn = False, triplet=True,  exp=False, offset=True),
    "1T1D+exp+offset":   dict(mode="1comp", rxn = False, triplet=True,  exp=True,  offset=True),

    # 2 comp
    "2D":                dict(mode="2comps", rxn = False, triplet=False, exp=False, offset=False),
    "2D+offset":         dict(mode="2comps", rxn = False, triplet=False, exp=False, offset=True),
    "2D+exp+offset":     dict(mode="2comps", rxn = False, triplet=False, exp=True,  offset=True),
    "2D+rxn+offset":     dict(mode="2comps", rxn = True, triplet=False, exp=False, offset=True),
    "2D+rxn+exp+offset": dict(mode="2comps", rxn = True, triplet=False, exp=True, offset=True),

    "1T2D":              dict(mode="2comps", rxn = False, triplet=True,  exp=False, offset=False),
    "1T2D+offset":       dict(mode="2comps", rxn = False, triplet=True,  exp=False, offset=True),
    "1T2D+exp+offset":   dict(mode="2comps", rxn = False, triplet=True,  exp=True,  offset=True),
}


# ディメンション（2D / 3D）と組み合わせて最終的な model_type を作る
DIMENSIONS = ["2D", "3D"]

def generate_model_type_list():
    """
    テンプレート x ディメンション から
    '1D(2D)', '1D+offset(2D)', ..., '1T2D+exp+offset(3D)' などを自動生成
    """
    model_types = []
    for base in MODEL_TEMPLATES.keys():
        for dim in DIMENSIONS:
            model_types.append(f"{base}({dim})")
    return model_types


# ===========================
# デフォルトパラメータ情報（p0, lower, upper）
# ===========================

DEFAULT_PARAM_INFO = {
    # 共通
    "N":        (50.0,  1e-6,  1e6),
    # 拡散時間（ms）
    "tau_D":    (5.0,    1.0,  1e3),
    "tau_D1":   (5.0,    1.0,  5e2),
    "tau_D2":   (50.0,   1.0,  5e3),
    # 分率
    "F":        (0.5,    0.0,  1.0),
    # Bind
    "Fbind":    (0.1,    0.0,  1.0),
    "tau_bind": (100.0,  1.0,  1e6),
    "A_rxn":    (0.5,    0.0,  5.0),
    "tau_rxn":  (100.0,  1.0,  1e6),
    # Triplet
    "T":        (0.1,    0.0,  1.0),
    "tau_T":    (1e-1,  1e-3,  1.0),   # ms
    # Exp
    "A1":       (0.02,   0.0,  1.0),
    "tau_exp1": (2e3,    1e3,  1e5),   # ms
    # offset
    "offset":   (0.0,   -0.1,  0.1),
}

# ===========================
# 動的モデル生成
# ===========================

def make_param_names(template_name: str):
    """
    base名（例: '1D+exp+offset', '1T2D', ...）から param_names の順序を構成
    """
    cfg = MODEL_TEMPLATES[template_name]
    mode   = cfg["mode"]
    rxn     = cfg["rxn"]
    triplet  = cfg["triplet"]
    exp_flag = cfg["exp"]
    offset   = cfg["offset"]

    names = ["N"]
    if mode == "1comp":
        names.append("tau_D")
    elif mode == "2comps":
        names.extend(["tau_D1", "tau_D2", "F"])
    elif mode == "binding":
        names.extend(["tau_D", "Fbind", "tau_bind"])
        
    if rxn:
        names.extend(["A_rxn", "tau_rxn"])

    if triplet:
        names.extend(["T", "tau_T"])

    if exp_flag:
        names.extend(["A1", "tau_exp1"])

    if offset:
        names.append("offset")

    return names

def make_model_function(template_name: str, dim: str, S: float):
    """
    テンプレート名 ('1D', '1T2D+exp+offset', etc.) と
    空間次元 ('2D' / '3D'), S から model(tau, *params) を生成
    tau, tau_D, tau_T, tau_exp1 はすべて [ms] 前提
    """
    cfg = MODEL_TEMPLATES[template_name]
    mode     = cfg["mode"]
    rxn      = cfg["rxn"]
    triplet  = cfg["triplet"]
    exp_flag = cfg["exp"]
    offset   = cfg["offset"]

    use_3D = (dim == "3D")

    # --- 1コンポーネント or 2コンポーネントの G0(N付き) を組み立てる ---
    def model(tau, *params):
        # tau, tau_D, tau_T, tau_exp1 は [ms] として扱う
        # ただし式そのものは単位に依存しない（tau と tau_D の単位が揃っていれば良い）

        idx = 0
        N = params[idx]; idx += 1

        if mode == "1comp":
            tau_D = params[idx]; idx += 1
            if use_3D:
                G0 = (1.0 / N) * G_comp_3D(tau, tau_D, S)
            else:
                G0 = (1.0 / N) * G_comp_2D(tau, tau_D)

        elif mode == "2comps":  # 2 comp
            tau_D1 = params[idx]; idx += 1
            tau_D2 = params[idx]; idx += 1
            F      = params[idx]; idx += 1
            if use_3D:
                G1 = G_comp_3D(tau, tau_D1, S)
                G2 = G_comp_3D(tau, tau_D2, S)
            else:
                G1 = G_comp_2D(tau, tau_D1)
                G2 = G_comp_2D(tau, tau_D2)
            G0 = (1.0 / N) * (F * G1 + (1.0 - F) * G2)
            
        # ---------- BINDING / UNBINDING ----------
        elif mode == "binding":
            tau_D = params[idx]; idx+=1
            Fbind = params[idx]; idx+=1
            tau_bind = params[idx]; idx+=1
            if use_3D:
                Gfree = G_comp_3D(tau, tau_D, S)
            else:
                Gfree = G_comp_2D(tau, tau_D)
            Gbound = np.exp(-tau/tau_bind) * Gfree
            G0 = (1.0/N)*((1-Fbind)*Gfree + Fbind*Gbound)
            
        # Bind
        if rxn:
            A_rxn   = params[idx]; idx += 1
            tau_rxn = params[idx]; idx += 1
            G0 = G0 * (1 + A_rxn*np.exp(-tau/tau_rxn))

            
        # Triplet
        if triplet:
            T     = params[idx]; idx += 1
            tau_T = params[idx]; idx += 1
            G0 = triplet_factor(tau, T, tau_T) * G0

        # Exp（加法）
        if exp_flag:
            A1        = params[idx]; idx += 1
            tau_exp1  = params[idx]; idx += 1
            G0 = G0 + single_exp(tau, A1, tau_exp1)

        # offset（加法）
        if offset:
            off = params[idx]; idx += 1
            G0 = G0 + off

        return G0

    return model


def get_model_config(model_type: str, S: float):
    """
    動的 get_model_config:
      model_type: '1D(2D)', '1D+offset(3D)', '1T2D+exp+offset(3D)' など
    戻り値 dict:
      - model: callable (tau, *params)
      - param_names: list[str]
      - p0: list[float]
      - bounds: (lower_list, upper_list)
    """
    # 例: '1T2D+exp+offset(3D)' → base_name='1T2D+exp+offset', dim='3D'
    if "(" not in model_type or not model_type.endswith(")"):
        raise ValueError(f"Invalid model_type: {model_type}")
    base_name, dim_part = model_type.split("(", 1)
    dim = dim_part.rstrip(")")

    if base_name not in MODEL_TEMPLATES:
        raise ValueError(f"Unknown base model: {base_name}")
    if dim not in DIMENSIONS:
        raise ValueError(f"Unknown dimension: {dim}")

    param_names = make_param_names(base_name)

    # p0, bounds をデフォルト定義から組み立て
    p0 = []
    lower = []
    upper = []
    for name in param_names:
        if name not in DEFAULT_PARAM_INFO:
            raise KeyError(f"No default param info for '{name}'")
        v0, vmin, vmax = DEFAULT_PARAM_INFO[name]
        p0.append(v0)
        lower.append(vmin)
        upper.append(vmax)

    model = make_model_function(base_name, dim, S)

    return {
        "model": model,
        "param_names": param_names,
        "p0": p0,
        "bounds": (lower, upper),
        "dimension": dim,
        "base_name": base_name,
    }


# ===========================
# フィット品質メトリクス
# ===========================

def compute_metrics(y_obs, y_fit):
    y_obs = np.asarray(y_obs)
    y_fit = np.asarray(y_fit)
    residuals = y_obs - y_fit
    ss_res = np.nansum(residuals ** 2)
    ss_tot = np.nansum((y_obs - np.nanmean(y_obs)) ** 2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    MSE = ss_res / np.sum(~np.isnan(y_obs))
    chi2 = np.nansum((residuals ** 2) / (np.abs(y_fit) + 1e-12))
    return residuals, R2, ss_res, MSE, chi2


# ===========================
# GUI クラス
# ===========================

class FCSMatrixFitGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FCS ACF Matrix Dynamic Fitting GUI")
        self.root.geometry("1200x800")

        # データ
        self.matrix = None   # shape: [positions, lags]
        self.tau = None      # shape: [lags]  (ms)
        self.curve = None    # 現在選択中の ACF
        self.pos_selection = None  # (min_idx, max_idx)
        self.fit_mask = None       # bool array on tau
        self.loaded_basename = ""  # for default save names

        # パラメータ
        self.pixel_scale = tk.DoubleVar(value=0.022)  # [um/pixel]
        self.S_ratio = tk.DoubleVar(value=5.0)
        self.w0_um   = tk.DoubleVar(value=0.2)  # w0 はここでは D換算等に使うだけ
        self.model_type = tk.StringVar(value="1D(3D)")
        self.param_names = []
        self.popt = None

        # 動的パラメータエントリ格納
        # {name: {"p0":Entry, "lo":Entry, "hi":Entry}}
        self.param_entries = {}
        self.param_vars = {}

        # ヒートマップ用 colorbar を追跡
        self.heat_cbar = None

        # Selection visuals
        self._hspan_heat = None   # heatmap averaged band
        self._hline_heat = None   # heatmap single position line

        # w0 computation panel (from D & tauD)
        self.D_for_w0 = tk.DoubleVar(value=400.0)      # [um^2/s]
        self.tauD_for_w0 = tk.DoubleVar(value=25.0)   # [us]
        self.w0_um = tk.DoubleVar(value=0.2)         # [um], updated by Compute

        self._build_gui()

    # --------------------------
    # GUI 構成
    # --------------------------
    def _build_gui(self):
        # 上部: ファイルとモデル選択
        top = tk.Frame(self.root, padx=8, pady=4)
        top.pack(fill="x")

        tk.Button(top, text="Load Matrix (.txt/.csv)", command=self.load_matrix).pack(side="left", padx=4)

        tk.Label(top, text=" Model:").pack(side="left", padx=(16, 4))
        self.model_box = ttk.Combobox(
            top, textvariable=self.model_type, width=24,
            values=generate_model_type_list()
        )
        self.model_box.pack(side="left")
        self.model_box.bind("<<ComboboxSelected>>", lambda e: self.build_param_table())

        tk.Label(top, text=" S:").pack(side="left", padx=(16, 4))
        tk.Entry(top, textvariable=self.S_ratio, width=6).pack(side="left")

        tk.Label(top, text=" w0 [µm]:").pack(side="left", padx=(16, 4))
        tk.Entry(top, textvariable=self.w0_um, width=6).pack(side="left")

        tk.Button(top, text="Fit", command=self.run_fit).pack(side="right", padx=4)

        # loaded file label
        self.loaded_label = tk.Label(top, text="No file loaded", fg="gray")
        self.loaded_label.pack(side="right")

        # Bottom: left (manual selection)
        bottom = tk.Frame(self.root, padx=8, pady=8)
        bottom.pack(fill="x")

        manual = tk.LabelFrame(bottom, text="Position Selection (Manual Input)")
        manual.pack(side="left", fill="x", expand=True, padx=6)

        # single position
        tk.Label(manual, text="Position index:").grid(row=0, column=0, sticky="w")
        self.pos_single = tk.IntVar(value=0)
        tk.Entry(manual, textvariable=self.pos_single, width=7).grid(row=0, column=1, padx=4)
        tk.Button(manual, text="Go", command=self.apply_single_position).grid(row=0, column=2, padx=4)

        # averaging range
        tk.Label(manual, text="Average range [min:max):").grid(row=1, column=0, sticky="w")
        self.pos_min = tk.IntVar(value=0)
        self.pos_max = tk.IntVar(value=5)
        tk.Entry(manual, textvariable=self.pos_min, width=7).grid(row=1, column=1, padx=2)
        tk.Entry(manual, textvariable=self.pos_max, width=7).grid(row=1, column=2, padx=2)
        tk.Button(manual, text="Apply", command=self.apply_avg_range).grid(row=1, column=3, padx=4)

        # Bottom: center (w0 compute)
        w0f = tk.LabelFrame(bottom, text="Calculate w0 from D and τD")
        w0f.pack(side="left", fill="x", expand=True, padx=6)

        tk.Label(w0f, text="D [µm²/s]:").grid(row=0, column=0, sticky="w")
        tk.Entry(w0f, textvariable=self.D_for_w0, width=10).grid(row=0, column=1, padx=4)

        tk.Label(w0f, text="τD [µs]:").grid(row=0, column=2, sticky="w")
        tk.Entry(w0f, textvariable=self.tauD_for_w0, width=10).grid(row=0, column=3, padx=4)

        tk.Label(w0f, text="→ w0 [µm]:").grid(row=0, column=4, sticky="w", padx=(10, 0))
        tk.Entry(w0f, textvariable=self.w0_um, width=10, state="readonly").grid(row=0, column=5, padx=4)

        tk.Button(w0f, text="Calculate", command=self.calculate_w0).grid(row=0, column=6, padx=8)

        # 中央: 左ヒートマップ、右ACF
        center = tk.Frame(self.root)
        center.pack(fill="both", expand=True)

        # 左: ヒートマップ
        self.fig_heat = Figure(figsize=(5.2, 4.4), dpi=100)
        self.ax_heat = self.fig_heat.add_subplot(111)
        self.canvas_heat = FigureCanvasTkAgg(self.fig_heat, master=center)
        self.canvas_heat.get_tk_widget().pack(side="left", fill="both", expand=True, padx=4, pady=4)

        # 右: ACF & residuals
        self.fig_fit = Figure(figsize=(5.2, 4.4), dpi=100)
        self.ax_acf = self.fig_fit.add_subplot(211)
        self.ax_res = self.fig_fit.add_subplot(212, sharex=self.ax_acf)
        self.canvas_fit = FigureCanvasTkAgg(self.fig_fit, master=center)
        self.canvas_fit.get_tk_widget().pack(side="left", fill="both", expand=True, padx=4, pady=4)

        self.ax_acf.set_ylabel("G(τ)")
        self.ax_res.set_xlabel("Lag time τ [ms]")
        self.ax_res.set_ylabel("Residuals")

        # ヒートマップの選択: 垂直ドラッグで平均範囲 / クリックで単一
        self.heat_selector = SpanSelector(
            self.ax_heat, self.on_select_positions, direction="vertical",
            useblit=True, minspan=1, interactive=True,
            props=dict(alpha=0.2, facecolor="orange")
        )
        self.canvas_heat.mpl_connect("button_press_event", self.on_heatmap_click)

        # ACF の選択: 水平ドラッグでフィット範囲指定
        self.acf_selector = SpanSelector(
            self.ax_acf, self.on_select_fit_range, direction="horizontal",
            useblit=True, minspan=1e-9, interactive=True,
            props=dict(alpha=0.2, facecolor="skyblue")
        )

        # 下部: パラメータ入力とメトリクス
        bottom = tk.Frame(self.root, padx=8, pady=4)
        bottom.pack(fill="x")

        # 左: パラメータテーブル
        self.param_frame = tk.LabelFrame(bottom, text="Fit parameters (p0 / lower / upper)")
        self.param_frame.pack(side="left", fill="x", expand=True, padx=4)

        # 右: メトリクス&保存
        right_tools = tk.Frame(bottom)
        right_tools.pack(side="right")

        self.metrics_label = tk.Label(right_tools, text="R²: --  SSR: --  MSE: --  χ²: --", fg="blue")
        self.metrics_label.pack(anchor="e", pady=(0, 6))

        tk.Button(right_tools, text="Save PDF (fit + residuals)", command=self.save_pdf).pack(fill="x", pady=2)
        tk.Button(right_tools, text="Save params CSV", command=self.save_params_csv).pack(fill="x", pady=2)
        tk.Button(right_tools, text="Save curve CSV", command=self.save_fit_curve_csv).pack(fill="x", pady=2)

        # ステータス
        self.status = tk.Label(self.root, text="Load an ACF matrix to begin.", fg="gray")
        self.status.pack(anchor="w", padx=8, pady=4)

        # 初期モデルに対してパラメータテーブルを準備
        self.build_param_table()

    # --------------------------
    # データ読み込み
    # --------------------------
    def load_matrix(self):
        path = filedialog.askopenfilename(
            title="Select ACF matrix (positions x lags)",
            filetypes=[("Text / CSV", "*_correlation.txt *_correlation.csv"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            # autodetect delimiter
            if path.lower().endswith(".csv"):
                arr = np.loadtxt(path, delimiter=",")
            else:
                try:
                    arr = np.loadtxt(path, delimiter=",")
                except Exception:
                    arr = np.loadtxt(path)

            if arr.ndim != 2:
                raise ValueError("Matrix must be 2D (positions x lags)")

            self.matrix = arr
            base_name = os.path.basename(path)
            base_root, _ = os.path.splitext(base_name)
            base_dir = os.path.dirname(path)
            self.loaded_label.config(text=f"Loaded: {base_name}", fg="black")
            self.loaded_basename = base_root
            self.status.config(text=f"Loaded matrix: {base_name}  shape={arr.shape}")
        except Exception as e:
            messagebox.showerror("Load error", f"Failed to load matrix:\n{e}")
            return

        # tau 自動ロード (同じフォルダから '..._tau.csv' 等)
        tau_candidates = [
            os.path.join(base_dir, base_root.replace("correlation", "tau") + ".csv"),
            os.path.join(base_dir, base_root.replace("correlation", "tau") + ".txt"),
            os.path.join(base_dir, "taus.csv"),
            os.path.join(base_dir, "tau.csv"),
            os.path.join(base_dir, "taus.txt"),
            os.path.join(base_dir, "tau.txt"),
        ]
        loaded_tau = False
        for tpath in tau_candidates:
            if os.path.exists(tpath):
                try:
                    tau = np.loadtxt(tpath, delimiter=",")
                    self.tau = tau.astype(float)
                    loaded_tau = True
                    self.status.config(text=self.status.cget("text") + f" | tau: {os.path.basename(tpath)}")
                    break
                except Exception:
                    pass

        if not loaded_tau:
            # fallback
            self.tau = np.arange(self.matrix.shape[1], dtype=float)
            messagebox.showwarning(
                "Tau not found",
                "Tau vector not found. Using index as tau (ms). Please load proper tau file if needed."
            )

        # ACF & 残差プロットもリセット
        self.ax_acf.clear()
        self.ax_res.clear()
        self.ax_acf.set_ylabel("G(τ)")
        self.ax_res.set_xlabel("Lag time τ [ms]")
        self.ax_res.set_ylabel("Residuals")
        self.canvas_fit.draw()

        self.fit_mask = None
        self.curve = None
        self.pos_selection = None

        self.plot_heatmap()

    # --------------------------
    # ヒートマップ描画
    # --------------------------
    def plot_heatmap(self):
        if self.matrix is None or self.tau is None:
            return

        self.ax_heat.clear()

        # 既存 colorbar を消す（縮小を防ぐ）
        if self.heat_cbar is not None:
            try:
                self.heat_cbar.remove()
            except Exception:
                pass
            self.heat_cbar = None
            self.fig_heat.clf()  # ← Figureを完全リセット
            self.ax_heat = self.fig_heat.add_subplot(111)

        dx_um = float(self.pixel_scale.get())  # µm/pixel
        n_pos = self.matrix.shape[0]
        pos_min = 0.0
        pos_max = n_pos * dx_um        
        
        extent = [self.tau[0], self.tau[-1], pos_min, pos_max]
        im = self.ax_heat.imshow(
            self.matrix,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="turbo"
        )
        self.ax_heat.set_xlabel("Lag time τ [ms]")
        self.ax_heat.set_ylabel("Position [µm]")
        self.ax_heat.set_title(f"ACF map (pixel={dx_um:.3f} µm)")

        self.heat_cbar = self.fig_heat.colorbar(im, ax=self.ax_heat, fraction=0.046, pad=0.04, label="G(τ)")

        self.canvas_heat.draw()

        if self.pos_selection is not None:
            pmin, pmax = self.pos_selection
            dx_um = float(self.pixel_scale.get())
            y0 = pmin * dx_um
            y1 = pmax * dx_um
            self.ax_heat.axhspan(y0, y1, color='white', alpha=0.25, lw=0)

    # --------------------------
    # Selection handlers & overlay visualization
    # --------------------------
    def _clear_position_overlay(self):
        if self._hspan_heat is not None:
            try:
                self._hspan_heat.remove()
            except Exception:
                pass
            self._hspan_heat = None
        if self._hline_heat is not None:
            try:
                self._hline_heat.remove()
            except Exception:
                pass
            self._hline_heat = None

    def _redraw_position_overlay(self):
        """Draw band/line on current heatmap for selection."""
        if self.matrix is None:
            return
        dx_um = float(self.pixel_scale.get())
        self._clear_position_overlay()
        if self.pos_selection is not None:
            pmin, pmax = self.pos_selection
            y0 = pmin * dx_um
            y1 = pmax * dx_um
            self._hspan_heat = self.ax_heat.axhspan(y0, y1, color="orange", alpha=0.25, lw=0)
        elif self.curve is not None:
            # 単一位置の場合、pos_single をもとに線を描く
            try:
                idx = int(self.pos_single.get())
            except Exception:
                idx = 0
            y = idx * dx_um
            self._hline_heat = self.ax_heat.axhline(y, color="orange", alpha=0.8, lw=1.0)

    # --------------------------
    # ヒートマップ上の選択
    # --------------------------
    def on_heatmap_click(self, event):
        if self.matrix is None or self.tau is None:
            return
        if event.inaxes != self.ax_heat:
            return
        if event.ydata is None:
            return
        dx_um = float(self.pixel_scale.get())
        pos_idx = int(np.clip(np.floor(event.ydata / dx_um), 0, self.matrix.shape[0]-1))
        self.pos_selection = None
        self.pos_single.set(pos_idx)
        self.curve = self.matrix[pos_idx, :].astype(float)
        self.status.config(text=f"Selected position: {pos_idx}")
        self._redraw_position_overlay()
        self.draw_acf_curve()

    def on_select_positions(self, y_min, y_max):
        if self.matrix is None:
            return
        dx_um = float(self.pixel_scale.get())
        pmin = int(np.clip(np.floor(min(y_min, y_max) / dx_um), 0, self.matrix.shape[0]-1))
        pmax = int(np.clip(np.ceil(max(y_min, y_max) / dx_um), pmin+1, self.matrix.shape[0]))
        if pmax <= pmin:
            return
        self.pos_selection = (pmin, pmax)
        self.curve = np.nanmean(self.matrix[pmin:pmax, :], axis=0).astype(float)
        self.status.config(text=f"Averaged positions: [{pmin}:{pmax}) (count={pmax-pmin})")
        self._redraw_position_overlay()
        self.draw_acf_curve()

    def apply_single_position(self):
        if self.matrix is None:
            messagebox.showwarning("No data", "Load matrix first.")
            return
        idx = int(self.pos_single.get())
        idx = int(np.clip(idx, 0, self.matrix.shape[0]-1))
        self.pos_selection = None
        self.curve = self.matrix[idx, :].astype(float)
        self.status.config(text=f"Selected position (manual): {idx}")
        self._redraw_position_overlay()
        self.draw_acf_curve()

    def apply_avg_range(self):
        if self.matrix is None:
            messagebox.showwarning("No data", "Load matrix first.")
            return
        pmin = int(self.pos_min.get())
        pmax = int(self.pos_max.get())
        pmin = int(np.clip(pmin, 0, self.matrix.shape[0]-1))
        pmax = int(np.clip(pmax, pmin+1, self.matrix.shape[0]))
        self.pos_selection = (pmin, pmax)
        self.curve = np.nanmean(self.matrix[pmin:pmax, :], axis=0).astype(float)
        self.status.config(text=f"Averaged positions (manual): [{pmin}:{pmax}) count={pmax-pmin}")
        self._redraw_position_overlay()
        self.draw_acf_curve()

    def draw_acf_curve(self):
        if self.curve is None:
            return
        self.ax_acf.clear()
        self.ax_res.clear()
        self.ax_acf.semilogx(self.tau, self.curve, "o", ms=3, label="ACF")
        self.ax_acf.set_ylabel("G(τ)")
        self.ax_acf.grid(True, which="both", ls=":")
        self.ax_acf.legend(loc="best", fontsize=8)

        self.ax_res.set_xlabel("Lag time τ [ms]")
        self.ax_res.set_ylabel("Residuals")
        self.ax_res.grid(True, which="both", ls=":")

        self.fig_fit.tight_layout()
        self.canvas_fit.draw()

    def on_select_fit_range(self, x_min, x_max):
        if self.tau is None or self.curve is None:
            return
        tmin, tmax = sorted([x_min, x_max])
        self.fit_mask = (self.tau >= tmin) & (self.tau <= tmax)
        self.status.config(text=f"Fit range: {tmin:.3f}–{tmax:.3f} ms")

    # --------------------------
    # パラメータテーブル（動的生成）
    # --------------------------
    def build_param_table(self):
        # 既存をクリア
        for w in self.param_frame.winfo_children():
            w.destroy()
        self.param_entries.clear()

        try:
            cfg = get_model_config(self.model_type.get(), S=self.S_ratio.get())
        except Exception as e:
            messagebox.showerror("Model error", f"Failed to build model config:\n{e}")
            return

        param_names = cfg["param_names"]
        p0 = cfg["p0"]
        lo, hi = cfg["bounds"]
        self.param_names = param_names

        # ヘッダ
        header = tk.Frame(self.param_frame)
        header.grid(row=0, column=0, sticky="w", padx=2, pady=2)
        tk.Label(header, text="Param", width=10).grid(row=0, column=0)
        tk.Label(header, text="p0", width=12).grid(row=0, column=1)
        tk.Label(header, text="lower", width=12).grid(row=0, column=2)
        tk.Label(header, text="upper", width=12).grid(row=0, column=3)

        # 行
        for i, name in enumerate(param_names):
            row = tk.Frame(self.param_frame)
            row.grid(row=i+1, column=0, sticky="w", padx=2, pady=1)

            tk.Label(row, text=name, width=10).grid(row=0, column=0)

            e_p0 = tk.Entry(row, width=12)
            e_lo = tk.Entry(row, width=12)
            e_hi = tk.Entry(row, width=12)

            e_p0.insert(0, f"{p0[i]:.6g}")
            e_lo.insert(0, f"{lo[i]:.6g}")
            e_hi.insert(0, f"{hi[i]:.6g}")

            e_p0.grid(row=0, column=1, padx=2)
            e_lo.grid(row=0, column=2, padx=2)
            e_hi.grid(row=0, column=3, padx=2)

            self.param_entries[name] = {"p0": e_p0, "lo": e_lo, "hi": e_hi}

    def collect_param_config_from_gui(self):
        """
        param_entries から p0, lower, upper を取得
        """
        p0 = []
        lo = []
        hi = []
        for name in self.param_names:
            ent = self.param_entries[name]
            try:
                v0 = float(ent["p0"].get())
                vmin = float(ent["lo"].get())
                vmax = float(ent["hi"].get())
            except ValueError:
                raise ValueError(f"Parameter '{name}' has invalid numeric input.")
            if vmax < vmin:
                vmin, vmax = vmax, vmin
            p0.append(v0)
            lo.append(vmin)
            hi.append(vmax)
        return p0, (lo, hi)

    # --------------------------
    # Compute w0 from D and τD
    # --------------------------
    def calculate_w0(self):
        try:
            D = float(self.D_for_w0.get())         # [um^2/s]
            tauD_us = float(self.tauD_for_w0.get())  # [us]
            tauD_s = tauD_us * 1e-6
            w0 = float(np.sqrt(4.0 * D * tauD_s))
            self.w0_um.set(w0)
            messagebox.showinfo("w0 calculated", f"w0 = {w0:.4f} µm")
        except Exception as e:
            messagebox.showerror("Calculate error", str(e))

    # --------------------------
    # フィット実行
    # --------------------------
    def run_fit(self):
        if self.curve is None:
            messagebox.showwarning("No curve", "Select a position or average range first.")
            return
        if self.tau is None:
            messagebox.showwarning("No tau", "Tau is not loaded.")
            return

        try:
            cfg = get_model_config(self.model_type.get(), S=self.S_ratio.get())
            model = cfg["model"]
            self.param_names = cfg["param_names"]

            # GUI から p0, bounds を取得して上書き
            p0, bounds = self.collect_param_config_from_gui()
        except Exception as e:
            messagebox.showerror("Model/param error", str(e))
            return

        # フィット対象データ
        if self.fit_mask is not None and np.any(self.fit_mask):
            tau_fit = self.tau[self.fit_mask]
            g_fit = self.curve[self.fit_mask]
        else:
            tau_fit = self.tau
            g_fit = self.curve

        try:
            popt, _ = curve_fit(model, tau_fit, g_fit, p0=p0, bounds=bounds, maxfev=20000)
            self.popt = popt
        except Exception as e:
            messagebox.showerror("Fit error", f"curve_fit failed:\n{e}")
            return

        # フル予測とメトリクス
        g_pred_full = model(self.tau, *self.popt)
        residuals, R2, SSR, MSE, chi2 = compute_metrics(
            g_fit if self.fit_mask is not None and np.any(self.fit_mask) else self.curve,
            model(tau_fit, *self.popt)
        )

        # プロット更新
        self.ax_acf.clear()
        self.ax_res.clear()

        self.ax_acf.semilogx(self.tau, self.curve, "o", ms=3, label="ACF")
        # フィットカーブ
        self.ax_acf.semilogx(self.tau, g_pred_full, "r-", lw=1.5, label=f"Fit ({self.model_type.get()})")
        self.ax_acf.set_ylabel("G(τ)")
        self.ax_acf.grid(True, which="both", ls=":")
        self.ax_acf.legend(loc="best", fontsize=8)

        # 残差
        if self.fit_mask is not None and np.any(self.fit_mask):
            resid_x = self.tau[self.fit_mask]
            resid_y = self.curve[self.fit_mask] - model(self.tau[self.fit_mask], *self.popt)
        else:
            resid_x = self.tau
            resid_y = self.curve - g_pred_full

        self.ax_res.semilogx(resid_x, resid_y, "k.-", ms=3)
        self.ax_res.axhline(0.0, color="gray", ls="--", lw=0.8)
        self.ax_res.set_xlabel("Lag time τ [ms]")
        self.ax_res.set_ylabel("Residuals")
        self.ax_res.grid(True, which="both", ls=":")

        self.fig_fit.tight_layout()
        self.canvas_fit.draw()

        self.metrics_label.config(text=f"R²={R2:.4f}  SSR={SSR:.3e}  MSE={MSE:.3e}  χ²={chi2:.3e}")
        self.status.config(text=f"Fit done: {self.model_type.get()}")

        # フィット結果を p0 エントリに反映（次回の初期値として）
        for name, val in zip(self.param_names, self.popt):
            self.param_entries[name]["p0"].delete(0, tk.END)
            self.param_entries[name]["p0"].insert(0, f"{val:.6g}")

    # Live sliders
    # --------------------------
    def build_live_sliders(self):
        for w in self.param_frame.winfo_children():
            w.destroy()
        if self.popt is None:
            return
        self.param_vars.clear()

        for i, (name, val) in enumerate(zip(self.param_names, self.popt)):
            row = tk.Frame(self.param_frame)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=name, width=10).pack(side="left")
            var = tk.DoubleVar(value=float(val))
            self.param_vars[name] = var
            low = val - abs(val) if val != 0 else -1.0
            high = val + abs(val) if val != 0 else 1.0
            if low > high: low, high = high, low
            res = max(abs(val) * 0.01, 1e-6)
            scale = tk.Scale(row, variable=var, from_=low, to=high, resolution=res,
                             orient="horizontal", length=320,
                             command=lambda _v, n=name: self.on_slider_change(n))
            scale.pack(side="left", padx=6)
            ent = tk.Entry(row, textvariable=var, width=10)
            ent.pack(side="left", padx=4)
            ent.bind("<Return>", lambda _e, n=name: self.on_slider_change(n))

        tk.Button(self.param_frame, text="Update fit (recompute metrics)",
                  command=self.update_live_fit).pack(pady=4)
        tk.Button(self.param_frame, text="Reset to last fitted params",
                  command=self.reset_params).pack(pady=2)

    def on_slider_change(self):
        self.preview_live_fit()

    def reset_params(self):
        if self.popt is None:
            return
        for name, val in zip(self.param_names, self.popt):
            self.param_vars[name].set(float(val))
        self.preview_live_fit()

    def collect_params(self):
        return [float(self.param_vars[n].get()) for n in self.param_names]

    def preview_live_fit(self):
        if self.curve is None or self.popt is None:
            return
        try:
            cfg = get_model_config(self.model_type.get(), S=self.S_ratio.get())
            params = self.collect_params()
            y_full = cfg["model"](self.tau, *params)
            x_n, y_n = self._normalize_curve_for_overlay(self.tau, y_full)
            self.ax_acf.semilogx(x_n, y_n, "-", lw=1, alpha=0.35, label="Preview")
            self.ax_acf.legend(loc="best", fontsize=8)
            self.canvas_fit.draw_idle()
        except Exception:
            pass

    def update_live_fit(self):
        if self.curve is None or self.popt is None:
            return
        try:
            cfg = get_model_config(self.model_type.get(), S=self.S_ratio.get())
            params = self.collect_params()
            model = cfg["model"]

            if self.fit_mask is not None and np.any(self.fit_mask):
                tau_fit = self.tau[self.fit_mask]
                g_fit = self.curve[self.fit_mask]
                y_pred_fit = model(tau_fit, *params)
                residuals, R2, SSR, MSE, chi2 = compute_metrics(g_fit, y_pred_fit)
            else:
                y_pred_fit = model(self.tau, *params)
                residuals, R2, SSR, MSE, chi2 = compute_metrics(self.curve, y_pred_fit)

            if not self.keep_overlay.get():
                self.ax_acf.clear()
                self.ax_res.clear()

            x_plot, y_plot = self._normalize_curve_for_overlay(self.tau, self.curve)
            self.ax_acf.semilogx(x_plot, y_plot, "o", ms=3, label="ACF")

            if self.fit_mask is not None and np.any(self.fit_mask):
                x_fit_n, y_fit_n = self._normalize_curve_for_overlay(self.tau[self.fit_mask], y_pred_fit)
                self.ax_acf.semilogx(x_fit_n, y_fit_n, "m--", lw=2, label="Live-fit")
                self.ax_res.semilogx(self.tau[self.fit_mask], residuals, "r.-", ms=3)
            else:
                x_full_n, y_full_n = self._normalize_curve_for_overlay(self.tau, y_pred_fit)
                self.ax_acf.semilogx(x_full_n, y_full_n, "m--", lw=2, label="Live-fit")
                self.ax_res.semilogx(self.tau, self.curve - y_pred_fit, "r.-", ms=3)

            self.ax_acf.grid(True, which="both", ls=":")
            self.ax_acf.legend(loc="best", fontsize=8)
            self.ax_res.axhline(0.0, color="gray", ls="--", lw=0.8)
            self.ax_res.set_xlabel("Lag time τ [ms]")
            self.ax_res.set_ylabel("Residuals")
            self.fig_fit.tight_layout()
            self.canvas_fit.draw()

            self.metrics_label.config(text=f"R²={R2:.4f}  SSR={SSR:.3e}  MSE={MSE:.3e}  χ²={chi2:.3e}")
        except Exception as e:
            messagebox.showwarning("Live update error", str(e))


    # --------------------------
    # 保存系
    # --------------------------
    def save_pdf(self):
        if self.curve is None:
            messagebox.showwarning("No curve", "No curve to save. Fit first.")
            return
        out = filedialog.asksaveasfilename(
            title="Save PDF", defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")],
            initialfile=(f"{getattr(self, 'loaded_basename', 'fit')}_{self.model_type.get()}_fit.pdf")
        )
        if not out:
            return
        try:
            with PdfPages(out) as pdf:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.8, 10.6),
                                               gridspec_kw={'height_ratios': [2.0, 1.0]}, sharex=True)
                # data
                x_plot, y_plot = self.tau, self.curve
                ax1.semilogx(x_plot, y_plot, "o", ms=4, label="ACF")

                # fit preview line（現在の slider 値）
                cfg = get_model_config(self.model_type.get(), S=self.S_ratio.get())
                model = cfg["model"]
                y_full = model(self.tau, *self.popt)

                x_full_n, y_full_n = self.tau, y_full
                ax1.semilogx(x_full_n, y_full_n, "k-", lw=1.6, label="Fit (full)")
                if self.fit_mask is not None and np.any(self.fit_mask):
                    x_fit = self.tau[self.fit_mask]
                    y_fit = model(x_fit, *self.popt)
                    x_fit_n, y_fit_n = x_fit, y_fit
                    ax1.semilogx(x_fit_n, y_fit_n, "m--", lw=2.2, label="Fit (range)")
                    resid = self.curve[self.fit_mask] - y_fit
                    ax2.semilogx(x_fit, resid, "r.-", ms=4)
                else:
                    resid = self.curve - y_full
                    ax2.semilogx(self.tau, resid, "r.-", ms=4)

                ax1.grid(True, which="both", ls=":")
                ax1.set_ylabel("G(τ)")
                ax1.legend(loc="best", fontsize=9)
                ax1.set_title(
                    f"Model: {self.model_type.get()} | S={self.S_ratio.get()} | file={self.loaded_basename or 'N/A'}"
                )
                ax2.axhline(0.0, color="gray", ls="--", lw=0.9)
                ax2.set_xlabel("Lag time τ [ms]")
                ax2.set_ylabel("Residuals")
                ax2.grid(True, which="both", ls=":")
                fig.tight_layout()
                pdf.savefig(fig, dpi=300)
                plt.close(fig)
            messagebox.showinfo("Saved", f"PDF saved:\n{out}")
        except Exception as e:
            messagebox.showerror("Save error", f"Failed to save PDF:\n{e}")
    
    def save_params_csv(self):
        if self.popt is None:
            messagebox.showwarning("No fit", "Nothing to save. Fit first.")
            return

        out = filedialog.asksaveasfilename(
            title="Save params CSV", defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=(f"{getattr(self, 'loaded_basename', 'fit')}_{self.model_type.get()}_params.csv")
        )
        if not out:
            return

        try:
            # metrics 再計算
            cfg = get_model_config(self.model_type.get(), S=self.S_ratio.get())
            model = cfg["model"]
            params = list(self.popt)

            if self.fit_mask is not None and np.any(self.fit_mask):
                tau_fit = self.tau[self.fit_mask]
                g_fit = self.curve[self.fit_mask]
                y_pred_fit = model(tau_fit, *params)
                residuals, R2, SSR, MSE, chi2 = compute_metrics(g_fit, y_pred_fit)
                tau_min_ms, tau_max_ms = float(tau_fit[0]), float(tau_fit[-1])
            else:
                y_pred_full = model(self.tau, *params)
                residuals, R2, SSR, MSE, chi2 = compute_metrics(self.curve, y_pred_full)
                tau_min_ms, tau_max_ms = float(self.tau[0]), float(self.tau[-1])

            row = {
                "file": getattr(self, "loaded_basename", ""),
                "Model": self.model_type.get()
            }

            for name, val in zip(self.param_names, params):
                row[name] = float(val)

            # τD → D 換算（あれば）
            w0 = self.w0_um.get()  # µm
            # 1コンポーネント
            if "tau_D" in self.param_names:
                tauD_ms = row["tau_D"]
                D = (w0 ** 2) / (4.0 * (tauD_ms * 1e-3))
                row["D_um2_s"] = D
            # 2コンポーネント
            if "tau_D1" in self.param_names and "tau_D2" in self.param_names:
                tauD1_ms = row["tau_D1"]
                tauD2_ms = row["tau_D2"]
                D1 = (w0 ** 2) / (4.0 * (tauD1_ms * 1e-3))
                D2 = (w0 ** 2) / (4.0 * (tauD2_ms * 1e-3))
                row["D1_um2_s"] = D1
                row["D2_um2_s"] = D2

            row.update({
                "S": float(self.S_ratio.get()),
                "w0_um": float(self.w0_um.get()),
                "R2": R2, "SSR": SSR, "MSE": MSE, "Chi2": chi2,
            })

            # 選択範囲
            if self.pos_selection is not None:
                row["pos_min"] = int(self.pos_selection[0])
                row["pos_max"] = int(self.pos_selection[1])
            else:
                row["pos_index"] = "single_click"

            row["tau_min_ms"] = tau_min_ms
            row["tau_max_ms"] = tau_max_ms

            df = pd.DataFrame([row])
            df.to_csv(out, index=False)
            messagebox.showinfo("Saved", f"Params CSV saved:\n{out}")
        except Exception as e:
            messagebox.showerror("Save error", f"Failed to save params CSV:\n{e}")

    def save_fit_curve_csv(self):
        if self.popt is None or self.curve is None:
            messagebox.showwarning("No fit", "Nothing to save. Fit first.")
            return

        out = filedialog.asksaveasfilename(
            title="Save curve CSV", defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=(f"{getattr(self, 'loaded_basename', 'fit')}_{self.model_type.get()}_curve.csv")
        )
        if not out:
            return

        try:
            cfg = get_model_config(self.model_type.get(), S=self.S_ratio.get())
            model = cfg["model"]
            y_fit = model(self.tau, *self.popt)
            resid = self.curve - y_fit

            df = pd.DataFrame({
                "tau_ms": self.tau,
                "G_obs": self.curve,
                "G_fit": y_fit,
                "Residuals": resid
            })
            df.to_csv(out, index=False)
            messagebox.showinfo("Saved", f"Curve CSV saved:\n{out}")
        except Exception as e:
            messagebox.showerror("Save error", f"Failed to save curve CSV:\n{e}")


# ===========================
# main
# ===========================

if __name__ == "__main__":
    root = tk.Tk()
    app = FCSMatrixFitGUI(root)
    root.mainloop()
