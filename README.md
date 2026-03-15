# FCS-analysis
このリポジトリは、蛍光相関分光法（FCS）解析するためのPythonベースのGUIツールセット一式を含んでいます。
収録ツール一覧
1. Photobleaching Correction Tool (bleach_correction_GUI.py)
蛍光褪色を補正するためのツールです。
•	主な機能:
o	.lsm および .czi ファイルの複数選択・一括処理。
o	多重指数関数フィッティング: 2成分指数関数モデルを用いた蛍光褪色補正。
o	補正済みデータ（.txt）、フィッティングプロット（.pdf）、およびパラメータ（.csv）の保存。
o	ファイル名からのスキャン時間（ms）の自動抽出。
2. GPU-based Autocorrelation function Tool
CuPyを利用して、GPU上で高速に自己相関計算を行うツールです。
•	主な機能:
o	Multiple-tau法: 広い時間レンジにわたる自己相関関数を効率的に計算。
o	時空間自己相関: 走査型FCSデータから G(Δx,τ) を計算。
o	GPU加速による大規模データセットの高速処理。
3. FCS Fitting GUI (FCS_fit_GUI_standard.py)
FCSの自己相関関数（ACF）行列を読み込み、動的にモデルを生成してフィッティングを行うツールです。
•	主な機能:
o	ACF行列（ポジション x ラグ）の読み込み。
o	ヒートマップ上での解析ポジション選択（単一または範囲平均）。
o	動的モデル生成: 2D/3D自由並進拡散、Triplet項、指数成分、オフセットの有無などを組み合わせたモデルを自動生成。
o	GUI上でのパラメータ初期値（p0）および境界条件（bounds）の直接編集。
o	フィッティング精度（R2, MSE, χ2）の算出と残差表示。
o	フィッティング結果のPDF出力。

セットアップ
必須ライブラリ
以下のライブラリがインストールされている必要があります。
pip install numpy pandas matplotlib scipy tifffile czifile
※ GPU自己相関ツールを使用する場合は、CUDA環境に応じた cupy のインストールが必要です。
実行方法
各スクリプトを直接実行するとGUIが起動します。

# 蛍光褪色補正GUIの起動
python bleach_correction_GUI.py

# 自己相関関数計算GUIの起動
python autocorrelation_gui.py

# フィッティングGUIの起動
python fcs_fitting_gui.py
 
注記: スキャン時間は、ファイル名に _1.0ms のように記述することも可能です。
