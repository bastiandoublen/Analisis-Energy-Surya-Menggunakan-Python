import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

# Konfigurasi

NASA_CSV_FILE = "POWER_Point_Monthly_20150101_20241231_000d13N_117d50E_UTC.csv"

PARAMS  = ["ALLSKY_SFC_SW_DWN", "T2M", "WS10M", "QV2M", "PS", "WSC"]
ETA_STC = 0.18
BETA_T  = 0.004
T_STC   = 25.0

OUT_CSV = "hasil_prediksi_6var_10tahun.csv"
OUT_TXT = "evaluasi_model_6var_10tahun.txt"
OUT_PNG = "plot_6var_10tahun.png"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Baca dan parsing file NASA POWER

csv_path = os.path.join(BASE_DIR, NASA_CSV_FILE)
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"File tidak ditemukan: {csv_path}")

skip_rows = 0
with open(csv_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if "-END HEADER-" in line:
            skip_rows = i + 1
            break

raw_df = pd.read_csv(csv_path, skiprows=skip_rows, header=0, na_values=["-999", -999])

# Reshape ke format bulanan

month_cols = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
              "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

records = []
for _, row in raw_df.iterrows():
    param = str(row["PARAMETER"]).strip()
    if param not in PARAMS:
        continue
    year = int(row["YEAR"])
    for mo_idx, mo_name in enumerate(month_cols, start=1):
        records.append({
            "date"     : datetime(year, mo_idx, 1),
            "parameter": param,
            "value"    : row[mo_name]
        })

long_df = pd.DataFrame(records)
wide_df = long_df.pivot_table(index="date", columns="parameter", values="value", aggfunc="first")
wide_df.sort_index(inplace=True)

missing_params = [p for p in PARAMS if p not in wide_df.columns]
if missing_params:
    raise ValueError(f"Parameter tidak ditemukan di CSV: {missing_params}")

wide_df.replace(-999, np.nan, inplace=True)
wide_df.dropna(inplace=True)

# Hitung variabel dependen Y (model PV)

G = wide_df["ALLSKY_SFC_SW_DWN"].values
T = wide_df["T2M"].values

Y_actual = G * ETA_STC * (1 - BETA_T * (T - T_STC))

# Bentuk matriks X tanpa intersep 

X = wide_df[PARAMS].values   # shape (n, 6)

# OLS — Normal Equation: koef = (X^T X)^{-1} X^T Y

XtX    = X.T @ X
XtY    = X.T @ Y_actual
coeffs = np.linalg.solve(XtX, XtY)

# Prediksi dan residual

Y_pred   = X @ coeffs
residual = Y_actual - Y_pred
ape      = np.abs(residual / Y_actual) * 100

# Metrik evaluasi 

n  = len(Y_actual)
k  = X.shape[1]

SS_res = np.sum(residual ** 2)
SS_tot = np.sum((Y_actual - Y_actual.mean()) ** 2)

R2     = 1 - SS_res / SS_tot
R2_adj = 1 - (1 - R2) * (n - 1) / (n - k)
MAE    = np.mean(np.abs(residual))
RMSE   = np.sqrt(np.mean(residual ** 2))
MAPE   = np.mean(ape)

# Simpan CSV hasil prediksi

result_df = wide_df[PARAMS].copy()
result_df.index.name = "Tanggal"
result_df["Y_aktual"] = Y_actual
result_df["Y_pred"]   = Y_pred
result_df["Residual"] = residual
result_df["APE_pct"]  = ape
result_df.index = result_df.index.strftime("%Y-%m-%d")
result_df.to_csv(os.path.join(BASE_DIR, OUT_CSV))

# Simpan TXT ringkasan evaluasi 

txt_out = os.path.join(BASE_DIR, OUT_TXT)
with open(txt_out, "w", encoding="utf-8") as f:
    f.write("=" * 65 + "\n")
    f.write("  EVALUASI MODEL REGRESI LINEAR BERGANDA OLS (TANPA INTERSEP)\n")
    f.write("  Prediksi Daya Energi Surya — NASA POWER Monthly\n")
    f.write("=" * 65 + "\n\n")

    f.write("--- RINGKASAN DATA ---\n")
    f.write(f"  Lokasi     : Bontang, Kalimantan Timur\n")
    f.write(f"  Koordinat  : 0.1333°N, 117.5°E\n")
    f.write(f"  Periode    : Januari 2015 – Desember 2024\n")
    f.write(f"  Jumlah sampel valid : {n}\n\n")

    f.write("--- KONSTANTA MODEL PV (STC) ---\n")
    f.write(f"  eta_STC  = {ETA_STC}   (efisiensi panel)\n")
    f.write(f"  beta_T   = {BETA_T}  (koefisien temperatur, /°C)\n")
    f.write(f"  T_STC    = {T_STC}  °C\n")
    f.write(f"  Formula  : Y = G × η_STC × [1 − β_T × (T − T_STC)]\n\n")

    f.write("--- MODEL REGRESI (TANPA INTERSEP) ---\n")
    f.write("  Y_pred = a·x1 + b·x2 + c·x3 + d·x4 + e·x5 + f·x6\n\n")
    f.write("  Variabel:\n")
    f.write("    x1 = ALLSKY_SFC_SW_DWN  (Radiasi matahari, kWh/m²/hari)\n")
    f.write("    x2 = T2M                (Suhu udara 2m, °C)\n")
    f.write("    x3 = WS10M              (Kecepatan angin 10m, m/s)\n")
    f.write("    x4 = QV2M               (Kelembaban spesifik 2m, g/kg)\n")
    f.write("    x5 = PS                 (Tekanan permukaan, kPa)\n")
    f.write("    x6 = WSC                (Tutupan awan, %)\n\n")

    f.write("--- KOEFISIEN REGRESI ---\n")
    var_names  = ["ALLSKY_SFC_SW_DWN", "T2M", "WS10M", "QV2M", "PS", "WSC"]
    coef_names = list("abcdef")
    for cn, vn, cv in zip(coef_names, var_names, coeffs):
        f.write(f"  {cn} ({vn:25s}) = {cv:+.8f}\n")

    f.write("\n--- METRIK EVALUASI ---\n")
    f.write(f"  R²            = {R2:.8f}\n")
    f.write(f"  Adjusted R²   = {R2_adj:.8f}\n")
    f.write(f"  MAE           = {MAE:.8f}  kWh/m²/hari\n")
    f.write(f"  RMSE          = {RMSE:.8f}  kWh/m²/hari\n")
    f.write(f"  MAPE          = {MAPE:.6f} %\n")

    f.write("\n--- STATISTIK RESIDUAL ---\n")
    f.write(f"  Min residual  = {residual.min():.6f}\n")
    f.write(f"  Max residual  = {residual.max():.6f}\n")
    f.write(f"  Std residual  = {residual.std():.6f}\n")

    f.write("\n" + "=" * 65 + "\n")
    f.write(f"  Dibuat otomatis pada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 65 + "\n")

# Visualisasi (dark theme)
plt.style.use("dark_background")

fig = plt.figure(figsize=(16, 14), facecolor="#0d1117")
gs  = gridspec.GridSpec(3, 2, figure=fig,
                        hspace=0.45, wspace=0.35,
                        left=0.08, right=0.97, top=0.93, bottom=0.06)

DATE_AXIS    = pd.to_datetime(wide_df.index)
COLOR_ACTUAL = "#58a6ff"
COLOR_PRED   = "#f78166"
COLOR_GRID   = "#30363d"
ACCENT       = "#d2a8ff"

# (a) Time series Y aktual vs prediksi
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(DATE_AXIS, Y_actual, color=COLOR_ACTUAL, lw=1.5,
         label="Y Aktual", marker="o", ms=3, zorder=3)
ax1.plot(DATE_AXIS, Y_pred, color=COLOR_PRED, lw=1.5,
         label="Y Prediksi", marker="s", ms=3, linestyle="--", zorder=3)
ax1.fill_between(DATE_AXIS, Y_actual, Y_pred, alpha=0.12, color="#ffffff")
ax1.set_title("(a) Time Series: Y Aktual vs Y Prediksi (Bulanan, 2015–2024)",
              color="white", fontsize=11, pad=8)
ax1.set_xlabel("Periode", color="#8b949e")
ax1.set_ylabel("Energi (kWh/m²/hari)", color="#8b949e")
ax1.tick_params(colors="#8b949e")
ax1.set_facecolor("#161b22")
ax1.grid(True, color=COLOR_GRID, linewidth=0.6, linestyle="--", alpha=0.7)
ax1.legend(loc="upper right", fontsize=9,
           facecolor="#161b22", edgecolor="#30363d", labelcolor="white")
for sp in ax1.spines.values():
    sp.set_color(COLOR_GRID)
for yr in range(2015, 2025):
    ax1.axvline(pd.Timestamp(f"{yr}-01-01"), color="#30363d", lw=0.8, linestyle=":")
ax1.text(0.01, 0.95,
         f"R²={R2:.4f}  MAPE={MAPE:.2f}%  RMSE={RMSE:.4f}",
         transform=ax1.transAxes, color=ACCENT, fontsize=9, va="top", ha="left",
         bbox=dict(boxstyle="round,pad=0.3", fc="#0d1117", ec=COLOR_GRID, alpha=0.8))

# (b) Scatter plot Y aktual vs prediksi
ax2 = fig.add_subplot(gs[1, 0])
sc = ax2.scatter(Y_actual, Y_pred, c=residual, cmap="coolwarm",
                 s=40, alpha=0.85, edgecolors="none", zorder=3)
lims = [min(Y_actual.min(), Y_pred.min()) * 0.97,
        max(Y_actual.max(), Y_pred.max()) * 1.03]
ax2.plot(lims, lims, "w--", lw=1.2, label="y = x (ideal)", zorder=4)
ax2.set_xlim(lims); ax2.set_ylim(lims)
ax2.set_title("(b) Scatter Plot: Y Aktual vs Y Prediksi",
              color="white", fontsize=11, pad=8)
ax2.set_xlabel("Y Aktual (kWh/m²/hari)", color="#8b949e")
ax2.set_ylabel("Y Prediksi (kWh/m²/hari)", color="#8b949e")
ax2.tick_params(colors="#8b949e")
ax2.set_facecolor("#161b22")
ax2.grid(True, color=COLOR_GRID, linewidth=0.6, linestyle="--", alpha=0.7)
ax2.legend(fontsize=9, facecolor="#161b22", edgecolor="#30363d", labelcolor="white")
for sp in ax2.spines.values():
    sp.set_color(COLOR_GRID)
cb = plt.colorbar(sc, ax=ax2, pad=0.02)
cb.set_label("Residual", color="#8b949e")
cb.ax.yaxis.set_tick_params(color="#8b949e")
plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8b949e")

# (c) Histogram distribusi residual
ax3 = fig.add_subplot(gs[1, 1])
counts, bin_edges, patches = ax3.hist(residual, bins=20, color=ACCENT,
                                       edgecolor="#161b22", alpha=0.85, zorder=3)
for patch, left in zip(patches, bin_edges[:-1]):
    patch.set_facecolor(COLOR_PRED if left < 0 else COLOR_ACTUAL)
ax3.axvline(0, color="white", lw=1.5, linestyle="--", label="Residual = 0")
ax3.axvline(residual.mean(), color="#ffa657", lw=1.2,
            linestyle="-.", label=f"Mean = {residual.mean():.4f}")
ax3.set_title("(c) Distribusi Residual", color="white", fontsize=11, pad=8)
ax3.set_xlabel("Residual (kWh/m²/hari)", color="#8b949e")
ax3.set_ylabel("Frekuensi", color="#8b949e")
ax3.tick_params(colors="#8b949e")
ax3.set_facecolor("#161b22")
ax3.grid(True, color=COLOR_GRID, linewidth=0.6, linestyle="--", alpha=0.7, axis="y")
ax3.legend(fontsize=9, facecolor="#161b22", edgecolor="#30363d", labelcolor="white")
for sp in ax3.spines.values():
    sp.set_color(COLOR_GRID)

# (d) APE per bulan
ax4 = fig.add_subplot(gs[2, 0])
bar_colors = [COLOR_PRED if v > MAPE else COLOR_ACTUAL for v in ape]
ax4.bar(range(n), ape, color=bar_colors, alpha=0.8, zorder=3)
ax4.axhline(MAPE, color="#ffa657", lw=1.4, linestyle="--",
            label=f"MAPE rata-rata = {MAPE:.2f}%")
ax4.set_title("(d) APE per Bulan (%)", color="white", fontsize=11, pad=8)
ax4.set_xlabel("Indeks Sampel (Bulan)", color="#8b949e")
ax4.set_ylabel("APE (%)", color="#8b949e")
ax4.tick_params(colors="#8b949e")
ax4.set_facecolor("#161b22")
ax4.grid(True, color=COLOR_GRID, linewidth=0.6, linestyle="--", alpha=0.7, axis="y")
ax4.legend(fontsize=9, facecolor="#161b22", edgecolor="#30363d", labelcolor="white")
for sp in ax4.spines.values():
    sp.set_color(COLOR_GRID)

# (e) Tabel ringkasan koefisien dan metrik
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis("off")
ax5.set_facecolor("#161b22")

table_data = [
    ["Parameter",              "Koefisien",            ""],
    ["a · ALLSKY_SFC_SW_DWN",  f"{coeffs[0]:+.5f}",   ""],
    ["b · T2M",                f"{coeffs[1]:+.5f}",   ""],
    ["c · WS10M",              f"{coeffs[2]:+.5f}",   ""],
    ["d · QV2M",               f"{coeffs[3]:+.5f}",   ""],
    ["e · PS",                 f"{coeffs[4]:+.5f}",   ""],
    ["f · WSC",                f"{coeffs[5]:+.5f}",   ""],
    ["",                       "",                     ""],
    ["Metrik",                 "Nilai",                ""],
    ["R²",                     f"{R2:.6f}",            ""],
    ["Adj. R²",                f"{R2_adj:.6f}",        ""],
    ["MAE",                    f"{MAE:.6f}",           "kWh/m²/hari"],
    ["RMSE",                   f"{RMSE:.6f}",          "kWh/m²/hari"],
    ["MAPE",                   f"{MAPE:.4f}",          "%"],
]

tbl = ax5.table(cellText=table_data, cellLoc="left", loc="center",
                bbox=[0.0, 0.0, 1.0, 1.0])
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.2)
for (row, col), cell in tbl.get_celld().items():
    cell.set_facecolor("#0d1117" if row % 2 == 0 else "#161b22")
    cell.set_text_props(color="white")
    cell.set_edgecolor(COLOR_GRID)
    if row in (0, 8):
        cell.set_facecolor("#1f6feb")
        cell.set_text_props(color="white", fontweight="bold")

ax5.set_title("(e) Ringkasan Koefisien & Metrik", color="white", fontsize=11, pad=8)

fig.suptitle(
    "Prediksi Daya Energi Surya — Regresi OLS Tanpa Intersep\n"
    "Bontang, Kalimantan Timur | NASA POWER Monthly | 2015–2024",
    color="white", fontsize=13, fontweight="bold", y=0.975
)

png_out = os.path.join(BASE_DIR, OUT_PNG)
fig.savefig(png_out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)

# Output terminal

print("  HASIL ANALISIS REGRESI OLS — PREDIKSI ENERGI SURYA")
print("=" * 60)

var_names  = ["ALLSKY_SFC_SW_DWN", "T2M", "WS10M", "QV2M", "PS", "WSC"]
coef_names = list("abcdef")
print("\n  Koefisien Regresi:")
for cn, vn, cv in zip(coef_names, var_names, coeffs):
    print(f"    {cn} ({vn:25s}) = {cv:+.8f}")

print(f"\n  Metrik Evaluasi:")
print(f"    R²          = {R2:.6f}")
print(f"    Adjusted R² = {R2_adj:.6f}")
print(f"    MAE         = {MAE:.6f}  kWh/m²/hari")
print(f"    RMSE        = {RMSE:.6f}  kWh/m²/hari")
print(f"    MAPE        = {MAPE:.4f} %")

print(f"\n  File output:")
print(f"    • {OUT_CSV}")
print(f"    • {OUT_TXT}")
print(f"    • {OUT_PNG}")
print("=" * 60)