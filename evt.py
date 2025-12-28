import numpy as np
import pandas as pd
from scipy.stats import genpareto, t, norm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ==========================================
# 1. データ準備
# ==========================================
def input_market_data():
    """
    Excelファイルから市場データを読み込み、対数リターンを計算する
    """
    # openpyxl が必要: pip install openpyxl
    path = r"C:\Users\yota-\Desktop\study\mystudy\金融経済分析\極値統計学\data\index_values.xlsx"
    df = pd.read_excel(path, index_col=0, parse_dates=True)
    df["return_TOPIX"] = np.log(1.0 + df["TOPIX"].pct_change())
    df["return_NIKKEI"] = np.log(1.0 + df["NIKKEI"].pct_change())
    return df

# ==========================================
# 2. EVT分析クラス
# ==========================================
class EVTAnalyzer:
    def __init__(self, data_series: pd.Series, threshold_quantile=0.95):
        # pd.Seriesをそのまま受け取り、日付インデックスを保持
        self.data = data_series # 損失データ（正の値）
        self.u = np.percentile(self.data, threshold_quantile * 100)
        
        # 日付情報を持つpd.Seriesとして超過データを保持
        self.excess_data = self.data[self.data > self.u] - self.u
        self.Nu = len(self.excess_data)
        self.n = len(self.data)
        self.model_fit = None
        
    def fit(self):
        # GPDパラメータ推定 (MLE)
        xi, _, beta = genpareto.fit(self.excess_data, floc=0)
        self.params = {'xi': xi, 'beta': beta}
        
        # 標準誤差 (Standard Error) の簡易計算 (漸近分散)
        self.se_xi = (1 + xi) / np.sqrt(self.Nu)
        self.ci_xi = (xi - 1.96 * self.se_xi, xi + 1.96 * self.se_xi)
        
        return self.params

    def calculate_return_period(self, loss_value):
        xi = self.params['xi']
        beta = self.params['beta']
        u = self.u
        
        if loss_value <= u:
            return 0
        
        # GPD生存関数
        term = 1 + xi * (loss_value - u) / beta
        if xi <= 0 or term <= 0:
            return float('inf')
        
        prob_exceed = (self.Nu / self.n) * (term ** (-1/xi))
        return 1 / (prob_exceed * ANNUAL_TRADING_DAYS) if prob_exceed > 0 else float('inf')

# ==========================================
# 3. プロット描画関数
# ==========================================
def plot_evt_diagnostics(analyzer, shock_name, actual_loss, return_period, t_params):
    """
    4つの重要な診断プロット（2x2）を描画し、ファイルに保存する
    """
    xi = analyzer.params['xi']
    beta = analyzer.params['beta']
    u = analyzer.u
    excess = analyzer.excess_data # 日付インデックスを持つSeries
    
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    fig.suptitle(f"EVT Diagnostics for NIKKEI: {shock_name}", fontsize=18, y=0.97)

    # --- Plot 1: Mean Residual Life Plot (MRL) ---
    ax1 = fig.add_subplot(gs[0, 0])
    thresholds = np.linspace(u * 0.5, np.percentile(analyzer.data, 99.9), 100)
    mrl = [np.mean(analyzer.data[analyzer.data > t_val] - t_val) if len(analyzer.data[analyzer.data > t_val]) > 5 else np.nan for t_val in thresholds]
            
    ax1.plot(thresholds * 100, np.array(mrl) * 100, marker='o', markersize=3, linestyle='-')
    ax1.axvline(u * 100, color='r', linestyle='--', label=f'Selected Threshold u={u*100:.2f}%')
    ax1.set_xlabel('Threshold (Loss %)')
    ax1.set_ylabel('Mean Excess Loss (%)')
    ax1.set_title('1. Mean Residual Life Plot')
    ax1.legend()
    ax1.grid(True, alpha=0.4)

    # --- Plot 2: QQ Plot with Top 5 Labels ---
    ax2 = fig.add_subplot(gs[0, 1])
    sorted_excess = excess.sort_values(ascending=True)
    n_excess = len(sorted_excess)
    empirical_prob = np.arange(1, n_excess + 1) / (n_excess + 1)
    
    theoretical_quantiles = genpareto.ppf(empirical_prob, c=xi, scale=beta)
    
    ax2.scatter(theoretical_quantiles, sorted_excess.values, alpha=0.6, color='b')
    max_val = max(sorted_excess.max(), theoretical_quantiles.max())
    ax2.plot([0, max_val], [0, max_val], 'r--')
    
    # 上位5件に日付ラベルを追加
    top5 = sorted_excess.tail(5)
    top5_theory_q = genpareto.ppf((n_excess - 4) / (n_excess + 1), c=xi, scale=beta) # おおよその位置
    
    for i in range(len(top5)):
        date_label = top5.index[i].strftime('%Y/%#m/%#d')
        x_pos = theoretical_quantiles[-(5-i)]
        y_pos = top5.values[i]
        ax2.text(x_pos, y_pos, date_label, fontsize=9, ha='right', va='bottom')

    ax2.set_xlabel('Theoretical Quantiles (Model)')
    ax2.set_ylabel('Empirical Quantiles (Real Data)')
    ax2.set_title(f'2. QQ Plot (Shape xi={xi:.2f})')
    ax2.grid(True, alpha=0.4)

    # --- Plot 3: Return Level Plot ---
    ax3 = fig.add_subplot(gs[1, 0])
    rps = np.logspace(0, 2, 100) # 1年〜100年
    prob = 1 / (ANNUAL_TRADING_DAYS * rps)
    
    # EVTモデルのリターンレベル
    if xi > 0:
        evt_levels = u + (beta / xi) * ((((analyzer.n / analyzer.Nu) * prob)**(-xi)) - 1)
        ax3.plot(rps, evt_levels * 100, color='red', label='EVT Model')

    # t分布モデルのリターンレベル
    t_levels = t.ppf(1 - prob, **t_params)
    ax3.plot(rps, t_levels * 100, color='green', linestyle='--', label='t-dist Model')

    ax3.scatter(return_period, actual_loss * 100, color='black', s=100, zorder=5, 
                label=f'Actual Shock\n(-{actual_loss*100:.1f}%)')
    
    ax3.set_xscale('log')
    ax3.set_xlabel('Return Period [Years] (Log Scale)')
    ax3.set_ylabel('Loss Magnitude (%)')
    ax3.set_title('3. Return Level Plot')
    ax3.legend()
    ax3.grid(True, which="both", ls="--", alpha=0.4)
    
    # --- Plot 4: Density Plot ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(excess, bins=30, density=True, alpha=0.7, label='Empirical Histogram')
    
    # GPDのPDFをプロット
    x_range = np.linspace(0, excess.max(), 200)
    pdf_gpd = genpareto.pdf(x_range, c=xi, scale=beta)
    ax4.plot(x_range, pdf_gpd, 'r-', lw=2, label='Fitted GPD PDF')
    
    ax4.set_xlabel('Excess Loss over Threshold u')
    ax4.set_ylabel('Probability Density')
    ax4.set_title('4. Density of Excess Losses')
    ax4.legend()
    ax4.grid(True, alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # --- ファイルに保存 ---
    safe_shock_name = shock_name.replace(' ', '_').replace('(', '').replace(')', '')
    output_path = f"output/NIKKEI_{safe_shock_name}_diagnostics_2x2.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig) # メモリ解放
    
    print(f"-> Diagnostic plot saved to: {output_path}")

# ==========================================
# 4. メイン実行プロセス (Walk-Forward)
# ==========================================

# --- 初期設定 ---
ANNUAL_TRADING_DAYS = 250
os.makedirs('output', exist_ok=True)
START_DATE = '2000-01-01'

# --- データ準備 ---
print("Loading market data from Excel...")
df_market = input_market_data()
df_market = df_market.loc[df_market.index >= START_DATE].copy()
df_market.dropna(subset=['return_NIKKEI'], inplace=True)
print(f"Data loaded. Using records from {df_market.index.min().date()} to {df_market.index.max().date()}.")


# --- 分析ターゲット定義 ---
targets = [
    {'name': 'BoJ Shock Scenario', 'date': '2024-08-05', 'loss': -0.124},
    {'name': 'Trump Shock Scenario', 'date': '2025-04-07', 'loss': -0.150}
]

# 仮定したショックをデータに追加
for tgt in targets:
    shock_date = pd.Timestamp(tgt['date'])
    if shock_date not in df_market.index:
        new_row = pd.DataFrame({'return_NIKKEI': tgt['loss']}, index=[shock_date])
        df_market = pd.concat([df_market, new_row])
df_market.sort_index(inplace=True)


# --- 分析実行 ---
print("\n" + "="*50)
print("Starting EVT analysis for NIKKEI return shocks...")
print("="*50 + "\n")

print(f"{'Event':<25} | {'Loss':<8} | {'Xi (EVT)':<12} | {'RP (EVT)':<15} | {'RP (Normal)':<15} | {'RP (t-dist)':<15}")
print("-" * 115)

for tgt in targets:
    shock_date = pd.Timestamp(tgt['date'])
    
    train_df = df_market.loc[: shock_date - pd.Timedelta(days=1)].copy()
    
    # 損失データ（プラス化）、pd.Seriesとして渡す
    losses = -train_df['return_NIKKEI']
    actual_loss = -df_market.loc[shock_date, 'return_NIKKEI']
    
    print(f"{tgt['name']:<25} | {actual_loss*100:>6.1f}% | ", end="")
    
    # --- 1. EVT (GPD) による分析 ---
    evt = EVTAnalyzer(losses, threshold_quantile=0.95)
    params_evt = evt.fit()
    rp_evt = evt.calculate_return_period(actual_loss)
    
    # --- 2. 正規分布による分析 ---
    loc_norm, scale_norm = norm.fit(losses)
    prob_norm = norm.sf(actual_loss, loc=loc_norm, scale=scale_norm)
    rp_norm = 1 / (prob_norm * ANNUAL_TRADING_DAYS) if prob_norm > 0 else float('inf')

    # --- 3. t分布による分析 ---
    df_t, loc_t, scale_t = t.fit(losses)
    t_params = {'df': df_t, 'loc': loc_t, 'scale': scale_t} # パラメータを辞書に格納
    prob_t = t.sf(actual_loss, **t_params)
    rp_t = 1 / (prob_t * ANNUAL_TRADING_DAYS) if prob_t > 0 else float('inf')
    
    # --- 結果表示 ---
    xi_str = f"{params_evt['xi']:.3f}"
    rp_evt_str = f"{rp_evt:.1f} Years"
    rp_norm_str = f"{rp_norm:.1f} Years"
    rp_t_str = f"{rp_t:.1f} Years"
    
    print(f"{xi_str:<12} | {rp_evt_str:<15} | {rp_norm_str:<15} | {rp_t_str:<15}")
    
    # EVTの診断プロットを描画・保存
    plot_evt_diagnostics(evt, tgt['name'], actual_loss, rp_evt, t_params)
    print("-" * 115)
