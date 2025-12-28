import numpy as np
import pandas as pd
from scipy.stats import genpareto, t, norm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ==========================================
# 1. データ準備（ダミーデータ生成）
# ==========================================
def generate_market_data():
    """
    ファットテールな市場データを生成し、特定のショックを注入する
    """
    np.random.seed(42)
    dates = pd.date_range(start='2015-01-01', end='2025-12-31', freq='B')
    n = len(dates)
    
    # 日常データ: t分布（自由度3.5）で厚い裾を再現
    returns = t.rvs(df=3.5, loc=0.0002, scale=0.012, size=n)
    df = pd.DataFrame({'Date': dates, 'Return': returns})
    df.set_index('Date', inplace=True)
    
    # シナリオ: 2024年8月 日銀ショック (-12.4%)
    if pd.Timestamp('2024-08-05') in df.index:
        df.loc['2024-08-05', 'Return'] = -0.124

    # シナリオ: 2025年4月 トランプショック (-15.0%)
    if pd.Timestamp('2025-04-15') in df.index:
        df.loc['2025-04-15', 'Return'] = -0.150
        
    return df

# ==========================================
# 2. EVT分析クラス
# ==========================================
class EVTAnalyzer:
    def __init__(self, data_series, threshold_quantile=0.95):
        self.data = data_series # 損失データ（正の値）
        self.u = np.percentile(self.data, threshold_quantile * 100)
        self.excess_data = self.data[self.data > self.u] - self.u
        self.Nu = len(self.excess_data)
        self.n = len(self.data)
        self.model_fit = None
        
    def fit(self):
        # GPDパラメータ推定 (MLE)
        # scipyのgenpareto形状: c=xi, loc=0, scale=beta
        xi, _, beta = genpareto.fit(self.excess_data, floc=0)
        self.params = {'xi': xi, 'beta': beta}
        
        # 標準誤差 (Standard Error) の簡易計算 (漸近分散)
        # Var(xi) approx (1+xi)^2 / Nu
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
        # xi > 0 の場合のみ定義
        if xi <= 0 or term <= 0:
            return float('inf')
        
        prob_exceed = (self.Nu / self.n) * (term ** (-1/xi))
        # 年換算 (ANNUAL_TRADING_DAYS はグローバルスコープから参照)
        return 1 / (prob_exceed * ANNUAL_TRADING_DAYS) if prob_exceed > 0 else float('inf')

# ==========================================
# 3. プロット描画関数
# ==========================================
def plot_evt_diagnostics(analyzer, shock_name, actual_loss, return_period):
    """
    3つの重要な診断プロットを描画し、ファイルに保存する
    """
    xi = analyzer.params['xi']
    beta = analyzer.params['beta']
    u = analyzer.u
    excess = analyzer.excess_data
    
    fig = plt.figure(figsize=(18, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
    fig.suptitle(f"EVT Diagnostics: {shock_name} (Input Data End: Prior Day)", fontsize=16)

    # --- Plot 1: Mean Residual Life Plot (MRL) ---
    ax1 = plt.subplot(gs[0])
    thresholds = np.linspace(analyzer.data.min(), np.percentile(analyzer.data, 98), 50)
    mrl = []
    for t_val in thresholds:
        e = analyzer.data[analyzer.data > t_val] - t_val
        if len(e) > 5: # データが少なすぎる場合は除外
            mrl.append(np.mean(e))
        else:
            mrl.append(np.nan)
            
    ax1.plot(thresholds, mrl, marker='o', markersize=4)
    ax1.axvline(u, color='r', linestyle='--', label=f'Selected Threshold u={u:.3f}')
    ax1.set_xlabel('Threshold (u)')
    ax1.set_ylabel('Mean Excess Loss')
    ax1.set_title('Mean Residual Life Plot')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: QQ Plot ---
    ax2 = plt.subplot(gs[1])
    sorted_excess = np.sort(excess)
    n_excess = len(excess)
    empirical_prob = np.arange(1, n_excess + 1) / (n_excess + 1)
    
    # 理論分位点 (GPD)
    theoretical_quantiles = genpareto.ppf(empirical_prob, c=xi, scale=beta)
    
    ax2.scatter(theoretical_quantiles, sorted_excess, alpha=0.6, color='b')
    max_val = max(sorted_excess.max(), theoretical_quantiles.max())
    ax2.plot([0, max_val], [0, max_val], 'r--')
    ax2.set_xlabel('Theoretical Quantiles (Model)')
    ax2.set_ylabel('Empirical Quantiles (Real Data)')
    ax2.set_title(f'QQ Plot (xi={xi:.2f})')
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Return Level Plot ---
    ax3 = plt.subplot(gs[2])
    rps = np.logspace(0, 4, 100) # 1年〜10000年
    # 再現期間Tから損失額を逆算
    prob = 1 / (ANNUAL_TRADING_DAYS * rps)
    # xi > 0 の場合
    if xi > 0:
        return_levels = u + (beta / xi) * ((((analyzer.n / analyzer.Nu) * prob)**(-xi)) - 1)
        ax3.plot(rps, return_levels * 100, color='red', label='EVT Model')

    # ショックの実績値をプロット
    ax3.scatter(return_period, actual_loss * 100, color='black', s=100, zorder=5, 
                label=f'Actual Shock\n(-{actual_loss*100:.1f}%)')
    
    ax3.set_xscale('log')
    ax3.set_xlabel('Return Period [Years] (Log Scale)')
    ax3.set_ylabel('Loss Magnitude (%)')
    ax3.set_title('Return Level Plot')
    ax3.legend()
    ax3.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # --- ファイルに保存 ---
    # ファイル名に使えない文字を置換
    safe_shock_name = shock_name.replace(' ', '_').replace('(', '').replace(')', '')
    output_path = f"output/{safe_shock_name}_diagnostics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig) # メモリ解放
    
    # 保存したことを通知
    print(f"-> Diagnostic plot saved to: {output_path}")


# ==========================================
# 4. メイン実行プロセス (Walk-Forward)
# ==========================================

# --- 初期設定 ---
# 年間営業日数を定数化
ANNUAL_TRADING_DAYS = 250
# 出力フォルダを作成
os.makedirs('output', exist_ok=True)

# --- データ準備 ---
df_market = generate_market_data()

# --- 分析ターゲット定義 ---
targets = [
    {'name': 'BoJ Shock (2024)', 'date': '2024-08-05'},
    {'name': 'Trump Shock (2025)', 'date': '2025-04-15'}
]

# --- 分析実行 ---
print(f"{ 'Event':<20} | { 'Loss':<8} | { 'Xi (EVT)':<12} | { 'RP (EVT)':<15} | { 'RP (Normal)':<15} | { 'RP (t-dist)':<15}")
print("-" * 105)

for tgt in targets:
    shock_date = pd.Timestamp(tgt['date'])
    
    # 【重要】ショック前日までのデータを取得 (Look-ahead Bias排除)
    train_df = df_market.loc[: shock_date - pd.Timedelta(days=1)].copy()
    
    # 損失データ（プラス化）
    losses = -train_df['Return'].values
    actual_loss = -df_market.loc[shock_date, 'Return']
    
    print(f"{tgt['name']:<20} | {actual_loss*100:>6.1f}% | ", end="")
    
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
    prob_t = t.sf(actual_loss, df=df_t, loc=loc_t, scale=scale_t)
    rp_t = 1 / (prob_t * ANNUAL_TRADING_DAYS) if prob_t > 0 else float('inf')
    
    # --- 結果表示 ---
    xi_str = f"{params_evt['xi']:.3f}"
    rp_evt_str = f"{rp_evt:.1f} Years"
    rp_norm_str = f"{rp_norm:.1f} Years"
    rp_t_str = f"{rp_t:.1f} Years"
    
    print(f"{xi_str:<12} | {rp_evt_str:<15} | {rp_norm_str:<15} | {rp_t_str:<15}")
    
    # EVTの診断プロットを描画・保存
    plot_evt_diagnostics(evt, tgt['name'], actual_loss, rp_evt)
    print("-" * 105)
