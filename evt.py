import numpy as np
import pandas as pd
from scipy.stats import genpareto, t, norm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import argparse
from datetime import datetime

# ==========================================
# 0. グローバル定数
# ==========================================
ANNUAL_TRADING_DAYS = 250
THRESHOLD_QUANTILE = 0.95
SHOCK_SCENARIOS = [
    {'name': 'BoJ Shock Scenario', 'date': '2024-08-05'},
    {'name': 'Trump Shock Scenario', 'date': '2025-04-07'},
]
DATA_FILE_PATH = r"C:\Users\yota-\Desktop\study\mystudy\金融経済分析\極値統計学\data\index_values.xlsx"


# ==========================================
# 1. データ準備
# ==========================================
def load_and_prepare_data(index_name: str, start_year: int) -> pd.DataFrame:
    """
    Excelファイルから市場データを読み込み、指定された期間とインデックスでフィルタリングし、対数リターンを計算する。
    """
    if not os.path.exists(DATA_FILE_PATH):
        raise FileNotFoundError(f"データファイルが見つかりません: {DATA_FILE_PATH}")

    df = pd.read_excel(DATA_FILE_PATH, index_col=0, parse_dates=True)
    
    # リターン計算
    return_col = f"return_{index_name}"
    df[return_col] = np.log(1.0 + df[index_name].pct_change())
    
    # 欠損値処理と期間フィルタリング
    df.dropna(subset=[return_col], inplace=True)
    df = df[df.index.year >= start_year]
    
    start_date = df.index.min().date()
    end_date = df.index.max().date()
    print(f"データ期間: {start_date} から {end_date} まで")
    
    return df


# ==========================================
# 2. EVT分析クラス
# ==========================================
class EVTAnalyzer:
    """極値理論（一般化パレート分布）に基づき分析を行うクラス"""
    def __init__(self, data_series: pd.Series, threshold_quantile=0.95):
        self.data = data_series  # 損失データ（正の値）
        self.u = np.percentile(self.data, threshold_quantile * 100)
        self.excess_data = self.data[self.data > self.u] - self.u
        self.Nu = len(self.excess_data)
        self.n = len(self.data)
        self.params = None

    def fit(self):
        """GPDパラメータを最尤推定法でフィッティングする"""
        if self.Nu < 10: # 推定に必要な最小サンプル数を設定
            raise RuntimeError("Threshold u を超えるサンプルが少なすぎます。")
            
        xi, _, beta = genpareto.fit(self.excess_data, floc=0)
        self.params = {'xi': xi, 'beta': beta}
        return self.params

    def calculate_return_period(self, loss_value: float) -> float:
        """与えられた損失値に対するリターンピリオド（年）を計算する"""
        if self.params is None:
            raise ValueError("モデルがフィッティングされていません。fit()を先に呼び出してください。")
        if loss_value <= self.u:
            return 0

        xi, beta = self.params['xi'], self.params['beta']
        term = 1 + xi * (loss_value - self.u) / beta
        
        if xi <= 0 or term <= 0:
            return float('inf')
        
        prob_exceed = (self.Nu / self.n) * (term ** (-1 / xi))
        return 1 / (prob_exceed * ANNUAL_TRADING_DAYS) if prob_exceed > 0 else float('inf')


# ==========================================
# 3. プロット描画関数
# ==========================================
def plot_evt_diagnostics(analyzer: EVTAnalyzer, shock_name: str, actual_loss: float, return_period: float, t_params: dict, index_name: str):
    """4つの診断プロット（MRL, QQ, Return Level, Density）を描画し、ファイルに保存する"""
    if analyzer.params is None:
        print(f"[{shock_name}] EVTモデルのパラメータがないため、プロットをスキップします。")
        return

    xi, beta = analyzer.params['xi'], analyzer.params['beta']
    
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"EVT Diagnostics for {index_name}: {shock_name}", fontsize=18, y=0.97)

    # 1. Mean Residual Life Plot
    ax1 = fig.add_subplot(2, 2, 1)
    thresholds = np.linspace(analyzer.u * 0.5, np.percentile(analyzer.data, 99.9), 100)
    mrl = [np.mean(analyzer.data[analyzer.data > t] - t) if len(analyzer.data[analyzer.data > t]) > 5 else np.nan for t in thresholds]
    ax1.plot(thresholds * 100, np.array(mrl) * 100, marker='o', markersize=3, linestyle='-')
    ax1.axvline(analyzer.u * 100, color='r', linestyle='--', label=f'Threshold u={analyzer.u*100:.2f}%')
    ax1.set(title='1. Mean Residual Life Plot', xlabel='Threshold (Loss %)', ylabel='Mean Excess Loss (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.4)

    # 2. QQ Plot
    ax2 = fig.add_subplot(2, 2, 2)
    sorted_excess = analyzer.excess_data.sort_values(ascending=True)
    empirical_prob = (np.arange(1, analyzer.Nu + 1)) / (analyzer.Nu + 1)
    theoretical_quantiles = genpareto.ppf(empirical_prob, c=xi, scale=beta)
    ax2.scatter(theoretical_quantiles, sorted_excess.values, alpha=0.6)
    max_val = max(sorted_excess.max(), theoretical_quantiles.max())
    ax2.plot([0, max_val], [0, max_val], 'r--')
    
    # 上位5件に日付ラベルを追加
    if analyzer.Nu >= 5:
        top5 = sorted_excess.tail(5)
        top5_theoretical_quantiles = theoretical_quantiles[-5:]
        for i in range(len(top5)):
            date_obj = top5.index[i]
            date_label = f"{date_obj.year}/{date_obj.month}/{date_obj.day}"
            x_pos = top5_theoretical_quantiles[i]
            y_pos = top5.values[i]
            ax2.text(x_pos, y_pos, date_label, fontsize=9, ha='right', va='bottom')
            
    ax2.set(title=f'2. QQ Plot (Shape xi={xi:.2f})', xlabel='Theoretical Quantiles (Model)', ylabel='Empirical Quantiles (Data)')
    ax2.grid(True, alpha=0.4)

    # 3. Return Level Plot
    ax3 = fig.add_subplot(2, 2, 3)
    rps = np.logspace(0, 2, 100) # 1年から1000年
    prob = 1 / (ANNUAL_TRADING_DAYS * rps)
    if xi > 0:
        evt_levels = analyzer.u + (beta / xi) * ((((analyzer.n / analyzer.Nu) * prob)**(-xi)) - 1)
        ax3.plot(rps, evt_levels * 100, color='red', label='EVT Model')
    t_levels = t.ppf(1 - prob, **t_params)
    ax3.plot(rps, t_levels * 100, color='green', linestyle='--', label='t-dist Model')
    ax3.scatter(return_period, actual_loss * 100, color='black', s=100, zorder=5, label=f'Actual Shock\n(-{actual_loss*100:.1f}%)')
    ax3.set(title='3. Return Level Plot', xlabel='Return Period [Years] (Log Scale)', ylabel='Loss Magnitude (%)')
    ax3.set_xscale('log')
    ax3.legend()
    ax3.grid(True, which="both", ls="--", alpha=0.4)
    
    # 4. Density Plot
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.hist(analyzer.excess_data, bins=30, density=True, alpha=0.7, label='Empirical Histogram')
    x_range = np.linspace(0, analyzer.excess_data.max(), 200)
    pdf_gpd = genpareto.pdf(x_range, c=xi, scale=beta)
    ax4.plot(x_range, pdf_gpd, 'r-', lw=2, label='Fitted GPD PDF')
    ax4.set(title='4. Density of Excess Losses', xlabel='Excess Loss over Threshold u', ylabel='Probability Density')
    ax4.legend()
    ax4.grid(True, alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    safe_shock_name = shock_name.replace(' ', '_')
    output_path = f"output/{index_name}_{safe_shock_name}_diagnostics_2x2.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"-> 診断プロットを保存しました: {output_path}")


# ==========================================
# 4. 分析実行プロセス
# ==========================================
def analyze_and_report_shock(df: pd.DataFrame, index_name: str, scenario: dict):
    """単一のショックシナリオについて分析し、結果を報告する"""
    shock_date = pd.Timestamp(scenario['date'])
    return_col = f"return_{index_name}"
    
    # ショック当日のリターンを実データから取得
    if shock_date not in df.index:
        print(f"{scenario['name']:<25} | データなし: {shock_date.date()} の市場データが存在しません。")
        return
        
    actual_loss = -df.loc[shock_date, return_col]
    if pd.isna(actual_loss) or actual_loss <= 0:
        print(f"{scenario['name']:<25} | スキップ: {shock_date.date()} の損失が0以下です。({-actual_loss*100:.2f}%)")
        return

    # Walk-forward: ショック発生前日までのデータでモデルを訓練
    train_df = df.loc[df.index < shock_date]
    losses = -train_df[return_col].dropna()
    #losses = losses[losses > 0] # 損失のみを対象

    if len(losses) < 200: # 分析に必要な最小データ数
        print(f"{scenario['name']:<25} | 分析不可: モデル訓練のためのデータが不足しています（{len(losses)}件）。")
        return

    print(f"{scenario['name']:<25} | {actual_loss*100:>6.1f}% | ", end="")
    
    # 1. EVT (GPD) 分析
    try:
        evt = EVTAnalyzer(losses, threshold_quantile=THRESHOLD_QUANTILE)
        params_evt = evt.fit()
        rp_evt = evt.calculate_return_period(actual_loss)
        xi_str = f"{params_evt['xi']:.3f}"
        rp_evt_str = f"{rp_evt:.1f} 年"
    except (ValueError, RuntimeError) as e:
        evt = None
        params_evt = None
        xi_str = "Error"
        rp_evt_str = str(e)

    # 2. 正規分布分析
    loc_norm, scale_norm = norm.fit(losses)
    prob_norm = norm.sf(actual_loss, loc=loc_norm, scale=scale_norm)
    rp_norm = 1 / (prob_norm * ANNUAL_TRADING_DAYS) if prob_norm > 0 else float('inf')

    # 3. t分布分析
    df_t, loc_t, scale_t = t.fit(losses)
    t_params = {'df': df_t, 'loc': loc_t, 'scale': scale_t}
    prob_t = t.sf(actual_loss, **t_params)
    rp_t = 1 / (prob_t * ANNUAL_TRADING_DAYS) if prob_t > 0 else float('inf')
    
    # 結果表示
    print(f"{xi_str:<12} | {rp_evt_str:<18} | {rp_norm:.1f} 年{'':<10} | {rp_t:.1f} 年")
    
    if evt is not None and params_evt is not None:
        plot_evt_diagnostics(evt, scenario['name'], actual_loss, rp_evt, t_params, index_name)

# ==========================================
# 5. メイン関数
# ==========================================
def main(args):
    """メインの分析実行関数"""
    os.makedirs('output', exist_ok=True)
    index_name = args.index.upper()

    print("=" * 80)
    print(f"極値分析 (EVT) を開始します")
    print(f"対象インデックス: {index_name}")
    print(f"分析開始年: {args.start_year}")
    print("=" * 80 + "\n")

    try:
        df_market = load_and_prepare_data(index_name, args.start_year)
    except (FileNotFoundError, KeyError) as e:
        print(f"エラー: {e}")
        return

    print("\n" + "=" * 100)
    print("ショックシナリオ分析")
    print("=" * 100)
    print(f"{ 'イベント':<25} | {'実損失':<8} | {'Xi (EVT)':<12} | {'リターンピリオド (EVT)':<18} | {'リターンピリオド (正規分布)':<22} | {'リターンピリオド (t分布)':<20}")
    print("-" * 100)

    for scenario in SHOCK_SCENARIOS:
        analyze_and_report_shock(df_market, index_name, scenario)
        print("-" * 100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="金融市場データに対する極値分析（EVT）を実行します。")
    parser.add_argument(
        "--index", 
        type=str, 
        default="NIKKEI", 
        choices=["NIKKEI", "TOPIX"],
        help="分析対象のインデックス名 (デフォルト: NIKKEI)"
    )
    parser.add_argument(
        "--start-year", 
        type=int, 
        default=2000,
        help="分析に使用するデータの開始年 (デフォルト: 2000)"
    )
    args = parser.parse_args()
    main(args)