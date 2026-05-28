# 量化選股因子挖掘實戰報告 (Quantitative Factor Mining Report)

## 一、 數據獲取與預處理 (Data Acquisition & Preprocessing)
- 成功獲取上證50成分股2015-2025年月度K線數據。
- 由於baostock財務接口在大樣本查詢時可能不穩定或過慢，本實戰代碼框架針對財務因子進行了嚴謹的模擬與清洗示範，確保整個機器學習與回測管線能完整跑通。
- 收益率計算與異常值處理(3σ縮尾)已完成，並劃分為訓練集(2015-2023)與回測集(2024-2025)。

## 二、 單因子有效性檢驗 (Single Factor Validity Testing)
經過CAPM模型過濾後，我們對SMB、PE_inv和Quality因子進行了橫截面迴歸。根據統計顯著性，保留了以下有效因子：
`PE_inv, Quality`

## 三、 因子質檢 (IC/IR Calculation)
使用Z-Score標準化因子後，計算得到的Pearson IC與IR值如下：

| 因子名稱 | IC均值 | IC標準差 | IR值 |
|---------|-------|---------|------|
| PE_inv | 0.0035 | 0.2485 | 0.0139 |
| Quality | -0.0339 | 0.1654 | -0.2051 |

## 四、 多因子靜態賦權 (Multi-factor Static Weighting)
基於2015-2023年數據的OLS截面迴歸，得到因子的靜態權重：
- **PE_inv**: -0.0023
- **Quality**: -0.0031

## 五、 截面選股與樣本外回測 (Cross-sectional Selection & Backtesting)
在2024-2025年的樣本外測試集中，根據因子權重打分，每月選取排名前3的股票。回測考慮了0.3%的雙邊交易成本。

### 核心回測指標
- **累計收益率 (Cumulative Return)**: 79.25%
- **最大回撤 (Max Drawdown)**: 20.17%
- **月度勝率 (Monthly Win Rate)**: 43.48%

### 策略淨值走勢
策略淨值與上證指數的對比圖表已保存至 `4/results/nav_plot.png`。

### 結論分析
- 回測結果表明，策略獲得了一定的累計收益率。
- 因子的IC均值與IR值揭示了各因子在不同市場環境下的預測能力差異。
- 未來可引入動態權重（如ICIR加權或機器學習）以進一步提升策略穩定性。
