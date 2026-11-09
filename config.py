"""模型参数 —— 调策略改这里，不用动代码"""

# ================= 股票池（硬过滤） =================
MIN_MARKET_CAP = 50e8          # 总市值下限（元）
MIN_AMOUNT = 1e8               # 当日成交额下限（元），保证流动性
EXCLUDE_PREFIX = ("8", "4", "92")  # 排除北交所
MIN_HISTORY_DAYS = 200         # 上市不足约 200 个交易日的次新股不参与

# ================= 第一阶段：截面多因子（全市场，行业中性化） =================
# ep = 1/PE（盈利收益率）  bp = 1/PB（账面市值比）
# roe = PB/PE（推算的 ROE，衡量盈利质量）  low_turn = -ln(换手率)（低换手溢价）
STAGE1_WEIGHTS = {"ep": 0.25, "bp": 0.15, "roe": 0.40, "low_turn": 0.20}
STAGE1_KEEP = 150              # 进入第二阶段的数量

# ================= 第二阶段：时间序列统计 =================
TREND_WINDOW = 60              # 趋势回归窗口（交易日）
MIN_TREND_R2 = 0.40            # 趋势拟合优度下限
STAGE2_WEIGHTS = {
    "trend": 0.35,             # 趋势强度 = 年化斜率 × R²
    "fundamental": 0.25,       # 第一阶段得分
    "low_vol": 0.20,           # 低波动
    "reversal": 0.10,          # 近 5 日短期反转（不追短期急涨）
    "drawdown": 0.10,          # 60 日最大回撤小
}

# ================= 进攻策略：底部启动 =================
BOTTOM_MIN_PCT = 1.0               # 粗筛：今日涨幅 ≥ 1%
BOTTOM_MIN_TURNOVER = 1.0          # 粗筛：换手率 ≥ 1%
BOTTOM_CANDIDATES = 200            # 粗筛后拉历史数据的数量
BOTTOM_MIN_DECLINE = 0.30          # 一年高点到近期低点至少跌 30%
BOTTOM_MIN_DOWN_DAYS = 40          # 下跌持续至少 40 个交易日
BOTTOM_REBOUND_RANGE = (0.05, 0.25)  # 距底部反弹幅度区间（超过 25% 不算刚启动）
BOTTOM_MAX_POSITION = 0.45         # 一年价格区间分位上限
BOTTOM_MIN_VOL_RATIO = 1.3         # 5 日均量 / 60 日均量
BOTTOM_WEIGHTS = {
    "vol_surge": 0.25,             # 放量强度
    "launch": 0.25,                # 启动质量：10 日回归斜率 × R²
    "depth": 0.15,                 # 前期跌幅深度
    "base": 0.15,                  # 底部收敛：筑底期波动 / 长期波动
    "low_pos": 0.10,               # 位置越低越好
    "early": 0.10,                 # 反弹越早越好
}

# ================= 风控 =================
ATR_MULT = 2.0                 # 参考止损 = 收盘价 − ATR_MULT × ATR(14)

# ================= 进攻策略：强势延续（押注连续上涨） =================
MOM_CANDIDATES = 250               # 粗筛后拉历史数据的数量（今日上涨、未涨停、按成交额）
MOM_TURNOVER_RANGE = (1.5, 20.0)   # 换手率区间
MOM_R20_RANGE = (0.08, 0.45)       # 20 日涨幅区间：涨了，但还没涨疯
MOM_MIN_R2 = 0.60                  # 20 日对数价格回归 R² 下限（涨得稳）
MOM_NEAR_HIGH = 0.95               # 收盘价 ≥ 60 日最高收盘价的 95%
MOM_MAX_BIAS20 = 0.15              # 偏离 20 日线上限
MOM_MAX_R5 = 0.18                  # 近 5 日涨幅上限，过滤短期急拉
MOM_VOL_TREND_RANGE = (0.9, 2.5)   # 5 日均量 / 20 日均量：温和放量，不要天量
MOM_WEIGHTS = {"trend": 0.35, "up_days": 0.20, "not_extended": 0.20, "volume": 0.15, "near_high": 0.10}

# ================= 进攻策略：趋势回踩 =================
PB_CANDIDATES = 250
PB_PCT_RANGE = (-1.0, 4.0)         # 今日涨跌幅区间（企稳，非大涨）
PB_MIN_R2 = 0.50                   # 60 日趋势 R² 下限
PB_PULL_RANGE = (-0.15, -0.04)     # 距 20 日最高收盘价回撤区间
PB_MAX_VOL_RATIO = 0.90            # 回调期缩量：5 日均量 / 20 日均量上限
PB_WEIGHTS = {"trend": 0.35, "shrink": 0.25, "support": 0.20, "strength": 0.20}

# ================= 名额分配与复盘 =================
CAND_PER_STRATEGY = 8          # 每个策略最多提供的候选数
TOTAL_SLOTS = 6                # 每日推荐总数（强势市场）
TOTAL_SLOTS_WEAK = 4           # 弱势市场（沪深300 在 60 日线下）
MAX_PER_INDUSTRY = 2           # 最终推荐中同一行业最多几只
RECENT_EXCLUDE_DAYS = 3        # 最近几个交易日推荐过的不再重复推荐（在跟踪池里继续看）
TRACK_DAYS = 10                # 跟踪池保留最近几个交易日的推荐
HALF_LIFE = 7                  # 策略表现的时间衰减半衰期（交易日）
PRIOR_N = 2                    # 先验样本数：数据少时向“不赚不亏”收缩的力度
PRIOR_SIGMA = 0.02             # 先验日超额收益标准差

# ================= 其他 =================
TOP_INDUSTRY_N = 5
INDUSTRY_CACHE_DAYS = 7        # 行业分类缓存天数
