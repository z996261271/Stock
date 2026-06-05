# 专业量化框架升级需求清单

整理日期：2026-06-03

## 1. 目标

将当前项目从“能跑回测的 A 股日频研究脚本”升级为“可信的专业量化研究与纸面交易框架”。

目标边界：

- 支持数据可信度检查。
- 支持严格样本外验证。
- 支持更真实的执行假设。
- 支持可解释的组合风控。
- 支持可复现的正式报告。
- 支持进入 60-90 个交易日纸面交易观察。
- 暂不包含自动实盘下单。

## 2. P0 红线需求

这些需求未完成前，不应把回测结果作为实盘或生产提醒依据。

| 编号 | 需求 | 验收标准 |
|---|---|---|
| P0-1 | 全量 ST/停牌状态接入回测 | 主板股票池状态覆盖率接近 100%；信号日 ST/退市过滤，成交日停牌禁止成交。 |
| P0-2 | 涨跌停规则按股票状态和板块区分 | 普通主板 10%，ST 5%，创业板/科创板 20%；不再固定返回 10%。 |
| P0-3 | 正式回测前强制数据质量 gate | raw/qfq/hfq、生命周期、状态表、最新交易日数据不达标时，正式回测直接失败。 |
| P0-4 | 固定训练/验证/测试集切分 | 报告明确写入 train/validation/test 时间段；测试集结果不得反向调参。 |
| P0-5 | 冻结参数机制 | 训练/验证完成后生成 frozen config；最终样本外只读取 frozen config，不重新搜索参数。 |
| P0-6 | 正式报告可信标记 | metrics 中必须包含 `is_formal_valid`、`data_quality_red_flags`、`status_coverage`、`split_policy`。 |
| P0-7 | 所有正式报告必须有 manifest | 每个正式 `.metrics.json` 必须对应 `.manifest.json`，记录命令、依赖、脚本 hash、数据库摘要。 |

## 3. 数据集切分需求

推荐三段切分：

| 数据集 | 时间 | 用途 |
|---|---|---|
| 训练/研究集 | 2006-2017 | 设计因子、初选参数池。 |
| 验证集 | 2018-2020 | 选择参数、风控阈值、`score_profile`。 |
| 最终样本外测试集 | 2021-2026 | 冻结后只跑一次，评估真实效果。 |

简化版切分：

| 数据集 | 时间 | 用途 |
|---|---|---|
| 训练集 | 2020 年以前 | 研究和调参。 |
| 测试集 | 2020 年以后 | 冻结验证。 |

建议将 2020 年作为压力年份，最终干净样本外从 2021 年开始。

关键约束：

- 看过最终样本外结果后，不得再用最终样本外结果修改因子、参数、过滤规则或风控阈值。
- 如必须修改策略，应重新声明新版本，并另行划分新的验证流程。
- 正式报告必须标注当前结果属于训练集、验证集还是最终样本外测试集。

## 4. P1 专业验证需求

| 编号 | 需求 | 验收标准 |
|---|---|---|
| P1-1 | 增加 nested validation / walk-forward 外层验证 | 内层选参，外层只评估，不混用。 |
| P1-2 | 参数敏感性分析 | `top_n`、持仓天数、止损、市场过滤、成本参数变化后，结果不能显著崩塌。 |
| P1-3 | 多重检验/过拟合折扣 | 报告记录扫描了多少参数组合，并给出过拟合风险提示。 |
| P1-4 | 年度/月度分解 | 输出每年收益、回撤、换手、交易次数、胜率。 |
| P1-5 | benchmark 对比 | 至少对比全市场等权、主板等权；后续可加入沪深300、中证全指。 |
| P1-6 | 专业绩效指标 | 输出 Sharpe、Sortino、Calmar、年化波动、超额收益、alpha/beta。 |
| P1-7 | 容量压力测试 | 不同资金规模、成交额占比、滑点假设下输出结果表。 |

## 5. 执行模型需求

| 编号 | 需求 | 验收标准 |
|---|---|---|
| E-1 | raw 价格用于成交，hfq/qfq 用于因子 | 报告明确展示 `factor_adjust_used`；正式回测禁止 raw fallback。 |
| E-2 | next-open 执行规则保持 | 所有成交日期必须晚于信号日期。 |
| E-3 | 部分成交模型 | 超过容量时不是简单全拒绝，而是允许部分成交并记录剩余未成交。 |
| E-4 | 冲击成本模型 | 成交占当日成交额越高，冲击成本越高。 |
| E-5 | 未成交持仓延续 | 卖不掉时继续持仓，并进入下一期风险计算。 |
| E-6 | 交易日志完整 | 每笔交易记录原因：买入、卖出、止损、换仓、涨跌停阻断、停牌阻断、容量阻断。 |

## 6. 风控需求

| 编号 | 需求 | 验收标准 |
|---|---|---|
| R-1 | 单票权重上限 | 默认 top-N 等权外，支持最大单票权重。 |
| R-2 | 行业暴露限制 | 支持单行业最大权重，并在报告中展示行业集中度。 |
| R-3 | 组合回撤降档 | 达到组合回撤阈值后降低仓位或空仓，恢复条件明确。 |
| R-4 | 黑名单/禁买池 | 支持 ST、退市风险、长期停牌、异常成交额过滤。 |
| R-5 | 换手约束 | 支持限制单日/单周换手，并输出换手压力。 |
| R-6 | 风险预算 | 报告说明每个策略/持仓承担的主要风险来源。 |

## 7. 纸面交易/提醒需求

| 编号 | 需求 | 验收标准 |
|---|---|---|
| T-1 | 每日信号生成脚本 | 盘后更新数据后生成 buy/sell/hold 信号。 |
| T-2 | 信号入库 | 写入 `signals`，带 `strategy`、`run_id`、`signal_date`、`score`、`reason`。 |
| T-3 | 持仓状态入库 | 写入 `positions`，支持卖出提醒和持仓延续。 |
| T-4 | 纸面成交入库 | 写入 `paper_trades`，成交假设与回测一致。 |
| T-5 | 推送去重/重试状态 | 后续扩展 `signal_runs`、`alert_attempts`，同一 `run_id` 不重复推送。 |
| T-6 | 纸面交易观察期 | 连续 60-90 个交易日无漏信号、无重复、无数据失败未告警。 |

## 8. 工程需求

| 编号 | 需求 | 验收标准 |
|---|---|---|
| G-1 | 核心逻辑从 `scripts/` 下沉到 `src/` | `scripts/` 只保留 CLI，核心在 `src/data`、`src/backtest`、`src/execution`、`src/risk`。 |
| G-2 | 增加 `pyproject.toml` | 管理 pytest、ruff、依赖和格式化配置。 |
| G-3 | 增加 CI/本地质量命令 | 一条命令跑 lint、测试、数据质量 smoke。 |
| G-4 | Git 仓库修复 | 当前 `.git` 不可用，需要恢复版本管理。 |
| G-5 | 测试覆盖扩展 | 覆盖 ST/停牌、涨跌停、部分成交、数据 split、正式 gate、manifest。 |
| G-6 | 报告规范统一 | 所有正式报告字段一致，历史研究报告与正式报告分目录。 |

## 9. 建议第一阶段改动范围

优先改这些文件：

| 文件 | 改动方向 |
|---|---|
| `scripts/backtest_dynamic_rebalance.py` | 接入 ST/停牌、涨跌停规则、状态覆盖、正式报告字段。 |
| `scripts/run_formal_dynamic.py` | 正式回测前增加数据质量 gate。 |
| `scripts/quant_data_quality.py` | 增加状态覆盖、最新交易日、正式可用性检查。 |
| `scripts/quant_schema.py` | 后续扩展 `signal_runs`、`alert_attempts`。 |
| `configs/formal_dynamic_default.json` | 加入固定 split、冻结配置字段。 |

## 10. 推荐实施顺序

1. 完成 P0-1：全量 ST/停牌状态接入回测。
2. 完成 P0-2：涨跌停规则按股票状态和板块区分。
3. 完成 P0-3：正式回测前强制数据质量 gate。
4. 完成 P0-4/P0-5：数据集切分和冻结参数机制。
5. 完成 P0-6/P0-7：正式报告可信标记和 manifest 强制化。
6. 补 P1 专业验证指标。
7. 扩展执行模型和风控模型。
8. 建立纸面交易/提醒流程。
9. 做工程化拆分、CI 和报告规范统一。

## 11. 当前实现备注

- P0-1/P0-3：本地正式门禁已验证主板 raw/qfq/hfq 行级覆盖率 1.0、生命周期覆盖 3,200/3,200、ST/停牌状态覆盖 11,152,173/11,152,173；发布前仍必须重新跑 `scripts/formal_readiness_gate.py`，不能靠历史截图放行。
- R-2/R-6：动态回测已接入 `--max-industry-weight` 和 `risk_budget` 报告字段；真实股票行业映射使用 `symbol_industries` 表，可通过 `scripts/import_symbol_industries.py` 从 CSV/JSON 导入，也可通过 `scripts/fetch_em_symbol_industries.py` 从东方财富行业成分表抓取。`quant_data_quality` 会检查行业映射覆盖率，若缺失会给出 `industry_symbols_incomplete` 红旗；当前主板覆盖为 3,199/3,200。
- R-4：动态回测已支持 `--blacklist-file` 禁买池，叠加 ST、退市/生命周期、停牌、异常成交额过滤；正式运行应把维护后的真实禁买池文件随报告 manifest 一起留痕。
- P0-4/P0-5：2006-2017 因子挖掘可用 `backtest_dynamic_rebalance.py --start-date 2018-01-02 --train-years 12 --freeze-selection-date 2017-12-29 --score-profile return40` 固定训练窗口，并把 `frozen_selected`/`frozen_train_candidate` 与候选 `validation_*` 指标写入诊断表；看中候选后可用 `--fixed-spec-json` 复盘单个规格。最近一轮严格验证里，`low_turnover_pullback` 是最接近 40% 的候选：2018-2020 年化 38.56%、最大回撤 -27.65%；`top2/top3` 分散后收益明显塌陷，`keep_top=1000` 也没有再找到验证年化 >= 40% 的候选。2018-2020 只做验证，2021-2026 最终样本外不得反复调参。
- P1-1/P1-2：`scripts/professional_validation_report.py` 可从 metrics、diagnostics、capacity_stress 产物生成 nested/walk-forward 与容量/滑点敏感性摘要；`scripts/run_sensitivity_matrix.py` 可围绕冻结正式配置生成或执行 `top_n`、持仓天数、止损、市场过滤、容量、滑点的一次一维变体重放矩阵。
- T-1/T-5：`scripts/generate_daily_signals.py` 可从审核后的 CSV/JSON 信号文件幂等写入状态表；`scripts/generate_daily_from_picks.py` 可从最新正式 `.picks.csv` 自动推导 buy/hold/sell 纸面信号；`scripts/run_daily_paper_pipeline.py` 作为每日入口，先校验 raw 数据新鲜度，失败时幂等记录 failed run 和告警尝试。所有脚本都只做纸面交易状态记录，不做自动实盘下单；每日 run 的 `config_json` 会记录 `manual_confirmation_required`、`manual_confirmed_at`、`data_delay_days` 和 `latest_raw_trade_date`。
- T-6：60-90 个交易日纸面观察期不能通过代码一次性完成，必须由每日运行记录证明；`scripts/state_report.py` 默认审计最后 60 个 raw 交易日是否都有 finished run 和信号记录，并列出失败/未完成 run、无信号 run、缺失交易日、重复/重试推送、未人工确认 run 和数据延迟 run；需要更长观察时显式使用 `--observation-days 90`。
- G-1：已建立 `src/professional_quant/{data,execution,risk,backtest}` 包，行业映射、涨跌停/停牌执行约束、行业暴露、报告指标、风险预算等核心纯逻辑已下沉；大型 walk-forward 主循环仍保留在 CLI 脚本中以避免一次性重写改变回测行为。
- G-3：本地质量门禁入口为 `python scripts/quality_gate.py`，覆盖编译、ruff（环境已安装时）、拆分后的 `tests/test_*.py` 和临时库数据质量 smoke；CI 入口为 `.github/workflows/quality.yml`，使用 Python 3.10/3.11 矩阵、pip 缓存和 `requirements-dev.txt`/`constraints.txt`。
- G-4：当前工作区默认 `.git` 不可写，本地版本记录使用 `.git-local`；所有本地提交命令应使用 `git --git-dir=.git-local --work-tree=.`。
- G-6：正式报告规范校验入口为 `scripts/validate_formal_reports.py`，要求 `.metrics.json` 具备专业字段且存在对应 `.manifest.json`；旧格式正式报告可用 `--quarantine-invalid --allow-empty` 整组移入 `reports/formal/_invalid/`。发布前硬门禁为 `scripts/formal_readiness_gate.py`，同时检查 raw/qfq/hfq、生命周期、ST/停牌、行业映射覆盖和正式报告字段。
