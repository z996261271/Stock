# 量化框架评估报告

评估日期：2026-06-04

## 1. 系统定位

当前项目应定位为：

> A股日频选股研究与回测框架，目标未来扩展为“只提醒、不自动下单”的信号系统。

这个定位决定了验收标准。研究框架可以保留实验脚本和候选参数扫描；提醒系统必须具备数据更新、状态管理、信号去重、失败重试、人工确认、监控和回滚能力。当前项目已具备正式研究报告发布门禁和纸面交易状态入口，仍不应绕过 60-90 个交易日观察直接作为真钱买卖提醒依据。

## 2. 当前证据

本地数据库 `data/stock_daily.sqlite3`：

| 数据表 | 口径 | 行数 | 标的数 | 日期范围 |
|---|---:|---:|---:|---|
| `daily_bars` | `raw` | 16,784,980 | 5,208 | 1990-12-19 至 2026-05-29 |
| `daily_bars` | `qfq/hfq` | 15,976,057 / 15,976,057 | 5,208 | 2000-10-09 至 2026-05-29 |
| `adj_factors` | derived | 15,976,057 | 5,208 | 2000-10-09 至 2026-05-29 |
| `symbol_lifecycle` | exchange lists | 5,873 | 5,873 | current + delisted metadata |
| `symbol_status_daily` | BaoStock main-board history | 11,910,003 | 3,442 | 2006-01-04 至 2026-05-29 |
| `symbol_industries` | Eastmoney board constituents | 5,605 | 5,605 | 203 industries |
| `industry_daily_bars` | `sw.industry.index.一级行业` | 116,600 | 31 | 2006-01-05 至 2026-05-29 |
| `industry_daily_bars` | `ths.industry.index` | 95,760 | 90 | 2022-01-04 至 2026-05-29 |

当前关键回测样例：

| 报告 | 年化 | 总收益 | 最大回撤 | 重要限制 |
|---|---:|---:|---:|---|
| `reports/formal/dynamic_rebalance_20210104_20260529_train4y_top3_aggressive_base_selected_credible_main_hfq_capinitial_pstop0_sector.metrics.json` | 5.80% | 35.55% | -27.66% | `is_formal_valid=true`，`data_quality_red_flags=[]`，样本外测试段 2021-01-04 至 2026-05-29 |

发布门禁证据：

- `scripts/formal_readiness_gate.py --required-adjust raw/qfq/hfq` 对主板 2006-01-01 至 2026-05-29 返回 `is_ready=true`、`blockers=[]`。
- raw/qfq/hfq 主板行级覆盖率均为 1.0；生命周期覆盖 3,200/3,200；ST/停牌状态覆盖 11,152,173/11,152,173；行业映射覆盖 3,199/3,200。
- `scripts/validate_formal_reports.py --require-formal-valid` 对 `reports/formal` 返回 1 个 active metrics、0 个 issue files；旧格式正式报告已隔离到 `reports/formal/_invalid/`。

结论：数据治理、正式报告闭环和工程门禁已从“建议项”升级为硬约束；策略收益稳健性、纸面观察期和生产提醒运营仍是主要短板。

## 3. 评分卡

| 维度 | 权重 | 当前分 | 评价 |
|---|---:|---:|---|
| 数据与股票池 | 20 | 18 | raw/qfq/hfq、adj_factors、生命周期、ST/停牌和行业映射均已纳入正式门禁；行业映射仍有 1 个主板标的缺口，免费接口存在局部失败风险。 |
| 回测可信度 | 20 | 17 | 已有 signal_date/entry_date 分离、next-open、固定 train/validation/test、冻结配置、正式报告字段和 manifest；多重检验折扣仍主要体现在报告提示和敏感性工具中。 |
| 执行真实性 | 15 | 13 | 动态脚本已加入状态区分涨跌停、停牌阻断、滑点、容量、部分成交、整手和 next-open；集合竞价细节仍未模拟。 |
| 策略稳健性 | 15 | 8 | 正式样本外回撤降至 -27.66%，但年度稳定性和参数敏感性仍需持续观察，不能用单次正式样本外替代策略生产验证。 |
| 风控与组合管理 | 10 | 8 | 已有单票、行业、换手、黑名单、组合止损和风险预算报告；真实组合运营阈值仍需纸面期校准。 |
| 实时提醒能力 | 10 | 6 | 已有状态表、每日 picks 到信号入口、数据新鲜度检查、幂等 run_id、失败记录和观察期审计；真实推送通道、人工确认和运营监控仍待接入。 |
| 工程与治理 | 10 | 8 | 已有本地 git、CI 矩阵、pip 缓存、dev/constraints 依赖约束、拆分测试、ruff 扩展和核心报告/风险模块下沉；主回测脚本仍偏大。 |

总分：78/100

评级：

- 研究框架：4/5
- 可上线提醒系统：3/5
- 可实盘依赖程度：暂不建议

## 4. 红线项

以下问题不是简单扣分，而是进入生产提醒前必须继续满足：

1. 正式发布必须通过 `scripts/formal_readiness_gate.py`；任何 raw/qfq/hfq、生命周期、ST/停牌或行业映射红旗都应阻断发布。
2. 正式报告目录必须通过 `scripts/validate_formal_reports.py --require-formal-valid`；旧格式或缺 manifest 的报告只能放在 `_invalid/`。
3. 当天信号必须由 `scripts/run_daily_paper_pipeline.py` 检查 raw 数据新鲜度；数据未更新时只记录 failed run/告警尝试，不生成可用信号。
4. 真实推送通道、人工确认、监控和回滚仍未接入；上线提醒前必须补齐运营闭环。
5. 仍未积累连续 60-90 个交易日纸面交易记录，不能直接进入真钱使用。

## 5. 优先级路线

P0：回测可信度

- 保持 raw/qfq/hfq 双价格流硬门禁，成交只用 raw，因子和长期收益用 hfq；若 hfq 缺失，正式门禁必须失败。
- 使用 `scripts/data_quality_report.py` 生成数据质量报告；使用 `scripts/quant_schema.py` 初始化 `adj_factors`、`symbol_lifecycle`、`symbol_status_daily`、`signals`、`positions`、`alerts`、`paper_trades` 等 P0 表。
- 使用 `scripts/fetch_symbol_lifecycle.py` 和 `scripts/fetch_symbol_status_daily.py` 分别补生命周期与历史状态入口。
- 使用 `scripts/backup_sqlite.py` 在数据大改、正式回测和纸面交易前创建 SQLite 一致性备份。
- 使用 `configs/formal_dynamic_default.json` + `scripts/run_formal_dynamic.py` 作为正式动态回测入口；零散参数命令只作为研究试跑。
- 建立历史股票池、上市日、退市日、历史 ST 状态；ST 和退市过滤必须按信号日判断。
- 执行层继续维护停牌、成交额容量、部分成交和历史涨跌幅规则。
- 每次报告写入 `run_manifest`：命令、依赖版本、数据快照、P0 schema 表存在性、脚本 hash、输出文件。

P1：策略验证

- 继续强化冻结参数流程：研究期选参数，冻结后跑独立样本外，不允许再用测试段反向修改参数。
- 增加 nested validation、年度分解、参数敏感性、最大回撤分解和换手压力测试。
- 报告同时展示收益、回撤、换手、交易阻断次数、容量假设和 factor_adjust 实际覆盖。

P2：提醒系统

- 使用 `scripts/run_daily_paper_pipeline.py` 从正式 picks 生成每日纸面信号，并用 `scripts/state_report.py` 审计当前状态。
- 后续如需真实推送，应在已有 `signal_runs`、`alert_attempts`、幂等 run_id 和重试状态上接入外部通道。
- 每日盘后先更新数据，再生成信号，再人工确认。
- 推送必须支持去重、失败重试、幂等 run_id 和异常告警。

## 6. 上线门槛

研究框架继续优化的最低门槛：

- 代码可编译，核心可信度测试通过。
- 正式运行前有 SQLite 备份 manifest。
- 正式动态回测只走 `scripts/run_formal_dynamic.py` 或等价 `backtest_dynamic_rebalance.py --formal --strict-factor-adjust`，且发布前通过 `scripts/formal_readiness_gate.py`。
- 每个回测报告都有 manifest。
- 报告中明确展示 `factor_adjust_used` 与 `factor_adjust_coverage`；正式报告用 `--strict-factor-adjust` 禁止隐藏 raw fallback。
- 所有策略结论必须标注是否样本外、是否 strict hfq、是否包含退市/ST。

提醒系统试运行门槛：

- 连续 60 个交易日纸面交易无漏信号、无重复推送、无数据更新失败未告警。
- 纸面交易成交假设和回测成交假设一致。
- 每日信号、持仓、买卖原因、失败重试和人工确认都有记录。
- 最大回撤、连续亏损和单票亏损都在用户可承受范围内。

## 7. 参考框架

- Federal Reserve SR 11-7, Model Risk Management: https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm
- Bailey, Borwein, López de Prado, Zhu, Statistical Overfitting and Backtest Performance: https://sdm.lbl.gov/oapapers/ssrn-id2507040-bailey.pdf
- NIST Cybersecurity Framework 2.0: https://www.nist.gov/cyberframework
