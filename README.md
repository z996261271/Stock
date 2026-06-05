# A股日K本地数据库

这个项目用免费接口拉取 A股日K历史数据，并保存到本地 SQLite 数据库。

当前支持的数据源：

| provider | 来源 | 特点 |
|---|---|---|
| `em` | 东方财富 kline JSON | 推荐优先使用；速度快，字段完整，但免费接口可能临时限流 |
| `tx` | AKShare 腾讯日K | 稳定性较好；字段较少，`amount` 会映射为 `volume`，成交额/换手率为空 |
| `baostock` | BaoStock | 较稳且字段完整；速度慢，不支持并发模式 |
| `akshare` | AKShare 东方财富封装 | 小范围近期数据可用；长历史请求可能断连 |
| `auto` | 自动兜底 | 依次尝试 `em -> akshare -> baostock -> tx`；不支持并发 |

## 安装

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

如果系统没有 `python3-venv`，可以用 `uv`：

```bash
uv venv .venv
. .venv/bin/activate
uv pip install -r requirements.txt
```

## 快速验证

先拉两只股票的小范围数据：

```bash
python scripts/fetch_akshare_daily.py \
  --provider em \
  --workers 2 \
  --symbols 000001,600519 \
  --start-date 20240101 \
  --end-date 20240601 \
  --adjust raw
```

## 本地质量门禁

改动专业量化框架后，先跑本地质量门禁：

```bash
python scripts/quality_gate.py
```

这个命令会执行编译检查、ruff（已安装时）、`tests/test_*.py` 全量测试，并创建一个临时 SQLite 数据库验证数据质量 gate。它不会写入 `data/stock_daily.sqlite3`。

当前工作区的默认 `.git` 是只读挂载，本地版本记录保存在 `.git-local`。查看状态或提交时使用：

```bash
git --git-dir=.git-local --work-tree=. status
```

## 拉取全A日K

默认会拉取当前全A股票的不复权日K，并写入 `data/stock_daily.sqlite3`：

```bash
python scripts/fetch_akshare_daily.py \
  --provider em \
  --workers 4 \
  --start-date 19900101 \
  --adjust raw \
  --progress-every 100
```

如果要同时保存真实交易价和后复权指标价：

```bash
python scripts/fetch_akshare_daily.py \
  --provider em \
  --workers 4 \
  --start-date 19900101 \
  --adjust both \
  --progress-every 100
```

脚本支持断点续跑。再次运行时会分别检查每只股票、每种复权类型的本地最小/最大日期：

- 本地最小日期晚于请求开始日期时，会向前补历史；
- 本地最大日期早于请求结束日期时，会向后补增量；
- `auto` 模式会记录实际使用的数据源，失败原因写入 `fetch_status.message`。

如果 `em` 出现 `RemoteDisconnected`，说明免费接口临时限流。可降低并发后重试，或切到更慢但稳的 BaoStock：

```bash
python scripts/fetch_akshare_daily.py \
  --provider baostock \
  --start-date 19900101 \
  --adjust raw \
  --progress-every 100
```

## 数据表

`symbols`

```text
symbol, name, fetched_at
```

`daily_bars`

```text
symbol, trade_date, adjust,
open, high, low, close,
volume, amount, amplitude, pct_chg, chg, turnover,
source, fetched_at
```

主键：

```text
symbol, trade_date, adjust
```

`fetch_status`

```text
symbol, adjust, requested_start, requested_end,
last_status, rows_fetched, source_used, message, fetched_at
```

`adj_factors`

```text
symbol, trade_date, adj_factor, forward_factor, backward_factor, source, fetched_at
```

`symbol_lifecycle`

```text
symbol, name, list_date, delist_date, board, market, source, fetched_at
```

`symbol_status_daily`

```text
symbol, trade_date, is_st, is_suspended, board, source, fetched_at
```

## 查询示例

```bash
sqlite3 data/stock_daily.sqlite3 \
  "select symbol, trade_date, open, close, volume from daily_bars where symbol='000001' order by trade_date desc limit 5;"
```

如果系统没有 `sqlite3` 命令行工具：

```bash
python - <<'PY'
import sqlite3

conn = sqlite3.connect("data/stock_daily.sqlite3")
for row in conn.execute("""
    select symbol, trade_date, open, close, volume
    from daily_bars
    where symbol='000001'
    order by trade_date desc
    limit 5
"""):
    print(row)
conn.close()
PY
```

## 日K因子回测

当前更可信的主入口是动态调仓 walk-forward 脚本：

```bash
python scripts/backtest_dynamic_rebalance.py \
  --start-date 2006-01-05 \
  --train-years 4 \
  --top-n 3 \
  --score-profile aggressive \
  --formula-set base \
  --formula-scope selected \
  --grid-profile smoke \
  --board-scope main \
  --factor-adjust hfq \
  --industry-source sw.industry.index.一级行业
```

它使用日频信号、次日开盘执行、年度 walk-forward 选参、主板股票池、板块过滤、涨跌停阻断、滑点、容量和整手约束。每次输出 `.metrics.json` 时会同步写 `.manifest.json`，记录命令、依赖版本、脚本 hash 和数据库覆盖情况。

2006-2017 训练挖掘、2018-2020 验证时使用冻结选择日期，不触碰 2021-2026 最终样本外测试：

```bash
python scripts/backtest_dynamic_rebalance.py \
  --output-dir reports/training_mining \
  --start-date 2018-01-02 \
  --end-date 2020-12-31 \
  --train-years 12 \
  --min-train-periods 2500 \
  --top-n 1 \
  --score-profile return40 \
  --formula-set expanded \
  --formula-scope all \
  --grid-profile credible \
  --board-scope main \
  --factor-adjust hfq \
  --strict-factor-adjust \
  --freeze-selection-date 2017-12-29 \
  --keep-top 100 \
  --workers 8 \
  --industry-source sw.industry.index.一级行业
```

`return40` 是研究期高收益评分档：它偏向训练期年化超过 40% 的候选，但仍保留活跃度、胜率、回撤和换手过滤。生成的 `diagnostics.csv` 会写入 `frozen_selected`、`frozen_train_candidate` 以及每个候选的 `validation_*` 指标；这些验证指标会从 2018-2020 验证窗口起点空仓重新模拟，不沿用训练期持仓状态。看中某个候选后可用 `--fixed-spec-json '{...}' --skip-capacity-stress` 快速复盘单个规格；能否进入正式报告仍要再跑容量压力矩阵，并用冻结配置一次性跑 2021-2026。

组合风控可以显式限制单票、行业和换手：

```bash
python scripts/backtest_dynamic_rebalance.py \
  --max-position-weight 0.20 \
  --max-industry-weight 0.35 \
  --max-turnover-pct 0.80
```

`--max-industry-weight` 会读取本地股票行业元数据列或映射表；如果只有 `symbol_lifecycle.board` 这类粗粒度字段，报告会在 `config.industry_coverage` 中显示覆盖口径，不能把它误读成真实申万/同花顺行业暴露。

正式报告必须显式处理复权覆盖：

```bash
python scripts/backtest_dynamic_rebalance.py \
  --start-date 2006-01-05 \
  --train-years 4 \
  --board-scope main \
  --factor-adjust hfq \
  --formal \
  --strict-factor-adjust
```

`--formal` 要求使用非 raw 因子并开启 `--strict-factor-adjust`。strict 会在因子窗口内逐行校验 `raw` 是否都有对应 `hfq/qfq`。只要缺失就失败，不再把缺失行静默回退成 `raw`。如果只是研究试跑、不加 strict，报告中的 `config.factor_adjust_used` 和 `config.factor_adjust_coverage` 仍会标明实际是否发生 raw fallback。

生成数据质量报告：

```bash
python scripts/data_quality_report.py \
  --db data/stock_daily.sqlite3 \
  --start-date 2006-01-01 \
  --end-date 2026-05-29 \
  --board-scope main \
  --output reports/data_quality_20060101_20260529_main.json
```

初始化 P0 参考/状态表（只建表，不回填数据）：

```bash
python scripts/data_quality_report.py --db data/stock_daily.sqlite3 --init-schema --required-adjust raw
```

补上市/退市生命周期表：

```bash
python scripts/fetch_symbol_lifecycle.py \
  --db data/stock_daily.sqlite3 \
  --output reports/symbol_lifecycle_fetch.json
```

补历史 ST/停牌状态表（BaoStock；可先用 `--symbols` 小范围验证，再全量跑）：

```bash
python scripts/fetch_symbol_status_daily.py \
  --db data/stock_daily.sqlite3 \
  --start-date 20060101 \
  --end-date 20260529 \
  --board-scope main \
  --progress-every 100
```

导入真实股票行业映射（CSV/JSON 至少包含 `symbol` 和 `industry_name` 或 `industry`）：

```bash
python scripts/import_symbol_industries.py \
  --db data/stock_daily.sqlite3 \
  --input data/reference/symbol_industries.csv \
  --provider sw \
  --source manual_sw_level1
```

也可以从东方财富行业成分表抓取并写入本地 canonical 映射：

```bash
python scripts/fetch_em_symbol_industries.py \
  --db data/stock_daily.sqlite3 \
  --workers 1 \
  --retries 3 \
  --min-unique-symbols 5000 \
  --output reports/em_symbol_industries_import_latest.json
```

数据质量 gate 会检查 `symbol_industries` 对回测股票池的覆盖率；启用 `--max-industry-weight` 前应先让行业映射覆盖率接近 100%。外部行业接口的局部失败会写入导入报告，最终是否可发布以 `formal_readiness_gate.py` 的覆盖率红线为准。

如果已有 `daily_bars` 的 qfq/hfq 行，但 `adj_factors` 为空，可从本地行情重建因子表：

```bash
python scripts/rebuild_adj_factors.py --db data/stock_daily.sqlite3
```

正式回测建议用版本化配置入口，避免临时命令漂移：

```bash
python scripts/run_formal_dynamic.py --config configs/formal_dynamic_default.json --dry-run
python scripts/run_formal_dynamic.py --config configs/formal_dynamic_default.json
```

正式报告发布前跑硬门禁，任何数据红旗、旧格式 metrics 或缺失 manifest 都会返回非零：

```bash
python scripts/formal_readiness_gate.py \
  --db data/stock_daily.sqlite3 \
  --start-date 2006-01-01 \
  --end-date 2026-05-29 \
  --board-scope main \
  --reports-dir reports/formal \
  --output reports/formal_readiness_latest.json
```

在任何数据大改、回测大跑或纸面交易前，先做一致性备份：

```bash
python scripts/backup_sqlite.py --db data/stock_daily.sqlite3 --output-dir backups --tag before-formal-run
```

纸面交易/提醒状态表的当前摘要：

```bash
python scripts/state_report.py --db data/stock_daily.sqlite3 --strategy dynamic_daily_checked_rebalance
```

从已审核的候选信号文件写入纸面交易状态表：

```bash
python scripts/generate_daily_signals.py \
  --db data/stock_daily.sqlite3 \
  --signals reports/latest_checked_picks.csv \
  --signal-date 2026-06-04 \
  --entry-date 2026-06-05 \
  --dry-run
```

确认无误后去掉 `--dry-run`。该脚本只写 `signal_runs`、`signals`、`positions`、`paper_trades`、`alerts` 和 `alert_attempts`，不自动下单。

也可以从最新正式 `.picks.csv` 自动生成 buy/hold/sell 纸面信号：

```bash
python scripts/generate_daily_from_picks.py \
  --db data/stock_daily.sqlite3 \
  --reports-dir reports/formal \
  --strategy dynamic_daily_checked_rebalance \
  --dry-run
```

每日生产化入口会先检查 raw 数据是否至少覆盖信号日，失败会幂等记录 `failed` run 和告警尝试：

```bash
python scripts/run_daily_paper_pipeline.py \
  --db data/stock_daily.sqlite3 \
  --reports-dir reports/formal \
  --strategy dynamic_daily_checked_rebalance \
  --dry-run
```

审计 60-90 个交易日纸面观察期：

```bash
python scripts/state_report.py \
  --db data/stock_daily.sqlite3 \
  --strategy dynamic_daily_checked_rebalance \
  --observation-days 90 \
  --output reports/paper_observation_latest.json
```

如果库中存在 `daily_bars`，观察期 readiness 会按最近 raw 交易日逐日校验 finished run 和信号记录；缺任一交易日不会算 60/90 日通过。

从正式报告产物生成专业验证摘要：

```bash
python scripts/professional_validation_report.py \
  --metrics reports/formal/run.metrics.json \
  --diagnostics reports/formal/run.diagnostics.csv \
  --capacity-stress reports/formal/run.capacity_stress.csv \
  --output reports/formal/run.professional_validation.json
```

围绕冻结正式配置生成参数敏感性矩阵计划：

```bash
python scripts/run_sensitivity_matrix.py \
  --config configs/formal_dynamic_default.json \
  --db data/stock_daily.sqlite3 \
  --output-dir reports/formal/sensitivity
```

确认计划后加 `--execute` 执行实际重放。矩阵覆盖 `top_n`、最短/最长持仓天数、止损、市场过滤、容量和滑点。

校验正式报告字段和 manifest 配对：

```bash
python scripts/validate_formal_reports.py \
  --reports-dir reports/formal \
  --require-formal-valid
```

清理旧格式正式报告时可把无效报告整组隔离到 `reports/formal/_invalid/<timestamp>/`，隔离目录不参与后续校验：

```bash
python scripts/validate_formal_reports.py \
  --reports-dir reports/formal \
  --require-formal-valid \
  --quarantine-invalid \
  --allow-empty
```

近 3 年、月末选股、下一交易日开盘调仓、只持有 5 只股票的高收益版本：

```bash
python scripts/backtest_momentum_lowvol.py \
  --years 3 \
  --top-n 5 \
  --score-mode balanced \
  --min-amount 50000000 \
  --min-price 10
```

反彩票/回撤控制版本会在动量之外惩罚近 20 日最大单日涨幅、高波动和高换手：

```bash
python scripts/backtest_momentum_lowvol.py \
  --years 3 \
  --top-n 5 \
  --score-mode anti_lottery_momentum \
  --min-amount 100000000 \
  --min-price 2
```

回测报告会写入 `reports/`，文件名包含 `top`、成交额阈值和股价阈值，避免不同参数互相覆盖。

进攻策略样本内挖掘器会扫描现有日K因子、调仓频率、市场过滤和持仓集中度，目标是快速寻找高收益组合：

```bash
python scripts/research_aggressive_fast.py \
  --start-date 2021-02-01 \
  --target-return 10 \
  --keep-top 80
```

这个脚本是研究工具。它会在样本内挖参数，命中高收益不代表未来可复制，必须再做样本外验证、涨跌停成交约束和滑点压力测试。

## 注意

- 正式评估口径见 `docs/quant_framework_evaluation.md`。
- `src/professional_quant/` 存放可复用核心逻辑；`scripts/` 保留 CLI 和大型回测编排入口。
- `scripts/quant_universe.py` 负责股票池口径；`scripts/quant_schema.py` 负责 P0 参考/状态表；`scripts/import_symbol_industries.py` 负责股票行业映射；`scripts/quant_state.py`、`scripts/state_report.py` 和 `scripts/run_daily_paper_pipeline.py` 负责纸面交易状态；`scripts/quant_data_quality.py`、`scripts/data_quality_report.py` 和 `scripts/formal_readiness_gate.py` 负责数据红线检查；`scripts/backup_sqlite.py` 负责 SQLite 一致性备份；`scripts/run_manifest.py` 负责报告复现清单。
- 免费数据接口没有商业 SLA，接口可能限流、断连或变更。
- 长期正式回测必须先补 `symbol_lifecycle` 并通过 `formal_readiness_gate.py`；仅用实时全A列表做历史研究会产生幸存者偏差。
- 回测成交建议用 `raw` 不复权价格，技术指标和长期收益可以用 `hfq` 后复权价格。
