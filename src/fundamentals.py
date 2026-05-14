"""港股基本面 — 财报指标 + 新闻 + 下次财报日期 + 现金跑道 + 公司名核验.
主要走 AKShare(东财源), 配 yfinance 拿财报日历.
本地拉不到时(网络屏蔽)优雅返回 None, GitHub Actions 上完整工作."""
import warnings
from typing import Optional

warnings.filterwarnings("ignore", category=FutureWarning)


def _safe_num(v):
    if v is None:
        return None
    try:
        f = float(v)
        if f != f:  # NaN
            return None
        return round(f, 2)
    except (ValueError, TypeError):
        return None


# 非公历年制的港股 (财年截止日 MM-DD)
# 阿里巴巴 财年 4月-3月, 京东物流/京东集团 公历年制不需登记
FISCAL_YEAR_END_BY_SYMBOL = {
    "09988": "03-31",  # 阿里巴巴 (财年 4月-次年3月)
    # 后续如有其他非公历年制公司, 在此追加
}


def _report_period_type(date_str: str, symbol: str = "") -> str:
    """从报告期日期 + 公司财年规则识别报告类型, 避免 LLM 把累计当单季陈述."""
    if not date_str:
        return "未知"
    fye = FISCAL_YEAR_END_BY_SYMBOL.get(symbol, "12-31")
    md = date_str[-5:]
    if fye == "12-31":
        # 标准公历年制 (绝大多数公司)
        return {
            "03-31": "Q1单季(港股一般不披露, A+H双重上市会有)",
            "06-30": "中报(半年累计, 即1-6月合计)",
            "09-30": "Q3累计(港股一般无, A+H双重上市会有)",
            "12-31": "年报(全年累计, 即1-12月合计)",
        }.get(md, "未知")
    if fye == "03-31":
        # 4月-次年3月制 (阿里巴巴等)
        return {
            "06-30": "Q1单季(财年 4-6月)",
            "09-30": "中报(财年半年累计, 即财年前6个月 4-9月)",
            "12-31": "Q3累计(财年前9个月 4-12月)",
            "03-31": "年报(财年全年累计, 即4月-次年3月)",
        }.get(md, "未知")
    return "未知"


def _anomaly_flags(record: dict) -> list:
    """检出异常财务数字, 提醒 LLM 不要当成正常表现陈述."""
    flags = []
    nm = record.get("net_margin_pct")
    if nm is not None and abs(nm) > 100:
        flags.append(
            f"⚠️ 净利率 {nm}% 绝对值超过 100%, 多半含一次性会计调整或非现金减值, 不代表实际经营情况"
        )
    roe = record.get("roe_pct")
    if roe is not None and (roe < -50 or roe > 100):
        flags.append(f"⚠️ ROE {roe}% 极端值, 股东权益可能被一次性损益扭曲")
    yoy = record.get("revenue_yoy_pct")
    if yoy is not None and abs(yoy) > 500:
        flags.append(f"⚠️ 营收同比 {yoy}% 异常剧烈, 可能因低基数或合并范围变化")
    return flags


def fetch_hk_financials(symbol: str, max_periods: int = 4) -> Optional[dict]:
    """AKShare 港股财务分析指标. 每期带 report_period_type 和 anomaly_flags."""
    try:
        import akshare as ak
        df = ak.stock_financial_hk_analysis_indicator_em(symbol=symbol, indicator="报告期")
        if df is None or df.empty:
            return None
        df = df.head(max_periods)
        records = []
        for _, row in df.iterrows():
            date_str = str(row.get("报告期", row.get("REPORT_DATE", "")))[:10]
            rec = {
                "report_date": date_str,
                "report_period_type": _report_period_type(date_str, symbol),
                "revenue": _safe_num(row.get("营业总收入", row.get("OPERATE_INCOME"))),
                "revenue_yoy_pct": _safe_num(
                    row.get("营业总收入同比增长率", row.get("OPERATE_INCOME_YOY"))
                ),
                "net_profit": _safe_num(row.get("净利润", row.get("PARENT_NETPROFIT"))),
                "net_profit_yoy_pct": _safe_num(
                    row.get("净利润同比增长率", row.get("PARENT_NETPROFIT_YOY"))
                ),
                "gross_margin_pct": _safe_num(row.get("毛利率", row.get("GROSS_PROFIT_RATIO"))),
                "net_margin_pct": _safe_num(row.get("净利率", row.get("NET_PROFIT_RATIO"))),
                "roe_pct": _safe_num(row.get("净资产收益率", row.get("ROE_AVG"))),
                "eps": _safe_num(row.get("摊薄每股收益", row.get("BASIC_EPS"))),
            }
            rec["anomaly_flags"] = _anomaly_flags(rec)
            records.append(rec)
        return {"periods": records}
    except Exception:
        return None


def fetch_hk_news(symbol: str, n: int = 5) -> Optional[list]:
    """AKShare 个股新闻, 带来源字段."""
    try:
        import akshare as ak
        df = ak.stock_news_em(symbol=symbol)
        if df is None or df.empty:
            return None
        df = df.head(n)
        items = []
        for _, row in df.iterrows():
            content = str(row.get("新闻内容", "")).strip()
            if len(content) > 200:
                content = content[:200] + "..."
            items.append({
                "title": str(row.get("新闻标题", "")).strip(),
                "summary": content,
                "time": str(row.get("发布时间", "")).strip(),
                "source": str(row.get("文章来源", "")).strip(),
            })
        return items
    except Exception:
        return None


def fetch_hk_balance_cash(symbol: str) -> Optional[float]:
    """尝试拉最新一期"现金及现金等价物"余额(原始单位元), 用于亏损公司现金跑道估算."""
    try:
        import akshare as ak
        df = ak.stock_financial_hk_report_em(
            stock=symbol, symbol="资产负债表", indicator="报告期"
        )
        if df is None or df.empty:
            return None
        # 找科目名列和金额列(AKShare 列名不太稳定, 尝试多种)
        item_col = next(
            (c for c in ["STD_ITEM_NAME", "STD_ITEM", "ITEM_NAME", "项目"] if c in df.columns),
            None,
        )
        amount_col = next(
            (c for c in ["AMOUNT", "金额", "VALUE", "本期金额"] if c in df.columns),
            None,
        )
        if not item_col or not amount_col:
            return None
        # 找包含"现金"的行
        cash_items = df[df[item_col].astype(str).str.contains("现金", na=False)]
        if cash_items.empty:
            return None
        return _safe_num(cash_items.iloc[0][amount_col])
    except Exception:
        return None


def fetch_next_earnings_date(symbol: str) -> Optional[str]:
    """yfinance 取**未来**最近一次财报日期 (过滤掉已过去的日期)."""
    try:
        from datetime import datetime
        import yfinance as yf
        yf_sym = symbol.lstrip("0").zfill(4) + ".HK"
        ticker = yf.Ticker(yf_sym)
        cal = ticker.calendar
        if not cal:
            return None
        dates = None
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date")
        if not dates:
            return None
        if not isinstance(dates, list):
            dates = [dates]

        today = datetime.now().date()
        future_dates = []
        for d in dates:
            try:
                date_obj = datetime.strptime(str(d)[:10], "%Y-%m-%d").date()
                if date_obj > today:
                    future_dates.append(date_obj)
            except (ValueError, TypeError):
                continue
        if not future_dates:
            return None
        return min(future_dates).strftime("%Y-%m-%d")
    except Exception:
        return None


def fetch_hk_company_name(symbol: str) -> Optional[str]:
    """从港股行情快照取公司名 — 用于核对配置中 name 是否匹配数据源."""
    try:
        import akshare as ak
        df = ak.stock_hk_spot_em()
        row = df[df["代码"] == symbol]
        if row.empty:
            return None
        return str(row.iloc[0]["名称"])
    except Exception:
        return None


def _compute_cash_runway(financials: dict, cash: Optional[float]) -> Optional[dict]:
    """对亏损公司估算现金跑道(月数). 盈利公司返回 None."""
    if not financials or not financials.get("periods") or cash is None or cash <= 0:
        return None
    latest = financials["periods"][0]
    np = latest.get("net_profit")
    if np is None or np >= 0:
        return None
    period_type = latest.get("report_period_type", "")
    if "年报" in period_type:
        months = 12
    elif "中报" in period_type:
        months = 6
    elif "Q3" in period_type:
        months = 9
    elif "Q1" in period_type:
        months = 3
    else:
        months = 6  # 默认
    monthly_burn = abs(np) / months
    if monthly_burn <= 0:
        return None
    return {
        "cash_balance": cash,
        "estimated_monthly_burn": round(monthly_burn, 2),
        "estimated_runway_months": round(cash / monthly_burn, 1),
        "based_on_period": latest.get("report_date"),
        "period_months": months,
    }


def fetch_fundamentals(symbol: str, configured_name: str = "") -> dict:
    """组合接口: 公司名核验 + 财报 + 新闻 + 下次财报日期 + 现金跑道."""
    financials = fetch_hk_financials(symbol)
    cash = fetch_hk_balance_cash(symbol) if financials else None
    return {
        "company_name_from_source": fetch_hk_company_name(symbol),
        "configured_name": configured_name,
        "financials": financials,
        "news": fetch_hk_news(symbol),
        "next_earnings_date": fetch_next_earnings_date(symbol),
        "cash_runway": _compute_cash_runway(financials, cash),
    }
