"""A股基本面 — 财报指标 + 新闻 + 下次财报日期 + 现金跑道 + 公司名核验.
主要走 AKShare(新浪/东财源), 配 yfinance 拿财报日历.
本地拉不到时(网络屏蔽)优雅返回 None, GitHub Actions 上完整工作."""
import warnings
from datetime import datetime
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


def _report_period_type(date_str: str) -> str:
    """A股全部公历年制, 从报告期日期识别报告类型."""
    if not date_str:
        return "未知"
    md = date_str[-5:]
    return {
        "03-31": "一季报(Q1单季, 1-3月)",
        "06-30": "中报(半年累计, 1-6月)",
        "09-30": "三季报(前三季度累计, 1-9月)",
        "12-31": "年报(全年, 1-12月)",
    }.get(md, "未知")


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


def fetch_financials(symbol: str, max_periods: int = 4) -> Optional[dict]:
    """AKShare A股财务分析指标(新浪源). 每期带 report_period_type 和 anomaly_flags."""
    try:
        import akshare as ak
        start_year = str(datetime.now().year - 2)
        df = ak.stock_financial_analysis_indicator(symbol=symbol, start_year=start_year)
        if df is None or df.empty:
            return None
        df = df.head(max_periods)
        records = []
        for _, row in df.iterrows():
            date_str = str(row.get("日期", ""))[:10]
            revenue = _safe_num(row.get("主营业务收入(万元)"))
            net_profit = _safe_num(row.get("净利润(万元)"))
            net_margin = None
            if revenue and net_profit and revenue != 0:
                net_margin = round(net_profit / revenue * 100, 2)
            rec = {
                "report_date": date_str,
                "report_period_type": _report_period_type(date_str),
                "revenue": revenue,
                "revenue_yoy_pct": None,
                "net_profit": net_profit,
                "net_profit_yoy_pct": None,
                "gross_margin_pct": _safe_num(row.get("主营业务利润率(%)")),
                "net_margin_pct": net_margin,
                "roe_pct": _safe_num(row.get("净资产收益率加权(%)")),
                "eps": _safe_num(row.get("摊薄每股收益")),
            }
            rec["anomaly_flags"] = _anomaly_flags(rec)
            records.append(rec)
        return {"periods": records}
    except Exception:
        return None


def fetch_news(symbol: str, n: int = 5) -> Optional[list]:
    """AKShare 个股新闻(东财源), 带来源字段."""
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


def fetch_balance_cash(symbol: str) -> Optional[float]:
    """拉最新一期货币资金余额(元), 用于亏损公司现金跑道估算."""
    try:
        import akshare as ak
        em_sym = ("SH" if symbol.startswith("6") else "SZ") + symbol
        df = ak.stock_balance_sheet_by_report_em(symbol=em_sym)
        if df is None or df.empty:
            return None
        for col in ["MONETARYFUNDS", "货币资金"]:
            if col in df.columns:
                return _safe_num(df.iloc[0][col])
        return None
    except Exception:
        return None


def fetch_next_earnings_date(symbol: str) -> Optional[str]:
    """yfinance 取**未来**最近一次财报日期 (过滤掉已过去的日期)."""
    try:
        import yfinance as yf
        yf_sym = symbol + (".SS" if symbol.startswith("6") else ".SZ")
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


def fetch_company_name(symbol: str) -> Optional[str]:
    """从A股行情快照取公司名 — 用于核对配置中 name 是否匹配数据源."""
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        row = df[df["代码"] == symbol]
        if row.empty:
            return None
        return str(row.iloc[0]["名称"])
    except Exception:
        return None


def _compute_cash_runway(financials: dict, cash: Optional[float]) -> Optional[dict]:
    """对亏损公司估算现金跑道(月数). 盈利公司返回 None.
    注意: 新浪源财务数据单位为万元, cash(东财资产负债表)单位为元."""
    if not financials or not financials.get("periods") or cash is None or cash <= 0:
        return None
    latest = financials["periods"][0]
    np_val = latest.get("net_profit")
    if np_val is None or np_val >= 0:
        return None
    period_type = latest.get("report_period_type", "")
    if "年报" in period_type:
        months = 12
    elif "中报" in period_type:
        months = 6
    elif "三季报" in period_type:
        months = 9
    elif "一季报" in period_type:
        months = 3
    else:
        months = 6
    # np_val 单位为万元, cash 单位为元, 统一到万元
    cash_wan = cash / 10000
    monthly_burn = abs(np_val) / months
    if monthly_burn <= 0:
        return None
    return {
        "cash_balance": round(cash_wan, 2),
        "estimated_monthly_burn": round(monthly_burn, 2),
        "estimated_runway_months": round(cash_wan / monthly_burn, 1),
        "based_on_period": latest.get("report_date"),
        "period_months": months,
    }


def fetch_fundamentals(symbol: str, configured_name: str = "") -> dict:
    """组合接口: 公司名核验 + 财报 + 新闻 + 下次财报日期 + 现金跑道."""
    financials = fetch_financials(symbol)
    cash = fetch_balance_cash(symbol) if financials else None
    return {
        "company_name_from_source": fetch_company_name(symbol),
        "configured_name": configured_name,
        "financials": financials,
        "news": fetch_news(symbol),
        "next_earnings_date": fetch_next_earnings_date(symbol),
        "cash_runway": _compute_cash_runway(financials, cash),
    }
