"""港股基本面 — 财报指标 + 新闻. 仅 AKShare(东财源).
本地拉不到时(网络屏蔽)优雅返回 None, 不影响主流程; GitHub Actions 上完整工作."""
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


def fetch_hk_financials(symbol: str, max_periods: int = 4) -> Optional[dict]:
    """AKShare 港股财务分析指标. 返回最近 N 期: 营收/净利润/同比/毛利率/ROE 等."""
    try:
        import akshare as ak
        df = ak.stock_financial_hk_analysis_indicator_em(symbol=symbol, indicator="报告期")
        if df is None or df.empty:
            return None
        df = df.head(max_periods)

        # AKShare 列名可能略有不同, 用 .get 容错
        records = []
        for _, row in df.iterrows():
            records.append({
                "report_date": str(row.get("报告期", row.get("REPORT_DATE", ""))),
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
            })
        return {"periods": records}
    except Exception as e:
        return None


def fetch_hk_news(symbol: str, n: int = 5) -> Optional[list]:
    """AKShare 个股新闻. 返回最近 N 条标题+摘要+时间."""
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


def fetch_fundamentals(symbol: str) -> dict:
    """组合接口: 财报 + 新闻. 任一失败不影响另一个, 都失败也返回空 dict."""
    return {
        "financials": fetch_hk_financials(symbol),
        "news": fetch_hk_news(symbol),
    }
