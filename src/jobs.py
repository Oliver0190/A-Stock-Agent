import sys
import traceback
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src import data, analyzer, llm, feishu, state
from src import fundamentals as fund

load_dotenv()


def load_config() -> dict:
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _analyze(item: dict, cfg: dict) -> tuple[dict, "pd.DataFrame"]:
    az_cfg = cfg.get("analyzer", {})
    df = data.fetch_hk_daily(item["symbol"], az_cfg.get("lookback_days", 365))
    analysis = analyzer.full_analysis(
        df,
        ma_window=az_cfg.get("ma_window", 60),
        sr_min_distance=az_cfg.get("support_resistance_min_distance", 15),
    )
    return analysis, df


# ---------- 盘前简报 (每股一张卡) ----------

def morning_brief() -> None:
    cfg = load_config()
    model = cfg["llm"]["model"]
    effort = cfg["llm"].get("effort", "medium")
    date_str = datetime.now().strftime("%Y-%m-%d")

    for item in cfg["watchlist"]:
        if state.already_sent_today("morning_brief", item["symbol"]):
            print(f"skip {item['symbol']}: already sent today")
            continue
        try:
            analysis, _ = _analyze(item, cfg)
            # 基本面 (财报+新闻+下次财报+现金跑道+公司名核验); 本地拉不到时各字段返回 None, 不影响主流程
            fund_data = fund.fetch_fundamentals(item["symbol"], configured_name=item["name"])
            text = llm.morning_brief(
                item["name"], item["symbol"], analysis, model, effort,
                fundamentals=fund_data,
            )
            title = f"📊 {item['name']}({item['symbol']}) 盘前简报 · {date_str}"
            feishu.send_card(title, text, color="blue")
            state.mark_sent("morning_brief", item["symbol"])
        except Exception as e:
            traceback.print_exc()
            feishu.send_text(f"⚠️ 盘前简报失败 {item['symbol']}: {e}")


# ---------- 盘后快照 (一张卡 = 全部股票) ----------

def _zone_badge(r: dict) -> str:
    pos = r.get("zone_position", "")
    if pos == "in_zone":
        return f"✅ 已在区间 ({r['zone_low']}-{r['zone_high']})"
    if pos == "above_zone":
        return f"⬆️ 高{r.get('zone_distance_pct', 0)}% (区间 {r['zone_low']}-{r['zone_high']})"
    if pos == "below_zone":
        return f"⬇️ 破位{r.get('zone_distance_pct', 0)}% (区间 {r['zone_low']}-{r['zone_high']})"
    return "状态未知"


def _trend_emoji(pct: float) -> str:
    if pct >= 3: return "🚀"
    if pct >= 0: return "🔺"
    if pct >= -3: return "🔻"
    return "📉"


def _pct(v: float) -> str:
    return f"+{v:.2f}%" if v >= 0 else f"{v:.2f}%"


def _extract_signals(ind: dict) -> list[str]:
    """从指标里挑出值得说的(金叉/死叉/超买/超卖/突破上下轨), 维持现状的不提."""
    out = []
    macd = ind.get("macd", {}).get("signal", "")
    if "金叉" in macd or "死叉" in macd:
        out.append(f"MACD {macd}")
    kdj = ind.get("kdj", {}).get("signal", "")
    if "金叉" in kdj or "死叉" in kdj or "超" in kdj:
        out.append(f"KDJ {kdj}")
    boll = ind.get("boll", {}).get("position", "")
    if "上轨" in boll or "下轨" in boll:
        out.append(f"BOLL {boll}")
    return out


def daily_snapshot() -> None:
    cfg = load_config()
    model = cfg["llm"]["model"]
    date_str = datetime.now().strftime("%Y-%m-%d")

    if state.already_sent_today("daily_snapshot", "all"):
        print("daily_snapshot already sent today")
        return

    rows = []
    failures = []
    for item in cfg["watchlist"]:
        try:
            analysis, _ = _analyze(item, cfg)
            rows.append({
                "name": item["name"],
                "symbol": item["symbol"],
                "close": analysis["kline"]["close"],
                "pct_change": analysis["kline"]["pct_change"],
                "volume_ratio": analysis["kline"]["volume_ratio"],
                "percentile": analysis["range"]["percentile"],
                "zone_low": analysis["value_zone"]["zone_low"],
                "zone_high": analysis["value_zone"]["zone_high"],
                "zone_position": analysis["value_zone"]["position"],
                "zone_distance_pct": analysis["value_zone"]["distance_pct"],
                "nearest_support": analysis["support_resistance"]["nearest_support"],
                "nearest_resistance": analysis["support_resistance"]["nearest_resistance"],
                "macd_signal": analysis["indicators"]["macd"]["signal"],
                "kdj_signal": analysis["indicators"]["kdj"]["signal"],
                "boll_position": analysis["indicators"]["boll"]["position"],
            })
        except Exception as e:
            traceback.print_exc()
            failures.append(f"{item['symbol']}: {e}")

    if not rows:
        feishu.send_text(f"⚠️ 盘后快照: 所有股票数据获取失败. {failures}")
        return

    # ---- 每只股票两行的紧凑布局 ----
    lines = []
    for r in rows:
        emoji = _trend_emoji(r["pct_change"])
        lines.append(
            f"**{emoji} {r['name']} ({r['symbol']})** · {r['close']} · {_pct(r['pct_change'])}"
        )
        lines.append(
            f"量比 {r['volume_ratio']:.2f} · 分位 {r['percentile']:.0f}% · {_zone_badge(r)}"
        )
        lines.append("")  # 股票之间空一行
    table = "\n".join(lines).rstrip()

    # ---- 关键信号 (只列触发了信号的股票) ----
    sig_lines = []
    for r in rows:
        sigs = _extract_signals({
            "macd": {"signal": r["macd_signal"]},
            "kdj": {"signal": r["kdj_signal"]},
            "boll": {"position": r["boll_position"]},
        })
        if sigs:
            sig_lines.append(f"- **{r['name']}**: {' · '.join(sigs)}")

    signals_block = ""
    if sig_lines:
        signals_block = "\n\n**关键信号**\n" + "\n".join(sig_lines)

    # ---- LLM 总结 ----
    try:
        summary = llm.snapshot_summary(rows, model)
    except Exception as e:
        traceback.print_exc()
        summary = f"(LLM 总结生成失败: {e})"

    # ---- 失败提示 ----
    failure_block = ""
    if failures:
        failure_block = "\n\n⚠️ 数据获取失败: " + ", ".join(failures)

    content = f"{table}{signals_block}\n\n**今日异动**\n{summary}{failure_block}"
    title = f"📋 关注列表 · 收盘快照 · {date_str}"
    feishu.send_card(title, content, color="purple")
    state.mark_sent("daily_snapshot", "all")


# ---------- 旧版盘后简报 (保留, 可手动调用) ----------

def closing_review() -> None:
    cfg = load_config()
    model = cfg["llm"]["model"]
    effort = cfg["llm"].get("effort", "medium")
    date_str = datetime.now().strftime("%Y-%m-%d")

    for item in cfg["watchlist"]:
        if state.already_sent_today("closing_review", item["symbol"]):
            print(f"skip {item['symbol']}: already sent today")
            continue
        try:
            analysis, _ = _analyze(item, cfg)
            text = llm.closing_review(item["name"], item["symbol"], analysis, model, effort)
            title = f"📈 {item['name']}({item['symbol']}) 盘后复盘 · {date_str}"
            feishu.send_card(title, text, color="green")
            state.mark_sent("closing_review", item["symbol"])
        except Exception as e:
            traceback.print_exc()
            feishu.send_text(f"⚠️ 盘后复盘失败 {item['symbol']}: {e}")


# ---------- 盘中预警 (异动驱动, 同一事件每天只发一次) ----------

def _detect_events(item: dict, price: float, analysis: dict) -> list[tuple[str, str]]:
    """检测当前价格触发的事件. 返回 [(event_id, 中文描述), ...]"""
    events = []
    sr = analysis["support_resistance"]
    vz = analysis["value_zone"]
    kline = analysis["kline"]

    # 1) 用户设定的硬阈值
    al = item.get("alert_low")
    ah = item.get("alert_high")
    if al is not None and price <= al:
        events.append(("alert_low", f"跌破设定下限 {al}"))
    if ah is not None and price >= ah:
        events.append(("alert_high", f"突破设定上限 {ah}"))

    # 2) 突破近期阻力 / 跌破近期支撑
    if sr.get("nearest_resistance") and price >= sr["nearest_resistance"]:
        events.append(("breakout_resistance", f"突破近期阻力位 {sr['nearest_resistance']}"))
    if sr.get("nearest_support") and price <= sr["nearest_support"]:
        events.append(("breakdown_support", f"跌破近期支撑位 {sr['nearest_support']}"))

    # 3) 进入参考买入区间
    if vz.get("position") == "in_zone":
        events.append(
            ("entered_value_zone",
             f"已进入参考买入区间 {vz['zone_low']}~{vz['zone_high']}")
        )

    # 4) 单日涨跌幅 > 5%
    pct = kline.get("pct_change", 0)
    if pct >= 5:
        events.append(("big_gain", f"今日大涨 {pct:.2f}%"))
    elif pct <= -5:
        events.append(("big_drop", f"今日大跌 {pct:.2f}%"))

    # 5) 量比 > 2 (放量异常)
    if kline.get("volume_ratio", 1) >= 2:
        events.append(("volume_spike", f"放量异常, 量比 {kline['volume_ratio']:.2f}"))

    return events


def intraday_alert() -> None:
    cfg = load_config()
    model = cfg["llm"]["model"]

    for item in cfg["watchlist"]:
        symbol = item["symbol"]
        try:
            spot = data.fetch_hk_spot(symbol)
            if spot is None:
                continue
            price = spot["price"]

            analysis, _ = _analyze(item, cfg)
            all_events = _detect_events(item, price, analysis)
            # 同一事件当天已发过的过滤掉
            new_events = [(eid, desc) for eid, desc in all_events
                          if not state.event_fired_today(symbol, eid)]
            if not new_events:
                continue

            text = llm.intraday_event_alert(item["name"], symbol, price, new_events, analysis, model)
            event_summary = ", ".join(desc for _, desc in new_events)
            title = f"🔔 {item['name']}({symbol}) · {price} · {event_summary}"

            # 颜色: 包含"跌破"或"大跌"用红, 包含"突破"或"大涨"用橙, 其他默认蓝
            color = "blue"
            txt = " ".join(desc for _, desc in new_events)
            if "跌破" in txt or "大跌" in txt:
                color = "red"
            elif "突破" in txt or "大涨" in txt:
                color = "orange"

            feishu.send_card(title, text, color=color)
            for eid, _ in new_events:
                state.mark_event_fired(symbol, eid)
        except Exception as e:
            traceback.print_exc()
            feishu.send_text(f"⚠️ 盘中预警失败 {symbol}: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m src.jobs [morning|snapshot|closing|intraday]")
        sys.exit(1)

    job = sys.argv[1]
    if job == "morning":
        morning_brief()
    elif job == "snapshot":
        daily_snapshot()
    elif job == "closing":
        closing_review()
    elif job == "intraday":
        intraday_alert()
    else:
        print(f"unknown job: {job}")
        sys.exit(1)
