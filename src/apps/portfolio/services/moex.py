from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen


MOEX_BASE_URL = "https://iss.moex.com/iss"


@dataclass(frozen=True)
class MoexHistoryResult:
    secid: str
    board: str
    dates: list[str]
    prices: list[float]

    @property
    def avg_price(self) -> float:
        if not self.prices:
            return 0.0
        return float(sum(self.prices) / len(self.prices))


def _fetch_json(path: str, params: dict[str, Any]) -> dict[str, Any]:
    query = urlencode(params)
    url = f"{MOEX_BASE_URL}{path}?{query}"
    with urlopen(url, timeout=12) as response:
        return json.loads(response.read().decode("utf-8"))


def _table_rows(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    block = payload.get(key) or {}
    columns = block.get("columns") or []
    data = block.get("data") or []
    rows: list[dict[str, Any]] = []
    for item in data:
        rows.append({columns[idx]: item[idx] for idx in range(min(len(columns), len(item)))})
    return rows


def search_tickers(query: str, limit: int = 8) -> list[dict[str, Any]]:
    clean_query = (query or "").strip().upper()
    if len(clean_query) < 1:
        return []

    payload = _fetch_json(
        "/securities.json",
        {
            "q": clean_query,
            "iss.meta": "off",
            "iss.only": "securities",
            "securities.columns": "secid,shortname,name,primary_boardid,is_traded",
        },
    )
    rows = _table_rows(payload, "securities")

    filtered = [
        row
        for row in rows
        if row.get("is_traded") == 1 and str(row.get("secid", "")).startswith(clean_query)
    ]
    if not filtered:
        filtered = [row for row in rows if row.get("is_traded") == 1]

    items: list[dict[str, Any]] = []
    for row in filtered[:limit]:
        items.append(
            {
                "secid": str(row.get("secid", "")).upper(),
                "shortname": str(row.get("shortname") or row.get("name") or ""),
                "board": str(row.get("primary_boardid") or "TQBR"),
            }
        )
    return items


def fetch_price_history(
    secid: str,
    board: str = "TQBR",
    from_date: str | None = None,
    till_date: str | None = None,
) -> MoexHistoryResult:
    if not secid:
        raise ValueError("secid is required")

    clean_secid = secid.strip().upper()
    clean_board = (board or "TQBR").strip().upper()

    till_value = till_date or date.today().isoformat()
    from_value = from_date or (date.today() - timedelta(days=365)).isoformat()

    all_rows: list[dict[str, Any]] = []
    start = 0
    page_size = 100

    while True:
        payload = _fetch_json(
            f"/history/engines/stock/markets/shares/boards/{clean_board}/securities/{clean_secid}.json",
            {
                "from": from_value,
                "till": till_value,
                "start": start,
                "iss.meta": "off",
                "iss.only": "history",
                "history.columns": "TRADEDATE,CLOSE,LEGALCLOSEPRICE,MARKETPRICE2",
            },
        )
        rows = _table_rows(payload, "history")
        if not rows:
            break

        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        start += page_size

    dates: list[str] = []
    prices: list[float] = []

    for row in all_rows:
        close = row.get("CLOSE")
        if close is None:
            close = row.get("LEGALCLOSEPRICE")
        if close is None:
            close = row.get("MARKETPRICE2")
        trade_date = row.get("TRADEDATE")
        if trade_date is None or close is None:
            continue
        try:
            price = float(close)
        except (TypeError, ValueError):
            continue
        if price <= 0:
            continue
        dates.append(str(trade_date))
        prices.append(price)

    if len(prices) < 20:
        raise ValueError(
            f"Недостаточно данных MOEX для {clean_secid} ({clean_board}). Получено точек: {len(prices)}"
        )

    return MoexHistoryResult(secid=clean_secid, board=clean_board, dates=dates, prices=prices)
