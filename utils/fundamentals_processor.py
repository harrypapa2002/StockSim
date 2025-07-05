"""
Fundamental Data Processing Utilities for StockSim

This module provides utilities for processing and filtering fundamental data
from various financial data sources, particularly Polygon.io fundamental data.

Functions:
    extract_polygon_fundamentals: Extract and filter fundamental data within date ranges
"""

from typing import Dict, Any, Optional, List

from .date_parsers import parse_iso_date, parse_iso_datetime


def extract_polygon_fundamentals(
    raw: Dict[str, Any],
    prev_cutoff: Optional[str],
    as_of_date: str
) -> Dict[str, Any]:
    """
    Extract and filter fundamental data within specified date range.
    
    This function processes raw fundamental data from Polygon.io and filters
    events based on their relevant dates to only include new information
    within the specified time window.
    
    Args:
        raw: Raw fundamental data dictionary
        prev_cutoff: Previous cutoff date (exclusive) in YYYY-MM-DD format
        as_of_date: Current date (inclusive) in YYYY-MM-DD format
        
    Returns:
        Filtered fundamental data dictionary with date-filtered events
    """
    cutoff_prev = parse_iso_date(prev_cutoff) if prev_cutoff else None
    cutoff_new = parse_iso_date(as_of_date)
    if not cutoff_new:
        raise ValueError("`as_of_date` must be a valid YYYY-MM-DD string")

    out: Dict[str, Any] = {}

    # Filter IPOs by announced_date
    ipos_clean: List[Dict[str, Any]] = []
    for ipo in raw.get("ipos", []):
        ann_str = ipo.get("announced_date")
        ann_dt = parse_iso_date(ann_str)
        if not ann_dt:
            continue
        if (cutoff_prev is None or ann_dt > cutoff_prev) and ann_dt <= cutoff_new:
            ipos_clean.append({
                "announced_date": ann_dt.isoformat(),
                "ipo_status": ipo.get("ipo_status"),
                "total_offer_size": float(ipo.get("total_offer_size", 0.0)),
            })
    ipos_clean.sort(key=lambda x: x["announced_date"], reverse=True)
    out["ipos"] = ipos_clean

    # Filter splits by execution_date
    splits_clean: List[Dict[str, Any]] = []
    for split in raw.get("splits", []):
        exec_str = split.get("execution_date")
        exec_dt = parse_iso_date(exec_str)
        if not exec_dt:
            continue
        if (cutoff_prev is None or exec_dt > cutoff_prev) and exec_dt <= cutoff_new:
            splits_clean.append({
                "execution_date": exec_dt.isoformat(),
                "split_from": int(split.get("split_from", 1)),
                "split_to": int(split.get("split_to", 1)),
                "ticker": split.get("ticker"),
                "id": split.get("id"),
            })
    splits_clean.sort(key=lambda x: x["execution_date"], reverse=True)
    out["splits"] = splits_clean

    # Filter dividends by ex_dividend_date
    dividends_clean: List[Dict[str, Any]] = []
    for d in raw.get("dividends", []):
        ex_str = d.get("ex_dividend_date")
        ex_dt = parse_iso_date(ex_str)
        if not ex_dt:
            continue
        if (cutoff_prev is None or ex_dt > cutoff_prev) and ex_dt <= cutoff_new:
            dividends_clean.append({
                "declaration_date": (
                    parse_iso_date(d.get("declaration_date")).isoformat()
                    if parse_iso_date(d.get("declaration_date")) else None
                ),
                "ex_dividend_date": ex_dt.isoformat(),
                "record_date": (
                    parse_iso_date(d.get("record_date")).isoformat()
                    if parse_iso_date(d.get("record_date")) else None
                ),
                "pay_date": (
                    parse_iso_date(d.get("pay_date")).isoformat()
                    if parse_iso_date(d.get("pay_date")) else None
                ),
                "cash_amount": float(d.get("cash_amount", 0.0)),
                "currency": d.get("currency"),
                "frequency": int(d.get("frequency", 0)),
                "ticker": d.get("ticker"),
                "id": d.get("id"),
            })
    dividends_clean.sort(key=lambda x: x["ex_dividend_date"], reverse=True)
    out["dividends"] = dividends_clean

    # Filter ticker events by event date
    te_raw = raw.get("ticker_events", {})
    te_clean: Dict[str, Any] = {
        "name": te_raw.get("name"),
        "cik": te_raw.get("cik"),
        "composite_figi": te_raw.get("composite_figi"),
    }
    events_list: List[Dict[str, Any]] = []
    for ev in te_raw.get("events", []):
        ev_date = parse_iso_date(ev.get("date"))
        if not ev_date:
            continue
        if (cutoff_prev is None or ev_date > cutoff_prev) and ev_date <= cutoff_new:
            details = {k: v for k, v in ev.items() if k not in {"type", "date"}}
            events_list.append({
                "type": ev.get("type"),
                "date": ev_date.isoformat(),
                "details": details
            })
    events_list.sort(key=lambda x: x["date"], reverse=True)
    te_clean["events"] = events_list
    out["ticker_events"] = te_clean

    # Filter financials by filing_date
    fin_clean: List[Dict[str, Any]] = []
    for entry in raw.get("financials", []):
        filing_str = entry.get("filing_date")
        filing_dt = parse_iso_date(filing_str)
        if not filing_dt:
            continue
        if (cutoff_prev is None or filing_dt > cutoff_prev) and filing_dt <= cutoff_new:
            fin_clean.append({
                "start_date": (
                    parse_iso_date(entry.get("start_date")).isoformat()
                    if parse_iso_date(entry.get("start_date")) else None
                ),
                "end_date": (
                    parse_iso_date(entry.get("end_date")).isoformat()
                    if parse_iso_date(entry.get("end_date")) else None
                ),
                "filing_date": filing_dt.isoformat(),
                "acceptance_datetime": (
                    parse_iso_datetime(entry.get("acceptance_datetime")).isoformat()
                    if parse_iso_datetime(entry.get("acceptance_datetime")) else None
                ),
                "timeframe": entry.get("timeframe"),
                "fiscal_period": entry.get("fiscal_period"),
                "fiscal_year": entry.get("fiscal_year"),
                "cik": entry.get("cik"),
                "sic": entry.get("sic"),
                "tickers": entry.get("tickers", []),
                "company_name": entry.get("company_name"),
                "source_filing_url": entry.get("source_filing_url"),
                "source_filing_file_url": entry.get("source_filing_file_url"),
                "financials": entry.get("financials", {}),
            })
    fin_clean.sort(key=lambda x: x["filing_date"], reverse=True)
    out["financials"] = fin_clean

    return out