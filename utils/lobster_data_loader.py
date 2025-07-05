import copy
import csv
import json
import os
import random
from datetime import datetime, timedelta
from statistics import mean, median
from typing import List, Dict, Any

from utils.time_utils import parse_datetime_utc


#############################
# Parsing and Analysis Tools
#############################
def parse_lobster_message_csv(file_path: str, trading_date: str) -> List[Dict[str, Any]]:
    """
    Reads a LOBSTER message CSV file and extracts new-order submissions.
    Returns a list of orders with {timestamp, side, price, quantity, order_type="LIMIT"}.
    """
    orders = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) < 6:
                continue
            time_str, event_type, _, size_str, price_str, direction_str = row
            if event_type.strip() != "1":  # Only new order submissions (Type "1")
                continue

            try:
                seconds_after_midnight = float(time_str.strip())
                base_date = parse_datetime_utc(trading_date)
                order_time = base_date + timedelta(seconds=seconds_after_midnight)
                price = float(price_str) / 10000
                quantity = int(size_str)
                side = "BUY" if direction_str.strip() == "1" else "SELL"
            except ValueError:
                continue

            orders.append({
                "timestamp": order_time.isoformat(),
                "side": side,
                "price": price,
                "quantity": quantity,
                "order_type": "LIMIT"
            })

    return sorted(orders, key=lambda o: o["timestamp"])


def analyze_orders(orders: List[Dict[str, Any]], title: str):
    """Perform analysis on orders and print summary statistics."""
    if not orders:
        print(f"\n{title}: No orders found.\n")
        return

    prices = [order["price"] for order in orders]
    quantities = [order["quantity"] for order in orders]

    print(f"\n=== {title} ===")
    print(f"Total Orders: {len(orders)}")
    print(
        f"Price Stats - Min: {min(prices):.2f}, Max: {max(prices):.2f}, Avg: {mean(prices):.2f}, Median: {median(prices):.2f}")
    print(
        f"Quantity Stats - Min: {min(quantities)}, Max: {max(quantities)}, Avg: {mean(quantities):.2f}, Median: {median(quantities):.2f}")
    print("===================================")

    return {
        "min_price": min(prices),
        "max_price": max(prices),
        "median_price": median(prices),
        "min_qty": min(quantities),
        "max_qty": max(quantities),
    }
#############################
# File Writing Utility
#############################
def split_orders_into_files(orders: List[Dict[str, Any]], num_files: int, output_dir: str, base_name: str, ticker: str):
    """Splits orders into multiple JSON files for structured output."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    partitions = [[] for _ in range(num_files)]
    for i, order in enumerate(orders):
        partitions[i % num_files].append(order)

    for idx, part in enumerate(partitions):
        output_path = os.path.join(output_dir, f"{base_name}_part{idx}.json")
        with open(output_path, "w") as f:
            json.dump({ticker: part}, f, indent=2)
        print(f"Wrote {len(part)} orders to {output_path}")


#############################
# Main Routine
#############################
def main():
    csv_file = "/Users/harry/PycharmProjects/StockSim/data/AAPL_2012-06-21_34200000_57600000_message_10.csv"
    trading_date = "2025-03-01"
    output_directory = "/Users/harry/PycharmProjects/StockSim/configs/orders"
    ticker = "AAPL"
    num_split_files = 10

    orders = parse_lobster_message_csv(csv_file, trading_date)
    analysis = analyze_orders(orders, "Original LOBSTER Data")

    scenarios = {
        "original": orders
    }

    for scenario_name, scenario_orders in scenarios.items():
        analyze_orders(scenario_orders, f"Scenario: {scenario_name.title()}")
        split_orders_into_files(scenario_orders, num_split_files, output_directory, scenario_name, ticker)


if __name__ == "__main__":
    main()