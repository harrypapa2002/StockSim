#!/usr/bin/env python3
"""
Simple Docker stats monitor for StockSim

Captures Docker container metrics for scaling analysis:

CPU %      - Percentage of CPU cores used (can exceed 100% on multi-core)
             Example: 200% = using 2 full CPU cores
MEM USAGE  - Actual memory consumption in MB/GB  
MEM %      - Percentage of container's memory limit used
NET I/O    - Network bytes: input / output (cumulative since container start)
BLOCK I/O  - Disk bytes: read / write (cumulative since container start)
PIDS       - Number of active processes in container

Usage:
    python utils/monitor_usage.py --duration 300
    python utils/monitor_usage.py --analyze reports/usage.csv
"""

import subprocess
import time
import argparse
import csv
import re
from datetime import datetime

def parse_percentage(value):
    """Extract percentage from string like '15.5%'"""
    return float(value.rstrip('%'))

def parse_memory(value):
    """Extract memory usage from '1.5GiB / 8GiB' format"""
    parts = value.split(' / ')
    used = parts[0].strip()
    limit = parts[1].strip()
    
    def to_mb(mem_str):
        num = float(re.findall(r'[\d.]+', mem_str)[0])
        if 'GiB' in mem_str or 'GB' in mem_str:
            return num * 1024
        elif 'MiB' in mem_str or 'MB' in mem_str:
            return num
        return num
    
    return to_mb(used), to_mb(limit)

def parse_io(value):
    """Extract IO from '1.5MB / 2.3MB' format and return total MB"""
    parts = value.split(' / ')
    input_val = float(re.findall(r'[\d.]+', parts[0])[0])
    output_val = float(re.findall(r'[\d.]+', parts[1])[0])
    
    # Convert to MB if needed
    def to_mb(val, unit_str):
        if 'GB' in unit_str:
            return val * 1024
        elif 'kB' in unit_str:
            return val / 1024
        return val  # Already MB
    
    input_mb = to_mb(input_val, parts[0])
    output_mb = to_mb(output_val, parts[1])
    return input_mb + output_mb

def monitor(duration=300, interval=5):
    """Monitor docker stats and save to CSV"""
    output_file = f"reports/usage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    print(f"Monitoring for {duration}s, saving to {output_file}")
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'container', 'cpu_pct', 'mem_used_mb', 'mem_limit_mb', 'mem_pct', 'net_io_mb', 'disk_io_mb', 'pids'])
        
        start_time = time.time()
        while time.time() - start_time < duration:
            try:
                # Monitor both containers
                containers = ['stocksim-simulation', 'stocksim-rabbitmq']
                timestamp = datetime.now().isoformat()
                
                for container in containers:
                    result = subprocess.run(['docker', 'stats', '--no-stream', '--format', 
                                           '{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}},{{.NetIO}},{{.BlockIO}},{{.PIDs}}', 
                                           container], 
                                          capture_output=True, text=True, timeout=10)
                    
                    if result.returncode == 0 and result.stdout.strip():
                        # Parse CSV format output
                        parts = result.stdout.strip().split(',')
                        if len(parts) == 6:
                            cpu_pct = parse_percentage(parts[0])
                            mem_used, mem_limit = parse_memory(parts[1])
                            mem_pct = parse_percentage(parts[2])
                            net_io_mb = parse_io(parts[3])
                            disk_io_mb = parse_io(parts[4])
                            pids = int(parts[5])
                            
                            writer.writerow([timestamp, container, cpu_pct, mem_used, mem_limit, mem_pct, net_io_mb, disk_io_mb, pids])
                            f.flush()  # Force write to disk
                            
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] {container[:8]:8} | CPU: {cpu_pct:5.1f}% | MEM: {mem_used:6.0f}MB ({mem_pct:4.1f}%) | NET: {net_io_mb:5.1f}MB | DISK: {disk_io_mb:5.1f}MB | PIDs: {pids:3d}")
                        else:
                            print(f"Unexpected format for {container}: {parts}")
                    else:
                        print(f"Docker command failed for {container}: {result.stderr}")
                
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
            
            time.sleep(interval)
    
    return output_file

def analyze(csv_file):
    """Analyze CSV and show max/average by container"""
    # Separate data by container
    containers = {'stocksim-simulation': {}, 'stocksim-rabbitmq': {}}
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            container = row['container']
            if container not in containers:
                containers[container] = {}
            
            for field in ['cpu_pct', 'mem_used_mb', 'mem_pct', 'net_io_mb', 'disk_io_mb', 'pids']:
                if field not in containers[container]:
                    containers[container][field] = []
                value = float(row[field]) if field != 'pids' else int(row[field])
                containers[container][field].append(value)
    
    print("\nSTOCKSIM USAGE ANALYSIS")
    print("="*70)
    
    all_stats = {}
    for container_name, data in containers.items():
        if not data:  # Skip if no data for this container
            continue
            
        print(f"\n{container_name.upper()}")
        print("-" * 40)
        
        container_stats = {}
        for field in data:
            values = data[field]
            if values:  # Only process if we have data
                max_val = max(values)
                avg_val = sum(values)/len(values)
                print(f"{field.upper():12} | Max: {max_val:8.1f} | Avg: {avg_val:8.1f}")
                
                container_stats[field] = {
                    'max': round(max_val, 2),
                    'avg': round(avg_val, 2),
                    'min': round(min(values), 2),
                    'count': len(values)
                }
        
        all_stats[container_name] = container_stats
    
    print("="*70)
    
    import json
    print("\nEXTRACTABLE STATS (JSON):")
    print(json.dumps(all_stats, indent=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", "-d", type=int, default=300)
    parser.add_argument("--interval", "-i", type=int, default=5)
    parser.add_argument("--analyze", "-a", type=str, help="Analyze existing CSV file")
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze(args.analyze)
    else:
        output_file = monitor(args.duration, args.interval)
        analyze(output_file)

if __name__ == "__main__":
    main()