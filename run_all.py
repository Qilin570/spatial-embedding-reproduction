#!/usr/bin/env python3
"""Main entry point for running spatial embedding experiments.

Usage:
    python run_all.py --tables 3 4 5 6 7 8 9 15 18 --data-dir ./downloaded_data/
    python run_all.py --tables all --download
    python run_all.py --tables 3  # Run only Table 3
"""
import argparse
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


TABLE_MODULES = {
    3: 'experiments.table3',
    4: 'experiments.table4',
    5: 'experiments.table5',
    6: 'experiments.table6',
    7: 'experiments.table7',
    8: 'experiments.table8',
    9: 'experiments.table9',
    15: 'experiments.table15',
    18: 'experiments.table18',
}

# Recommended execution order
EXECUTION_ORDER = [3, 4, 15, 5, 6, 7, 18, 8, 9]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run spatial embedding experiments (Table 3-18)"
    )
    parser.add_argument(
        '--tables', nargs='+', default=['all'],
        help='Table numbers to run (e.g., 3 4 5) or "all"'
    )
    parser.add_argument(
        '--data-dir', default=None,
        help='Directory containing input data (default: ./downloaded_data/)'
    )
    parser.add_argument(
        '--output-dir', default=None,
        help='Directory for output results (default: ./results/)'
    )
    parser.add_argument(
        '--download', action='store_true',
        help='Download/link data before running experiments'
    )
    parser.add_argument(
        '--spatial-emb-dir', default=None,
        help='Path to existing spatial-embedding directory for data linking'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    project_dir = os.path.dirname(os.path.abspath(__file__))

    # Set data directory
    if args.data_dir:
        data_dir = os.path.abspath(args.data_dir)
    else:
        data_dir = os.path.join(project_dir, "downloaded_data")

    # Set output directory
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = os.path.join(project_dir, "results")

    os.makedirs(output_dir, exist_ok=True)

    # Download data if requested
    if args.download:
        from download_data import download_data
        download_data(data_dir, args.spatial_emb_dir)

    # Parse table numbers
    if 'all' in args.tables:
        tables = EXECUTION_ORDER
    else:
        tables = [int(t) for t in args.tables]
        # Sort by execution order
        tables = [t for t in EXECUTION_ORDER if t in tables]
        # Add any tables not in the standard order
        for t in [int(t) for t in args.tables]:
            if t not in tables:
                tables.append(t)

    print("=" * 60)
    print("SPATIAL EMBEDDING EXPERIMENTS")
    print("=" * 60)
    print(f"Data directory:   {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Tables to run:    {tables}")
    print("=" * 60)

    # Run experiments
    total_time = 0
    results = {}
    for table_num in tables:
        if table_num not in TABLE_MODULES:
            print(f"\nWARNING: Table {table_num} not implemented, skipping")
            continue

        module_name = TABLE_MODULES[table_num]
        print(f"\n{'#' * 60}")
        print(f"# Running Table {table_num}")
        print(f"{'#' * 60}")

        t0 = time.time()
        try:
            import importlib
            module = importlib.import_module(module_name)
            df = module.run(data_dir, output_dir)
            results[table_num] = df
            elapsed = time.time() - t0
            total_time += elapsed
            print(f"\nTable {table_num} completed in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            total_time += elapsed
            print(f"\nERROR running Table {table_num}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for table_num in tables:
        if table_num in results:
            df = results[table_num]
            if df is not None and not df.empty:
                print(f"  Table {table_num}: {len(df)} results -> results/table{table_num}.csv")
            else:
                print(f"  Table {table_num}: no results (missing data?)")
        else:
            print(f"  Table {table_num}: FAILED")
    print(f"\nTotal time: {total_time:.1f}s")
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()
