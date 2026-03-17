#!/usr/bin/env python
"""Run difficulty validation stage standalone."""
import sys
import os

# Ensure we're in the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import asyncio
from mirror.data.difficulty_validator import validate_all_difficulties
from mirror.data.provenance import build_provenance_table, compile_final_question_bank
import json
from datetime import datetime

print("=" * 60)
print("Running difficulty validation...")
print("=" * 60)

stats = asyncio.run(validate_all_difficulties())

print("\n" + "=" * 60)
print("Difficulty stats summary:")
for domain, domain_stats in stats.items():
    print(f"\n{domain}:")
    print(f"  8B accuracy: {domain_stats.get('llama8b_accuracy', 0):.1%}")
    dist = domain_stats.get('difficulty_distribution', {})
    total = domain_stats.get('total', 0)
    for diff, count in sorted(dist.items()):
        pct = count / total * 100 if total else 0
        print(f"  {diff}: {count} ({pct:.0f}%)")

print("\n" + "=" * 60)
print("Compiling final question bank...")
total = compile_final_question_bank()
print(f"Total questions compiled: {total}")

print("\n" + "=" * 60)
print("Building provenance table...")
prov_stats = build_provenance_table()
print(f"Provenance stats: {prov_stats}")

print("\nDone!")
