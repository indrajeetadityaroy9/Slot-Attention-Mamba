#!/usr/bin/env python3
"""
Download and prepare ContraPro dataset for pronoun disambiguation evaluation.

ContraPro (Contrastive Pronoun Test Set) by Müller et al. (2018) tests
English→German pronoun translation accuracy through contrastive scoring.

Reference:
    Müller, M., Rios, A., Voita, E., & Sennrich, R. (2018).
    A Large-Scale Test Set for the Evaluation of Context-Dependent Pronoun
    Translation in Neural Machine Translation. WMT 2018.

Usage:
    python scripts/setup_contrapro.py
    python scripts/setup_contrapro.py --output data/contrapro/contrapro.json
"""

import argparse
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Any

try:
    import requests
except ImportError:
    print("Error: requests library required. Install with: pip install requests")
    sys.exit(1)


# ContraPro GitHub repository URL
CONTRAPRO_URL = "https://github.com/ZurichNLP/ContraPro/archive/refs/heads/master.zip"
DEFAULT_OUTPUT = "data/contrapro/contrapro.json"


def download_contrapro(url: str = CONTRAPRO_URL) -> zipfile.ZipFile:
    """Download ContraPro from GitHub."""
    print(f"Downloading ContraPro from {url}...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return zipfile.ZipFile(io.BytesIO(response.content))


def parse_contrapro_file(z: zipfile.ZipFile, filename: str) -> List[Dict[str, Any]]:
    """Parse a single ContraPro JSON file from the archive."""
    try:
        content = z.read(filename)
        return json.loads(content)
    except Exception as e:
        print(f"Warning: Could not parse {filename}: {e}")
        return []


def convert_to_expected_format(raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert ContraPro format to the format expected by ContraProDataset.

    Expected output format:
    {
        "src_context": str,      # Previous context sentences
        "src_current": str,      # Current source sentence with pronoun
        "ref": str,              # Correct German translation
        "contrastive": [str],    # Wrong German translations
        "pronoun_type": str,     # Pronoun category (it, they, you, etc.)
        "ante_distance": int,    # Distance to antecedent
        "antecedent": str        # The antecedent noun phrase
    }
    """
    examples = []

    for item in raw_data:
        # ContraPro format varies, handle different field names
        example = {
            # Source context (previous sentences)
            "src_context": item.get("src context", item.get("src_context", "")),

            # Current source sentence
            "src_current": item.get("src segment", item.get("src_current", item.get("src", ""))),

            # Correct translation
            "ref": item.get("ref", item.get("correct", item.get("trg segment", ""))),

            # Contrastive (wrong) translations - may be a list or need to be constructed
            "contrastive": item.get("contrastive", item.get("contrastive_translations", [])),

            # Pronoun type
            "pronoun_type": item.get("class", item.get("pronoun_type", "unknown")),

            # Antecedent distance
            "ante_distance": item.get("ante distance", item.get("ante_distance", 0)),

            # Antecedent phrase
            "antecedent": item.get("ante", item.get("antecedent", "")),
        }

        # Handle case where contrastive is not a list
        if not isinstance(example["contrastive"], list):
            if example["contrastive"]:
                example["contrastive"] = [example["contrastive"]]
            else:
                example["contrastive"] = []

        # Try to extract contrastive translations from errors field
        if not example["contrastive"] and "errors" in item:
            errors = item["errors"]
            if isinstance(errors, list):
                example["contrastive"] = [e.get("replacement", e) if isinstance(e, dict) else str(e) for e in errors]
            elif isinstance(errors, dict):
                example["contrastive"] = [errors.get("replacement", str(errors))]

        # Only include examples with valid data
        if example["src_current"] and example["ref"]:
            examples.append(example)

    return examples


def setup_contrapro(output_path: str = DEFAULT_OUTPUT) -> int:
    """
    Download and prepare ContraPro dataset.

    Returns:
        Number of examples processed
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Download archive
    z = download_contrapro()

    # List all JSON files in archive
    json_files = [f for f in z.namelist() if f.endswith('.json') and 'contrapro' in f.lower()]

    if not json_files:
        # Fallback: look for any JSON file
        json_files = [f for f in z.namelist() if f.endswith('.json')]

    print(f"Found {len(json_files)} JSON files in archive")

    # Parse and combine all examples
    all_examples = []

    for json_file in json_files:
        print(f"  Processing: {json_file}")
        raw_data = parse_contrapro_file(z, json_file)
        if raw_data:
            examples = convert_to_expected_format(raw_data)
            print(f"    Extracted {len(examples)} examples")
            all_examples.extend(examples)

    if not all_examples:
        # Try the main contrapro.json file directly
        main_file = "ContraPro-master/contrapro.json"
        if main_file in z.namelist():
            print(f"  Processing main file: {main_file}")
            raw_data = parse_contrapro_file(z, main_file)
            all_examples = convert_to_expected_format(raw_data)

    # Save to output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_examples, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(all_examples)} examples to {output_path}")

    # Print statistics
    if all_examples:
        pronoun_types = {}
        distances = {}
        for ex in all_examples:
            pt = ex.get("pronoun_type", "unknown")
            pronoun_types[pt] = pronoun_types.get(pt, 0) + 1
            dist = ex.get("ante_distance", 0)
            distances[dist] = distances.get(dist, 0) + 1

        print("\nPronoun type distribution:")
        for pt, count in sorted(pronoun_types.items(), key=lambda x: -x[1])[:10]:
            print(f"  {pt}: {count}")

        print("\nAntecedent distance distribution:")
        for dist, count in sorted(distances.items()):
            print(f"  {dist}: {count}")

    return len(all_examples)


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare ContraPro dataset for pronoun disambiguation evaluation"
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT,
        help=f"Output JSON file path (default: {DEFAULT_OUTPUT})"
    )
    args = parser.parse_args()

    try:
        count = setup_contrapro(args.output)
        if count > 0:
            print(f"\nContraPro setup complete. {count} examples ready for evaluation.")
            print(f"\nTo run evaluation:")
            print(f"  python scripts/evaluate.py checkpoint=<path> eval_mode=contrapro")
        else:
            print("\nWarning: No examples extracted. Check ContraPro repository structure.")
            sys.exit(1)
    except Exception as e:
        print(f"\nError setting up ContraPro: {e}")
        print("\nYou can also manually download from:")
        print("  https://github.com/ZurichNLP/ContraPro")
        sys.exit(1)


if __name__ == "__main__":
    main()
