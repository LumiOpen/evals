#!/usr/bin/env python3
"""
Prepare Eurovoc dataset for multilingual LongPPL evaluation.

This script:
1. Loads the EuropeanParliament/Eurovoc dataset
2. Analyzes document lengths per language
3. Filters documents with ≥16k tokens (using reference tokenizer)
4. Selects 50 longest documents per language
5. Saves filtered datasets for LongPPL evaluation

EU Languages (24):
bg, cs, da, de, el, en, es, et, fi, fr, ga, hr, hu, it, lt, lv, mt, nl, pl, pt, ro, sk, sl, sv
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# 24 EU languages
EU_LANGUAGES = [
    "bg",  # Bulgarian
    "cs",  # Czech
    "da",  # Danish
    "de",  # German
    "el",  # Greek
    "en",  # English
    "es",  # Spanish
    "et",  # Estonian
    "fi",  # Finnish
    "fr",  # French
    "ga",  # Irish
    "hr",  # Croatian
    "hu",  # Hungarian
    "it",  # Italian
    "lt",  # Lithuanian
    "lv",  # Latvian
    "mt",  # Maltese
    "nl",  # Dutch
    "pl",  # Polish
    "pt",  # Portuguese
    "ro",  # Romanian
    "sk",  # Slovak
    "sl",  # Slovenian
    "sv",  # Swedish
]

LANGUAGE_NAMES = {
    "bg": "Bulgarian", "cs": "Czech", "da": "Danish", "de": "German",
    "el": "Greek", "en": "English", "es": "Spanish", "et": "Estonian",
    "fi": "Finnish", "fr": "French", "ga": "Irish", "hr": "Croatian",
    "hu": "Hungarian", "it": "Italian", "lt": "Lithuanian", "lv": "Latvian",
    "mt": "Maltese", "nl": "Dutch", "pl": "Polish", "pt": "Portuguese",
    "ro": "Romanian", "sk": "Slovak", "sl": "Slovenian", "sv": "Swedish"
}


def load_reference_tokenizer(model_name: str = "meta-llama/Meta-Llama-3-8B"):
    """Load reference tokenizer for length estimation."""
    print(f"Loading reference tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def analyze_eurovoc_dataset(
    dataset_name: str = "EuropeanParliament/Eurovoc",
    min_tokens: int = 16384,
    samples_per_language: int = 50,
    max_samples_to_scan: int = 10000,
    output_dir: str = "data/eurovoc_filtered"
) -> Dict[str, List[Dict]]:
    """
    Analyze Eurovoc dataset and extract long documents per language.

    Returns:
        Dict mapping language code to list of document dicts
    """
    print(f"Loading Eurovoc dataset: {dataset_name}")
    print(f"Target: {samples_per_language} documents per language with ≥{min_tokens} tokens")
    print(f"Scanning up to {max_samples_to_scan} samples per config")
    print()

    # Load reference tokenizer for length estimation
    tokenizer = load_reference_tokenizer()

    # Storage for filtered documents per language
    language_docs: Dict[str, List[Tuple[int, Dict]]] = {lang: [] for lang in EU_LANGUAGES}

    # Try to load dataset
    try:
        # Eurovoc is organized by date configs (e.g., "1996-03", "2004-06")
        # We'll try a few different approaches to load it

        print("Attempting to load dataset...")

        # Approach 1: Try loading without config (may work if there's a default)
        try:
            dataset = load_dataset(dataset_name, split="train", streaming=True)
            print("✓ Loaded dataset in streaming mode")
        except Exception as e:
            print(f"Could not load without config: {e}")
            print("\nNote: Eurovoc may require specific date configs.")
            print("Checking available configs...")

            # Try to get dataset info
            from datasets import get_dataset_config_names
            try:
                configs = get_dataset_config_names(dataset_name)
                print(f"Available configs: {len(configs)} found")
                print(f"Sample configs: {configs[:5]}")

                # Load first few configs to sample documents
                dataset = load_dataset(dataset_name, configs[0], split="train", streaming=True)
                print(f"✓ Loaded first config: {configs[0]}")
            except Exception as e2:
                print(f"Error loading configs: {e2}")
                return language_docs

        # Process documents
        print("\nProcessing documents...")
        scanned = 0

        for doc in tqdm(dataset, desc="Scanning documents", total=max_samples_to_scan):
            if scanned >= max_samples_to_scan:
                break

            # Extract language and text
            lang = doc.get("lang", "").lower()
            text = doc.get("text", "")

            # Skip if not EU language or no text
            if lang not in EU_LANGUAGES or not text:
                scanned += 1
                continue

            # Tokenize and check length
            tokens = tokenizer.encode(text, add_special_tokens=False)
            token_count = len(tokens)

            # If long enough, add to language's list
            if token_count >= min_tokens:
                doc_info = {
                    "title": doc.get("title", ""),
                    "text": text,
                    "lang": lang,
                    "token_count": token_count,
                    "url": doc.get("url", ""),
                    "date": doc.get("date", ""),
                    "eurovoc_concepts": doc.get("eurovoc_concepts", [])
                }

                # Store with token count for sorting
                language_docs[lang].append((token_count, doc_info))

            scanned += 1

        print(f"\nScanned {scanned} documents")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nTrying alternative approach: loading multiple configs...")

        # Alternative: Try loading multiple date configs
        try:
            from datasets import get_dataset_config_names
            configs = get_dataset_config_names(dataset_name)

            # Sample from multiple configs to get diverse documents
            for config in tqdm(configs[:20], desc="Loading configs"):  # Try first 20 configs
                try:
                    config_dataset = load_dataset(dataset_name, config, split="train")

                    for doc in config_dataset:
                        lang = doc.get("lang", "").lower()
                        text = doc.get("text", "")

                        if lang not in EU_LANGUAGES or not text:
                            continue

                        # Check if we already have enough for this language
                        if len(language_docs[lang]) >= samples_per_language * 2:  # Get 2x for safety
                            continue

                        tokens = tokenizer.encode(text, add_special_tokens=False)
                        token_count = len(tokens)

                        if token_count >= min_tokens:
                            doc_info = {
                                "title": doc.get("title", ""),
                                "text": text,
                                "lang": lang,
                                "token_count": token_count,
                                "url": doc.get("url", ""),
                                "date": doc.get("date", ""),
                                "eurovoc_concepts": doc.get("eurovoc_concepts", [])
                            }
                            language_docs[lang].append((token_count, doc_info))

                except Exception as e:
                    print(f"Skipping config {config}: {e}")
                    continue

        except Exception as e:
            print(f"Could not load configs: {e}")
            return language_docs

    # Filter to top N longest documents per language
    print("\n" + "=" * 80)
    print("FILTERING RESULTS")
    print("=" * 80)

    filtered_docs = {}
    for lang in EU_LANGUAGES:
        docs = language_docs[lang]

        if not docs:
            print(f"{lang.upper()} ({LANGUAGE_NAMES[lang]:12s}): No documents found with ≥{min_tokens} tokens")
            continue

        # Sort by token count (descending) and take top N
        docs.sort(key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in docs[:samples_per_language]]

        filtered_docs[lang] = top_docs

        avg_tokens = sum(d["token_count"] for d in top_docs) / len(top_docs)
        print(f"{lang.upper()} ({LANGUAGE_NAMES[lang]:12s}): {len(top_docs):3d} documents, "
              f"avg {avg_tokens:6.0f} tokens, max {top_docs[0]['token_count']:6d} tokens")

    return filtered_docs


def save_filtered_datasets(
    language_docs: Dict[str, List[Dict]],
    output_dir: str = "data/eurovoc_filtered"
):
    """Save filtered documents to JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("SAVING FILTERED DATASETS")
    print("=" * 80)

    summary = {
        "dataset": "EuropeanParliament/Eurovoc",
        "reference_tokenizer": "meta-llama/Meta-Llama-3-8B",
        "languages": {}
    }

    for lang, docs in language_docs.items():
        if not docs:
            continue

        # Save language-specific file
        output_file = output_path / f"eurovoc_{lang}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "language": lang,
                "language_name": LANGUAGE_NAMES[lang],
                "num_documents": len(docs),
                "documents": docs
            }, f, indent=2, ensure_ascii=False)

        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"✓ Saved {lang}: {output_file} ({file_size_mb:.1f} MB)")

        # Add to summary
        summary["languages"][lang] = {
            "language_name": LANGUAGE_NAMES[lang],
            "num_documents": len(docs),
            "avg_tokens": sum(d["token_count"] for d in docs) / len(docs),
            "max_tokens": max(d["token_count"] for d in docs),
            "min_tokens": min(d["token_count"] for d in docs)
        }

    # Save summary
    summary_file = output_path / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Saved summary: {summary_file}")

    print("\n" + "=" * 80)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 80)
    print(f"Languages with data: {len(summary['languages'])}/24")
    print(f"Total documents: {sum(s['num_documents'] for s in summary['languages'].values())}")
    print(f"Output directory: {output_path.absolute()}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Eurovoc for multilingual LongPPL")
    parser.add_argument("--min-tokens", type=int, default=16384,
                        help="Minimum token count for documents (default: 16384)")
    parser.add_argument("--samples-per-language", type=int, default=50,
                        help="Number of documents to select per language (default: 50)")
    parser.add_argument("--max-scan", type=int, default=50000,
                        help="Maximum documents to scan (default: 50000)")
    parser.add_argument("--output-dir", type=str, default="data/eurovoc_filtered",
                        help="Output directory for filtered datasets")

    args = parser.parse_args()

    print("=" * 80)
    print("EUROVOC MULTILINGUAL LONGPPL DATASET PREPARATION")
    print("=" * 80)
    print(f"Minimum tokens: {args.min_tokens}")
    print(f"Samples per language: {args.samples_per_language}")
    print(f"Max documents to scan: {args.max_scan}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target languages: {len(EU_LANGUAGES)} EU languages")
    print("=" * 80)
    print()

    # Analyze and filter dataset
    language_docs = analyze_eurovoc_dataset(
        min_tokens=args.min_tokens,
        samples_per_language=args.samples_per_language,
        max_samples_to_scan=args.max_scan,
        output_dir=args.output_dir
    )

    # Save filtered datasets
    if language_docs:
        save_filtered_datasets(language_docs, args.output_dir)
    else:
        print("\nERROR: No documents found. Check dataset access and configuration.")
        sys.exit(1)


if __name__ == "__main__":
    main()
