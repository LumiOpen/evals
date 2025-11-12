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

# 24 EU languages (2-letter ISO 639-1 codes)
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

# EuroVoc dataset uses 3-letter ISO 639-2 codes, map to 2-letter ISO 639-1
LANG_3_TO_2 = {
    "bul": "bg", "ces": "cs", "dan": "da", "deu": "de", "ger": "de",
    "ell": "el", "gre": "el", "eng": "en", "spa": "es", "est": "et",
    "fin": "fi", "fra": "fr", "fre": "fr", "gle": "ga", "hrv": "hr",
    "hun": "hu", "ita": "it", "lit": "lt", "lav": "lv", "mlt": "mt",
    "nld": "nl", "dut": "nl", "pol": "pl", "por": "pt", "ron": "ro",
    "rum": "ro", "slk": "sk", "slo": "sk", "slv": "sl", "swe": "sv"
}

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
    print(f"Max to scan per config: {max_samples_to_scan}")
    print()

    # Load reference tokenizer for length estimation
    tokenizer = load_reference_tokenizer()

    # Storage for filtered documents per language
    language_docs: Dict[str, List[Tuple[int, Dict]]] = {lang: [] for lang in EU_LANGUAGES}

    # Try to load dataset - Eurovoc requires date configs
    try:
        from datasets import get_dataset_config_names

        print("Getting available Eurovoc configs...")
        configs = get_dataset_config_names(dataset_name)
        print(f"Found {len(configs)} configs")

        # Prioritize more recent configs (likely to have longer documents)
        # Configs are in format YYYY-MM, so sorting gives chronological order
        configs_sorted = sorted(configs, reverse=True)  # Start with most recent
        print(f"Scanning configs from {configs_sorted[0]} (newest) to {configs_sorted[-1]} (oldest)")
        print()

        # Sample from multiple configs to get diverse documents
        configs_processed = 0

        for config in tqdm(configs_sorted, desc="Scanning configs"):
            # Stop if we have enough documents for all languages
            all_languages_satisfied = all(
                len(language_docs[lang]) >= samples_per_language * 2  # Get 2x for safety
                for lang in EU_LANGUAGES
            )
            if all_languages_satisfied:
                print(f"\n✓ Found sufficient documents for all {len(EU_LANGUAGES)} languages")
                break

            try:
                print(f"\n→ Loading config {config}...", flush=True)
                config_dataset = load_dataset(dataset_name, config, split="train")
                print(f"✓ Loaded {len(config_dataset)} documents from {config}", flush=True)
                configs_processed += 1

                scanned_in_config = 0
                found_in_config = 0

                for doc in config_dataset:
                    # Limit per config, not total
                    if scanned_in_config >= max_samples_to_scan:
                        break

                    lang_3 = doc.get("lang", "").lower()
                    text = doc.get("text", "")

                    scanned_in_config += 1

                    # Progress update every 1000 documents
                    if scanned_in_config % 1000 == 0:
                        print(f"  → Scanned {scanned_in_config} docs in {config}, found {found_in_config} long docs so far", flush=True)

                    # Convert 3-letter code to 2-letter code
                    lang = LANG_3_TO_2.get(lang_3, lang_3)

                    if lang not in EU_LANGUAGES or not text:
                        continue

                    # Check if we already have enough for this language
                    if len(language_docs[lang]) >= samples_per_language * 2:  # Get 2x for safety
                        continue

                    tokens = tokenizer.encode(text, add_special_tokens=False)
                    token_count = len(tokens)

                    if token_count >= min_tokens:
                        # Convert date to string if it's a datetime object
                        date_value = doc.get("date", "")
                        if hasattr(date_value, 'isoformat'):
                            date_value = date_value.isoformat()

                        doc_info = {
                            "title": doc.get("title", ""),
                            "text": text,
                            "lang": lang,
                            "token_count": token_count,
                            "url": doc.get("url", ""),
                            "date": str(date_value) if date_value else "",
                            "eurovoc_concepts": doc.get("eurovoc_concepts", [])
                        }
                        language_docs[lang].append((token_count, doc_info))
                        found_in_config += 1

                # Progress update
                langs_with_docs = sum(1 for lang in EU_LANGUAGES if len(language_docs[lang]) > 0)
                print(f"\nConfig {config}: scanned {scanned_in_config}, found {found_in_config} long docs. "
                      f"Languages with docs: {langs_with_docs}/24")

                # Print current counts per language
                if found_in_config > 0:
                    for lang in EU_LANGUAGES:
                        if len(language_docs[lang]) > 0:
                            print(f"  {lang}: {len(language_docs[lang])} docs")

            except Exception as e:
                print(f"\nWarning: Skipping config {config}: {e}")
                continue

        print(f"\n✓ Processed {configs_processed} configs")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
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
