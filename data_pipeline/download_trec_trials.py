"""Download the TREC 2023 ClinicalTrials dataset ZIP files."""

import argparse
from pathlib import Path
import requests
from loguru import logger

DEFAULT_URLS = [
    "https://www.trec-cds.org/2023_data/ClinicalTrials.2023-05-08.trials0.zip",
    "https://www.trec-cds.org/2023_data/ClinicalTrials.2023-05-08.trials1.zip",
    "https://www.trec-cds.org/2023_data/ClinicalTrials.2023-05-08.trials2.zip",
    "https://www.trec-cds.org/2023_data/ClinicalTrials.2023-05-08.trials3.zip",
    "https://www.trec-cds.org/2023_data/ClinicalTrials.2023-05-08.trials4.zip",
    "https://www.trec-cds.org/2023_data/ClinicalTrials.2023-05-08.trials5.zip",
]


def download_file(url: str, output_dir: Path, timeout: int = 300, overwrite: bool = False) -> Path | None:
    """Download a file from a URL to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    output_path = output_dir / filename

    if output_path.exists() and not overwrite:
        logger.info(f"Skipping existing file: {output_path}")
        return output_path

    try:
        logger.info(f"Downloading {url}")
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-US,en;q=0.9",
        }
        response = requests.get(url, stream=True, timeout=timeout, headers=headers)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 8192
        last_log = 0
        log_interval = 10 * 1024 * 1024

        with output_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded - last_log >= log_interval:
                    last_log = downloaded
                    if total_size:
                        pct = downloaded / total_size * 100
                        logger.info(
                            f"Downloaded {downloaded / (1024**2):.1f}MB / {total_size / (1024**2):.1f}MB ({pct:.1f}%)"
                        )
                    else:
                        logger.info(f"Downloaded {downloaded / (1024**2):.1f}MB")

        size_mb = output_path.stat().st_size / (1024**2)
        logger.success(f"Saved {output_path} ({size_mb:.1f}MB)")
        return output_path

    except requests.exceptions.RequestException as err:
        logger.error(f"Failed to download {url}: {err}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Download TREC 2023 ClinicalTrials ZIP files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data/trec_trials"),
        help="Directory where TREC ZIP files will be saved",
    )
    parser.add_argument(
        "--urls",
        nargs="+",
        default=DEFAULT_URLS,
        help="List of files to download. Defaults to the six TREC ClinicalTrials zip files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files instead of skipping them",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Request timeout in seconds for each download",
    )
    args = parser.parse_args()

    logger.info("Starting TREC ClinicalTrials download")
    failed_urls = []

    for url in args.urls:
        result = download_file(url, args.output_dir, timeout=args.timeout, overwrite=args.overwrite)
        if result is None:
            failed_urls.append(url)

    if failed_urls:
        logger.error("Download completed with failures")
        for url in failed_urls:
            logger.error(f"Failed: {url}")
        return 1

    logger.success("✓ All TREC ClinicalTrials files downloaded successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
