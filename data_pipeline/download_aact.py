"""Download AACT database dumps from clinicaltrials.org."""

import argparse
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import requests
from loguru import logger


def get_target_date(date_str: str = None) -> str:
    """
    Get the target date for downloading AACT dump.

    Args:
        date_str: Optional date string in YYYY-MM-DD format. If None, uses yesterday's date.

    Returns:
        Date string in YYYYMMDD format
    """
    if date_str:
        target_date = datetime.strptime(date_str, "%Y-%m-%d")
    else:
        # Get yesterday's date (AACT dumps are published for the previous day)
        target_date = datetime.now() - timedelta(days=1)

    return target_date.strftime("%Y%m%d")


def construct_download_url(date_str: str) -> str:
    """
    Construct the download URL for AACT dump.

    Args:
        date_str: Date string in YYYYMMDD format

    Returns:
        Full download URL
    """
    base_url = "https://aact.ctti-clinicaltrials.org/static/static_db_copies/daily"
    filename = f"{date_str}_clinical_trials.zip"
    return f"{base_url}/{filename}"


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
    """
    Download file from URL with progress tracking.

    Args:
        url: URL to download from
        output_path: Path to save the downloaded file
        chunk_size: Size of chunks to download at a time

    Returns:
        True if download successful, False otherwise
    """
    try:
        logger.info(f"Downloading from: {url}")
        logger.info("Connecting to server...")

        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Stream download with longer timeout for initial connection
        logger.info("Sending request (timeout: 300s)...")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        logger.info("✓ Connection established")

        # Get file size if available
        total_size = int(response.headers.get("content-length", 0))
        if total_size:
            logger.info(f"File size: {total_size / (1024**2):.1f}MB")
        else:
            logger.info("File size: Unknown")

        logger.info("Starting download...")

        # Download with progress
        downloaded = 0
        last_log = 0
        log_interval = 10 * 1024 * 1024  # Log every 10MB

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Log progress every 10MB
                    if downloaded - last_log >= log_interval:
                        if total_size:
                            progress = (downloaded / total_size) * 100
                            logger.info(
                                f"Progress: {downloaded / (1024**2):.1f}MB / {total_size / (1024**2):.1f}MB ({progress:.1f}%)"
                            )
                        else:
                            logger.info(f"Downloaded: {downloaded / (1024**2):.1f}MB")
                        last_log = downloaded

        logger.success(f"✓ Download complete: {output_path}")
        logger.info(f"Total downloaded: {output_path.stat().st_size / (1024**2):.1f}MB")
        return True

    except requests.exceptions.Timeout as e:
        logger.error(f"Download timed out after 300 seconds: {e}")
        logger.error("The AACT server might be slow or unavailable. Try again later or use --skip-download")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during download: {e}")
        return False


def extract_zip(zip_path: Path, output_dir: Path) -> Path | None:
    """
    Extract zip file and return path to .dmp file.

    Args:
        zip_path: Path to zip file
        output_dir: Directory to extract files to

    Returns:
        Path to extracted .dmp file, or None if extraction failed
    """
    try:
        logger.info(f"Extracting {zip_path}...")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # List contents
            file_list = zip_ref.namelist()
            logger.info(f"Archive contains {len(file_list)} file(s)")

            # Extract all
            zip_ref.extractall(output_dir)

        # Find .dmp file
        dmp_files = list(output_dir.glob("*.dmp"))

        if not dmp_files:
            logger.error("No .dmp file found in archive")
            return None

        if len(dmp_files) > 1:
            logger.warning(f"Multiple .dmp files found, using first one: {dmp_files[0]}")

        dmp_path = dmp_files[0]
        logger.info(f"Extracted dump file: {dmp_path}")
        logger.info(f"Dump file size: {dmp_path.stat().st_size / (1024**2):.1f}MB")

        return dmp_path

    except zipfile.BadZipFile as e:
        logger.error(f"Invalid zip file: {e}")
        return None
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return None


def cleanup_old_versions(base_dir: Path, keep_count: int = 3):
    """
    Remove old version directories, keeping only the latest N versions.

    Args:
        base_dir: Base downloads directory
        keep_count: Number of latest versions to keep (default: 3)
    """
    try:
        # Find all version directories (format: YYYYMMDD)
        version_dirs = []
        for item in base_dir.iterdir():
            if item.is_dir() and item.name.isdigit() and len(item.name) == 8:
                version_dirs.append(item)

        if len(version_dirs) <= keep_count:
            logger.info(f"Found {len(version_dirs)} version(s), no cleanup needed (keeping {keep_count})")
            return

        # Sort by name (date) in descending order
        version_dirs.sort(key=lambda x: x.name, reverse=True)

        # Keep the latest N, delete the rest
        versions_to_keep = version_dirs[:keep_count]
        versions_to_delete = version_dirs[keep_count:]

        logger.info(f"Keeping {keep_count} latest version(s): {[v.name for v in versions_to_keep]}")
        logger.info(f"Deleting {len(versions_to_delete)} old version(s): {[v.name for v in versions_to_delete]}")

        for version_dir in versions_to_delete:
            try:
                # Delete all files in the directory
                for file in version_dir.iterdir():
                    file.unlink()
                    logger.info(f"  Deleted file: {file.name}")

                # Delete the directory
                version_dir.rmdir()
                logger.success(f"  ✓ Deleted version directory: {version_dir.name}")
            except Exception as e:
                logger.error(f"  Failed to delete {version_dir}: {e}")

    except Exception as e:
        logger.error(f"Cleanup failed: {e}")


def main():
    """Main function to download and extract AACT dump."""
    parser = argparse.ArgumentParser(description="Download AACT database dump")
    parser.add_argument("--date", type=str, help="Target date in YYYY-MM-DD format (default: yesterday)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/downloads",
        help="Base output directory for downloads (default: ./data/downloads)",
    )
    parser.add_argument("--keep-zip", action="store_true", help="Keep zip file after extraction")
    parser.add_argument("--keep-versions", type=int, default=3, help="Number of latest versions to keep (default: 3)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be downloaded without actually downloading"
    )

    args = parser.parse_args()

    # Get target date
    date_str = get_target_date(args.date)
    logger.info(f"Target date: {date_str}")

    # Construct URL
    url = construct_download_url(date_str)

    if args.dry_run:
        logger.info(f"[DRY RUN] Would download from: {url}")
        logger.info(f"[DRY RUN] Would save to: {args.output_dir}/{date_str}/")
        return

    # Setup paths - organize by version
    base_dir = Path(args.output_dir)
    version_dir = base_dir / date_str
    version_dir.mkdir(parents=True, exist_ok=True)

    zip_path = version_dir / f"{date_str}_clinical_trials.zip"

    logger.info(f"Version directory: {version_dir}")

    # Check if already downloaded
    if zip_path.exists():
        logger.warning(f"Zip file already exists: {zip_path}")
        user_input = input("Do you want to re-download? (y/n): ")
        if user_input.lower() != "y":
            logger.info("Skipping download")
            # Still extract if .dmp doesn't exist
            dmp_files = list(version_dir.glob("*.dmp"))
            if not dmp_files:
                logger.info("No .dmp file found, extracting from existing zip...")
                dmp_path = extract_zip(zip_path, version_dir)
                if dmp_path:
                    logger.success("✓ Successfully extracted dump file")
                    logger.info(f"Dump file path: {dmp_path}")
            else:
                logger.info(f"Using existing dump file: {dmp_files[0]}")

            # Cleanup old versions before exiting
            cleanup_old_versions(base_dir, args.keep_versions)
            return
        else:
            zip_path.unlink()

    # Download file
    if not zip_path.exists():
        success = download_file(url, zip_path)
        if not success:
            logger.error("Download failed, exiting")
            return

    # Extract zip
    dmp_path = extract_zip(zip_path, version_dir)

    if dmp_path:
        logger.success("✓ Successfully downloaded and extracted dump file")
        logger.info(f"Dump file path: {dmp_path}")

        # Cleanup zip file if requested
        if not args.keep_zip:
            logger.info(f"Removing zip file: {zip_path}")
            zip_path.unlink()

        # Cleanup old versions
        logger.info("")
        logger.info("=" * 60)
        logger.info("Cleaning up old versions...")
        logger.info("=" * 60)
        cleanup_old_versions(base_dir, args.keep_versions)

    else:
        logger.error("Failed to extract dump file")


if __name__ == "__main__":
    main()
