"""Main AACT data pipeline orchestrator.

This script orchestrates the complete pipeline:
1. Download AACT dump for previous day
2. Import to PostgreSQL
3. Index to Elasticsearch
4. Cleanup old files
"""

import argparse
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import yaml
from loguru import logger


def load_config(config_path: str = "data_pipeline/config.yaml") -> dict:
    """Load pipeline configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)


def setup_logging(log_dir: str):
    """Setup logging configuration."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Add file logger with rotation
    log_file = log_path / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(
        log_file,
        rotation="100 MB",
        retention="30 days",
        level="INFO"
    )
    logger.info(f"Logging to: {log_file}")


def run_download(date_str: str = None, output_dir: str = "./data/downloads", keep_zip: bool = False) -> Path | None:
    """
    Run download step.
    
    Returns:
        Path to extracted .dmp file, or None if failed
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Downloading AACT dump")
    logger.info("=" * 60)
    
    cmd = [
        "python", "data_pipeline/download_aact.py",
        "--output-dir", output_dir
    ]
    
    if date_str:
        cmd.extend(["--date", date_str])
    
    if keep_zip:
        cmd.append("--keep-zip")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(result.stdout)
        
        # Find the .dmp file
        output_path = Path(output_dir)
        dmp_files = list(output_path.glob("*.dmp"))
        
        if dmp_files:
            # Get the most recent .dmp file
            dmp_file = max(dmp_files, key=lambda p: p.stat().st_mtime)
            logger.success(f"✓ Download complete: {dmp_file}")
            return dmp_file
        else:
            logger.error("No .dmp file found after download")
            return None
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Download failed: {e}")
        logger.error(e.stderr)
        return None


def run_import(dmp_file: Path) -> bool:
    """
    Run PostgreSQL import step.
    
    Returns:
        True if import successful, False otherwise
    """
    logger.info("=" * 60)
    logger.info("STEP 2: Importing to PostgreSQL")
    logger.info("=" * 60)
    
    cmd = ["bash", "data_pipeline/import_aact_data.sh", str(dmp_file)]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(result.stdout)
        logger.success("✓ PostgreSQL import complete")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"PostgreSQL import failed: {e}")
        logger.error(e.stderr)
        return False




def run_materialized_view_refresh() -> bool:
    """
    Create or refresh materialized view for optimized Elasticsearch indexing.
    
    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 60)
    logger.info("STEP 2.5: Creating/Refreshing Materialized View")
    logger.info("=" * 60)
    
    # Try to refresh first (for incremental updates)
    cmd_refresh = ["python", "data_pipeline/create_materialized_view.py", "--refresh"]
    
    try:
        result = subprocess.run(cmd_refresh, check=True, capture_output=True, text=True)
        logger.info(result.stdout)
        logger.success("✓ Materialized view refreshed")
        return True
    except subprocess.CalledProcessError:
        # If refresh fails, try to create (first time)
        logger.info("Materialized view doesn't exist, creating new one...")
        cmd_create = ["python", "data_pipeline/create_materialized_view.py"]
        
        try:
            result = subprocess.run(cmd_create, check=True, capture_output=True, text=True)
            logger.info(result.stdout)
            logger.success("✓ Materialized view created")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create materialized view: {e}")
            logger.error(e.stderr)
            return False


def run_indexing() -> bool:
    """
    Run Elasticsearch indexing step.
    
    Returns:
        True if indexing successful, False otherwise
    """
    logger.info("=" * 60)
    logger.info("STEP 3: Indexing to Elasticsearch")
    logger.info("=" * 60)
    
    # Use optimized script that queries from materialized view
    cmd = ["python", "data_pipeline/sql2es_optimized.py"]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(result.stdout)
        logger.success("✓ Elasticsearch indexing complete")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Elasticsearch indexing failed: {e}")
        logger.error(e.stderr)
        return False



def cleanup_old_files(download_dir: str, retention_days: int):
    """
    Cleanup old dump files.
    
    Args:
        download_dir: Directory containing dump files
        retention_days: Number of days to retain files
    """
    logger.info("=" * 60)
    logger.info("STEP 4: Cleanup old files")
    logger.info("=" * 60)
    
    download_path = Path(download_dir)
    if not download_path.exists():
        logger.warning(f"Download directory does not exist: {download_dir}")
        return
    
    cutoff_date = datetime.now() - timedelta(days=retention_days)
    
    # Find old files
    old_files = []
    for file in download_path.glob("*"):
        if file.is_file():
            file_mtime = datetime.fromtimestamp(file.stat().st_mtime)
            if file_mtime < cutoff_date:
                old_files.append(file)
    
    if not old_files:
        logger.info(f"No files older than {retention_days} days found")
        return
    
    # Delete old files
    for file in old_files:
        try:
            file_age = (datetime.now() - datetime.fromtimestamp(file.stat().st_mtime)).days
            logger.info(f"Deleting {file.name} (age: {file_age} days)")
            file.unlink()
        except Exception as e:
            logger.error(f"Failed to delete {file}: {e}")
    
    logger.success(f"✓ Cleanup complete: removed {len(old_files)} old file(s)")


def main():
    """Main pipeline orchestration."""
    parser = argparse.ArgumentParser(description="AACT Data Pipeline Orchestrator")
    parser.add_argument(
        "--date",
        type=str,
        help="Target date in YYYY-MM-DD format (default: yesterday)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="data_pipeline/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step (use existing dump file)"
    )
    parser.add_argument(
        "--skip-import",
        action="store_true",
        help="Skip PostgreSQL import step"
    )
    parser.add_argument(
        "--skip-indexing",
        action="store_true",
        help="Skip Elasticsearch indexing step"
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Skip cleanup step"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(config['pipeline']['log_dir'])
    
    logger.info("🚀 Starting AACT Data Pipeline")
    logger.info(f"Pipeline started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = datetime.now()
    dmp_file = None
    
    try:
        # Step 1: Download
        if not args.skip_download:
            dmp_file = run_download(
                date_str=args.date,
                output_dir=config['aact']['download_dir'],
                keep_zip=False
            )
            
            if not dmp_file:
                logger.error("❌ Pipeline failed at download step")
                sys.exit(1)
        else:
            logger.info("Skipping download step")
            # Find most recent .dmp file
            download_path = Path(config['aact']['download_dir'])
            dmp_files = list(download_path.glob("*.dmp"))
            if dmp_files:
                dmp_file = max(dmp_files, key=lambda p: p.stat().st_mtime)
                logger.info(f"Using existing dump file: {dmp_file}")
            else:
                logger.error("No .dmp file found in download directory")
                sys.exit(1)
        
        # Step 2: Import to PostgreSQL
        if not args.skip_import:
            if not run_import(dmp_file):
                logger.error("❌ Pipeline failed at import step")
                sys.exit(1)
        else:
            logger.info("Skipping import step")
        
        # Step 2.5: Create/Refresh Materialized View
        if not args.skip_indexing:  # Only needed if we're going to index
            if not run_materialized_view_refresh():
                logger.error("❌ Pipeline failed at materialized view refresh step")
                sys.exit(1)
        
        # Step 3: Index to Elasticsearch
        if not args.skip_indexing:
            if not run_indexing():
                logger.error("❌ Pipeline failed at indexing step")
                sys.exit(1)
        else:
            logger.info("Skipping indexing step")
        
        # Step 4: Cleanup old files
        if not args.skip_cleanup:
            cleanup_old_files(
                config['aact']['download_dir'],
                config['aact']['retention_days']
            )
        else:
            logger.info("Skipping cleanup step")
        
        # Success!
        duration = (datetime.now() - start_time).total_seconds()
        logger.success("=" * 60)
        logger.success("✅ Pipeline completed successfully!")
        logger.success(f"Total duration: {duration / 60:.1f} minutes")
        logger.success("=" * 60)
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed with unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
