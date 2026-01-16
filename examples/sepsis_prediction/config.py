from pathlib import Path


class PATHS:
    """Paths for different data sources"""

    # Base paths
    NEMSIS_BASE = Path("/mnt/dataset/nemsis")

    # Multi-year paths
    NEMSIS_COMBINED = Path("/workspaces/prehosp-vitals-gap/data/nemsis_combined")

    # Other datasets
    MIMIC_IV_ED = Path("/mnt/dataset/mimic-iv-ed/mimic-iv-ed-2.2/ed")

    @staticmethod
    def get_nemsis_raw_path(year):
        """Get the path to raw NEMSIS data for a specific year"""
        return Path(f"{PATHS.NEMSIS_BASE}/{year}/extracted_sas")

    @staticmethod
    def get_nemsis_path(year):
        """Get the path to NEMSIS data for a specific year"""
        return Path(f"/workspaces/prehosp-vitals-gap/data/nemsis{year}")


class VARIABLES:
    NEMSIS_YEAR_STR = "2024"  # Can be a single year or multiple years like "2021_2022" or "_combined", only used for get_nemsis_path
