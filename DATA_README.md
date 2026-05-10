# Dataset

This folder is the destination for the LendingClub historical loan dataset used by the pilot study and all three solutions.

The dataset is too large for direct GitHub hosting (approximately 1.1GB raw CSV) and must be downloaded separately.

## Download instructions

1. Visit the Kaggle dataset page:

   https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1

2. If you do not have a Kaggle account, create one (free). Click **Download** on the dataset page.

3. The download produces a ZIP archive. Extract it.

4. Place the extracted CSV files (or the primary `Loan_status_2007-2020Q3.gzip` / equivalent) directly in this `data/` folder.

5. Verify by running from the repository root:

   ```bash
   cd code
   python pilot_study.py
   ```

   The script's first few lines load the dataset and report the row count. If the dataset is in place correctly, you will see approximately 2.93 million loans loaded.

## Citation

LendingClub Corporation (2020) _Loan Statistics 2007-2020Q3_. Compiled and distributed by user `ethon0426` on Kaggle. Original source: LendingClub Corporation public loan statistics releases (no longer hosted by LendingClub directly following the company's 2020 retail-lending exit).

## Why the dataset isn't included in the repo

GitHub's hard file-size limit is 100 MB. The LendingClub dataset is approximately 1.1 GB. Direct inclusion would require Git LFS, which adds setup complexity and bandwidth costs. Linking to the canonical Kaggle source is academic standard and ensures markers reproduce the analysis against the same data the dissertation used.

## Reproducibility note

The pilot study was developed against the version of the dataset available on Kaggle as of February 2026. The dataset is static (LendingClub no longer releases public loan statistics), so all future downloads from this Kaggle source will produce identical results.
