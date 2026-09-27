# Dataset setup and attribution

Dataset: NSL-KDD, developed by the University of New Brunswick research team.
Original description: https://www.unb.ca/cic/datasets/nsl.html
Paper: Tavallaee, Bagheri, Lu and Ghorbani (2009), *A Detailed Analysis of the KDD CUP 99 Data Set*, DOI 10.1109/CISDA.2009.5356528.
Feature definitions: https://kdd.ics.uci.edu/databases/kddcup99/task.html

The original UNB page currently says downloads are no longer available. This project uses the public mirror https://github.com/jmnwong/NSL-KDD-Dataset pinned to commit 9d544d0eb9b87d7e2f43ff65733bdb644631d12f. The mirror is a distribution source, not the dataset author. Retrieved 27 September 2026. No ownership of the dataset is claimed.

- `KDDTrain+_20Percent.txt`: 25,192 training rows.
- `KDDTest+.txt`: 22,544 test rows.
- No header: 41 features, attack name, difficulty (43 columns).
- `manifest.json` records URLs, file sizes and SHA-256 hashes.
- `demo_features.csv` contains ten test records with only the 41 feature columns.

The submission ZIP contains the two raw files for offline use. The public repository provides a downloader instead of republishing the full dataset. From the project root run `python src/download_data.py`, or manually save those two files into `data/`. The downloader rejects a hash mismatch.

Important: one duplicate training feature record was removed. Another 117 training feature records matching test features were removed. The benchmark test set is kept intact, including 57 repeated feature rows. Therefore 25,074 rows are used to fit the model. This is feature-level hygiene; no test labels guide fitting or parameter selection. Test metrics are row-weighted and not fully independent because repeated test features remain.
