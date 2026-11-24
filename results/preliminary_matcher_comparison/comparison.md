# Preliminary SP2 matcher comparison | processed TLC scenarios

Wall times come from unprofiled runs. Memory peaks come from separate replays.

| Scenario | Fleet | Sample % | Seed | Cutoff (cells) | Algorithm | Successful assignments | Completed trips | Expired requests | Request to pickup (s) | Assignment to pickup (s) | Driver utilization (%) | Matching wall time (ms) | Peak traced memory (KiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| midday | 20 | 1.0 | 42 | 18 | Greedy | 205 | 197 | 0 | 89.40 | 61.79 | 11.69 | 1.07 | 0.66 |
| midday | 20 | 1.0 | 42 | 18 | Centralized batch | 205 | 197 | 0 | 88.20 | 60.60 | 11.69 | 2.33 | 1.90 |
| midday | 40 | 1.0 | 42 | 18 | Greedy | 205 | 198 | 0 | 68.62 | 40.99 | 5.85 | 2.03 | 0.60 |
| midday | 40 | 1.0 | 42 | 18 | Centralized batch | 205 | 198 | 0 | 69.51 | 41.88 | 5.85 | 4.85 | 3.30 |
| midday | 20 | 2.0 | 42 | 18 | Greedy | 411 | 404 | 0 | 99.05 | 72.09 | 24.69 | 1.56 | 0.80 |
| midday | 20 | 2.0 | 42 | 18 | Centralized batch | 411 | 404 | 0 | 94.04 | 67.08 | 24.72 | 3.17 | 2.00 |
| midday | 40 | 2.0 | 42 | 18 | Greedy | 411 | 404 | 0 | 77.23 | 50.29 | 12.39 | 3.56 | 0.94 |
| midday | 40 | 2.0 | 42 | 18 | Centralized batch | 411 | 404 | 0 | 77.67 | 50.74 | 12.39 | 6.25 | 4.16 |
| midday | 20 | 5.0 | 42 | 18 | Greedy | 640 | 625 | 332 | 657.18 | 144.77 | 38.44 | 1.95 | 3.34 |
| midday | 20 | 5.0 | 42 | 18 | Centralized batch | 943 | 928 | 62 | 207.83 | 42.57 | 54.25 | 8.17 | 6.94 |
| midday | 40 | 5.0 | 42 | 18 | Greedy | 1027 | 1011 | 0 | 82.02 | 52.49 | 30.29 | 5.67 | 1.29 |
| midday | 40 | 5.0 | 42 | 18 | Centralized batch | 1027 | 1011 | 0 | 77.06 | 47.55 | 30.31 | 9.76 | 6.08 |
| peak | 20 | 1.0 | 42 | 18 | Greedy | 371 | 368 | 0 | 102.46 | 71.97 | 23.06 | 1.81 | 0.80 |
| peak | 20 | 1.0 | 42 | 18 | Centralized batch | 371 | 368 | 0 | 99.55 | 68.89 | 23.06 | 5.49 | 2.57 |
| peak | 40 | 1.0 | 42 | 18 | Greedy | 371 | 369 | 0 | 75.62 | 46.09 | 11.54 | 4.10 | 0.95 |
| peak | 40 | 1.0 | 42 | 18 | Centralized batch | 371 | 369 | 0 | 75.29 | 45.77 | 11.54 | 10.96 | 4.95 |
| peak | 20 | 2.0 | 42 | 18 | Greedy | 627 | 610 | 93 | 609.07 | 146.17 | 38.64 | 1.87 | 2.85 |
| peak | 20 | 2.0 | 42 | 18 | Centralized batch | 737 | 721 | 6 | 155.89 | 76.36 | 45.25 | 5.16 | 4.20 |
| peak | 40 | 2.0 | 42 | 18 | Greedy | 743 | 733 | 0 | 71.74 | 42.15 | 23.06 | 7.91 | 1.27 |
| peak | 40 | 2.0 | 42 | 18 | Centralized batch | 743 | 733 | 0 | 70.37 | 40.78 | 23.06 | 15.22 | 6.58 |
| peak | 20 | 5.0 | 42 | 18 | Greedy | 612 | 595 | 1171 | 712.78 | 157.23 | 37.89 | 3.86 | 6.17 |
| peak | 20 | 5.0 | 42 | 18 | Centralized batch | 1059 | 1041 | 735 | 372.26 | 13.96 | 63.47 | 45.76 | 18.30 |
| peak | 40 | 5.0 | 42 | 18 | Greedy | 1411 | 1375 | 381 | 633.00 | 114.24 | 42.15 | 5.11 | 6.83 |
| peak | 40 | 5.0 | 42 | 18 | Centralized batch | 1822 | 1789 | 30 | 159.32 | 45.02 | 54.44 | 28.50 | 18.42 |
