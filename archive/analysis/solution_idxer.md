# DSA Top-K Indexer — Workload Analysis

## Per-Workload Detail — batch_size and seq_lens

`block_table shape` = `[batch, max_num_pages]`. `seq_lens` = KV cache size (in tokens) per request; trailing 0s in `block_table` are padding for `max_num_pages` alignment, never read.

| # | UUID | block table shape | Ref ms | max_len | seq_lens (tokens/req) | n>2048 |
|---:|---|---|---:|---:|---|---:|
|   1 | 30cecff1 | [1, 1] | 0.975 |    2 | [2] | 0 |
|   2 | cd594d26 | [1, 2] | 0.969 |   65 | [65] | 0 |
|   3 | 44ddaa65 | [1, 3] | 0.968 |  129 | [129] | 0 |
|   4 | b2098949 | [2, 2] | 1.122 |   92 | [92, 48] | 0 |
|   5 | 4667f9ad | [2, 1] | 1.119 |   52 | [33, 52] | 0 |
|   6 | 8f2fde6c | [2, 6] | 1.142 |  337 | [6, 337] | 0 |
|   7 | 02fa7f90 | [2, 5] | 1.142 |  288 | [288, 4] | 0 |
|   8 | 545f8a85 | [2, 3] | 1.136 |  129 | [129, 85] | 0 |
|   9 | e49574dd | [2, 7] | 1.133 |  385 | [54, 385] | 0 |
|  10 | 0ebafac4 | [2, 4] | 1.126 |  193 | [193, 149] | 0 |
|  11 | 82a8a885 | [2, 8] | 1.134 |  449 | [118, 449] | 0 |
|  12 | 4279d75e | [4, 17] | 1.460 | 1037 | [33, 52, 72, 1037] | 0 |
|  13 | 83cb81c5 | [3, 2] | 1.299 |   73 | [34, 53, 73] | 0 |
|  14 | d8a73470 | [4, 34] | 1.476 | 2161 | [18, 11, 2161, 20] | 1 |
|  15 | 16feeab1 | [4, 43] | 1.461 | 2693 | [63, 9, 2693, 212] | 1 |
|  16 | 9754a4e7 | [4, 16] | 1.469 | 1002 | [18, 19, 1002, 31] | 0 |
|  17 | 05775386 | [4, 18] | 1.483 | 1105 | [6, 337, 1105, 9] | 0 |
|  18 | 46f236c0 | [4, 1] | 1.444 |   32 | [19, 20, 32, 1] | 0 |
|  19 | d54c1568 | [4, 2] | 1.443 |   74 | [35, 54, 74, 1] | 0 |
|  20 | 17ced9b8 | [4, 30] | 1.472 | 1887 | [17, 13, 1887, 16] | 0 |
|  21 | cd3434ac | [4, 35] | 1.458 | 2177 | [34, 27, 2177, 36] | 1 |
|  22 | 101a39ac | [4, 31] | 1.460 | 1921 | [51, 47, 1921, 50] | 0 |
|  23 | 55be3dc3 | [4, 3] | 1.447 |  129 | [90, 109, 129, 1] | 0 |
|  24 | ef12ac76 | [4, 44] | 1.470 | 2753 | [123, 69, 2753, 272] | 1 |
|  25 | 4c7705ad | [4, 36] | 1.601 | 2241 | [98, 91, 2241, 100] | 1 |
|  26 | ef0d0deb | [4, 32] | 1.664 | 1985 | [115, 111, 1985, 114] | 0 |
|  27 | 28a9fa48 | [4, 19] | 1.663 | 1153 | [201, 157, 1153, 123] | 0 |
|  28 | 899c2d2f | [4, 4] | 1.465 |  193 | [154, 173, 193, 1] | 0 |
|  29 | 03fc111f | [4, 45] | 1.496 | 2817 | [187, 133, 2817, 336] | 1 |
|  30 | b017f77a | [8, 34] | 2.357 | 2161 | [18, 11, 2161, 20, 25, 45, 135, 326] | 1 |
|  31 | 82bd3e70 | [8, 16] | 2.095 | 1002 | [18, 19, 1002, 31, 11, 316, 24, 2] | 0 |
|  32 | f897f64e | [8, 17] | 2.106 | 1044 | [92, 48, 1044, 14, 411, 30, 16, 8] | 0 |
|  33 | df80c00b | [7, 18] | 1.968 | 1089 | [33, 52, 72, 17, 18, 401, 1089] | 0 |
|  34 | 99920dc5 | [7, 34] | 1.940 | 2162 | [19, 12, 2162, 21, 26, 46, 136] | 1 |
|  35 | 9410ad1e | [6, 1] | 1.778 |   32 | [19, 20, 32, 12, 25, 3] | 0 |
|  36 | 3e91afa0 | [8, 43] | 2.124 | 2693 | [63, 9, 2693, 212, 11, 25, 6, 50] | 1 |
|  37 | 9a2bb7f8 | [6, 26] | 1.803 | 1639 | [6, 9, 9, 14, 1639, 71] | 0 |
|  38 | 2774963f | [8, 38] | 2.101 | 2405 | [288, 4, 1884, 21, 136, 2405, 42, 335] | 1 |
|  39 | e64a4ebc | [8, 1] | 2.075 |   34 | [21, 22, 34, 14, 27, 5, 1, 1] | 0 |
|  40 | a4cdaee6 | [8, 30] | 2.081 | 1885 | [289, 5, 1885, 22, 137, 43, 336, 1] | 0 |
|  41 | 6caf09cf | [8, 18] | 2.102 | 1092 | [36, 55, 75, 20, 21, 404, 1092, 1] | 0 |
|  42 | 7752dda1 | [8, 26] | 2.078 | 1641 | [8, 11, 11, 16, 1641, 73, 1, 1] | 0 |
|  43 | 67216408 | [8, 32] | 2.095 | 1986 | [17, 13, 1887, 16, 180, 1986, 413, 1] | 0 |
|  44 | 8638fe06 | [8, 35] | 2.060 | 2177 | [34, 27, 2177, 36, 41, 61, 151, 1] | 1 |
|  45 | d04ea89f | [8, 27] | 2.083 | 1665 | [32, 35, 35, 40, 1665, 97, 1, 1] | 0 |
|  46 | 13dad24c | [8, 2] | 2.044 |   65 | [52, 53, 65, 45, 58, 36, 1, 1] | 0 |
|  47 | 1152c61f | [8, 31] | 2.065 | 1921 | [51, 47, 1921, 50, 214, 447, 1, 1] | 0 |
|  48 | 77279062 | [8, 44] | 2.073 | 2753 | [123, 69, 2753, 272, 71, 85, 66, 110] | 1 |
|  49 | 9c313fc4 | [8, 19] | 2.086 | 1153 | [97, 116, 136, 81, 82, 465, 1153, 1] | 0 |
|  50 | 8f3fe9ff | [8, 36] | 2.076 | 2241 | [98, 91, 2241, 100, 105, 125, 215, 1] | 1 |
|  51 | 06ec358c | [8, 28] | 2.066 | 1729 | [96, 99, 99, 104, 1729, 161, 1, 1] | 0 |
|  52 | 8f1a5846 | [8, 3] | 2.283 |  129 | [117, 129, 109, 100, 1, 1, 1, 1] | 0 |
|  53 | 3240d5fa | [8, 8] | 2.048 |  506 | [138, 157, 177, 122, 123, 506, 1, 1] | 0 |
|  54 | c729310b | [8, 9] | 2.129 |  513 | [145, 164, 184, 130, 513, 1, 1, 1] | 0 |
|  55 | 81a953ea | [8, 45] | 2.054 | 2817 | [187, 133, 2817, 336, 135, 149, 130, 174] | 1 |
|  56 | 09bb020f | [6, 28] | 1.768 | 1766 | [133, 136, 136, 141, 1766, 198] | 0 |
|  57 | e515e20a | [7, 32] | 1.928 | 2010 | [414, 130, 2010, 147, 262, 168, 461] | 0 |
|  58 | 10b4eebe | [6, 32] | 1.782 | 2012 | [142, 138, 2012, 141, 305, 538] | 0 |
|  59 | 1ece7fb3 | [15, 18] | 3.182 | 1089 | [33, 52, 72, 17, 18, 401, 1089, 19, 6, 31, 50, 15, 8, 8, 12] | 0 |
|  60 | 9810dadf | [16, 34] | 3.339 | 2161 | [18, 11, 2161, 20, 25, 45, 135, 326, 7, 8, 10, 22, 17, 11, 30, 130] | 1 |
|  61 | 9a95a10e | [15, 34] | 3.142 | 2162 | [19, 12, 2162, 21, 26, 46, 136, 8, 9, 11, 23, 18, 12, 31, 131] | 1 |
|  62 | 696dbfa4 | [16, 43] | 3.353 | 2693 | [63, 9, 2693, 212, 11, 25, 6, 50, 77, 22, 25, 10, 52, 769, 76, 11] | 1 |
|  63 | ee603b53 | [15, 43] | 3.191 | 2694 | [64, 10, 2694, 213, 12, 26, 7, 51, 78, 23, 26, 11, 53, 77, 12] | 1 |
|  64 | 70d53807 | [12, 82] | 2.743 | 5194 | [18, 19, 31, 11, 24, 2, 12, 12, 336, 20, 5194, 78] | 1 |
|  65 | 19e7663d | [16, 38] | 3.327 | 2405 | [288, 4, 1884, 21, 136, 2405, 42, 335, 21, 15, 304, 436, 14, 9, 15, 1231] | 1 |
|  66 | ed3e595b | [16, 33] | 3.380 | 2112 | [92, 48, 1044, 14, 411, 30, 16, 8, 3, 35, 21, 5, 7, 2112, 10, 2015] | 1 |
|  67 | e4ecb462 | [15, 82] | 3.157 | 5195 | [19, 20, 32, 12, 25, 3, 13, 13, 337, 21, 5195, 79, 1, 1, 1] | 1 |
|  68 | f1fc35d4 | [15, 38] | 3.274 | 2406 | [289, 5, 1885, 22, 137, 2406, 43, 336, 22, 16, 305, 437, 15, 10, 16] | 1 |
|  69 | 67c09e9c | [15, 30] | 3.766 | 1887 | [17, 13, 1887, 16, 180, 413, 22, 276, 17, 8, 102, 54, 24, 1, 1] | 0 |
|  70 | 786b5173 | [15, 89] | 3.227 | 5679 | [6, 9, 9, 14, 1639, 71, 32, 7, 75, 25, 9, 148, 5679, 15, 1] | 1 |
|  71 | 08a752fc | [15, 35] | 3.244 | 2177 | [34, 27, 2177, 36, 41, 61, 151, 23, 24, 26, 38, 33, 27, 46, 146] | 1 |
|  72 | cdc0ff86 | [15, 90] | 3.207 | 5697 | [24, 27, 27, 32, 1657, 89, 50, 25, 93, 43, 27, 166, 5697, 33, 1] | 1 |
|  73 | f457feb2 | [15, 39] | 3.151 | 2433 | [316, 32, 1912, 49, 164, 2433, 70, 363, 49, 43, 332, 464, 42, 37, 43] | 1 |
|  74 | e977c163 | [15, 31] | 3.096 | 1921 | [51, 47, 1921, 50, 447, 56, 310, 51, 42, 136, 88, 58, 1, 1, 1] | 0 |
|  75 | 6bdb38e6 | [15, 83] | 3.131 | 5249 | [73, 74, 86, 66, 57, 67, 67, 391, 75, 5249, 133, 1, 1, 1, 1] | 1 |
|  76 | e26d02ef | [15, 44] | 3.179 | 2753 | [123, 69, 2753, 272, 71, 85, 66, 110, 137, 82, 85, 70, 112, 136, 71] | 1 |
|  77 | 97a6d5c2 | [15, 19] | 3.762 | 1153 | [97, 116, 136, 81, 82, 465, 1153, 83, 70, 95, 114, 79, 72, 72, 76] | 0 |
|  78 | 175849a8 | [15, 36] | 3.168 | 2241 | [98, 91, 2241, 100, 105, 125, 215, 87, 88, 102, 97, 91, 110, 210, 1] | 1 |
|  79 | 3eab2c37 | [15, 91] | 3.169 | 5761 | [88, 91, 91, 96, 1721, 153, 114, 89, 157, 107, 91, 230, 5761, 97, 1] | 1 |
|  80 | 60605091 | [15, 40] | 3.127 | 2497 | [380, 96, 1976, 113, 228, 2497, 134, 427, 113, 107, 396, 106, 101, 107, 1] | 1 |
|  81 | bb22d09a | [15, 32] | 3.118 | 1985 | [115, 111, 1985, 114, 511, 120, 374, 115, 200, 152, 122, 1, 1, 1, 1] | 0 |
|  82 | fb1ceff0 | [15, 84] | 3.120 | 5313 | [138, 150, 130, 121, 131, 131, 455, 139, 5313, 197, 1, 1, 1, 1, 1] | 1 |
|  83 | b83c4150 | [15, 45] | 3.148 | 2817 | [187, 133, 2817, 336, 135, 149, 130, 174, 201, 146, 149, 134, 176, 200, 135] | 1 |
|  84 | 752c2ee5 | [15, 1] | 2.947 |    1 | [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] | 0 |
|  85 | e1a185dc | [14, 84] | 3.040 | 5321 | [146, 158, 138, 129, 139, 139, 463, 147, 5321, 205, 1, 1, 1, 1] | 1 |
|  86 | 7f03b670 | [14, 32] | 2.988 | 2013 | [143, 139, 2013, 142, 539, 148, 402, 143, 228, 180, 150, 1, 1, 1] | 0 |
|  87 | d0c00dd5 | [14, 1] | 2.711 |    1 | [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] | 0 |
|  88 | f7f61b05 | [14, 91] | 3.008 | 5805 | [132, 135, 135, 140, 1765, 197, 158, 133, 201, 151, 135, 274, 5805, 141] | 1 |
|  89 | 7f1cd9c2 | [14, 40] | 2.977 | 2532 | [415, 131, 2011, 148, 263, 2532, 169, 462, 148, 142, 431, 141, 136, 142] | 1 |
|  90 | f59fd3e2 | [14, 35] | 3.015 | 2239 | [219, 175, 1171, 141, 143, 135, 130, 162, 148, 132, 134, 2239, 137, 1] | 1 |
|  91 | 5f7e6f22 | [11, 32] | 2.534 | 2014 | [144, 140, 2014, 143, 540, 149, 403, 144, 229, 181, 151] | 0 |
|  92 | bda73497 | [31, 26] | 5.681 | 1623 | [33, 52, 72, 17, 18, 401, 1089, 19, 6, 31, 50, 15, 8, 8, 12, 41, 746, 51, 600, 1623, 388, 181, 65, 32, 9, 9, 42, 28, 20, 16, 14] | 0 |
|  93 | 37098ea3 | [29, 18] | 5.681 | 1090 | [34, 53, 73, 18, 19, 402, 1090, 20, 7, 32, 51, 16, 9, 9, 13, 42, 747, 52, 601, 389, 182, 33, 10, 10, 43, 29, 21, 17, 15] | 0 |
|  94 | a03d722b | [30, 34] | 5.562 | 2161 | [18, 11, 2161, 20, 25, 45, 135, 7, 8, 10, 22, 17, 11, 30, 130, 9, 600, 255, 1033, 58, 2, 21, 954, 7, 14, 1015, 28, 64, 5, 18] | 1 |
|  95 | 34195ade | [27, 82] | 5.195 | 5194 | [18, 19, 31, 11, 24, 2, 12, 12, 336, 20, 5194, 78, 23, 963, 656, 12, 101, 9, 18, 2, 7, 18, 54, 9, 99, 15, 46] | 1 |
|  96 | 03910df4 | [29, 30] | 5.811 | 1887 | [17, 13, 1887, 16, 180, 413, 22, 276, 17, 8, 102, 54, 24, 860, 21, 65, 713, 33, 16, 18, 9, 67, 693, 42, 114, 5, 122, 43, 56] | 0 |
|  97 | a876010b | [29, 89] | 5.477 | 5679 | [6, 9, 9, 14, 1639, 71, 32, 7, 75, 25, 9, 148, 5679, 15, 50, 365, 63, 175, 51, 99, 7, 10, 38, 117, 10, 11, 3, 73, 2] | 1 |
|  98 | 7f20565a | [30, 30] | 5.490 | 1884 | [288, 4, 1884, 21, 136, 42, 335, 21, 15, 304, 436, 14, 9, 15, 13, 1732, 39, 665, 470, 349, 8, 24, 79, 69, 16, 61, 128, 1205, 33, 33] | 0 |
|  99 | a52c09bc | [31, 43] | 5.817 | 2693 | [63, 9, 2693, 212, 11, 25, 6, 50, 77, 22, 25, 10, 52, 76, 11, 30, 1822, 25, 1034, 1811, 7, 13, 23, 1896, 13, 285, 18, 166, 16, 1421, 12] | 1 |
| 100 | bb0f8277 | [30, 89] | 5.588 | 5680 | [7, 10, 10, 15, 1640, 72, 33, 8, 76, 26, 10, 149, 5680, 16, 51, 366, 64, 176, 52, 100, 8, 11, 39, 118, 11, 12, 4, 74, 3, 1] | 1 |
| 101 | 6b4b9d2b | [30, 82] | 5.656 | 5196 | [20, 21, 33, 13, 26, 4, 14, 14, 338, 22, 5196, 80, 25, 965, 658, 14, 103, 11, 20, 4, 9, 20, 56, 11, 101, 17, 48, 1, 1, 1] | 1 |
| 102 | a30b4f8d | [30, 18] | 5.518 | 1094 | [38, 57, 77, 22, 23, 406, 1094, 24, 11, 36, 55, 20, 13, 13, 17, 46, 751, 56, 605, 393, 186, 37, 14, 14, 47, 33, 25, 21, 19, 1] | 0 |
| 103 | 8bdd4f88 | [30, 43] | 5.805 | 2694 | [64, 10, 2694, 213, 12, 26, 7, 51, 78, 23, 26, 11, 53, 77, 12, 31, 1823, 26, 1035, 1812, 8, 14, 24, 1897, 14, 286, 19, 167, 17, 13] | 1 |
| 104 | 30a90fa5 | [30, 33] | 5.623 | 2112 | [92, 48, 1044, 14, 411, 30, 16, 8, 3, 35, 21, 5, 7, 2112, 10, 17, 1758, 11, 1312, 300, 314, 286, 11, 1746, 29, 74, 62, 300, 1439, 96] | 1 |
| 105 | de54c4e6 | [30, 35] | 5.564 | 2177 | [34, 27, 2177, 36, 41, 61, 151, 23, 24, 26, 38, 33, 27, 46, 146, 25, 616, 271, 1049, 74, 18, 37, 970, 23, 30, 1031, 44, 80, 21, 34] | 1 |
| 106 | 22207643 | [30, 90] | 5.480 | 5697 | [24, 27, 27, 32, 1657, 89, 50, 25, 93, 43, 27, 166, 5697, 33, 68, 383, 81, 193, 69, 117, 25, 28, 56, 135, 28, 29, 21, 91, 20, 1] | 1 |
| 107 | 6832006b | [30, 28] | 5.509 | 1778 | [112, 68, 1064, 34, 431, 50, 36, 28, 23, 55, 41, 25, 27, 30, 37, 1778, 31, 1332, 320, 306, 31, 1766, 49, 94, 82, 320, 116, 1, 1, 1] | 0 |
| 108 | 9f252ffa | [30, 31] | 5.492 | 1921 | [51, 47, 1921, 50, 447, 56, 310, 51, 42, 136, 88, 58, 894, 55, 99, 747, 67, 50, 52, 43, 101, 727, 76, 148, 39, 156, 77, 90, 1, 1] | 0 |
| 109 | 4a0e0529 | [30, 29] | 5.607 | 1793 | [127, 83, 1079, 49, 446, 65, 51, 43, 38, 70, 56, 40, 42, 45, 52, 1793, 46, 1347, 335, 321, 46, 1781, 64, 109, 97, 335, 131, 1, 1, 1] | 0 |
| 110 | f362edf4 | [30, 83] | 5.686 | 5249 | [73, 74, 86, 66, 57, 67, 67, 391, 75, 5249, 133, 78, 1018, 711, 67, 156, 64, 73, 57, 62, 73, 109, 64, 154, 70, 101, 1, 1, 1, 1] | 1 |
| 111 | e0488cb7 | [30, 19] | 5.929 | 1153 | [97, 116, 136, 81, 82, 465, 1153, 83, 70, 95, 114, 79, 72, 72, 76, 105, 810, 115, 664, 452, 245, 96, 73, 73, 106, 92, 84, 80, 78, 1] | 0 |
| 112 | ee6946e7 | [30, 44] | 5.576 | 2753 | [123, 69, 2753, 272, 71, 85, 66, 110, 137, 82, 85, 70, 112, 136, 71, 90, 1882, 85, 1094, 1871, 67, 73, 83, 1956, 73, 345, 78, 226, 76, 72] | 1 |
| 113 | 2f3b7321 | [30, 36] | 5.507 | 2241 | [98, 91, 2241, 100, 105, 125, 215, 87, 88, 102, 97, 91, 110, 210, 89, 680, 335, 1113, 138, 82, 101, 1034, 87, 94, 1095, 144, 85, 98, 1, 1] | 1 |
| 114 | 5db1b172 | [30, 91] | 5.580 | 5761 | [88, 91, 91, 96, 1721, 153, 114, 89, 157, 107, 91, 230, 5761, 97, 132, 447, 145, 257, 133, 181, 89, 92, 120, 199, 92, 93, 85, 155, 84, 1] | 1 |
| 115 | 8ba75447 | [30, 32] | 5.522 | 1985 | [115, 111, 1985, 114, 511, 120, 374, 115, 106, 200, 152, 122, 958, 119, 163, 811, 131, 114, 116, 107, 165, 791, 140, 103, 220, 141, 154, 1, 1, 1] | 0 |
| 116 | e667d2ac | [29, 32] | 5.306 | 1993 | [123, 119, 1993, 122, 519, 128, 382, 123, 114, 208, 160, 130, 966, 127, 171, 819, 139, 122, 124, 115, 173, 799, 148, 111, 228, 149, 162, 1, 1] | 0 |
| 117 | 6b10b6da | [29, 91] | 5.465 | 5784 | [111, 114, 114, 119, 1744, 176, 137, 112, 180, 130, 114, 253, 5784, 120, 155, 470, 168, 280, 156, 204, 112, 115, 143, 222, 115, 116, 108, 178, 107] | 1 |
| 118 | 1571c14a | [29, 19] | 5.420 | 1198 | [142, 161, 181, 126, 127, 510, 1198, 128, 115, 140, 159, 124, 117, 117, 121, 150, 855, 160, 709, 497, 290, 141, 151, 137, 129, 125, 123, 1, 1] | 0 |
| 119 | 0c4f5578 | [29, 83] | 5.383 | 5300 | [125, 137, 117, 108, 118, 118, 442, 126, 5300, 184, 129, 1069, 762, 118, 115, 124, 108, 113, 124, 160, 115, 205, 121, 152, 1, 1, 1, 1, 1] | 1 |
| 120 | fc14d852 | [29, 44] | 5.515 | 2798 | [168, 114, 2798, 317, 116, 130, 111, 155, 182, 127, 130, 157, 181, 116, 135, 1927, 130, 1139, 1916, 112, 118, 128, 2001, 118, 390, 123, 271, 121, 117] | 1 |
| 121 | e63194e7 | [29, 36] | 5.390 | 2268 | [125, 118, 2268, 127, 132, 152, 242, 114, 115, 129, 124, 118, 137, 237, 116, 707, 362, 1140, 165, 109, 128, 1061, 114, 121, 1122, 171, 112, 125, 1] | 1 |
| 122 | 27afdcea | [29, 84] | 6.450 | 5313 | [138, 150, 130, 121, 131, 131, 455, 139, 5313, 197, 142, 1082, 775, 131, 128, 137, 121, 126, 137, 173, 128, 218, 134, 165, 1, 1, 1, 1, 1] | 1 |
| 123 | abc9d12c | [29, 1] | 5.289 |    1 | [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] | 0 |
| 124 | 27c3374f | [29, 45] | 5.282 | 2817 | [187, 133, 2817, 336, 135, 149, 130, 174, 201, 146, 149, 176, 200, 135, 154, 1946, 149, 1158, 1935, 131, 137, 147, 2020, 137, 409, 142, 290, 140, 136] | 1 |
| 125 | 6e4e9b37 | [26, 30] | 5.009 | 1884 | [218, 174, 1170, 140, 537, 142, 134, 129, 161, 147, 131, 133, 136, 143, 1884, 137, 1438, 426, 412, 137, 1872, 155, 200, 188, 426, 222] | 0 |
| 126 | 4a616af2 | [27, 32] | 5.668 | 2011 | [415, 131, 2011, 148, 263, 169, 462, 148, 142, 431, 563, 141, 136, 142, 140, 1859, 166, 792, 597, 135, 151, 206, 196, 143, 188, 160, 160] | 0 |
| 127 | 8635db8f | [27, 91] | 5.069 | 5806 | [133, 136, 136, 141, 1766, 198, 159, 134, 202, 152, 136, 275, 5806, 142, 177, 492, 190, 302, 178, 226, 134, 137, 165, 244, 130, 200, 129] | 1 |
| 128 | dba1e960 | [25, 30] | 4.796 | 1885 | [219, 175, 1171, 141, 143, 135, 130, 162, 148, 132, 134, 137, 144, 1885, 138, 1439, 427, 413, 138, 1873, 156, 201, 189, 427, 223] | 0 |

---

## Individual Requests with seq_len > 2048 (Slow-Path GEMM Required)

59 individual requests across 128 workloads require the full score GEMM. Each row = one batch element `b` where `seq_len > 2048`. The kernel receives `block_table[b, :pages]` — a row of page indices into the global `k_index_cache_fp8[11923, 64, 1, 132]` pool.

**Critical: pages are NOT contiguous** — 42/59 have gaps or backwards jumps. Must use block_table indirection.

| WL# | UUID | b | sl | pg | contig | block_table[b, :pages] |
|---:|---|---:|---:|---:|---|---|
|  14 | d8a73470 | 2 | 2161 | 34 | ✓ | [3, 4, 5, ..., 36] |
|  15 | 16feeab1 | 2 | 2693 | 43 | ✓ | [3, 4, 5, ..., 45] |
|  21 | cd3434ac | 2 | 2177 | 35 | ✗ | [3..36, **38**] |
|  24 | ef12ac76 | 2 | 2753 | 44 | ✗ | [3..45, **53**] |
|  25 | 4c7705ad | 2 | 2241 | 36 | ✗ | [3..36, **38, 42**] |
|  29 | 03fc111f | 2 | 2817 | 45 | ✗ | [3..45, **53, 57**] |
|  30 | b017f77a | 2 | 2161 | 34 | ✓ | [3, 4, 5, ..., 36] |
|  34 | 99920dc5 | 2 | 2162 | 34 | ✓ | [3, 4, 5, ..., 36] |
|  36 | 3e91afa0 | 2 | 2693 | 43 | ✓ | [3, 4, 5, ..., 45] |
|  38 | 2774963f | 5 | 2405 | 38 | ✓ | [41, 42, 43, ..., 78] |
|  44 | 8638fe06 | 2 | 2177 | 35 | ✗ | [3..36, **48**] |
|  48 | 77279062 | 2 | 2753 | 44 | ✗ | [3..45, **61**] |
|  50 | 8f3fe9ff | 2 | 2241 | 36 | ✗ | [3..36, **48, 55**] |
|  55 | 81a953ea | 2 | 2817 | 45 | ✗ | [3..45, **61, 69**] |
|  60 | 9810dadf | 2 | 2161 | 34 | ✓ | [3, 4, 5, ..., 36] |
|  61 | 9a95a10e | 2 | 2162 | 34 | ✓ | [3, 4, 5, ..., 36] |
|  62 | 696dbfa4 | 2 | 2693 | 43 | ✓ | [3, 4, 5, ..., 45] |
|  63 | ee603b53 | 2 | 2694 | 43 | ✓ | [3, 4, 5, ..., 45] |
|  64 | 70d53807 | 10 | 5194 | 82 | ✗ | [44..64, **25, 18**, 65..94, **42, 33**, 95..121] |
|  65 | 19e7663d | 5 | 2405 | 38 | ✓ | [41, 42, 43, ..., 78] |
|  66 | ed3e595b | 13 | 2112 | 33 | ✓ | [37, 38, 39, ..., 69] |
|  67 | e4ecb462 | 10 | 5195 | 82 | ✗ | [44..64, **25, 18**, 65..94, **42, 33**, 95..121] |
|  68 | f1fc35d4 | 5 | 2406 | 38 | ✓ | [41, 42, 43, ..., 78] |
|  70 | 786b5173 | 12 | 5679 | 89 | ✗ | [**7**, 65, 66, ..., 152] |
|  71 | 08a752fc | 2 | 2177 | 35 | ✗ | [3..36, **48**] |
|  72 | cdc0ff86 | 12 | 5697 | 90 | ✗ | [**7**, 65..152, **154**] |
|  73 | f457feb2 | 5 | 2433 | 39 | ✗ | [41..78, **125**] |
|  75 | 6bdb38e6 | 9 | 5249 | 83 | ✗ | [44..64, **25, 18**, 65..94, **42, 33**, 95..121, **132**] |
|  76 | e26d02ef | 2 | 2753 | 44 | ✗ | [3..45, **89**] |
|  78 | 175849a8 | 2 | 2241 | 36 | ✗ | [3..36, **48, 68**] |
|  79 | 3eab2c37 | 12 | 5761 | 91 | ✗ | [**7**, 65..152, **154, 168**] |
|  80 | 60605091 | 5 | 2497 | 40 | ✗ | [41..78, **125, 137**] |
|  82 | fb1ceff0 | 8 | 5313 | 84 | ✗ | [44..64, **25, 18**, 65..94, **42, 33**, 95..121, **132, 140**] |
|  83 | b83c4150 | 2 | 2817 | 45 | ✗ | [3..45, **89, 104**] |
|  85 | e1a185dc | 8 | 5321 | 84 | ✗ | [44..64, **25, 18**, 65..94, **42, 33**, 95..121, **132, 140**] |
|  88 | f7f61b05 | 12 | 5805 | 91 | ✗ | [**7**, 65..152, **154, 168**] |
|  89 | 7f1cd9c2 | 5 | 2532 | 40 | ✗ | [41..78, **125, 137**] |
|  90 | f59fd3e2 | 11 | 2239 | 35 | ✗ | [37..69, **102, 117**] |
|  94 | a03d722b | 2 | 2161 | 34 | ✓ | [3, 4, 5, ..., 36] |
|  95 | 34195ade | 10 | 5194 | 82 | ✗ | [44..64, **25, 18**, 65..94, **42, 33**, 95..121] |
|  97 | a876010b | 12 | 5679 | 89 | ✗ | [**7**, 65, 66, ..., 152] |
|  99 | a52c09bc | 2 | 2693 | 43 | ✓ | [3, 4, 5, ..., 45] |
| 100 | bb0f8277 | 12 | 5680 | 89 | ✗ | [**7**, 65, 66, ..., 152] |
| 101 | 6b4b9d2b | 10 | 5196 | 82 | ✗ | [44..64, **25, 18**, 65..94, **42, 33**, 95..121] |
| 103 | 8bdd4f88 | 2 | 2694 | 43 | ✓ | [3, 4, 5, ..., 45] |
| 104 | 30a90fa5 | 13 | 2112 | 33 | ✓ | [37, 38, 39, ..., 69] |
| 105 | de54c4e6 | 2 | 2177 | 35 | ✗ | [3..36, **150**] |
| 106 | 22207643 | 12 | 5697 | 90 | ✗ | [**7**, 65..152, **198**] |
| 110 | f362edf4 | 9 | 5249 | 83 | ✗ | [44..64, **25, 18**, 65..94, **42, 33**, 95..121, **194**] |
| 112 | ee6946e7 | 2 | 2753 | 44 | ✗ | [3..45, **248**] |
| 113 | 2f3b7321 | 2 | 2241 | 36 | ✗ | [3..36, **150, 176**] |
| 114 | 5db1b172 | 12 | 5761 | 91 | ✗ | [**7**, 65..152, **198, 227**] |
| 117 | 6b10b6da | 12 | 5784 | 91 | ✗ | [**7**, 65..152, **198, 227**] |
| 119 | 0c4f5578 | 8 | 5300 | 83 | ✗ | [44..64, **25, 18**, 65..94, **42, 33**, 95..121, **194**] |
| 120 | fc14d852 | 2 | 2798 | 44 | ✗ | [3..45, **248**] |
| 121 | e63194e7 | 2 | 2268 | 36 | ✗ | [3..36, **150, 176**] |
| 122 | 27afdcea | 8 | 5313 | 84 | ✗ | [44..64, **25, 18**, 65..94, **42, 33**, 95..121, **194, 214**] |
| 124 | 27c3374f | 2 | 2817 | 45 | ✗ | [3..45, **248, 275**] |
| 127 | 8635db8f | 12 | 5806 | 91 | ✗ | [**7**, 65..152, **198, 227**] |

**Summary:**

- **17 contiguous** (29%): mostly the small-page-count b=2 cases starting at page 3
- **42 non-contiguous** (71%): pages have gaps, backwards jumps (e.g., `64→25→18→65`), scattered tail pages
- **Pattern**: initial run of ~32 contiguous pages, then 1–8 scattered "overflow" pages appended at the end
- **Implication**: kernel MUST do per-page indirect gather via block_table — cannot assume flat memory

| Pages | M tokens | Count | Distribution |
|---:|---:|---:|---|
| 33–45 | 2112–2880 | 39 | ██████████████████████████████████████ (66%) |
| 82–91 | 5248–5824 | 20 | ████████████████████ (34%) |

Min: 33 pages (M=2112), Max: 91 pages (M=5824). Always exactly **1 request per workload** exceeds 2048.

---

## Input Tensor Shapes

All constants (never change across workloads):
- `page_size = 64`
- `num_index_heads = 64`
- `index_head_dim = 128`
- `topk = 2048`
- `num_pages = 11923` (global KV cache pool, always the same)

| Tensor | Shape | Dtype | Source |
| `q_index_fp8` | `[batch, 64, 128]` | fp8_e4m3fn (packed as uint8) | random |
| `k_index_cache_fp8` | `[11923, 64, 1, 132]` | int8 (fp8+scale packed) | random |
| `weights` | `[batch, 64]` | float32 | random |
| `seq_lens` | `[batch]` | int32 | safetensors |
| `block_table` | `[batch, max_num_pages]` | int32 | safetensors |
| **output** `topk_indices` | `[batch, 2048]` | int32 | pre-allocated (DPS=true) |

The `k_index_cache_fp8` packing: `132 = 128 (fp8 data) + 4 (float32 scale)` bytes per token slot.
Stored as `[num_pages, page_size=64, 1, 132]` — each "row" of 132 bytes holds 128 fp8 values + 1 float32 scale.

### Concrete Examples — Stored Tensors

**WL 1 (uuid=30cecff1)** — axes: batch_size=1, max_num_pages=1, num_pages=11923
```
seq_lens:    shape=[1]    int32  →  [2]
block_table: shape=[1, 1] int32  →  [[1]]
```
1 request, 2 tokens, occupies page 1.

**WL 13 (uuid=83cb81c5)** — axes: batch_size=3, max_num_pages=2, num_pages=11923
```
seq_lens:    shape=[3]    int32  →  [34, 53, 73]
block_table: shape=[3, 2] int32  →  [[1, 0],
                                     [2, 0],
                                     [3, 4]]
```
3 requests, `page_size=64`:
- req 0: `seq_len=34` → `ceil(34/64)=1` page → uses page **1** only. `block_table[0,1]=0` is **padding, never read**.
- req 1: `seq_len=53` → `ceil(53/64)=1` page → uses page **2** only. `block_table[1,1]=0` is **padding, never read**.
- req 2: `seq_len=73` → `ceil(73/64)=2` pages → uses pages **3 and 4** (64 tokens from page 3, 9 tokens from page 4).

`max_num_pages=2` because req 2 needs 2 pages — all rows are padded to that width.
The code slices off padding: `block_table[b, :ceil(seq_lens[b]/64)]` → `[1]`, `[2]`, `[3,4]`.

---

## What Changes Across Workloads

Only two things vary:

### 1. `batch_size` (1 – 31)
Unique values: `1, 2, 3, 4, 6, 7, 8, 11, 12, 14, 15, 16, 25, 26, 27, 29, 30, 31`

Controls the number of sequences processed in parallel. Shapes that scale with batch:
- `q_index_fp8`: `[batch, 64, 128]`
- `weights`: `[batch, 64]`
- `seq_lens`: `[batch]`
- `block_table`: `[batch, max_num_pages]`
- `topk_indices` (output): `[batch, 2048]`

### 2. `max_num_pages` (1 – 91)
The second dimension of `block_table`. Equals `ceil(max(seq_lens) / 64)`.
Reflects the longest sequence in the batch — determines how many page slots the block table has per row.

### Derived: `seq_lens` values (1 – 5806 tokens per sequence)
- Real sequence lengths. Min = 1 token (trivially short), max observed = 5806 tokens = 91 pages.
- Most workloads are a mix of short (single-digit token counts) and one long sequence with thousands of tokens.

---

## Latency Grouping by Batch Size

| Batch range | Typical latency | Workload range |
| 1–3   | ~1.0 ms | [1]–[3] |
| 2–4   | ~1.2–1.7 ms | [4]–[29] |
| 6–8   | ~1.9–2.5 ms | [30]–[58] |
| 12–16 | ~3.0–3.7 ms | [59]–[91] |
| 25–31 | ~5.1–6.5 ms | [92]–[128] |

Latency scales roughly linearly with batch size — dominated by the sequential Python loop over batch elements in the reference. A vectorized/fused GPU kernel would collapse this.

---

## Histogram — Workloads by max_num_pages (bin = 4)

`max_num_pages` = second dim of `block_table`; each page = 64 tokens, so bin width = 4 pages = 256 tokens.

| Bin (pages) | Token range       | Count | Distribution                           |
|  1– 4       |     1 –   256     |    19 | ███████████████████                    |
|  5– 8       |   257 –   512     |     5 | █████                                  |
|  9–12       |   513 –   768     |     1 | █                                      |
| 13–16       |   769 – 1,024     |     2 | ██                                     |
| 17–20       | 1,025 – 1,280     |    13 | █████████████                          |
| 21–24       | 1,281 – 1,536     |     0 |                                        |
| 25–28       | 1,537 – 1,792     |     7 | ███████                                |
| 29–32       | 1,793 – 2,048     |    22 | ██████████████████████                 |
| 33–36       | 2,049 – 2,304     |    18 | ██████████████████                     |
| 37–40       | 2,305 – 2,560     |     6 | ██████                                 |
| 41–44       | 2,561 – 2,816     |    11 | ███████████                            |
| 45–48       | 2,817 – 3,072     |     4 | ████                                   |
| 49–80       | 3,073 – 5,120     |     0 | *(gap — no workloads)*                 |
| 81–84       | 5,121 – 5,376     |    10 | ██████████                             |
| 85–88       | 5,377 – 5,632     |     0 |                                        |
| 89–92       | 5,633 – 5,888     |    10 | ██████████                             |
| **Total**   |                   |   **128** |                                    |

The distribution is bimodal: a cluster of short-context workloads (≤ 3,072 tokens) peaking at [29–32] pages (2 048 tokens max), and a high-context cluster at [81–92] pages (5,100–5,900 tokens) with a hard gap in between — no workload has a longest sequence in the 3,073–5,120 token range.

### Fast-path vs slow-path split (TOPK = 2048)

The fast/slow boundary is `max_num_pages <= 32` ↔ `max_sl = max_num_pages * 64 <= 2048`:

| Path | Condition | Workloads | What happens |
| **Fast** | `max_num_pages ≤ 32` → `max_sl ≤ 2,048` | **69** | Skip GEMM entirely — scatter all valid global token ids straight into `topk_indices` |
| **Slow** | `max_num_pages > 32` → `max_sl > 2,048` | **59** | Full dequant → gather → GEMM → relu → weighted reduce → top-2048 |

**Edge case — what if `max_seq_len = 2049`?**  
`ceil(2049 / 64) = 33` pages → `max_sl = 33 × 64 = 2112` — just 1 token over budget. This still triggers the full slow path. Even though only 1 token is genuinely competing for the last slot, you cannot skip the GEMM: you must score *all* 2112 candidates to know which 2048 win. There is no way to cheaply identify the single weakest without computing every score. For 1 extra token the overhead is negligible in practice, but architecturally the threshold is hard at `max_sl > 2048`.

---

## Key Compute Pattern

For each batch element `b`:
```
scores = relu(q[b] @ K[b].T)          # [64, seq_len]   — GEMM: 64×128 × seq_len×128^T
final  = (scores * weights[b,:,None]).sum(dim=0)  # [seq_len]  — weighted reduce over heads
topk   = topk(final, 2048)             # index select
```

The dominant cost is the GEMM: `batch × 64 heads × seq_len × 128 dim`.
For the largest workloads: `31 × 64 × ~5000 × 128 ≈ 1.3 billion multiply-adds`.

The reference uses PyTorch eager per batch element — no batching of GEMMs, no GPU parallelism across batch.

---

## Optimization Opportunities

1. **Batch the GEMM**: `q` is `[batch, 64, 128]`, `K` varies per batch element due to paging but could be gathered first. Use `torch.bmm` or a fused paged-GEMM.
2. **Fuse dequant + GEMM**: `k_index_cache_fp8` decode + multiply can be a single Triton kernel.
3. **Use `flashinfer.top_k_page_table_transform`**: The baseline (`flashinfer_deepgemm_wrapper_2ba145`) uses `deep_gemm.fp8_paged_mqa_logits` for the score phase (fp8 GEMM) + `flashinfer.top_k_page_table_transform` for selection. These are optimized CUDA kernels.
4. **Avoid Python loop over batch**: Reference loops `for b in range(batch_size)` — kills GPU utilization for the GEMM phases.

---

## sA SMEM Layout: Sw<3,4,3> (tcgen05.mma A operand)

`sm100_utils.make_smem_layout_a(tiled_mma, cta_tile_mnk=(128,64,128), fp8, stages=1)`
returns a composed layout with inner swizzle **Sw<3,4,3>** (also written `S<3,4,3>` or
`Swizzle<3,4,3>`) and outer row-major layout for the 128×128 fp8 tile.

### What Sw<3,4,3> does

```
Sw<B=3, M=4, S=3>:  addr_swizzled = addr XOR ((addr >> (M+S)) << M)
                                   = addr XOR ((addr >> 7) & 0x78) -- XOR bits [9:6] onto bits [6:3]
```

Equivalently in byte-offset terms:
```python
def swizzle_343(byte_off):
    return byte_off ^ (((byte_off >> 7) & 0xF) << 3)
```

- **Period**: 128 bytes along the column axis (= one 128-fp8 row).
- **XOR groups**: 16 groups × 8 rows = 128-row repeat.  
  Each group of rows [8k … 8k+7] maps to a distinct XOR pattern, eliminating some bank conflicts for `ldmatrix` / `tcgen05.mma`.

### sA tile dimensions

| Property | Value |
|---|---|
| dtype | fp8 (E4M3) |
| shape | 128 rows × 128 cols |
| total bytes | 16 384 |
| page size (rows) | 64 |
| top page bytes | 8 192 (rows 0–63) |
| bottom page bytes | 8 192 (rows 64–127) |

### Page partitioning and index reconstruction

The KV cache stores K in paged blocks of 64 rows each.  When the kernel needs
two pages (BM=128), it loads them into the **top half** (rows 0–63) and
**bottom half** (rows 64–127) of `sA`.

Given a paged GMEM pointer (`page_ptr`) for one 64-row page:

```python
# linear byte offset inside sA (for a given page)
linear = page_offset + col_idx + 128 * row_idx_in_page

# apply Sw<3,4,3> to get the actual SMEM byte
swizzled = swizzle_343(linear)

# recover (smem_row, smem_col) from swizzled byte offset
smem_row = swizzled // 128
smem_col = swizzled  % 128
```

The pattern repeats every 8 rows, so rows 0–7 and rows 64–71 have the same XOR
group assignment modulo 8.

### SMEM allocation (CuTe DSL)

```python
sA = smem.allocate_tensor(
    element_type=fp8_dtype,
    layout=a_smem_layout.outer,   # (128, 128) row-major
    byte_alignment=128,
    swizzle=a_smem_layout.inner,  # Sw<3,4,3>
)
```

**Critical constraint**: `sA` must be the *first* allocation (SMEM offset 0).
`Sw<3,4,3>` XORs absolute address bits — it does **not** subtract the SMEM
base.  Placing `sA` at any offset > 0 will XOR against the wrong bits, causing
misplaced data (observed max error ≈ 347 with a non-zero offset).

---

## cp.async A-load: TV-Layout PISL i32 (512-thread CTA)

Instead of autovectorized loads (which under-utilize memory bandwidth for small
tiles), we drive **all 512 CTA threads** to issue `cp.async` and saturate the
L2→SMEM pipeline.

### Why 512 threads?

`cp.async` throughput scales with *bytes-in-flight*, not just bytes-per-thread.
With 128-compute-thread autovec (4 × 128-bit = 64 B/thread per issue), peak
in-flight is limited.  Spreading the same 16 384 bytes over 512 threads at
32 bits each (128-bit per thread × 4 threads = same data) keeps the pipeline
full and measured latency drops from **6.07 µs → 1.89 µs (3.2×)**.

### PISL = Per-thread Issued Sub-Load

Each thread issues an `i32` (32-bit) `cp.async`.  The copy atom is:

```python
atom_cpa = cute.make_copy_atom(
    cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
    cutlass.Int32,
    num_bits_per_copy=cutlass.Int32.width,   # 32 bits / thread / issue
)
```

### TV layout construction

The 128×128 fp8 tile viewed as `i32` is shape **(128 rows, 32 cols)** (`HEAD_DIM_I32 = 128 // 4 = 32`).  
We want 512 threads × 8 i32 values each = 4096 i32 = 16 384 bytes.

```python
N_PER_THREAD_I32 = (BM * HEAD_DIM_I32) // THREADS_PER_CTA   # = 8

# Thread layout: 16 rows of threads × 32 col threads  →  512 threads
#   stride=(HEAD_DIM_I32, 1) = (32,1) means contiguous across the column axis
thr_layout_load = cute.make_layout((16, HEAD_DIM_I32),   # (16, 32)
                                    stride=(HEAD_DIM_I32, 1))

# Value layout: 8 i32 values per thread, stacked along row axis
val_layout_load = cute.make_layout((N_PER_THREAD_I32, 1),  # (8, 1)
                                    stride=(1, 1))

tiled_copy_a = cute.make_tiled_copy_tv(atom_cpa, thr_layout_load, val_layout_load)
thr_copy_a   = tiled_copy_a.get_slice(tidx)
```

### SMEM destination: `Sw<3,2,3>` composed view

`cp.async` writes to an `i32` view of `sA` swizzled with **`Sw<3,2,3>`** (not
`Sw<3,4,3>`).  The reason: `Sw<3,4,3>` operates on byte offsets; viewing as
`i32` means 4× fewer "columns", which shifts the effective M parameter by 2
bits: 4→2 in the composed layout.

```python
sA_load_layout = cute.make_composed_layout(
    cute.make_swizzle(3, 2, 3), 0,
    cute.make_layout((BM, HEAD_DIM_I32),       # (128, 32) i32 shape
                     stride=(HEAD_DIM_I32, 1)),
)
sA_i32_ptr = cute.recast_ptr(sA.iterator, dtype=cutlass.Int32)
sA_load    = cute.make_tensor(sA_i32_ptr, sA_load_layout)
```

### Sync sequence after cp.async

```python
# Both cp.async and TMA-B complete before MMA:
cute.arch.mbarrier_wait(tma_mbar, 0)       # wait for TMA-B
cute.arch.cp_async_commit_group()
cute.arch.cp_async_wait_group(0)           # wait for all cp.async
cute.arch.sync_threads()
cute.arch.fence_view_async_shared()        # proxy fence: SMEM visible to MMA
```

`fence_view_async_shared` is **mandatory** between `cp_async_wait_group` and
the first `tcgen05.mma` — without it the MMA may see stale data.

### Named barriers

| Barrier | ID | Count | Purpose |
|---|---|---|---|
| `INIT_BAR_ID` | 1 | 512 | tmem alloc + mbarrier init visible to all 512 threads |
| `EPI_BAR_ID` | 2 | 128 | epilogue compute group synchronization |

### Constants summary

| Constant | Value | Notes |
|---|---|---|
| `PAGE_SIZE` | 64 | rows per KV page |
| `HEAD_DIM` | 128 | fp8 columns |
| `ROW_STRIDE` | 132 | bytes per paged row (includes padding) |
| `BM, BN, BK` | 128, 64, 128 | CTA tile shape |
| `THREADS_PER_CTA` | 512 | all issue cp.async A |
| `COMPUTE_THREADS` | 128 | TMA / MMA / epilogue |
| `HEAD_DIM_I32` | 32 | `HEAD_DIM // 4` |
| `ROW_STRIDE_I32` | 33 | `ROW_STRIDE // 4` |
| `N_PER_THREAD_I32` | 8 | i32 values per thread |

---

## Topk-input value distribution (sl > 2048 workloads)

Collected on B200 by `src/modal/dump_topk_inputs.py`, replaying
`src.kernels.idxer_tc.run` for every workload in
`dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl` whose `max_sl > 2048`,
and dumping the post-`_score_and_reduce` `final` tensor (the input to
`torch.topk`) to `archive/analysis/idxer_topk_inputs/wl{N}.pt`.

### Workload coverage

| metric | value |
|---|---|
| total workloads in JSONL | 128 |
| workloads with `max_sl > 2048` | **59** |
| max_sl bucket 2k-3k | 39 workloads |
| max_sl bucket 5k-6k | 20 workloads |
| min `max_sl` | 2112 |
| max `max_sl` | **5806** (well under the 10240 SMEM cap) |
| aggregate valid scalars | 205,160 |

### Zero-count statistics (the original optimization hypothesis)

> Hypothesis: scores fed to `torch.topk` contain many exact zeros (from the
> `relu` in `_score_and_reduce`) that could be filtered out cheaply, turning
> them into `-1` in the index output and shrinking phase 1/2 work.

| metric | value |
|---|---|
| total zero values across all 59 workloads | **0** |
| zero fraction | **0.0000 %** |

**Conclusion: the zero-skip optimization yields zero work reduction on this
dataset.** Reason: `_score_and_reduce` does `relu(bmm(q, K))` *per-head*
(producing many zeros at the [B, 64, max_sl] level) but then
`einsum("bhs,bh->bs", scores, weights)` collapses across all 64 heads with
random ±weights. Getting an exact 0 at the output requires every one of 64
random head-scores to be ≤ 0 simultaneously (≈ 2⁻⁶⁴), which never happens
in practice.

### Other distributional facts (sample: wl126, B=27, max_sl=5806)

Across the valid prefix of all rows in the sampled workload:

| bucket | count | fraction |
|---|---|---|
| `== 0`  | 0       | 0.00 % |
| `< 0`   | ~1,340  | **~32 %** |
| `<= 0`  | ~1,340  | ~32 % |

So ~32 % of values are *negative* (positive heads × negative weights). Since
`TOPK = 2048` and rows with `sl ≥ 2048` typically have ≥ 2048 positive
values, the top-k winners are essentially all positive — meaning a
**negative-skip** filter would actually be effective even though zero-skip
is not.

(Note: the synthetic fp8 dequant from `randint(0,256)` also occasionally
produces `inf` and very large negatives via fp32-cast scale × fp8 product;
these are fp8-quant artifacts of the random benchmark inputs.)

### Implication for kernel design

- **SMEM cache is safe**: max real `sl` is 5806, so a 10240-int32
  (40 KB) SMEM buffer comfortably holds an entire row of radix-converted
  bits. This is what `topk_aten_cutedsl_v4_fuse_earlyexit_intra_smem_v1.py`
  does.
- **Zero-skip filter is futile** on this dataset (0 zeros → 0 work saved).
- A **negative-skip** (or equivalently `<= 0`) prefilter could remove
  ~32 % of inputs before phase 1/2; this is the variant worth implementing
  if a "remove sentinel-marked elements" stage is desired.

---

## Top-K Algorithm: Histogram Radix Select (`idxerv4_hist_pdl.py`)

### Problem statement

For each batch request `b` with `seq_len[b] > TOP_K` (= 2048), the scoring
kernel writes a float32 score per token into `score_output[b, 0:sl]`.  The
topk kernel must then find the 2048 highest-scoring global token indices —
without sorting the full array.

### Key insight: `float_to_radix`

IEEE-754 float32 values are bit-cast to uint32 with a monotonicity fix so
that the natural integer order matches the float order:

```
if value < 0:  flip all 32 bits  (0xFFFFFFFF XOR x)
else:          flip only the sign bit  (0x80000000 XOR x)
NaN → 0xFFFFFFFF  (treated as +∞, always "above" any finite value)
```

After this transform a plain uint32 comparison `a > b` is equivalent to
`float_a > float_b`.  All histogram passes operate on these radix bits
rather than the original floats.

### Phase 0 — setup (SMEM bit-cache)

When `USE_LIMIT_TOPK_SEQ_LEN=True` (always the case here), every thread
converts `score_output[b, tidx + k*topk_threads]` to radix bits and writes
them into `smem_bits[0:sl]` (SMEM, 4 bytes each).  This one-time pass costs
one global load + one inline-asm per element.  All subsequent passes read
from SMEM instead of GMEM, replacing expensive L2 hits with ~20-cycle SMEM
reads.

### Phase 1 — 4-pass 8-bit histogram radix select

The algorithm runs four sequential passes over the `sl` radix bits, each
pass resolving one byte of the 32-bit radix key (MSB first: bits [31:24],
[23:16], [15:8], [7:0]).

**State** in `smem_tau[5]`:

| idx | name | meaning |
|-----|------|---------|
| 0 | `desired` | fixed upper bits of the threshold value so far |
| 1 | `desired_mask` | bitmask of already-resolved bits (0 initially) |
| 2 | `above_total` | number of elements strictly above the threshold |
| 3 | `k_to_find` | remaining elements needed in the top-k window |
| 4 | `early_exit` | set to 1 when an exact bin count matches `k_to_find` |

**Per-pass steps:**

1. **Clear** 32 warp-private sub-histograms of 256 bins (`smem_warp_hist`,
   shape `[32, 256]`).
2. **Bin pass** — each thread iterates over its slice of `smem_bits`.
   An element is counted only if its upper bits match `desired & desired_mask`
   (elements already known to be below the threshold are skipped).  The digit
   for the current byte is extracted and added to the warp's sub-histogram
   via a relaxed CTA-scoped `atomicAdd`.  Using 32 independent sub-histograms
   avoids inter-warp contention entirely.
3. **Merge** — threads `0..255` each sum their column across all 32
   sub-histograms into `smem_hist[256]`.
4. **τ-find** (thread 0 only) — scan `smem_hist` from bin 255 down to 0,
   accumulating counts until the running sum would reach `k_to_find`.  The
   bin where the sum first reaches or crosses `k_to_find` is `tau_b`.
   - `above_total` += count of elements in bins `[255 .. tau_b+1]`
   - `k_to_find`  -= count of elements in bins `[255 .. tau_b+1]`
   - `desired` and `desired_mask` are extended to pin `tau_b` in the current
     byte position.
   - If `smem_hist[tau_b] == k_to_find`, the tie bin is exact — set
     `early_exit=1` to skip remaining passes.

After 4 passes the final threshold is a full 32-bit radix key.  Elements
with `radix > threshold` are "above" (unconditionally selected); elements
with `radix == threshold` are "ties" (selected greedily up to `k_to_find`).

**Work saved by `desired_mask`**: in later passes only elements whose upper
bits match the already-determined prefix are histogram-counted.  In the
worst case (uniform scores) this is all elements; in practice the histogram
concentrates quickly and later passes skip a large fraction.

### Phase 2 — fused scatter pass

A single warp-level parallel scan over all `sl` tokens writes indices into
`topk_indices[b, 0:TOP_K]`:

- **"above"** elements (`radix > desired_pin`): written to positions
  `[0 .. above_total-1]` in order of appearance.
- **"tie"** elements (`radix == desired_pin`): written to positions
  `[above_total .. above_total+need_ties-1]`, stopping once `need_ties`
  slots are filled.

Each tile of `topk_threads` elements is processed together.  Within a tile:
1. Each warp independently prefix-sums its `is_above` and `is_tie` flags
   (5-step Hillis-Steele with `shuffle_sync_up`).
2. Warp 0 prefix-sums the 32 per-warp totals, writing exclusive offsets back
   to `smem_warp_above` / `smem_warp_tie` and computing the tile-level totals
   into `smem_above_round` / `smem_tie_round`.
3. Each thread computes its global output slot as
   `above_cursor + warp_b_off + my_b_excl` (above) or
   `above_total + tie_cursor + warp_t_off + my_t_excl` (tie) and writes the
   global token index `page_id * PAGE_SIZE + token_in_page`.

The two cursors are advanced by the tile totals after each tile, giving a
deterministic but non-sorted output order (matches `check_topk_indices`
order-independent check).

### PDL (Programmatic Dependent Launch) integration

The score kernel (`indexer_ksplit_kernel`) calls
`griddepcontrol_launch_dependents()` immediately after SMEM allocation and
copy-atom construction — before the first `cp.async` — so the topk CTAs can
be scheduled onto free SMs and run their own independent prologue
(SMEM allocation + `smem_tau` init) in parallel with the score kernel's
main MMA work.

The topk kernel calls `griddepcontrol_wait()` at its very first line, before
any read of `score_output`.  The wait completes only after the score kernel's
last `score_output` write is visible, so there is no data race.

The net effect is that the topk kernel launch latency (~1–3 µs) and the
topk prologue (SMEM alloc + τ init + optional bit-cache setup) are
overlapped with the score kernel's GEMM tiles, effectively hiding them from
the critical path.

---

## Kernel Latency Sorted — All 128 Workloads

`seq_len` = longest sequence in the batch (determines fast vs slow path).
`num_pages` = `max_num_pages` from `block_table` shape = `ceil(max_seq_len / 64)`.

**Fast-path (pass-through): 69 workloads** — all seq_lens ≤ 2048 (num_pages ≤ 32).
No GEMM required; scatter valid token indices directly into `topk_indices`.

| # | Workload | seq_len | num_pages | ms |
|--:|---|---:|---:|---:|
| 1–69 | *(69 workloads — see table above: 30cecff1, cd594d26, 44ddaa65, b2098949, 4667f9ad, 8f2fde6c, 02fa7f90, 545f8a85, e49574dd, 0ebafac4, 82a8a885, 4279d75e, 83cb81c5, 9754a4e7, 16feeab1 (WL16), 05775386, 46f236c0, d54c1568, 17ced9b8, 101a39ac, ef0d0deb, 28a9fa48, 55be3dc3, ef12ac76 (WL24), 899c2d2f, …)* | ≤ 2048 | ≤ 32 | **0.002** |

**Slow-path (GEMM required): 59 workloads** — exactly 1 request per workload has seq_len > 2048.

| # | Workload | seq_len | num_pages | ms |
|--:|---|---:|---:|---:|
|  1 | 81a953ea | 2,817 | 45 | 0.014 |
|  2 | 19e7663d | 2,405 | 38 | 0.015 |
|  3 | 16feeab1 | 2,693 | 43 | 0.016 |
|  4 | a52c09bc | 2,693 | 43 | 0.016 |
|  5 | ee603b53 | 2,694 | 43 | 0.016 |
|  6 | 9a95a10e | 2,162 | 34 | 0.016 |
|  7 | a03d722b | 2,161 | 34 | 0.016 |
|  8 | 3e91afa0 | 2,693 | 43 | 0.016 |
|  9 | 9810dadf | 2,161 | 34 | 0.016 |
| 10 | 175849a8 | 2,241 | 36 | 0.016 |
| 11 | f457feb2 | 2,433 | 39 | 0.017 |
| 12 | f59fd3e2 | 2,239 | 35 | 0.017 |
| 13 | e26d02ef | 2,753 | 44 | 0.017 |
| 14 | 03fc111f | 2,817 | 45 | 0.017 |
| 15 | 7f1cd9c2 | 2,532 | 40 | 0.017 |
| 16 | d8a73470 | 2,161 | 34 | 0.017 |
| 17 | 30a90fa5 | 2,112 | 33 | 0.017 |
| 18 | ef12ac76 | 2,753 | 44 | 0.017 |
| 19 | ed3e595b | 2,112 | 33 | 0.017 |
| 20 | 2f3b7321 | 2,241 | 36 | 0.017 |
| 21 | b83c4150 | 2,817 | 45 | 0.017 |
| 22 | 696dbfa4 | 2,693 | 43 | 0.017 |
| 23 | b017f77a | 2,161 | 34 | 0.017 |
| 24 | 2774963f | 2,405 | 38 | 0.017 |
| 25 | 99920dc5 | 2,162 | 34 | 0.018 |
| 26 | de54c4e6 | 2,177 | 35 | 0.018 |
| 27 | 8635db8f | 5,806 | 91 | 0.018 |
| 28 | cd3434ac | 2,177 | 35 | 0.018 |
| 29 | f1fc35d4 | 2,406 | 38 | 0.018 |
| 30 | 8bdd4f88 | 2,694 | 43 | 0.018 |
| 31 | 8f3fe9ff | 2,241 | 36 | 0.018 |
| 32 | 77279062 | 2,753 | 44 | 0.018 |
| 33 | e63194e7 | 2,268 | 36 | 0.019 |
| 34 | ee6946e7 | 2,753 | 44 | 0.019 |
| 35 | 27c3374f | 2,817 | 45 | 0.019 |
| 36 | 08a752fc | 2,177 | 35 | 0.019 |
| 37 | f362edf4 | 5,249 | 83 | 0.019 |
| 38 | 70d53807 | 5,194 | 82 | 0.019 |
| 39 | fc14d852 | 2,798 | 44 | 0.019 |
| 40 | 8638fe06 | 2,177 | 35 | 0.019 |
| 41 | 60605091 | 2,497 | 40 | 0.019 |
| 42 | e4ecb462 | 5,195 | 82 | 0.020 |
| 43 | 4c7705ad | 2,241 | 36 | 0.020 |
| 44 | 34195ade | 5,194 | 82 | 0.020 |
| 45 | 27afdcea | 5,313 | 84 | 0.021 |
| 46 | 6bdb38e6 | 5,249 | 83 | 0.021 |
| 47 | bb0f8277 | 5,680 | 89 | 0.021 |
| 48 | 0c4f5578 | 5,300 | 83 | 0.021 |
| 49 | 22207643 | 5,697 | 90 | 0.021 |
| 50 | 6b10b6da | 5,784 | 91 | 0.021 |
| 51 | f7f61b05 | 5,805 | 91 | 0.021 |
| 52 | e1a185dc | 5,321 | 84 | 0.021 |
| 53 | 6b4b9d2b | 5,196 | 82 | 0.021 |
| 54 | cdc0ff86 | 5,697 | 90 | 0.022 |
| 55 | 3eab2c37 | 5,761 | 91 | 0.022 |
| 56 | a876010b | 5,679 | 89 | 0.022 |
| 57 | fb1ceff0 | 5,313 | 84 | 0.023 |
| 58 | 786b5173 | 5,679 | 89 | 0.023 |
| 59 | 5db1b172 | 5,761 | 91 | 0.024 |

> **Latency bands:**
> - 0.002 ms — pass-through (scatter only, no GEMM)
> - 0.014–0.019 ms — 1× GEMM, pages 33–45 (~2100–2850 tokens) or pages 82–91 with anomalously fast measurement (#27, #37–38)
> - 0.019–0.021 ms — 1× GEMM, pages 82–91 (~5200–5810 tokens)
> - 0.022–0.024 ms — same page range, slower due to batch/memory pressure
>
> **Key observation:** latency is almost entirely determined by `num_pages` (GEMM M-dimension), not batch size. Workloads #3–#10 (batch 8–31, pages 33–45) land at the same 0.016 ms as batch=4 equivalents, confirming 1 slow-path request per workload regardless of batch_size.
