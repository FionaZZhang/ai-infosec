| Variant | BU | UAA | ASR | Benign KC | Attacked KC | ASR_soft |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 1.00 | 0.10 | 0.90 | 0.8667 | 0.8667 | 0.65 |
| defended (heuristic) | 0.90 | 0.80 | 0.00 | 0.6667 | 0.6000 | 0.00 |
| defended (heuristic + gemini) | 1.00 | 1.00 | 0.00 | 0.7333 | 0.8000 | 0.00 |

Run files:
- baseline: `results/baseline/run_20260318_231522_219639.json`
- defended (heuristic): `results/defended/run_20260318_231522_223879.json`
- defended (heuristic + gemini): `results/defended/run_20260318_231522_236168.json`
