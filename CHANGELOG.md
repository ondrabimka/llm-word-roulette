# Changelog

## 0.1.0

- Implemented a full baseline pipeline from README spec foundations:
  - corpus config loading,
  - checksum-verified download and cleaning,
  - tokenization and word counts,
  - salience computation with `wordfreq`,
  - reference baseline distributions (`P_freq`, `P_type`, `P_salience`),
  - metrics and static plots.
- Added CLI commands: `prepare`, `baselines`, `metrics`, `plots`, `web`, `run-all`.
- Added an interactive GitHub Pages dashboard (`docs/index.html`) and automated Pages deployment workflow.
