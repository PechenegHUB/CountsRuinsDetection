# Working on CountsRuinsDetection

- Read README.md, ASSETS_REQUIRED.md and docs/reproducibility.md first.
- Preserve the frozen D53 reference and distinguish it from unverified ports.
- Never describe cached reconstruction as full image inference or model training.
- Do not invent working commands, timings, metrics, assets or download links.
- Keep the main README sufficient for supported workflows. Mark pending work there.
- Keep official images, annotations, per-row predictions, caches, model weights and
  credentials outside Git, under ignored data/ or artifacts/ directories.
- Record original source commits, checkpoint SHA-256, environment and data scope.
- Historical EXP-044 pseudo training iterates all rows of tables.csv and aux;
  preserve that selector and do not call pseudo val an independent evaluation.
- Use date/episode-separated evaluation for supervised heads; report official and
  manually reviewed labels separately.
- Do not transfer the old repository's conversational style instructions.
- Push only when requested or as part of an explicitly requested publication.
- Before every push run git pull --rebase origin main and preserve others' changes.
- For the present scaffold, training and full image inference are pending. Do not
  add placeholder launchers that appear to implement these stages.
