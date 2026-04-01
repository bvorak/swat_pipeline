---
description: "Use when checking whether a thesis sentence, figure claim, metric interpretation, or reviewer-response paragraph matches the implemented SWAT stats/export/chart workflow and the current thesis context files."
name: "Verify Thesis Claim"
argument-hint: "Paste the thesis claim, paragraph, figure caption, or reviewer comment to verify against the workflow."
agent: "SWAT Stats Workflow"
---
Verify the provided thesis claim against the implemented SWAT statistics workflow.

Your job is to check the claim against:
- trabajoFM/python_pipeline_scripts/stats.py
- trabajoFM/python_pipeline_scripts/dashboard.py
- trabajoFM/notebooks/03_dashboard_RCH_analysis.ipynb
- trabajoFM/notebooks/04_sensitivity_stats.ipynb
- trabajoFM/Context/TFM Results.md
- trabajoFM/Context/TFM Discussion.md
- trabajoFM/Context/TFM Intro & Methods.md

Use the code and exported JSON workflow as the source of truth for calculations and metric definitions.
Use the three thesis text files as interpretation context for terminology, scenario naming, figure intent, and unresolved reviewer comments.

For the user input below:

{{input}}

Return:
- A short workflow trace showing where the relevant logic lives.
- Whether the claim is verified, partly verified, unsupported, or contradicted by the implementation.
- The exact metric keys and JSON paths involved, when applicable.
- Any mismatch between thesis wording and implemented computation.
- The smallest change needed if the wording should be fixed in the thesis, or if the code/workflow should be changed instead.
- A note on what must be rerun if the implementation changes.

Do not invent a parallel interpretation pipeline. If the implementation and thesis text disagree, state that explicitly.