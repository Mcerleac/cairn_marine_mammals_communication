# Risk Register

Ranked from highest to lowest risk.

| Rank | Risk | Type | Owner | What we're doing / going to do about it |
|---|---|---|---|---|
| 1 | Training needs more than the 16 GB of GPU memory we have now. Without a bigger GPU, training new models will be slow or impossible. | Technical | Ciaran / Wren | Getting access to and learning to run models on OSU HPC servers. |
| 2 | The repo has no tests or CI, so a change could break the pipeline without anyone noticing. | Technical | Tiernan | Add a small test suite that runs on every pull request. |
| 3 | Publishing models and datasets on Hugging Face requires Cairn Institute's approval and account. | Business | Adam | Ask the project partner for approval early (not needed until around winter/spring). |
| 4 | Training a new model may take a lot longer than anticipated and divert attention from the existing models. | Team | Ciaran | Set a time limit for training a new model; if it runs too long, fall back to the previous model. |
| 5 | We have no way to automatically grade generated audio. Our only check is listening, which is highly subjective. | Technical | Wren | Write an automated check script (length ≥ 3 s, frequencies within our hydrophone data's range), alongside listening checks. |
| 6 | Some code and documentation referenced in `/docs` is still in Cairn's private repo, and we don't have access to it. | Team | Ciaran | Ask the project partner for access to these files to fill gaps, such as `scripts/grade_audio_quality.py`. |

## External Prerequisites

| Prerequisite | Needed for | Asked of | Asked by | Date asked | Status |
|---|---|---|---|---|---|
| GPU with more than 16 GB VRAM (OSU HPC) | Training | OSU HPC | Team | 10/23/26 | Complete |
| Approval and access to publish on Cairn's Hugging Face | Release | Ken Janik | Team | Not yet asked | Not started |

## Staging and Rollback Plan

| Item | Plan | Date |
|---|---|---|
| Staging | Run the pipeline on one small SanctSound file before merging any change to the data, model, or checkpoint code. | By week 10 |
| Rollback | Revert the merge commit on main. Tag releases so we can get back to a known good version. | By week 10 |
