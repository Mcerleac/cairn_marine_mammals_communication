# Contributing Guide

How to set up, code, test, review, and release so contributions meet our Definition of Done.

add on, rather than detract from the codebase. We are building on existing work, do not push without passing unit tests.

We do expect to run at least one code review done by another member of the team before merging into the codebase.

## Code of Conduct

Please combine reports into one place if possible to cut down on the number of reports made.

## Getting Started

Prerequisites
Linux or macOS, or Windows via WSL2
Python 3.10 or newer 
Git
A GPU
Some disk space (Preferably a lot)
Setup Steps
git clone https://github.com/cairninstitute/cairn_marine_mammals_communication.git
cd cairn_marine_mammals_communication
python 3 -m venv .venv
. .venv/bin/activate
pip install -e “.[dev]”
pip install -e “.[dataset,audio-codex]”
Environment Variables/Secrets Handling
None currently required
How to run app locally
Generate audio from a released checkpoint:
	PYTHONPATH=. python3 scripts/generate_dac_9cb_unified.py \
  	--checkpoint runs/audio_large_swa_moe_sanctsound_humpback_dac_9cb_10k_v2/best_model.pt \
 	 --n-samples 3 --max-new-seconds 6 --decode-audio generated
Train a model: PYTHONPATH=. python3 scripts/train.py configs/audio_large_swa_moe_sanctsound_humpback_dac_9cb_10k.yaml
Rebuild dataset from raw audio:
	PYTHONPATH=. python3 scripts/download_sanctsound.py --station hi01 --deployment 1
PYTHONPATH=. python3 scripts/process_sanctsound_humpback.py --codec dac --station hi01
Inspect a checkpoint: PYTHONPATH=. python3 scripts/describe_checkpoint.py runs/.../best_model.pt 
	

## Branching & Workflow

Workflow: Gitflow
Default branch: Master
Branch naming: Will be based on the work being done in the branch, eg if someone is working on the model, the branch name would be something like “Model” or “Modeling” 
Rebase vs merge: Merge will be mainly used since it is consistently safer than rebase, but rebase can be used if not working on a branch someone else is working on. 
## Issues & Planning

Planning handled through docs/board.txt

Issues can be raised on the github issues page
(if page not working put them on the board)

## Commit Messages

State the convention (e.g., Conventional Commits), include examples, and how to reference issues.

Every Commit should have a message with specifications. If your commit message is too long, your commit itself was likely too large. Commit often, and with a message for each change. Ex: ‘Fixed cloud connection error with for loop’. ‘Edited comments to be grammatically correct’.

## Code Style, Linting & Formatting

Name the formatter/linter, config file locations, and the exact commands to check/fix locally.

Formatter/linter: None currently
Config file locations: Configs folder
Commands: None currently 

## Testing

Define required test types, how to run tests, expected coverage thresholds, and when new/updated tests are mandatory.

Currently, there is no test suite in our repo for code changes. We will be adding unit tests and an overall test suite, making sure to cover edge cases for each individual Python file. That way, it will be a large set of tests.

We will be expected to run audio through the script grade_audio_quality.py, which will provide a more quantifiable way to measure the performance of the output.

The grade_audio_quality.py script grades on the following metrics:
Signal-to-Noise Ratio (SNR): Comparison of vocalization energy to background
Spectral Clarity: Presence of distinct frequency peaks characteristic of whale calls
Peak Frequency Distribution: Expected frequency ranges for each species
Spectral Centroid: Center of mass of frequency spectrum
Harmonic Structure: Detection of harmonic relationships in orca/dolphin calls

Grade Scale: A (excellent) → F (unusable)

## Definition of Done

List the checks a change must pass before it merges (tests, review, docs updated, lint and security checks), each with the CI job, hook, or test that enforces it. For a gate not built yet, name the sprint it lands in and its owner.

This is going to represent a future state of the projects, but we want to create it hopefully by the end of fall term while we are recreating previous work.

A suite of unit tests for each Python file is planned to be created. These tests will cover edge cases for each major step, and ensure the output looks accurate. If a unit test cannot be created for part of our pipeline, it will be replaced with manual human testing.

For example, in the case of our AI model, it will be hard to run automated tests to ensure the output works. Instead, that will require human judgement and pattern recognition to recognize if a whale song sounds like a whale song based on our knowledge, and not something wildly different. If all checks pass, a commit may be sent. Merging will only happen after the code is reviewed by one different team member, assuming it passed the test suite without resorting to falsifying test suite results. (Adding if statements to set it to true by default).

In the case of a release, we plan to be more strict in our checks, and have someone specialized in the field to ensure that our output sounds accurate in comparison to that specific marine creature.


## Contribution Norm

State the floor (every contributor lands at least one piece of accepted work per sprint), where each kind of work lives (the repository, the issue tracker, the partner's system), and what accepted means for each.

Floor:
At least one piece of accepted work per sprint.

Kind of Work
Live in
Accepted when
Code
Repo: cairninstitute/cairn_marine_mammals_communication 
The PR is merged with main, and any checks are passed 
Documentation
Repo: cairninstitute/cairn_marine_mammals_communication 
Merged to main after being reviewed by a group member
Experiments and analysis 
In repo under docs or analysis folders 
Experiment is reproduced by another team member, and the analysis has been looked over and confirmed by another group member
Datasets
In Hugging face, under cairninstitute, and the processing code goes in the repo
The data set is complete, with its source, license and token stats. Processing code is accepted when it gets merged.
Models/Checkpoints 
Hugging face models under cairninstitute, and configs go in the repo 
The config PR is merged


## Pull Requests & Reviews

Outline PR requirements (template, checklist, size limits), reviewer expectations, approval rules, and required status checks.

Pull request temp:
## Summary
What changed and why. Closes #<issue>

## Type
- [ ] Code  - [ ] Docs  - [ ] Config  - [ ] Data pipeline  - [ ] CI/tooling

## How tested
Commands run + results (e.g., `pytest -m "not slow"`, tiny training run, generated sample)

## Checklist
- [ ] Linked to an issue
- [ ] Tests added/updated (or N/A with reason)
- [ ] Docs/README updated for CLI, config, or data-format changes
- [ ] No secrets, tokens, raw audio, checkpoints, or `.npy` data committed
- [ ] New data sources attributed in NOTICE / DATASET_TERMS.md
- [ ] Checkpoint/tokenizer format unchanged, or backward-compat noted

Reviewer Expectations: 
Reviewers will check for correctness, tests, readability, reproducibility and check any new/updated docs
The review of code should happen within 3 business days
If there are any issue give polite and detailed feedback 

Approval rules:
Authors cannot approve their own pull requests
All conversations must be resolved before merge
No direct pushes to main 

## CI/CD

Link to pipeline definitions, list mandatory jobs, how to view logs/re-run jobs, and what must pass before merge/release.

Currently no CI/CD pipeline 

## Security & Secrets

State how to report vulnerabilities, prohibited patterns (hard-coded secrets), dependency update policy, and scanning tools.

Please do not make a public pull request for anything pertaining to security threats. Please contact one of the contributors directly, as we want this to be as safe as possible. DO make a pull request in the case that there is no response for an extended period of time (more than two weeks, assuming more pressing pull requests do not exist).

## Documentation Expectations

Specify what must be updated (README, docs/, API refs, CHANGELOG) and docstring/comment standards.

No file must be updated as of now, for now we will use git issues with the template we have for pull requests to keep track of changes/

## Release Process

Describe versioning scheme, tagging, changelog generation, packaging/publishing steps, and rollback process.

Code Scheme: MAJOR.MINOR.PATCH, currently 0.1.0
Datasets: mmc-<animal>-dac9-<num>.tar e.g. mmc-humpback-dac9-00000.tar
Models: <context>-v<N>, e.g. 10k - v2

Packaging: 
yproject.toml defines a buildable package: setuptools backend, name cairn-marine-mammals-communication, version 0.1.0, src* packages, and the extras audio-codec, dataset, and dev. So pip install -e . and python -m build work. But the version does get bumped by hand 

Currently no tagging, changelog generation, publishing, Artifact versioning(kind has it just informal), Rollback

## Support & Contact

If you want to ask questions about the research and codebase, contact Ken Janik at ken.janik@gmail.com.
