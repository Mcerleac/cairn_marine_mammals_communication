# Inherited-Codebase Audit: `cairn_marine_mammals_communication`

**Repository Audited:** `https://github.com/Mcerleac/cairn_marine_mammals_communication` (forked from CAIRN Institute `cairn-marine-mammals-communication` v0.1.0, Apache 2.0)[cite: 1]  
**Audit Period:** Fall Weeks 1–5 (November 2026)  
**Auditors:** Ciaran McErlean, Tiernan Flanagan-Caldwell, Wren Gilbert, Adam Moeglein

---

## 1. What Runs and How, and What Does Not

### 1.1 What Runs Out of the Box
* **Core Model Instantiation, Forward Pass, and KV-Cached Generation (`src/model/`):** `CausalTransformer`, `RoPE`, `CausalSelfAttention`, `SlidingWindowAttention`, `CompressedGlobalAttention`, `MoEFeedForward`, and `checkpoints.py` run cleanly on both CPU and CUDA when invoked with `PYTHONPATH=.`[cite: 1].
* **Checkpoint Inspection & Telemetry Utilities (`scripts/`):**
  * `PYTHONPATH=. python3 scripts/describe_checkpoint.py <ckpt.pt> --json` runs on CPU and reconstructs architecture metadata directly from saved `TransformerConfig` payloads[cite: 1].
  * `PYTHONPATH=. python3 scripts/analyze_checkpoint_weights.py <ckpt.pt>` and `scripts/analyze_model_weights.py` run on CPU and output deduplicated parameter and attention-head statistics[cite: 1].
  * `PYTHONPATH=. python3 scripts/measure_forward_branches.py <ckpt.pt>` runs on CPU using synthetic DAC-style interleaved token IDs when `--tokens` is omitted[cite: 1].
  * `PYTHONPATH=. python3 scripts/draw_model_diagram.py configs/<config>.yaml --output <out.svg>` runs on CPU and generates standalone SVG architecture diagrams[cite: 1].
* **Acoustic Heuristic & Round-Trip Scripts:** `scripts/grade_audio_quality.py` and `scripts/analyze_generation_quality.py` run on CPU over any directory of `.wav`/`.flac` files using `librosa` and `soundfile`[cite: 1].

### 1.2 What Does Not Run Without External Artifacts, Hardware, or Fixes
* **Training (`scripts/train.py`) and Held-Out Checkpoint Evaluation (`scripts/evaluate_checkpoint.py`):** Do not run from a clean clone because `.gitignore` excludes `/data/` and `runs/`[cite: 1]. They require downloading the 28 GB+ tokenized dataset (`data/tokenized/sanctsound_humpback_dac/*.npy` and `chunk_scores.csv`) and `.pt` checkpoints from Hugging Face[cite: 1], or synthesizing local fixtures.
* **Prompted Audio Generation (`scripts/generate_dac_9cb_unified.py`):** Defaults to `--device cuda` and initializes `DACTokenizer(device="cpu")`, which attempts to download ~574 MB of Descript Audio Codec weights via `dac.utils.download(model_type="44khz")`[cite: 1]. Fails if the `dac` optional dependency is not installed or if run offline.
* **SanctSound Preprocessing (`scripts/process_sanctsound_humpback.py`) on Non-Linux OS:** Contains Linux-specific memory guards (`open(f'/proc/{os.getpid()}/status')` and `ctypes.CDLL('libc.so.6').malloc_trim(0)`)[cite: 1] that raise `FileNotFoundError` / `OSError` on macOS and Windows.
* **Blackwell (`sm_120`) GPU Compatibility with TensorFlow Hub Detector:** `load_humpback_detector()` in `scripts/process_sanctsound_humpback.py` uses `tensorflow-cpu>=2.21.0` and explicitly hides GPUs (`tf.config.set_visible_devices([], 'GPU')`) because TensorFlow 2.21 lacks native Blackwell `sm_120` kernels[cite: 1].

---

## 2. State of the Tests and the CI
* **Inherited Automated Tests:** **None (0 test files).** The inherited repository contained no `tests/` directory, no unit tests, no integration tests, and no test runner in `pyproject.toml`[cite: 1].
* **Inherited Continuous Integration:** **None.** There was no `.github/workflows/` directory or external CI configuration[cite: 1].
* **Remediation Implemented in Week 5:** Added `pytest>=8.0` to `pyproject.toml` `[project.optional-dependencies] dev`, created `tests/test_pipeline.py` with 6 CPU unit and integration tests covering `src/model`, `src/data`, `src/training`, and `scripts/`, and configured `.github/workflows/ci.yml` to run on every push and pull request.

---

## 3. Dependencies and Their Age
From `pyproject.toml` (`requires-python = ">=3.10"`), the inherited dependency stack is modern (2024–2026 releases) but split across base and hardware/network-coupled optional groups[cite: 1]:

| Dependency Group | Packages & Version Bounds | Age / Ecosystem Status | Audit Finding |
| :--- | :--- | :--- | :--- |
| **Core (`dependencies`)** | `torch>=2.5.0`, `torchaudio>=2.5.0`, `librosa>=0.10.0`, `soundfile>=0.12.0`, `matplotlib>=3.8.0`, `numpy>=1.26.0`, `scipy>=1.12.0`, `pyyaml>=6.0`, `tqdm>=4.66.0`, `datasets>=2.16.0`, `pandas>=2.1.0`[cite: 1] | Current (2024–2026) | Installs cleanly on Python 3.10–3.12; CPU wheels available for CI. |
| **`[training]`** | `bitsandbytes>=0.43.0`[cite: 1] | Current (2024–2026) | Requires CUDA runtime (`use_8bit_adam: true` in configs)[cite: 1]. Keep optional so CPU CI does not fail. |
| **`[audio-codec]`** | `dac>=0.4.0`, `lac @ git+https://github.com/hugofloresgarcia/lac.git`, `descript-audiotools @ git+https://github.com/hugofloresgarcia/audiotools.git`[cite: 1] | Unpinned Git URLs (2023–2024) | Unpinned `git+https://` dependencies risk upstream breakage; isolated from core CI tests. |
| **`[dataset]`** | `google-cloud-storage>=2.16.0`, `pyloudnorm>=0.1.1`, `tensorflow-cpu>=2.21.0`, `tensorflow-hub>=0.16.1`[cite: 1] | Current (`tf-cpu 2.21`) | Heavy install footprint (~500 MB+); only required when running raw FLAC ingestion. |
| **`[dev]`** | `ipykernel>=6.0`, `ipywidgets>=8.0`, `jupyter>=1.0`[cite: 1] | Current | Missing `pytest`; updated in this audit to include `pytest>=8.0`. |

---

## 4. Open Risks and Known Technical Debt
1. **128K NSA+MoE Semantic Degradation Debt:** As documented in the inherited root `design.md` (Section 8.1 & 9.2), the 128K NSA+MoE model (`375M` params) learns token distributions (`val_loss ~5.388`) but generates low-energy/noise-like continuations compared to the 32K SWA+MoE model (`205M` params), which produces clear humpback vocalizations at 73% of prompt RMS[cite: 1].
2. **Legacy Script Duplication:** `scripts/generate_dac_9cb_prompted.py` and `scripts/generate_dac_9cb_prompted_128k.py` duplicate 90% of `scripts/generate_dac_9cb_unified.py`[cite: 1].
3. **Malformed CSV & Empty Filename Debt:** Historical `chunk_scores.csv` files occasionally contain blank `npy_file` or `detector_score` cells for chunks skipped by the detector (`min_detector_score < 0.1`)[cite: 1].
4. **Hardcoded CUDA Calls in `Trainer`:** `src/training/trainer.py` calls `torch.cuda.empty_cache()` unconditionally inside the training/eval loop[cite: 1].
5. **Open Operational Risks:** Tracked in `docs/risks.md` (Hugging Face artifact hosting sync, GPU VRAM overflow at 128K context, and Linux-only `/proc` calls).

---

## 5. Predecessor Handoff Document & Forward-Looking Retrospective
* **Predecessor Handoff Document (`/design.md` & `blog/model_release_details.html`):** The upstream CAIRN Institute team left two handoff artifacts in the repo: the historical pipeline report (`/design.md`, created `2026-05-15`) and the technical release note (`blog/model_release_details.html`, published `August 2026`)[cite: 1].
* **Forward-Looking Section of Predecessor Retrospective:**
  * In `/design.md` **Section 12 ("Future Directions")**, the predecessor team listed 7 forward-looking items: (1) Scale SanctSound to additional stations (`oc02` orcas, Pacific margin), (2) Multi-station temporal chaining, (3) Species-specific models, (4) Comparative generation against real SanctSound recordings, (5) Investigate 128K NSA+MoE token-distribution vs. semantic structure issues, (6) Online inference system, and (7) Downstream classification/detection tasks[cite: 1].
  * In `blog/model_release_details.html` **("Limits and next measurements")**, the predecessor team explicitly identified the immediate next engineering priorities: *"The next useful additions are dataset-wide next-token calibration, codebook-conditioned accuracy, expert-selection stability, attention-map analysis, and targeted ablations of global refresh layers"*[cite: 1].
* **How Our Team Uses This Handoff:** Our project directly implements the predecessor retrospective's call for **codebook-conditioned accuracy (token-space CB0 accuracy)** and **comparative audio-space spectral loss evaluation** across multiple prompts and model continuations.

---

## 6. Open-Source Project Contribution Rules
* **Upstream License & Attribution Rules:** Governed by the **Apache License 2.0** (`LICENSE`) and `NOTICE` file[cite: 1]. Any modified files must carry prominent modification notices (Apache 2.0 Section 4b), retain all CAIRN Institute copyright and NOAA/NCEI source-data citations (`https://doi.org/10.25921/saca-sp25`), and state that NOAA and NCEI do not endorse derived works[cite: 1].
* **Artifact & Repository Hygiene Rules:** Raw audio (`.wav`, `.flac`, `.mp3`), tokenized datasets (`/data/`), and checkpoints (`runs/`, `checkpoints/`, `models/`) must never be committed to Git (`.gitignore`)[cite: 1].
* **Team Fork Contribution Rules (`CONTRIBUTING.md`):** Because upstream did not include a `CONTRIBUTING.md`, we established formal contribution rules at `/CONTRIBUTING.md`: feature-branch-only workflow, mandatory green GitHub Actions CI (`pytest`), at least one teammate review approval per PR, squash-and-merge history, and backward compatibility with `src.*` imports and released `.pt` checkpoints.

---

## 7. What We Will Change First (And Where Findings Were Recorded)
1. **First Changes Executed (Fall Week 5):**
   * Relocate and update system architecture and the walking skeleton production target in `docs/design.md` (removing misplaced root `design.md`).
   * Add `/CONTRIBUTING.md`, `/AGENTS.md`, `docs/requirements.md` (with CI test acceptance criteria), `docs/risks.md`, and `docs/audit.md`.
   * Add `pytest` to `pyproject.toml`, create `tests/test_pipeline.py` (6 CPU unit/integration tests), add `.github/workflows/ci.yml`, and document the Cold-Clone CPU Quickstart in `README.md`.
2. **Next Changes for Repo Checkpoint 2 (Fall Weeks 6–10):**
   * Implement the unified continuation evaluation CLI (`scripts/evaluate_continuations.py`) computing codebook-0 (CB0) token accuracy and multi-resolution STFT spectral loss to deploy the Walking Skeleton end-to-end.
3. **Where Audit Findings Were Placed for the Next Developer:**
   * **Constraints (`docs/requirements.md` Section 3):** Recorded checkpoint schema compatibility, `9,219` vocab invariants, 16 GB VRAM hardware limits, and NOAA/CAIRN licensing constraints.
   * **Conventions & Pitfalls (`CONTRIBUTING.md` & `AGENTS.md`):** Recorded `PYTHONPATH=.` import rules, `DACTokenizer` weight download behavior, Linux `/proc` pitfalls, and `.gitignore` binary rules.
   * **Keep/Replace Decisions (`docs/design.md` Section 3):** Recorded the decisions to keep `CausalTransformer` and `generate_dac_9cb_unified.py`, deprecate legacy generation scripts, and add automated CI and unified evaluation.
