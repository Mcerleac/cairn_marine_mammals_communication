# Multi-GPU training

`scripts/train.py` runs on one GPU as before, or on any number of GPUs with
PyTorch DistributedDataParallel (DDP) when started by `torchrun`. No GPU count
is hardcoded: each process reads `WORLD_SIZE`, `RANK` and `LOCAL_RANK` from the
launcher.

```bash
# one GPU (unchanged)
PYTHONPATH=. python3 scripts/train.py configs/<config>.yaml

# 8 GPUs on one node
PYTHONPATH=. torchrun --nproc_per_node=8 scripts/train.py configs/<config>.yaml
```

From an sbatch script, request the GPUs and start one `torchrun` per node, e.g.
`srun torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE scripts/train.py <config>`.
For more than one node, also pass `--nnodes`, `--node_rank` (or a c10d
rendezvous) and a shared `output_dir`.

All released configs work unchanged, and checkpoints are identical in format to
single-GPU ones: a run can be resumed on a different number of GPUs.

## What changes under DDP

Each GPU holds a full copy of the model and trains on its own share of the
data (`DistributedSampler`). Gradients are averaged across GPUs before every
optimizer update, so all copies stay identical. Only rank 0 prints, writes logs
and saves checkpoints.

**Batch size.** By default (`keep_effective_batch: true`) the YAML's
`grad_accumulation_steps` is split across GPUs, so the number of sequences per
optimizer update matches a single-GPU run:

| GPUs | grad accumulation per GPU | sequences per update (batch_size 1, YAML accumulation 8) |
|------|---------------------------|----------------------------------------------------------|
| 1    | 8                         | 8                                                        |
| 2    | 4                         | 8                                                        |
| 4    | 2                         | 8                                                        |
| 8    | 1                         | 8                                                        |

The GPU count must divide `grad_accumulation_steps` evenly. With
`keep_effective_batch: false`, every GPU keeps the YAML value and the
effective batch grows with the GPU count (consider adjusting the learning rate).

**Steps.** `step` (in logs, checkpoints, `warmup_steps`, `log_interval`,
`eval_interval`, `save_interval`) is counted in single-GPU units: one batch on
one GPU. A run on N GPUs advances N steps per iteration. The LR schedule and
intervals therefore cover the same amount of data whatever the GPU count.

**Evaluation.** The validation set is split across GPUs and the loss is
averaged over all of them. `max_eval_batches` is a total, divided across GPUs.

**Mixture-of-experts.** Experts that receive no tokens in a step get no
gradient. DDP is told to expect this (`find_unused_parameters`, on
automatically for MoE models), at a small speed cost.

## Optional `training:` keys

| key                          | default | meaning                                                      |
|------------------------------|---------|--------------------------------------------------------------|
| `keep_effective_batch`       | `true`  | split accumulation across GPUs (see above)                   |
| `ddp_find_unused_parameters` | auto    | force on/off; auto = on for MoE models                       |
| `num_workers`                | `4`     | data-loader workers **per GPU process** (8 GPUs = 32 workers) |
