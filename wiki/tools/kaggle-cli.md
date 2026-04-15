---
title: Kaggle CLI Usage
category: tools
tags: [kaggle, cli, notebook, submission]
created: 2026-04-15
updated: 2026-04-15
---

# Kaggle CLI Usage

## Setup

```bash
pip install kaggle
# Place kaggle.json in ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## Notebook Push Workflow

### 1. Create kernel-metadata.json

```json
{
  "id": "username/kernel-slug",
  "title": "Notebook Title",
  "code_file": "notebook.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": false,
  "dataset_sources": [],
  "competition_sources": ["competition-slug"],
  "kernel_sources": []
}
```

### 2. Push

```bash
kaggle kernels push -p /path/to/folder/
```

### 3. Monitor Status

```bash
kaggle kernels status username/kernel-slug
# Returns: KernelWorkerStatus.RUNNING, .COMPLETE, or .CANCEL_ACKNOWLEDGED
```

### 4. Download Output

```bash
kaggle kernels output username/kernel-slug -p /tmp/output/
```

### 5. Submit

```bash
kaggle competitions submit -c competition-slug \
  -f /tmp/output/submission.csv \
  -m "Description of this submission"
```

### 6. Check Scores

```bash
kaggle competitions submissions -c competition-slug
kaggle competitions leaderboard competition-slug -s
```

## Common Issues

- **403 on status check**: Kernel slug doesn't match. The slug in the URL may differ from the `id` in metadata if Kaggle auto-renamed it.
- **Status stays RUNNING for hours**: Kaggle queue time can be significant. The 12h execution limit starts when the kernel actually begins running, not when queued.
- **CANCEL_ACKNOWLEDGED**: Kernel exceeded time limit or errored. Output may still be available if the notebook saved files before being killed.

## See Also
- [[tools/kaggle-cpu-notebooks]]
