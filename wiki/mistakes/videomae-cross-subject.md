---
title: "Mistake: VideoMAE Features Destroy Cross-Subject Generalization"
category: mistakes
tags: [videomae, video, cross-subject, generalization, multimodal]
created: 2026-04-16
updated: 2026-04-16
---

# Mistake: VideoMAE Features Destroy Cross-Subject Generalization

**What happened:** Integrated pretrained VideoMAE features (768-dim) into PatchTST as an additional token in self-attention. OOF improved dramatically (0.5204 → 0.6087) but LB collapsed (0.5858 → 0.1321).

**The pattern:** VideoMAE features encode subject-specific and environment-specific visual information (the person's appearance, surroundings, camera viewpoint). Within the training population, these features distinguish activities well. But for new test subjects in new environments, the features are out-of-distribution noise.

| Version | Approach | OOF F1 | LB F1 | Gap |
|---------|----------|--------|-------|-----|
| V10 | PatchTST inertial only | 0.5204 | 0.5858 | +0.065 |
| V13 | GBM + VideoMAE RP | 0.5901 | 0.2432 | -0.347 |
| V15 | PatchTST + VideoMAE token | 0.6087 | 0.1321 | -0.477 |

**Key insight:** Adding video features made things WORSE than GBM alone (0.13 vs 0.24). The transformer learned to attend heavily to the video token, and when those features were OOD at test time, it collapsed to predicting 61% null class.

**The rule:** For cross-subject HAR with pretrained video features:
- Do NOT feed raw pretrained video features to any model
- The only approach that generalizes is learned representations on raw inertial data
- Video features would need domain adaptation or subject-invariant extraction to be useful

**Untested alternatives that might work:**
- Video dropout (90%+ rate) to prevent model relying on video
- Domain-adversarial training to force subject-invariant video representations
- Multi-task learning: video classifier as auxiliary loss, discard at test time
- Contrastive learning: pull same-activity video embeddings together across subjects

## See Also
- [[patterns/cross-subject-generalization]]
- [[patterns/oof-vs-lb-divergence]]
- [[mistakes/gbm-for-cross-subject-har]]
- [[techniques/patchtst]]
