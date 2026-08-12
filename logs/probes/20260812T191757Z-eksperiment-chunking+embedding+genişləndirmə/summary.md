# Chunking / embedding / sorğu genişləndirmə (dev)

`probe_id`: `20260812T191757Z-eksperiment-chunking+embedding+genişləndirmə`

```bash
python tools/retrieval_experiments.py --workdir <müvəqqəti-qovluq>
```

| eksperiment | indeks chunk | örtük | sızma | retrieval büdcəsi |
|---|---|---|---|---|
| baseline (800/200) | 19 | 7/8 | 1/2 | 48 |
| chunking 500/150 | 29 | 8/8 | 1/2 | 48 |
| chunking 400/120 | 35 | 8/8 | 1/2 | 48 |
| chunking 300/90 | 47 | 8/8 | 2/2 | 48 |
| chunking 250/60 | 54 | 8/8 | 2/2 | 48 |
| embedding 3-large | 19 | 7/8 | 1/2 | 48 |
| 3-large + chunking 300/90 | 47 | 8/8 | 1/2 | 48 |
| genişləndirmə (büdcə bərabər) | 19 | 8/8 | 1/2 | 46 ↓ büdcə az |
| genişləndirmə (büdcə sərbəst) | 19 | 8/8 | 1/2 | 64 ✗ BÜDCƏ ARTIQ |

> ⚠️ **Büdcə ARTIQ** (genişləndirmə (büdcə sərbəst)): bu sətirlər baseline-dan (48) ÇOX retrieval büdcəsi ilə ölçülüb, ona görə üstünlükləri dəyişiklikdən deyil, əlavə chunk-dan gələ bilər və eyni cədvəldə müqayisə edilə bilməz.

> Baseline retrieval büdcəsi: **48** çəkilən chunk. «↓ büdcə az» işarəsi problem deyil: eyni örtüyü daha ucuz almaq nəticəni gücləndirir.
