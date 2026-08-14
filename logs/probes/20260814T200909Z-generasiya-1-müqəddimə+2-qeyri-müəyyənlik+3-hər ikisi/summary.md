# Generasiya qolları (dev, saxlanmış kontekst)

`probe_id`: `20260814T200909Z-generasiya-1-müqəddimə+2-qeyri-müəyyənlik+3-hər ikisi`

```bash
python tools/generation_probe.py --run 20260812T094516Z-dev-baseline
```

| case | 0-nəzarət | 1-müqəddimə | 2-qeyri-müəyyənlik | 3-hər ikisi |
|---|---|---|---|---|
| dev_false_premise_free_cache | imtina (3/3) | imtina (3/3) | imtina (3/3) | imtina (3/3) |
| dev_ambiguous_limit | tek_oxunus (3/3) | tek_oxunus (3/3) | tek_oxunus (3/3) | tek_oxunus (3/3) |
| dev_out_of_corpus_graphql | imtina (3/3) | imtina (3/3) | imtina (3/3) | imtina (3/3) |
