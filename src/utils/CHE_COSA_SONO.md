# CHE_COSA_SONO — src/utils

> Utility trasversali, usate un po' ovunque nella pipeline.

| File | Che cos'è |
|---|---|
| `config.py` | Caricamento delle config YAML (`load_config`). |
| `seeds.py` | Seeding deterministico (`set_seed`) per riproducibilità. |
| `freq.py` | Compatibilità degli alias di frequenza pandas (es. `"h"` → `"H"`) con lo stack pesante GluonTS usato da M2/M3. |
| `__init__.py` | API pubblica del package utils. |
