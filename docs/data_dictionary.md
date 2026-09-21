# Data dictionary and feature availability

The raw columns originate from the UCI Bank Marketing dataset. The operational model uses only information that is available before a new call begins.

| Field group | Fields | Availability | Model treatment |
|---|---|---|---|
| Customer profile | `age`, `job`, `marital`, `education`, `default`, `housing`, `loan` | Pre-contact | Included |
| Campaign planning | `contact`, `month`, `day_of_week` | Pre-contact | Included |
| Previous-contact history | `pdays`, `previous`, `poutcome`, `had_previous_contact` | Pre-contact | Included; the last field is derived from `previous` |
| Economic context | `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed` | Pre-contact for the historical campaign period | Included, with temporal validation caveat |
| Current-call outcomes | `duration`, `campaign` | Known during or after the current call | Excluded from operational model |
| Target and descriptive helpers | `y`, `success`, `age_group`, `duration_category` | Outcome or derived analysis field | Excluded |

`duration` is intentionally excluded because it is recorded after a call begins. `campaign` is also excluded because the number of contacts in the current campaign is not known before the first call. The source-order temporal holdout is a proxy for forward validation because the dataset does not include a complete call timestamp.
