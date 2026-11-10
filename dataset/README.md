# Dataset — Bitext customer support tickets

`customer_support_data.csv` · 26,872 rows · 18.3 MiB (19,229,347 bytes)

```
SHA-256  6cf0433ed3eb7c8fd8d5afa1a6de0aa482945a1f77f712c63a7b673cd4bb8569
```

## Source

Downloaded from the Hugging Face Hub:

**[`bitext/Bitext-customer-support-llm-chatbot-training-dataset`](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)**

The `train` split was converted to CSV and saved locally to allow offline work
(see the first code cell of the notebook):

```python
import pandas as pd
from datasets import load_dataset

dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
pd.DataFrame(dataset["train"]).to_csv("customer_support_data.csv", index=False)
```

Because the file is a pandas re-export rather than the original Hub file, its byte size
and line endings differ from the upstream download even though the rows are the same.

## Schema

| Column | Type | Used here | Notes |
|---|---|---|---|
| `flags` | string | no | 2–4 letter marker for the paraphrase template used |
| `instruction` | string | **input** | The customer's message |
| `category` | string | **label** | 11 values: ACCOUNT, CANCEL, CONTACT, DELIVERY, FEEDBACK, INVOICE, ORDER, PAYMENT, REFUND, SHIPPING, SUBSCRIPTION |
| `intent` | string | no | 27 fine-grained intents (e.g. `cancel_order`, `get_refund`) |
| `response` | string | no | Reference agent reply |

Class counts in the file as committed:

| Category | Rows | | Category | Rows |
|---|---:|---|---|---:|
| ACCOUNT | 5,986 | | ORDER | 3,988 |
| REFUND | 2,992 | | INVOICE | 1,999 |
| CONTACT | 1,999 | | PAYMENT | 1,998 |
| FEEDBACK | 1,997 | | DELIVERY | 1,994 |
| SHIPPING | 1,970 | | SUBSCRIPTION | 999 |
| CANCEL | 950 | | **total** | **26,872** |

There are no null values, 2,237 duplicated `instruction` values, and no duplicate rows.

## About the content

The data is **synthetic**. Messages are generated from templates and paraphrases, and
placeholders such as `{{Order Number}}` or `{{Customer Support Hours}}` are left in the
text. The only e-mail addresses present are obvious placeholders — 55 in total, all on
`example.com` (26), `company.com` (28) or `companyname.com` (1). No real person or
organisation is referenced.

**It contains no real customer records**, so it is safe to commit to a public repository.

## Licence

The dataset is released under the **Community Data License Agreement – Sharing, Version 1.0
(CDLA-Sharing-1.0)**, © Bitext. The full text is in
[`LICENSE-CDLA-Sharing-1.0.md`](LICENSE-CDLA-Sharing-1.0.md).

In short: you may use, share and modify the data, provided you keep this notice and
distribute any derived data under the same agreement.

If you use this dataset, credit Bitext and link the Hugging Face page above.
