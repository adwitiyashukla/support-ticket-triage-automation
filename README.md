# support-ticket-triage-automation

A bank support desk gets around 4,200 tickets a day. Right now a person reads each one, tags it, sends it to the right team and takes a standard action. This project does that job with two models I trained myself and a policy layer that decides when a person still has to look.

Everything is driven by `spec.toml`. That file holds the intent list, which team each intent belongs to, what actions each team may run, the confidence thresholds and the accuracy floor the client asked for. Point the same code at a different spec and you get a different model for a different client.

## The two models

The screener is a decoder only transformer I wrote from scratch in PyTorch. RMSNorm, rotary position embeddings, SwiGLU, tied embeddings. It uses its own byte level BPE tokenizer, also written from scratch and trained on the ticket text. 5.7M parameters. First it is pretrained on the tickets with next token prediction, then fine tuned to pick one of 77 intents or say the ticket is out of scope. It answers in about 6 milliseconds on a CPU.

The resolver is Qwen2.5-0.5B-Instruct post trained with LoRA. It returns one JSON object holding the intent, the entities it found, the action and a draft reply. It only sees the tickets the screener is unsure about.

If neither model is confident enough, the ticket goes to a person.

## The policy layer

The model never picks the queue or the action. It picks the intent, and `spec.toml` turns that into a queue, a priority and the actions that queue is allowed to run. Two queues are marked as never automatic. Identity checks always go to a person. Fraud freezes the card straight away and then goes to a person.

The action the resolver writes is checked against the actions the policy allows, and the policy wins. Every executed action goes into an audit log with the confidence and the model that decided it. Tools are idempotent per ticket, so replaying a ticket does not open a second case.

## Data

banking77 for the real customer messages, 10,003 train and 3,080 test rows over 77 intents. The clinc out of scope set for messages that are not banking at all. Both are downloaded by the prepare step.

Cleaning removed 31 exact duplicates, 25 training rows that also appear in the test set, 31 near duplicates inside training and 15 more training rows that MinHash matched against test rows. What is left is 8,814 train, 1,284 validation and 4,078 test rows.

The tokenizer ends up with 3,425 tokens from 3,166 merges. On held out tickets that is 4.19 characters per token and 1.19 tokens per word.

## Results

Test set, 4,078 rows.

| model | accuracy | macro F1 |
| --- | --- | --- |
| majority class | 0.98% | 0.02% |
| TF-IDF nearest neighbour | 65.11% | 70.16% |
| TF-IDF and logistic regression | 87.54% | 87.99% |
| screener, trained from scratch | **89.68%** | **87.86%** |

The 95 percent bootstrap interval on screener accuracy is 88.67 to 90.55.

Out of scope detection is the part I care about most, because that is where a wrong automatic answer costs the client. Recall 93.49 percent, precision 98.00 percent, AUROC 0.9978.

Calibration matters too, since the routing threshold is only meaningful if the confidence means something. Temperature scaling fitted on validation gives T = 0.96 and expected calibration error of 0.0225 on test.

Speed on two CPU cores: 247 tickets per second batched, and 5.6 ms median for a single ticket with 8.2 ms at p95. The client budget is 800 ms.

## Does pretraining from scratch help

Same model, same data, same 12 fine tuning epochs. The only difference is whether the 8 epochs of next token pretraining on the ticket corpus happen first.

| run | validation accuracy | calibration error |
| --- | --- | --- |
| pretrain then fine tune | **89.56%** | 0.0313 |
| fine tune only | 87.69% | 0.0431 |

Worth 1.87 points and a better calibrated model, on a corpus of only about 140,000 tokens.

## How much of the queue it takes over

The threshold sweep is the part a client actually buys. Higher threshold means fewer tickets handled without a person, but higher accuracy on the ones that are.

| confidence threshold | tickets handled with no person | accuracy on those |
| --- | --- | --- |
| 0.50 | 66.38% | 90.54% |
| 0.75 | 61.99% | 93.71% |
| 0.85 | 59.37% | 94.84% |
| 0.90 | 56.55% | 96.27% |
| 0.95 | 49.22% | 97.76% |

The spec asks for 95 percent accuracy on anything handled automatically. At the 0.90 threshold in `spec.toml`, the screener alone clears that with 56.55 percent of the queue taken over at 96.27 percent accuracy. Another 1,240 tickets are pushed to a person because their queue is not allowed to be automatic, which is the policy doing its job rather than the model failing.

Using the client numbers in `spec.toml`, 4,200 tickets a day at 4.5 minutes each is 315 agent hours a day. Taking over 56.55 percent of that removes 178 hours a day. Compute for the screener path costs about 0.04 cents per 1,000 tickets. Sending the same tickets to a frontier API with the full taxonomy in the prompt would cost about 5.10 dollars per 1,000 tickets at 3 and 15 dollars per million tokens, which is the assumption sitting in `config.toml` rather than a quote.

## Which intents it mixes up

The confusions are the pairs a new human agent would also mix up. why_verify_identity against verify_my_identity 11 times, pending_transfer against failed_transfer 6 times, fiat_currency_support against exchange_via_app 6 times. Those three intents have the lowest F1 in the whole taxonomy.

## Running it

```
pip install -r requirements.txt
python -m triage prepare
python -m triage train-screener
python -m triage baselines
python -m triage evaluate
```

Screener training takes about 26 minutes on two CPU cores and much less on a GPU. The resolver wants a GPU:

```
python -m triage train-resolver
python -m triage evaluate --resolver
```

Run the queue over a file of tickets, or start the service:

```
python -m triage queue --limit 200
python -m triage serve
```

The service exposes `POST /triage`, `POST /triage/batch`, `GET /healthz` and `GET /metrics`.

Tests:

```
python -m pytest tests -q
```

66 tests. They cover the config and spec validation, split leakage, tokenizer round trips, causal masking and padding invariance in the transformer, the routing thresholds, tool idempotency and the API. The resolver tests build a tiny local base model so they run without downloading anything.

## Layout

`spec.toml` is the client requirement. `config.toml` holds every tunable with validation on load. `triage/` has the tokenizer, the transformer, both training loops, the router, the tools, the evaluation and the service. `python -m triage` is the single entry point.
