# support-ticket-triage-automation

A bank support desk gets around 4,200 tickets a day. Right now a person reads each one, tags it, sends it to the right team and takes a standard action. This project does that job with two models I trained myself and a policy layer that decides when a person still has to look.

Everything is driven by `spec.toml`. That file holds the intent list, which team each intent belongs to, what actions each team may run, the confidence thresholds and the accuracy floor the client asked for. Point the same code at a different spec and you get a different model for a different client.

## The two models

The screener is a decoder only transformer I wrote from scratch in PyTorch. RMSNorm, rotary position embeddings, SwiGLU, tied embeddings. It uses its own byte level BPE tokenizer, also written from scratch and trained on the ticket text. 5.7M parameters. First it is pretrained on the tickets with next token prediction, then fine tuned to pick one of 77 intents or say the ticket is out of scope.

The resolver is Qwen2.5-0.5B-Instruct post trained with LoRA, 8.8M trainable parameters out of 502.8M. It returns one JSON object holding the intent, the entities it found, the action and a draft reply. It only sees the tickets the screener is unsure about.

If neither model is confident enough, the ticket goes to a person.

## The policy layer

The model never picks the queue or the action. It picks the intent, and `spec.toml` turns that into a queue, a priority and the actions that queue is allowed to run. Two queues are marked as never automatic. Identity checks always go to a person. Fraud freezes the card straight away and then goes to a person.

The action the resolver writes is checked against the actions the policy allows, and the policy wins. Across the whole test run the two agreed 100 percent of the time, so the check never had to override the model, but it is there for when it does. Every executed action goes into an audit log with the confidence and the model that decided it. Tools are idempotent per ticket, so replaying a ticket does not open a second case.

## Data

banking77 for the real customer messages, 10,003 train and 3,080 test rows over 77 intents. The clinc out of scope set for messages that are not banking at all. Both are downloaded by the prepare step.

Cleaning removed 31 exact duplicates, 25 training rows that also appear in the test set, 31 near duplicates inside training and 15 more training rows that MinHash matched against test rows. What is left is 8,814 train, 1,284 validation and 4,078 test rows.

The tokenizer ends up with 3,425 tokens from 3,166 merges. On held out tickets that is 4.19 characters per token and 1.19 tokens per word.

## Results

Everything below was measured on a Kaggle T4. The screener trained in 118 seconds, the resolver in 28 minutes for one epoch.

Test set, 4,078 rows.

| model | accuracy | macro F1 |
| --- | --- | --- |
| majority class | 0.98% | 0.02% |
| TF-IDF nearest neighbour | 65.11% | 70.16% |
| TF-IDF and logistic regression | 87.54% | 87.99% |
| screener, trained from scratch | **87.59%** | **87.04%** |

The classical baseline is right behind a transformer I wrote from scratch, which is worth knowing before anyone reaches for a model.

Out of scope detection is the part I care about most, because that is where a wrong automatic answer costs the client. AUROC 0.9954.

Calibration matters too, since the routing threshold is only meaningful if the confidence means something. Expected calibration error is 0.0284 after temperature scaling fitted on validation.

The threshold sweep picks 0.925 as the lowest confidence gate that holds the client's 95 percent floor. At that gate the screener alone handles 57.75 percent of the queue at 95.97 percent accuracy.

## What the resolver adds

Routing every deferred ticket to the resolver, measured on 1,500 test tickets:

| setup | share handled with no person | accuracy on those |
| --- | --- | --- |
| screener alone at its chosen gate | 57.75% | 95.97% |
| screener plus resolver | **68.67%** | 90.68% |

So the resolver buys about 11 points of extra coverage and gives up more accuracy than the spec allows. The client floor is 95 percent and this setup lands at 90.68, which means at this spec the resolver's answers go to a person for review rather than straight through, or its confidence gate moves well above the 0.70 in `spec.toml`.

The resolver itself is solid on its own terms. It produced valid JSON on 98.44 percent of a validation sample, needed zero repair attempts across 271 calls in the test run, and its intent accuracy on that sample was 87.3 percent. The tickets it sees are the ones the screener could not call, so its share of the blended number is doing the hard part.

Latency splits the same way. A ticket the screener answers takes 6.8 ms at the median. A ticket that reaches the resolver takes about 2.5 seconds, so p95 across the mix is 2,486 ms against the client's 800 ms budget. The cheap tier is inside the budget and the expensive one is not, which is the tradeoff you take on when a 0.5B model joins the path.

## Cost

Using the client numbers in `spec.toml`, 4,200 tickets a day at 4.5 minutes each is 315 agent hours a day. Taking over 68.67 percent of that removes 216 hours a day. Compute costs about half a cent per 1,000 tickets. Sending the same tickets to a frontier API with the full taxonomy in the prompt would cost about 5.10 dollars per 1,000 tickets at 3 and 15 dollars per million tokens, which is the assumption sitting in `config.toml` rather than a quote. The fine tuned resolver needs 52.8 prompt tokens per ticket because the taxonomy lives in its weights instead of the prompt.

## The same thing on a laptop CPU

I also ran the screener path end to end on my own laptop, an Intel Core 5 120U with no GPU. Training took 94 minutes there against 118 seconds on the T4. Serving is the interesting part: 108 tickets per second batched, 21.9 ms median for a single ticket and 29.5 ms at p95, which is well inside the client budget.

That run scored 85.73 percent on the test set against 87.59 on the T4, same config and same seed. Floating point reduction order changes with thread count, so expect a point or two of spread between machines.

## Running it

```
pip install -r requirements.txt
python -m triage prepare
python -m triage train-screener
python -m triage baselines
python -m triage evaluate
```

The resolver wants a GPU:

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
