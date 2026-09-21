Code used in the Deep Ignorance paper.

Includes an improved version of Circuit Breakers that uses a better loss coefficient scheduler.

## Finetune Attack

Run a tampering/finetuning attack on an unlearned model to recover bio knowledge:

```bash
python unlearn/reference/cas/finetune_attack.py \
  --model_name <path_to_unlearned_model> \
  --num_train_examples 512 \
  --eval_every 50 \
  --epochs 1
```
