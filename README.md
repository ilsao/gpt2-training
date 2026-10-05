# gpt2-training

## Hugging Face Hub

Before training, run `hf auth login` on the training machine with a token that
has write access to `asciibase64/gpt2-c4`. Alternatively, provide the token through
the `HF_TOKEN` environment variable. Do not commit the token to this repository.

`train.py` uploads the model and tokenizer to
https://huggingface.co/asciibase64/gpt2-c4 at each checkpoint save (every 250 steps).
Intermediate uploads are asynchronous; a save may be skipped if the previous
upload is still running. At the end of training, it saves the final model and
tokenizer directly in the run's output directory and waits for the final upload
to complete.
