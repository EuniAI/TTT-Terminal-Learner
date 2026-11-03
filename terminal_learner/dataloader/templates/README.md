# Qwen3 chat template fixture

`qwen3.jinja` is the unmodified `chat_template` field from:

https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/tokenizer_config.json

It is used by the byte-level tiny model and offline parity tests. Production
Qwen3 imports use the template from the selected pretrained tokenizer and save
that exact template in the checkpoint; they do not substitute this fixture.
