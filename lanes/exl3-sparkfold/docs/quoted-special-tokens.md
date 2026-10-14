# Special tokens quoted in text

Patch 0001 (a). TensorFold's chat template turns reserved strings in message text into the model's real special tokens. A fetched web page, a
tool result or a pasted file that merely mentions them can therefore change what the model sees.

## What we saw

- An image added in the second turn of a chat whose first turn fetched a page got HTTP 400, "The rendered prompt must contain exactly one
  image marker for each image". The page text contained the literal `<|vision_start|><|image_pad|><|vision_end|>`; the history replay then
  had two markers for one image. Without an image the same literal was silently tokenized to the real special tokens (73 prompt tokens became
  77), so the model could not even see the text.
- Every one of `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, `<think>`, `</think>`, `<tool_call>`, `</tool_call>` in text is tokenized to a
  single real token. Effects seen:
  - quoting `<|im_end|>` in an answer cut the reply there (`finish_reason=stop`);
  - quoting `</think>` in the reasoning cut the reasoning at it, the rest leaked into `content` and the final answer was lost;
  - a pasted `<|im_end|><|im_start|>system ...` was obeyed in one probe (with a zero-width space after `<|` the model called it a prompt
    injection; a more document-like injection was resisted both ways, so the effect depends on the text);
  - a fetched page quoting a `<tool_call>` block, asked to be copied, made the server return `finish_reason=tool_calls` with a real call.

## The fix

Fourteen strings are escaped in every string of the messages and the tool definitions, by writing a zero-width space (U+200B) after the
opening `<`: `<|image_pad|> <|video_pad|> <|vision_start|> <|vision_end|> <|vision_pad|> <|im_start|> <|im_end|> <|endoftext|> <think>
</think> <tool_call> </tool_call> <tool_response> </tool_response>`. Escaping alone made the model echo the invisible character when quoting
the text, so the model's own copies are restored on the way out, in three ordered steps in `cuda/server.py`:

1. after the reply is split into reasoning and answer, restore everything except the tool tags (restoring `</think>` before the split would
   turn an escaped quote back into a split point);
2. the tool-call parsers read text in which the tool tags are still escaped, so an escaped copy is never mistaken for a call;
3. then restore everything: the visible content, the parsed calls' arguments and the streamed argument fragments.

A trap on the way: the first edit went into `server/prompts.py::prepare_prompt`, which the CUDA server does not use (it has its own
`_prepare`); a live check showed no change and caught it.

Verified live: the failing shape (marker in an earlier tool result plus an image turn) gives 200 with a correct description; quoting each
string returns it exactly, streamed and not, with thinking on and off; ordinary `<div>` and `a<b` text is unchanged. Tests:
`tests/test_media_marker_defuse.py` (6, including one with the real tokenizer showing the literal is special without the fix).

## Limits

- A model that decides on its own to emit real tool-call tags still produces a call: asked to "copy the excerpt exactly", it returned a real
  `tool_calls` even with the page escaped. The escape stops the parser mistaking an escaped copy for a call; it cannot stop the model's choice.
- The escaped text made the model refuse one edit as "prompt injection" when a file's prose contained `<|im_end|>`-style strings. Realistic
  code edits worked, but a quirky refusal is possible on files about chat templates.
- A quote cut off by `max_tokens` in the middle of a string stays escaped.
- `<function=`, `<parameter=` and other text that is not a token are untouched.
- This changes the text the model is shown. SparkFold's contribution guide asks that a resumed prompt equal a fresh one and that no output
  bits change without saying so: this does change the input for these strings, on purpose.
