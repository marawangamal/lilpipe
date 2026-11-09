"""Run pre-rendered text tasks with lm-eval 0.4.11 and vLLM 0.23."""

import runpy


def raw_prompt_chat_template(*, tokenizer, chat_template=None, tools=None, **kwargs):
    """Resolve only the unused default template for our pre-rendered text prompts."""
    if chat_template is not None or tools:
        raise ValueError("This launcher supports pre-rendered text prompts only")
    return tokenizer.chat_template


def configure_backend():
    import vllm.entrypoints.chat_utils as chat_utils

    # lm-eval imports this removed helper even without --apply_chat_template.
    if not hasattr(chat_utils, "resolve_hf_chat_template"):
        chat_utils.resolve_hf_chat_template = raw_prompt_chat_template


if __name__ == "__main__":
    configure_backend()
    runpy.run_module("lm_eval", run_name="__main__")
