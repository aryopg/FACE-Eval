# Vulture whitelist — false positives that should not be flagged.
# Logger.progress() description parameter is part of the public API.

description  # noqa

# run_judge_stage stub params — kept for API stability until agentic judge is implemented.
judge_config_path  # noqa
concurrency  # noqa

# test_inspect_task.py helper — parameter unused in body but part of the function signature.
a_explanation  # noqa

# InklingClient._chat_template_kwargs override — arg matches the base VLLMClient signature.
enable_thinking  # noqa
