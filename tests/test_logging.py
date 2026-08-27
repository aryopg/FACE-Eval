from __future__ import annotations

from io import StringIO

from rich.console import Console
from rich.progress import Progress

from src.utils import logging as logging_module
from src.utils.logging import Logger, get_logger, set_verbose


def _make_logger(verbose: bool = True) -> tuple[Logger, StringIO]:
    """Create a Logger whose output is captured to a StringIO buffer."""
    buf = StringIO()
    logger = Logger(verbose=verbose)
    logger.console = Console(file=buf, force_terminal=True, width=120)
    return logger, buf


# ---------- verbose / quiet behavior ----------


class TestLoggerVerbose:
    def test_info_shown_in_verbose_mode(self):
        logger, buf = _make_logger(verbose=True)
        logger.info("hello world")
        assert "hello world" in buf.getvalue()

    def test_info_suppressed_in_quiet_mode(self):
        logger, buf = _make_logger(verbose=False)
        logger.info("hidden message")
        assert buf.getvalue() == ""

    def test_info_force_overrides_quiet(self):
        logger, buf = _make_logger(verbose=False)
        logger.info("forced info", force=True)
        assert "forced info" in buf.getvalue()

    def test_success_shown_in_verbose_mode(self):
        logger, buf = _make_logger(verbose=True)
        logger.success("it worked")
        assert "it worked" in buf.getvalue()

    def test_success_suppressed_in_quiet_mode(self):
        logger, buf = _make_logger(verbose=False)
        logger.success("secret success")
        assert buf.getvalue() == ""

    def test_success_force_overrides_quiet(self):
        logger, buf = _make_logger(verbose=False)
        logger.success("forced success", force=True)
        assert "forced success" in buf.getvalue()

    def test_warning_shown_in_quiet_mode_by_default(self):
        """warning() has force=True by default, so it always shows."""
        logger, buf = _make_logger(verbose=False)
        logger.warning("watch out")
        assert "watch out" in buf.getvalue()

    def test_warning_suppressed_when_force_false_and_quiet(self):
        logger, buf = _make_logger(verbose=False)
        logger.warning("optional warning", force=False)
        assert buf.getvalue() == ""

    def test_warning_shown_in_verbose_mode(self):
        logger, buf = _make_logger(verbose=True)
        logger.warning("verbose warning")
        assert "verbose warning" in buf.getvalue()

    def test_error_always_shown_in_quiet_mode(self):
        logger, buf = _make_logger(verbose=False)
        logger.error("something broke")
        assert "something broke" in buf.getvalue()

    def test_error_always_shown_in_verbose_mode(self):
        logger, buf = _make_logger(verbose=True)
        logger.error("bad thing")
        assert "bad thing" in buf.getvalue()

    def test_header_shown_in_verbose_mode(self):
        logger, buf = _make_logger(verbose=True)
        logger.header("My Header")
        assert "My Header" in buf.getvalue()

    def test_header_suppressed_in_quiet_mode(self):
        logger, buf = _make_logger(verbose=False)
        logger.header("Hidden Header")
        assert buf.getvalue() == ""

    def test_rule_shown_in_verbose_mode(self):
        logger, buf = _make_logger(verbose=True)
        logger.rule("section")
        output = buf.getvalue()
        assert "section" in output

    def test_rule_suppressed_in_quiet_mode(self):
        logger, buf = _make_logger(verbose=False)
        logger.rule("hidden rule")
        assert buf.getvalue() == ""


# ---------- table ----------


class TestLoggerTable:
    def test_table_renders_title_and_data(self):
        logger, buf = _make_logger(verbose=True)
        logger.table("Stats", {"accuracy": "0.95", "loss": "0.12"})
        output = buf.getvalue()
        assert "Stats" in output
        assert "accuracy" in output
        assert "0.95" in output
        assert "loss" in output
        assert "0.12" in output

    def test_table_converts_values_to_str(self):
        logger, buf = _make_logger(verbose=True)
        logger.table("Numbers", {"count": 42, "ratio": 3.14})
        output = buf.getvalue()
        assert "42" in output
        assert "3.14" in output

    def test_table_suppressed_in_quiet_mode(self):
        logger, buf = _make_logger(verbose=False)
        logger.table("Hidden Table", {"key": "value"})
        assert buf.getvalue() == ""

    def test_table_force_overrides_quiet(self):
        logger, buf = _make_logger(verbose=False)
        logger.table("Forced Table", {"key": "value"}, force=True)
        output = buf.getvalue()
        assert "Forced Table" in output
        assert "key" in output

    def test_table_empty_dict(self):
        logger, buf = _make_logger(verbose=True)
        logger.table("Empty", {})
        output = buf.getvalue()
        assert "Empty" in output


# ---------- progress ----------


class TestLoggerProgress:
    def test_progress_returns_progress_object(self):
        logger, _buf = _make_logger(verbose=True)
        prog = logger.progress("Working...")
        assert isinstance(prog, Progress)

    def test_progress_usable_as_context_manager(self):
        logger, _buf = _make_logger(verbose=True)
        prog = logger.progress("Loading...")
        with prog:
            task = prog.add_task("Loading...", total=10)
            prog.update(task, advance=5)
        # No exception means it works as a context manager.


# ---------- print ----------


class TestLoggerPrint:
    def test_print_outputs_text(self):
        logger, buf = _make_logger(verbose=True)
        logger.print("raw output")
        assert "raw output" in buf.getvalue()

    def test_print_works_in_quiet_mode(self):
        """print() is a direct passthrough - no verbose gating."""
        logger, buf = _make_logger(verbose=False)
        logger.print("always visible")
        assert "always visible" in buf.getvalue()


# ---------- get_logger / set_verbose ----------


class TestGetLoggerSetVerbose:
    def setup_method(self):
        """Reset the global logger before each test."""
        logging_module._logger = None

    def test_get_logger_returns_logger_instance(self):
        logger = get_logger()
        assert isinstance(logger, Logger)

    def test_get_logger_default_is_verbose(self):
        logger = get_logger()
        assert logger.verbose is True

    def test_get_logger_verbose_false(self):
        logger = get_logger(verbose=False)
        assert logger.verbose is False

    def test_get_logger_returns_same_instance(self):
        a = get_logger(verbose=True)
        b = get_logger(verbose=False)
        assert a is b

    def test_set_verbose_changes_state(self):
        logger = get_logger(verbose=True)
        assert logger.verbose is True
        set_verbose(False)
        assert logger.verbose is False

    def test_set_verbose_creates_logger_if_none(self):
        assert logging_module._logger is None
        set_verbose(True)
        assert logging_module._logger is not None
        assert logging_module._logger.verbose is True

    def test_set_verbose_to_false_creates_quiet_logger(self):
        set_verbose(False)
        assert logging_module._logger is not None
        assert logging_module._logger.verbose is False
