import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--eager", action="store_true", default=False, help="whether to run all functions eagerly"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--eager"):
        import tensorflow as tf

        tf.config.run_functions_eagerly(True)
