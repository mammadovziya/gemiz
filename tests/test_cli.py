"""Basic CLI smoke tests."""

from click.testing import CliRunner

from gemiz.cli import main


def test_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "reconstruct" in result.output


def test_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_reconstruct_help():
    runner = CliRunner()
    result = runner.invoke(main, ["reconstruct", "--help"])
    assert result.exit_code == 0
    assert "Prodigal" in result.output
