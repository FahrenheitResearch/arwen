"""Keep shipped docs from requiring an account for public ERA5 acquisition."""

import argparse
from pathlib import Path
import re

from gpuwm.era5_member import validate_selection
from gpuwm.fetch import register_cli


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
REQUIREMENT = r"\b(?:requires?|needs?)\b(?!\s+no\b)"
ACCOUNT = r"\b(?:credentials?|configuration|accounts?|keys?)\b"


def _provider_credentials():
    # The registry declares credentials per source. The CLI describes them
    # per provider, so read that distinction without probing any credentials.
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers()
    register_cli(commands)
    action = next(action for action in commands.choices["fetch"]._actions
                  if "--era5-provider" in action.option_strings)
    descriptions = dict(clause.strip().split(" ", 1)
                        for clause in action.help.split(":", 1)[1].split(";"))
    assert set(descriptions) == set(action.choices)
    facts = {}
    for provider, description in descriptions.items():
        keyless = bool(re.search(r"without (?:a )?key|keyless", description))
        keyed = bool(re.search(r"uses .*credentials", description))
        assert keyless != keyed, f"Unclassified provider credentials: {provider}"
        facts[provider] = keyed
    assert any(facts.values()) and not all(facts.values())
    return facts


def _sentences(path):
    # Paragraphs and table rows are separate contexts. A provider in a
    # heading must not qualify an unrelated requirement in the next block.
    blocks = re.split(r"\n\s*\n|\n(?=\s*\|)", path.read_text(encoding="utf-8"))
    for block in blocks:
        yield from re.split(r"(?<=[.!?])\s+", " ".join(block.split()).lower())


def test_shipped_docs_era5_credentials_are_provider_specific():
    facts = _provider_credentials()
    paths = [README, *sorted((ROOT / "docs" / "public").glob("*.md")),
             ROOT / "docs" / "run-plan.md",
             ROOT / "tools/release/cut/drivers/linux-drivers/prepare_stage.py"]
    assert len(paths) > 2, "The public documentation scan must not be empty"
    failures = []
    for path in paths:
        for sentence in _sentences(path):
            if ("era5" not in sentence or not re.search(REQUIREMENT, sentence)
                    or not re.search(ACCOUNT, sentence)):
                continue
            subject = re.split(REQUIREMENT, sentence, maxsplit=1)[0]
            if not any(re.search(rf"\b{re.escape(provider)}\b", subject)
                       for provider, keyed in facts.items() if keyed):
                failures.append(f"{path.relative_to(ROOT)}: {sentence}")
    assert not failures, (
        "Shipped docs require an account for ERA5 without qualifying the provider:\n"
        + "\n".join(failures)
    )


def test_first_run_guides_describe_keyless_era5():
    facts = _provider_credentials()
    failures = []
    for name in ("DATA.md", "FIRST-LIGHT.md", "SOURCES.md"):
        path = ROOT / "docs" / "public" / name
        sentences = list(_sentences(path))
        for provider, keyed in facts.items():
            if not keyed and not any(
                re.search(rf"\b{re.escape(provider)}\b", sentence)
                and re.search(r"keyless|without (?:an? )?(?:account|key)|no (?:account|key)",
                              sentence)
                for sentence in sentences
            ):
                failures.append(f"{name}: missing keyless provider {provider}")
    assert not failures, "\n".join(failures)


def test_readme_documents_keyless_selection_and_ensemble_restriction():
    facts = _provider_credentials()
    text = " ".join(README.read_text(encoding="utf-8").split()).lower()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for provider, keyed in facts.items():
        mentioned = [sentence for sentence in sentences
                     if re.search(rf"\b{re.escape(provider)}\b", sentence)]
        if keyed:
            assert any(re.search(REQUIREMENT, sentence)
                       and re.search(ACCOUNT, sentence) for sentence in mentioned)
        else:
            assert any(re.search(r"keyless|without (?:an? )?(?:account|key)", sentence)
                       for sentence in mentioned), f"Missing public route: {provider}"
            assert f"--era5-provider {provider}" in text

    # Derive the supported ensemble providers from the actual selection rules.
    ensemble_providers = set()
    for provider in facts:
        try:
            validate_selection(product_type="ensemble_members", provider=provider,
                               member=0, cadence=3)
        except ValueError:
            continue
        ensemble_providers.add(provider)
    assert ensemble_providers
    ensemble_claims = [sentence for sentence in sentences
                       if "ensemble" in sentence and re.search(REQUIREMENT, sentence)]
    assert ensemble_claims, "README must state the ensemble provider restriction"
    documented = {provider for provider in facts
                  if any(re.search(rf"\b{re.escape(provider)}\b", sentence)
                         for sentence in ensemble_claims)}
    assert documented == ensemble_providers


def test_fetch_summary_describes_provider_credentials():
    parser = argparse.ArgumentParser()
    register_cli(parser.add_subparsers())
    summary = " ".join(parser.format_help().lower().split())
    for provider, keyed in _provider_credentials().items():
        assert re.search(rf"\b{re.escape(provider)}\b", summary), provider
        if keyed:
            assert "credentials" in summary
        else:
            assert re.search(r"keyless|without (?:a )?key", summary)
