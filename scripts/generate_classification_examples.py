"""Generate local PNG/JPEG/CSV and an explicit supervised toy classifier, no downloads."""

from pathlib import Path

from opweaver.classification_examples import generate_classification_examples

generate_classification_examples(
    Path(__file__).resolve().parents[1] / "examples" / "classification"
)
