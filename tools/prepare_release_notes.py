import argparse
import json
import re
from pathlib import Path

# GitHub's release body limit is measured in characters.
BODY_LIMIT = 125000


def release_body(notes):
    if len(notes) <= BODY_LIMIT:
        return notes
    # Keep the next section and the changelog link, even without contributors.
    return re.sub(
        r"^## What's Changed[^\S\r\n]*\r?\n.*?(?=^#{1,2} |^\*\*Full Changelog\*\*:|\Z)",
        "",
        notes,
        count=1,
        flags=re.MULTILINE | re.DOTALL,
    )


def prepare_notes(source, body, full):
    notes = json.loads(source.read_text(encoding="utf-8"))["body"]
    if not isinstance(notes, str):
        raise ValueError("Generated release notes body must be a string")
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(notes, encoding="utf-8", newline="")
    # Do not shorten anything else: GitHub should reject an oversized remainder.
    body.write_text(release_body(notes), encoding="utf-8", newline="")


def main():
    parser = argparse.ArgumentParser(description="Prepare size-safe GitHub release notes")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--body", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    args = parser.parse_args()
    prepare_notes(args.input, args.body, args.full)


if __name__ == "__main__":
    main()
