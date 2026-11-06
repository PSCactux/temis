"""Print the Markdown table for the README from the latest evals/results/*.json (or the one given)."""

import json
import sys
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(Path("evals/results").glob("*.json"))[-1]
    data = json.loads(path.read_text())
    date = path.stem[:8]
    print(f"Model `{data['model']}` (fast: `{data['fast_model']}`), privacy mode `{data['privacy_mode']}`, {date}.\n")
    print("| Kind | Passed | Median seconds |")
    print("|---|---|---|")
    for kind, row in data["by_kind"].items():
        print(f"| {kind} | {row['passed']}/{row['cases']} | {row['median_seconds']} |")
    print(f"| **total** | **{data['passed']}/{data['cases']}** ({data['pass_rate']:.0%}) | |")


if __name__ == "__main__":
    main()
