"""Capture the complete installed distribution versions for frozen runs."""

from importlib.metadata import distributions


def installed_versions() -> dict[str, str]:
    return dict(
        sorted(
            (d.metadata["Name"].lower().replace("_", "-"), d.version)
            for d in distributions()
            if d.metadata["Name"]
        )
    )
