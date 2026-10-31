#!/usr/bin/env python
"""Train and evaluate one fold of one configuration."""

import argparse
import dataclasses
import json

from petslice.training.config import RunConfig
from petslice.training.runner import train_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in dataclasses.fields(RunConfig):
        required = field.default is dataclasses.MISSING
        kind = {"int": int, "float": float, "str": str}[
            field.type.__name__ if hasattr(field.type, "__name__") else str(field.type)
        ]
        parser.add_argument(
            f"--{field.name.replace('_', '-')}",
            type=kind,
            required=required,
            default=None if required else field.default,
        )
    print(json.dumps(train_run(RunConfig(**vars(parser.parse_args()))), indent=2))


if __name__ == "__main__":
    main()
