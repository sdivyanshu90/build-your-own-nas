"""Configuration loading and the precedence chain.

Precedence, lowest to highest
-----------------------------
1. **Built-in defaults** — every field in :mod:`nas_engine.config.models` has one, so a
   configuration file may be empty and the run still works.
2. **YAML file** — the reviewable, version-controlled source of truth.
3. **Environment variables** — for deployment-specific values (device, worker count, paths)
   that should not live in a committed file.
4. **Command-line overrides** — ``--set key.path=value``, for one-off experiments.

Later sources win, and merging is *deep*: setting ``NAS_ENGINE__TRAINING__OPTIMIZER__LEARNING_RATE``
replaces exactly that leaf and leaves the rest of the training section alone. A shallow
merge would silently discard every sibling field, which is a classic source of "my
configuration file stopped working when I set one environment variable".

Environment variable syntax
---------------------------
``NAS_ENGINE__<SECTION>__<FIELD>``, with double underscores as the nesting separator. Single
underscores are preserved because field names contain them
(``NAS_ENGINE__BUDGET__MAX_EVALUATIONS`` sets ``budget.max_evaluations``).

Value parsing
-------------
Values from the environment and the command line arrive as strings and are parsed with
``yaml.safe_load``, so ``3`` becomes an integer, ``true`` a boolean, ``[1, 2]`` a list, and
``null`` becomes ``None``. ``safe_load`` never constructs arbitrary Python objects, so a
crafted value cannot execute code.

Security
--------
YAML is parsed with ``yaml.safe_load`` throughout. ``yaml.load`` with the default loader
instantiates arbitrary Python objects named by ``!!python/object`` tags — a remote code
execution vector in any file the user did not write themselves.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from nas_engine.config.models import CONFIG_VERSION, SearchConfig
from nas_engine.exceptions import ConfigurationError
from nas_engine.observability.logging import get_logger

_LOGGER = get_logger(__name__)

#: Prefix identifying environment variables that configure a search.
ENV_PREFIX: str = "NAS_ENGINE__"

#: Separator marking one level of nesting in an environment variable name.
ENV_SEPARATOR: str = "__"

#: Maximum size of a configuration file. A configuration is a few kilobytes; a larger file
#: is either wrong or hostile, and parsing it would waste memory before validation runs.
MAX_CONFIG_BYTES: int = 1024 * 1024


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new mapping.

    Nested mappings merge key by key; every other type is replaced wholesale. A list is
    replaced rather than concatenated, because "append to the list in the file" is almost
    never what an override means and concatenation cannot be undone.

    Args:
        base: Lower-precedence mapping.
        override: Higher-precedence mapping.

    Returns:
        A new merged dictionary. Neither input is modified.
    """
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def parse_scalar(text: str) -> Any:
    """Parse a string into a Python scalar using YAML rules.

    Args:
        text: Raw text from the environment or the command line.

    Returns:
        The parsed value; the original string when it is not valid YAML.
    """
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        # Not valid YAML: treat it as a plain string rather than failing. A path such as
        # `C:\runs` is a perfectly reasonable value that YAML would choke on.
        return text


def assign_path(target: dict[str, Any], path: Sequence[str], value: Any) -> None:
    """Set a nested key, creating intermediate mappings.

    Args:
        target: Mapping to modify in place.
        path: Key path, outermost first.
        value: Value to assign at the leaf.

    Raises:
        ConfigurationError: If the path is empty or traverses a non-mapping value.
    """
    if not path:
        msg = "cannot assign a configuration value with an empty key path"
        raise ConfigurationError(msg)
    cursor = target
    for index, key in enumerate(path[:-1]):
        nested = cursor.get(key)
        if nested is None:
            nested = {}
            cursor[key] = nested
        elif not isinstance(nested, dict):
            traversed = ".".join(path[: index + 1])
            msg = (
                f"cannot set {'.'.join(path)}: '{traversed}' is already a scalar value "
                f"({nested!r}), not a section"
            )
            raise ConfigurationError(msg, details={"path": ".".join(path), "conflict": traversed})
        cursor = nested
    cursor[path[-1]] = value


def parse_environment(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Extract configuration overrides from environment variables.

    Args:
        environ: Environment mapping; defaults to :data:`os.environ`.

    Returns:
        A nested mapping of overrides; empty when no matching variables are set.

    Raises:
        ConfigurationError: If a variable name is malformed.
    """
    source = environ if environ is not None else os.environ
    overrides: dict[str, Any] = {}
    for name, raw_value in sorted(source.items()):
        if not name.startswith(ENV_PREFIX):
            continue
        remainder = name[len(ENV_PREFIX) :]
        if not remainder:
            msg = (
                f"environment variable {name!r} has the configuration prefix but no field "
                f"path; use {ENV_PREFIX}SECTION{ENV_SEPARATOR}FIELD"
            )
            raise ConfigurationError(msg, details={"variable": name})
        path = [part.lower() for part in remainder.split(ENV_SEPARATOR) if part]
        if not path:
            msg = f"environment variable {name!r} does not name a configuration field"
            raise ConfigurationError(msg, details={"variable": name})
        assign_path(overrides, path, parse_scalar(raw_value))
    return overrides


def parse_overrides(assignments: Sequence[str]) -> dict[str, Any]:
    """Parse ``key.path=value`` command-line assignments.

    Args:
        assignments: Raw ``--set`` arguments.

    Returns:
        A nested mapping of overrides.

    Raises:
        ConfigurationError: If an assignment has no ``=`` or an empty key.
    """
    overrides: dict[str, Any] = {}
    for assignment in assignments:
        if "=" not in assignment:
            msg = (
                f"override {assignment!r} is not of the form key.path=value; for example "
                "--set budget.max_evaluations=20"
            )
            raise ConfigurationError(msg, details={"assignment": assignment})
        key, _, raw_value = assignment.partition("=")
        path = [part for part in key.strip().split(".") if part]
        if not path:
            msg = f"override {assignment!r} has an empty key"
            raise ConfigurationError(msg, details={"assignment": assignment})
        assign_path(overrides, path, parse_scalar(raw_value.strip()))
    return overrides


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML configuration file safely.

    Args:
        path: File to read.

    Returns:
        The parsed mapping; empty for an empty file.

    Raises:
        ConfigurationError: If the file is missing, oversized, unparseable, or does not
            contain a mapping at the top level.
    """
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        msg = (
            f"configuration file not found: {resolved}. Create one with "
            "'nas-engine init', or pass --config with a valid path."
        )
        raise ConfigurationError(msg, details={"path": str(resolved)})
    size = resolved.stat().st_size
    if size > MAX_CONFIG_BYTES:
        msg = (
            f"configuration file {resolved} is {size} bytes, above the {MAX_CONFIG_BYTES} "
            "byte limit; this guard exists because a configuration should never be large"
        )
        raise ConfigurationError(msg, details={"path": str(resolved), "size": size})
    try:
        # `safe_load` refuses `!!python/...` tags, so a configuration file cannot construct
        # arbitrary objects or call arbitrary code.
        loaded = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"configuration file {resolved} is not valid YAML: {exc}"
        raise ConfigurationError(msg, details={"path": str(resolved), "error": str(exc)}) from exc
    except UnicodeDecodeError as exc:
        msg = f"configuration file {resolved} is not valid UTF-8 text: {exc}"
        raise ConfigurationError(msg, details={"path": str(resolved), "error": str(exc)}) from exc

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        msg = (
            f"configuration file {resolved} must contain a YAML mapping at the top level, "
            f"but it contains a {type(loaded).__name__}"
        )
        raise ConfigurationError(msg, details={"path": str(resolved)})
    return loaded


def _format_validation_error(error: ValidationError, *, source: str) -> ConfigurationError:
    """Turn a Pydantic error into an actionable :class:`ConfigurationError`.

    Args:
        error: The validation error.
        source: Where the configuration came from, for the message.

    Returns:
        A configuration error naming every invalid field with its received value.
    """
    problems: list[dict[str, Any]] = []
    lines: list[str] = []
    for item in error.errors():
        field = ".".join(str(part) for part in item["loc"]) or "<root>"
        received = item.get("input")
        message = item["msg"]
        problems.append({"field": field, "message": message, "received": repr(received)})
        lines.append(f"  - {field}: {message} (received {received!r})")
    summary = "\n".join(lines)
    text = (
        f"configuration from {source} is invalid:\n{summary}\n"
        "Fix the listed fields, or run 'nas-engine validate-config --config <file>' to "
        "check a file before using it."
    )
    return ConfigurationError(text, details={"source": source, "problems": problems})


def build_config(payload: Mapping[str, Any], *, source: str = "mapping") -> SearchConfig:
    """Validate a plain mapping into a :class:`SearchConfig`.

    Args:
        payload: Merged configuration data.
        source: Where the data came from, used in error messages.

    Returns:
        The validated configuration.

    Raises:
        ConfigurationError: If validation fails.
    """
    try:
        return SearchConfig.model_validate(dict(payload))
    except ValidationError as exc:
        raise _format_validation_error(exc, source=source) from exc


def load_config(
    path: Path | str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    overrides: Sequence[str] | None = None,
    use_environment: bool = True,
) -> SearchConfig:
    """Load a configuration through the full precedence chain.

    Args:
        path: YAML file to read; ``None`` uses defaults only.
        environ: Environment mapping; defaults to the process environment.
        overrides: ``key.path=value`` assignments from the command line.
        use_environment: Whether to consult the environment at all. Tests disable it so a
            developer's shell cannot change test outcomes.

    Returns:
        The validated configuration.

    Raises:
        ConfigurationError: If any layer is malformed or the result fails validation.
    """
    layers: list[tuple[str, Mapping[str, Any]]] = []

    if path is not None:
        layers.append((str(path), read_yaml(Path(path))))
    if use_environment:
        environment_layer = parse_environment(environ)
        if environment_layer:
            layers.append(("environment", environment_layer))
    if overrides:
        layers.append(("command line", parse_overrides(overrides)))

    merged: dict[str, Any] = {}
    for _, layer in layers:
        merged = deep_merge(merged, layer)

    source = " < ".join(["defaults", *(name for name, _ in layers)])
    config = build_config(merged, source=source)
    _LOGGER.debug(
        "config.loaded",
        source=source,
        config_hash=config.config_hash(),
        version=config.version,
    )
    return config


def dump_yaml(config: SearchConfig, path: Path | None = None) -> str:
    """Render a configuration as YAML, optionally writing it to disk.

    Args:
        config: Configuration to render.
        path: Destination file; ``None`` renders without writing.

    Returns:
        The YAML text.

    Raises:
        ConfigurationError: If the file cannot be written.
    """
    text = yaml.safe_dump(config.to_dict(), sort_keys=True, default_flow_style=False)
    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        except OSError as exc:
            msg = f"cannot write configuration to {path}: {exc}"
            raise ConfigurationError(msg, details={"path": str(path), "error": str(exc)}) from exc
    return text


def check_config_compatibility(stored: Mapping[str, Any], current: SearchConfig) -> list[str]:
    """Compare a stored configuration against the one being used to resume.

    Returns *warnings*, not errors, for most differences: adjusting the log level or the
    device between runs is legitimate. Differences that change what the search *means* —
    the strategy, the space, the seed — are reported first because they invalidate the
    comparison between the two halves of the run.

    Args:
        stored: Configuration recorded when the search was created.
        current: Configuration now in use.

    Returns:
        Human-readable difference descriptions; empty when the configurations match.

    Raises:
        ConfigVersionError: If the stored configuration is from an incompatible version.
    """
    from nas_engine.exceptions import ConfigVersionError

    stored_version = int(stored.get("version", 0))
    if stored_version > CONFIG_VERSION:
        msg = (
            f"the stored configuration is version {stored_version} but this build supports "
            f"at most version {CONFIG_VERSION}; upgrade nas-engine to resume this search"
        )
        raise ConfigVersionError(
            msg, details={"stored": stored_version, "supported": CONFIG_VERSION}
        )
    if stored_version < 1:
        msg = (
            f"the stored configuration reports version {stored_version}, which is not a "
            "valid configuration version; the search record is corrupt"
        )
        raise ConfigVersionError(msg, details={"stored": stored_version})

    current_data = current.to_dict()
    critical = (
        ("algorithm", "search strategy"),
        ("search_space", "search space"),
        ("reproducibility", "seeding"),
        ("objectives", "objectives"),
    )
    differences: list[str] = []
    for section, label in critical:
        if stored.get(section) != current_data.get(section):
            differences.append(
                f"the {label} section changed since this search was created; results from "
                "before and after the resume are not directly comparable"
            )
    for section in sorted(set(current_data) | set(stored)):
        if any(section == name for name, _ in critical):
            continue
        if stored.get(section) != current_data.get(section):
            differences.append(f"section '{section}' differs from the stored configuration")
    return differences


__all__ = [
    "ENV_PREFIX",
    "ENV_SEPARATOR",
    "MAX_CONFIG_BYTES",
    "assign_path",
    "build_config",
    "check_config_compatibility",
    "deep_merge",
    "dump_yaml",
    "load_config",
    "parse_environment",
    "parse_overrides",
    "parse_scalar",
    "read_yaml",
]
