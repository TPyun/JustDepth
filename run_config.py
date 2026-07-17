import configparser
import sys
from pathlib import Path


def load_run_config(path, section='run'):
    parser = configparser.ConfigParser()
    parser.optionxform = str
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = path.read_text()
    if not text.lstrip().startswith("["):
        text = "[run]\n" + text
    parser.read_string(text)
    if section not in parser:
        available = ", ".join(parser.sections())
        raise KeyError(f"Config section [{section}] not found in {path}. Available: {available}")
    return parser[section]


def cfg_get(cfg, key, default=None):
    value = cfg.get(key, fallback=default)
    return value


def cfg_get_with_cli_priority(cfg, key, current, option, argv=None):
    """Use an explicitly supplied CLI option before the config value.

    Empty config values are treated as unset so entries such as
    ``confidence_path =`` do not erase a useful argparse default.
    """
    argv = sys.argv[1:] if argv is None else argv
    if any(arg == option or arg.startswith(option + '=') for arg in argv):
        return current
    value = cfg_get(cfg, key, current)
    if isinstance(value, str) and not value.strip():
        return current
    return value


def cfg_get_bool(cfg, key, default=False):
    if key not in cfg:
        return default
    return cfg.getboolean(key)


def cfg_get_int(cfg, key, default=0):
    if key not in cfg:
        return default
    return cfg.getint(key)


def cfg_get_float(cfg, key, default=0.0):
    if key not in cfg:
        return default
    return cfg.getfloat(key)


def cfg_get_list(cfg, key, default=None, cast=str):
    if key not in cfg:
        return default if default is not None else []
    raw = cfg.get(key, fallback="")
    if not raw.strip():
        return []
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]
