#!/usr/bin/env python3
"""Configure and build the pymdp FFI library from [tool.pymdp.ffi]."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, TypedDict, cast

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
PYPROJECT: Final[Path] = REPO_ROOT / "pyproject.toml"

FfiCommand = Literal["configure", "build", "clean"]
CudaMode = Literal["auto", "on", "off"]
CmakeCudaMode = Literal["AUTO", "ON", "OFF"]
FfiConfigKey = Literal[
    "source_dir",
    "build_dir",
    "generator",
    "build_type",
    "python",
    "cuda",
]

_CUDA_TO_CMAKE: Final[dict[CudaMode, CmakeCudaMode]] = {
    "auto": "AUTO",
    "on": "ON",
    "off": "OFF",
}


class FfiConfig(TypedDict, total=False):
    """``[tool.pymdp.ffi]`` section from pyproject.toml."""

    source_dir: str
    build_dir: str
    generator: str
    build_type: str
    python: str
    cuda: CudaMode
    cmake_defines: dict[str, str | int | float | bool]


# pyproject uses hyphenated keys; map to snake_case field names above.
_FFI_KEY_ALIASES: Final[dict[str, str]] = {
    "source-dir": "source_dir",
    "build-dir": "build_dir",
    "build-type": "build_type",
    "cmake-defines": "cmake_defines",
}


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        if sys.version_info >= (3, 11):
            import tomllib

            return tomllib.load(handle)
        import tomli

        return tomli.load(handle)


def _normalize_ffi_config(raw: dict[str, Any]) -> FfiConfig:
    cfg: dict[str, object] = {}
    for key, value in raw.items():
        field = _FFI_KEY_ALIASES.get(key, key)
        cfg[field] = value
    return cast(FfiConfig, cfg)


def _ffi_config() -> FfiConfig:
    tool = _load_toml(PYPROJECT).get("tool", {})
    if not isinstance(tool, dict):
        return {}
    pymdp = tool.get("pymdp", {})
    if not isinstance(pymdp, dict):
        return {}
    ffi = pymdp.get("ffi", {})
    if not isinstance(ffi, dict):
        return {}
    return _normalize_ffi_config(ffi)


def _cfg_str(cfg: FfiConfig, key: FfiConfigKey, default: str) -> str:
    value = cfg.get(key, default)
    return str(value)


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _python_executable(cfg: FfiConfig) -> str:
    override = os.environ.get("PYTHON")
    if override:
        return override
    configured = _cfg_str(cfg, "python", ".venv/bin/python")
    path = Path(configured)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if path.exists():
        # Keep the venv entrypoint; resolving the symlink drops site-packages.
        return str(path)
    return sys.executable


def _cmake_executable() -> str:
    venv_cmake = REPO_ROOT / ".venv" / "bin" / "cmake"
    if venv_cmake.exists():
        return str(venv_cmake)
    return "cmake"


def _cuda_mode_explicitly_set(cfg: FfiConfig) -> bool:
    defines = cfg.get("cmake_defines", {})
    if "PYMDP_FFI_CUDA" in defines:
        return True
    extra = os.environ.get("CMAKE_FLAGS", "")
    return "PYMDP_FFI_CUDA" in extra


def _parse_cuda_mode(cfg: FfiConfig) -> CudaMode:
    mode = _cfg_str(cfg, "cuda", "auto").strip().lower()
    if mode not in _CUDA_TO_CMAKE:
        raise ValueError(
            f"invalid [tool.pymdp.ffi].cuda value {mode!r}; expected auto, on, or off"
        )
    return cast(CudaMode, mode)


def _cuda_cmake_arg(cfg: FfiConfig) -> list[str]:
    """Map ``[tool.pymdp.ffi].cuda`` / ``PYMDP_FFI_CUDA`` env to a ``-D`` flag."""
    if _cuda_mode_explicitly_set(cfg):
        return []

    env = os.environ.get("PYMDP_FFI_CUDA", "").strip()
    if env:
        return [f"-DPYMDP_FFI_CUDA={env.upper()}"]

    mode = _parse_cuda_mode(cfg)
    return [f"-DPYMDP_FFI_CUDA={_CUDA_TO_CMAKE[mode]}"]


def _cmake_define_args(cfg: FfiConfig) -> list[str]:
    args: list[str] = []
    for key, value in cfg.get("cmake_defines", {}).items():
        args.append(f"-D{key}={value}")
    args.extend(_cuda_cmake_arg(cfg))
    extra = os.environ.get("CMAKE_FLAGS", "")
    if extra.strip():
        args.extend(extra.split())
    return args


def _maybe_clean_stale_cache(source_dir: Path, build_dir: Path) -> None:
    cache = build_dir / "CMakeCache.txt"
    if not cache.exists():
        return
    content = cache.read_text(encoding="utf-8", errors="replace")
    source = str(source_dir)
    build = str(build_dir)
    if source in content and build in content:
        return
    print(
        f"Removing stale CMake cache at {build_dir} (project path changed)",
        file=sys.stderr,
    )
    shutil.rmtree(build_dir)


def configure(cfg: FfiConfig, *, quiet: bool = False) -> Path:
    source_dir = _resolve_path(_cfg_str(cfg, "source_dir", "pymdp/ffi"))
    build_dir = _resolve_path(_cfg_str(cfg, "build_dir", "pymdp/ffi/build"))
    build_dir.mkdir(parents=True, exist_ok=True)
    _maybe_clean_stale_cache(source_dir, build_dir)

    cmd: list[str] = [
        _cmake_executable(),
        "-B",
        str(build_dir),
        "-S",
        str(source_dir),
        "-G",
        _cfg_str(cfg, "generator", "Unix Makefiles"),
        f"-DPython_EXECUTABLE={_python_executable(cfg)}",
        f"-DCMAKE_BUILD_TYPE={_cfg_str(cfg, 'build_type', 'Release')}",
        *_cmake_define_args(cfg),
    ]
    stdout: int | None = subprocess.DEVNULL if quiet else None
    _ = subprocess.run(cmd, check=True, stdout=stdout)

    compile_db = build_dir / "compile_commands.json"
    if not compile_db.exists():
        raise FileNotFoundError("compile_commands.json was not generated")
    return build_dir


def build(cfg: FfiConfig) -> None:
    build_dir = configure(cfg)
    _ = subprocess.run(
        [_cmake_executable(), "--build", str(build_dir)],
        check=True,
    )


def clean(cfg: FfiConfig) -> None:
    build_dir = _resolve_path(_cfg_str(cfg, "build_dir", "pymdp/ffi/build"))
    if build_dir.exists():
        shutil.rmtree(build_dir)


@dataclass(frozen=True, slots=True)
class CliArgs:
    command: FfiCommand
    quiet: bool


def parse_args(argv: list[str] | None = None) -> CliArgs:
    parser = argparse.ArgumentParser(
        description="Configure/build the pymdp FFI library using pyproject.toml settings."
    )
    _ = parser.add_argument(
        "command",
        choices=("configure", "build", "clean"),
        help="configure: cmake configure only; build: configure + compile; clean: remove build dir",
    )
    _ = parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress configure output (used by ffi-compile-db).",
    )
    ns = parser.parse_args(argv)
    return CliArgs(command=ns.command, quiet=ns.quiet)


def main() -> int:
    args = parse_args()
    cfg = _ffi_config()
    try:
        if args.command == "configure":
            _ = configure(cfg, quiet=args.quiet)
        elif args.command == "build":
            build(cfg)
        else:
            clean(cfg)
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
