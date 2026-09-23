"""Fetch the third-party NAO meshes into this checkout.

The NAO geometry is published by Aldebaran / SoftBank Robotics under the
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International
license, and upstream permits redistribution only through an installer that
obtains the user's explicit assent. This repository therefore ships no mesh
file; this script downloads them onto your machine after you accept the
license.

Source: the official ROS Noetic binary package of ``nao_meshes``, published by
the ROS build farm. A ``.deb`` is an ``ar`` archive wrapping ``data.tar.xz``,
and both formats are handled by the Python standard library, so this runs on
native Windows with no WSL, no ROS and no third-party tooling.

Usage::

    python scripts/fetch_nao_meshes.py
    python scripts/fetch_nao_meshes.py --accept-license   # non-interactive
    python scripts/fetch_nao_meshes.py --force            # re-extract

See ``nao/assets/licenses/README.md`` for provenance and the licensing decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXTENSION_ROOT = _REPO_ROOT / "source" / "humanoid_transfer"
if str(_EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTENSION_ROOT))

from humanoid_transfer.nao.assets.nao_paths import (  # noqa: E402
    ASSET_CACHE_DIR,
    NAO_ASSET_DIR,
    NAO_MESH_DIR,
    NAO_MESH_V40_DIR,
    NAO_TEXTURE_DIR,
    NAO_URDF_PATH,
    REPO_ROOT,
    missing_mesh_files,
    required_mesh_files,
)

# Pinned upstream artifact. Bump URL and digest together; never relax the check.
DEB_FILENAME = "ros-noetic-nao-meshes_0.1.13-0focal.20250426.003811_amd64.deb"
DEB_URL = (
    "http://packages.ros.org/ros/ubuntu/pool/main/r/ros-noetic-nao-meshes/" + DEB_FILENAME
)
DEB_SHA256 = "8149c70bd13a7cc89938f3c57c23d0da9f7ea9683aefabbfad7f6860f49a0b63"

# Marker that separates the ROS install prefix from the payload we want.
PAYLOAD_MARKER = "share/nao_meshes/"

LICENSE_RELPATH = Path("nao") / "assets" / "licenses" / "LICENSE.nao_meshes.txt"

ACCEPT_PHRASE = "I ACCEPT"

LICENSE_NOTICE = f"""
================================================================================
  THIRD-PARTY LICENSE — NAO 3D GEOMETRY
================================================================================

  Work        : NAO robot meshes and texture
  Copyright   : Aldebaran Robotics / SoftBank Robotics
  Packaged by : ros-naoqi/nao_meshes (Vincent Rabaud, Mikael Arguedas;
                maintained by Maxime Busy and Surya Ambrose)
  License     : Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
                International Public License (CC BY-NC-ND 4.0)

  Full license text:
    {LICENSE_RELPATH.as_posix()}

  Key terms you are agreeing to:

    * NonCommercial. You may use this geometry only for purposes that are not
      primarily intended for or directed towards commercial advantage or
      monetary compensation.
    * NoDerivatives. You may produce adapted material (for example the USD
      that Isaac Sim generates from these meshes) for your own non-commercial
      use, but you may not share it.
    * Attribution. You must keep the attribution in nao/assets/licenses/README.md.
    * These files are downloaded onto this machine only. Do not commit them to
      this or any other repository.

  This is a good-faith summary, not a substitute for the full license text.

================================================================================
"""


def _print_license_notice() -> None:
    print(LICENSE_NOTICE)
    full_license = REPO_ROOT / LICENSE_RELPATH
    if full_license.is_file():
        print(f"  The complete license is at:\n    {full_license}\n")
    else:
        print(f"  WARNING: expected license file not found at {full_license}\n")


def _prompt_for_acceptance() -> bool:
    """Require an explicit typed acceptance before anything is downloaded."""
    if not sys.stdin or not sys.stdin.isatty():
        print(
            "This shell is not interactive, so the license cannot be accepted here.\n"
            "Re-run with --accept-license if you accept the terms shown above.",
            file=sys.stderr,
        )
        return False
    print(f"  Type exactly '{ACCEPT_PHRASE}' to accept, or anything else to abort.")
    try:
        answer = input("  > ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.", file=sys.stderr)
        return False
    if answer != ACCEPT_PHRASE:
        print("License not accepted. Nothing was downloaded.", file=sys.stderr)
        return False
    return True


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _download(url: str, destination: Path) -> bytes:
    """Download ``url``, caching the payload at ``destination``."""
    if destination.is_file():
        cached = destination.read_bytes()
        if _sha256(cached) == DEB_SHA256:
            print(f"[fetch] using cached download: {destination}")
            return cached
        print("[fetch] cached download failed its checksum, re-downloading")
        destination.unlink()

    print(f"[fetch] downloading {url}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
            payload = response.read()
    except urllib.error.URLError as error:
        raise SystemExit(f"[fetch] download failed: {error}\n[fetch] URL: {url}") from error

    destination.write_bytes(payload)
    print(f"[fetch] wrote {len(payload):,} bytes to {destination}")
    return payload


def _verify_checksum(payload: bytes) -> None:
    digest = _sha256(payload)
    if digest != DEB_SHA256:
        raise SystemExit(
            "[fetch] checksum mismatch — refusing to extract\n"
            f"        expected {DEB_SHA256}\n"
            f"        actual   {digest}"
        )
    print(f"[fetch] sha256 verified: {digest}")


def _read_ar_member(archive: bytes, wanted: str) -> bytes:
    """Extract one member from a Unix ``ar`` archive (the .deb container)."""
    if not archive.startswith(b"!<arch>\n"):
        raise SystemExit("[fetch] downloaded file is not an ar archive")
    offset = 8
    while offset + 60 <= len(archive):
        header = archive[offset : offset + 60]
        name = header[0:16].decode("ascii", "replace").strip().rstrip("/")
        try:
            size = int(header[48:58].decode("ascii", "replace").strip())
        except ValueError as error:
            raise SystemExit("[fetch] malformed ar header in .deb") from error
        body_start = offset + 60
        if name == wanted:
            return archive[body_start : body_start + size]
        offset = body_start + size + (size % 2)
    raise SystemExit(f"[fetch] member '{wanted}' not found in .deb")


def _payload_destination(member_name: str) -> Path | None:
    """Map a tar entry onto its location in this repository, or None to skip."""
    normalized = member_name.lstrip("./")
    index = normalized.find(PAYLOAD_MARKER)
    if index < 0:
        return None
    relative = normalized[index + len(PAYLOAD_MARKER) :]
    if not relative:
        return None
    top = relative.split("/", 1)[0]
    if top not in {"meshes", "texture"}:
        return None
    # Built from the sanitized relative path only, so no traversal is possible.
    return NAO_ASSET_DIR / Path(relative)


def _extract_payload(deb: bytes) -> int:
    data_tar = _read_ar_member(deb, "data.tar.xz")
    print(f"[fetch] extracting data.tar.xz ({len(data_tar):,} bytes)")

    written = 0
    with tarfile.open(fileobj=io.BytesIO(data_tar), mode="r:xz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            destination = _payload_destination(member.name)
            if destination is None:
                continue
            source = tar.extractfile(member)
            if source is None:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source, open(destination, "wb") as handle:
                shutil.copyfileobj(source, handle)
            written += 1
    return written


def _verify_installation() -> None:
    missing = missing_mesh_files()
    required = required_mesh_files()
    if missing:
        preview = "\n".join(f"          {name}" for name in missing[:10])
        more = f"\n          ... and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise SystemExit(
            f"[fetch] {len(missing)} of {len(required)} required meshes are still missing:\n"
            f"{preview}{more}"
        )
    print(f"[fetch] all {len(required)} meshes referenced by the URDF resolve on disk")


def _clean_existing() -> None:
    for directory in (NAO_MESH_DIR, NAO_TEXTURE_DIR):
        if directory.exists():
            print(f"[fetch] removing {directory}")
            shutil.rmtree(directory)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the CC BY-NC-ND licensed NAO meshes into this checkout.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help=(
            "Accept the CC BY-NC-ND 4.0 license non-interactively. Equivalent to "
            f"typing '{ACCEPT_PHRASE}' at the prompt."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete any existing mesh tree and extract again.",
    )
    parser.add_argument(
        "--keep-download",
        action="store_true",
        help="Keep the downloaded .deb in .asset_cache/ for offline re-runs.",
    )
    args = parser.parse_args()

    if not NAO_URDF_PATH.is_file():
        raise SystemExit(f"[fetch] NAO URDF not found at {NAO_URDF_PATH}")

    if args.force:
        _clean_existing()
    elif not missing_mesh_files():
        print(f"[fetch] NAO meshes already present under {NAO_ASSET_DIR}")
        print("[fetch] nothing to do (use --force to re-extract)")
        return 0

    _print_license_notice()
    if not args.accept_license and not _prompt_for_acceptance():
        return 1
    if args.accept_license:
        print("[fetch] license accepted via --accept-license")

    cached_deb = ASSET_CACHE_DIR / DEB_FILENAME
    payload = _download(DEB_URL, cached_deb)
    _verify_checksum(payload)

    count = _extract_payload(payload)
    print(f"[fetch] wrote {count} files under {NAO_ASSET_DIR}")
    _verify_installation()

    if not args.keep_download and cached_deb.is_file():
        cached_deb.unlink()
        print(f"[fetch] removed download cache {cached_deb}")

    print(f"[fetch] meshes   : {NAO_MESH_V40_DIR}")
    print(f"[fetch] texture  : {NAO_TEXTURE_DIR}")
    print("[fetch] done. These files are untracked by design — do not commit them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
