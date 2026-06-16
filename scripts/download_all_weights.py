#!/usr/bin/env python3
"""Download all public model weights used by this repository.

The script intentionally uses only the Python standard library so it can run on
login nodes without installing `huggingface_hub`, `gcsfs`, or `requests`.
"""

import argparse
import contextlib
import http.cookiejar
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


DEFAULT_HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
DEFAULT_ROOT = Path(os.environ.get("NWP_WEIGHTS_ROOT", "assets/weights"))


class WeightSpec:
    def __init__(
        self,
        weight_id: str,
        model: str,
        target: str,
        urls: Tuple[str, ...],
        note: str = "",
        optional: bool = False,
        archive: bool = False,
        extract_dir: Optional[str] = None,
        expected_outputs: Tuple[str, ...] = (),
        env_url: Optional[str] = None,
        min_bytes: int = 1,
        output_min_bytes: int = 1,
    ) -> None:
        self.weight_id = weight_id
        self.model = model
        self.target = target
        self.urls = urls
        self.note = note
        self.optional = optional
        self.archive = archive
        self.extract_dir = extract_dir
        self.expected_outputs = expected_outputs
        self.env_url = env_url
        self.min_bytes = min_bytes
        self.output_min_bytes = output_min_bytes


def hf_url(repo_id: str, filename: str) -> str:
    quoted = urllib.parse.quote(filename, safe="/")
    return f"{DEFAULT_HF_ENDPOINT}/{repo_id}/resolve/main/{quoted}"


def sharepoint_urls(url: str) -> Tuple[str, ...]:
    urls = [f"{url}?download=1"]
    if "/:u:/" in url:
        urls.append(url.replace("/:u:/", "/:u:/download"))
    return tuple(urls)


SPECS: Tuple[WeightSpec, ...] = (
    WeightSpec(
        "aifs-single-1.1",
        "aifs",
        "aifs/aifs-single-mse-1.1.ckpt",
        (hf_url("ecmwf/aifs-single-1.1", "aifs-single-mse-1.1.ckpt"),),
        "AIFS single v1.1 checkpoint used by src/aifs and src/models/aifs_runner.py.",
        min_bytes=100_000_000,
    ),
    WeightSpec(
        "aurora-0.1-finetuned",
        "aurora",
        "aurora/aurora-0.1-finetuned.ckpt",
        (hf_url("microsoft/aurora", "aurora-0.1-finetuned.ckpt"),),
        "Checkpoint selected by src/aurora/download_weights.py.",
        min_bytes=100_000_000,
    ),
    WeightSpec(
        "aurora-0.25-pretrained",
        "aurora",
        "aurora/aurora-0.25-pretrained.ckpt",
        (hf_url("microsoft/aurora", "aurora-0.25-pretrained.ckpt"),),
        "Checkpoint loaded by src/aurora/inference.py and src/models/aurora_runner.py.",
        min_bytes=100_000_000,
    ),
    WeightSpec(
        "aurora-0.25-finetuned",
        "aurora",
        "aurora/aurora-0.25-finetuned.ckpt",
        (hf_url("microsoft/aurora", "aurora-0.25-finetuned.ckpt"),),
        "IFS Aurora checkpoint loaded by src/models/aurora_runner_ifs.py.",
        min_bytes=100_000_000,
    ),
    WeightSpec(
        "fengwu-v1",
        "fengwu",
        "fengwu/fengwu_v1.onnx",
        sharepoint_urls(
            "https://pjlab-my.sharepoint.cn/:u:/g/personal/chenkang_pjlab_org_cn/"
            "EVA6V_Qkp6JHgXwAKxXIzDsBPIddo5RgDtGCBQ-sQbMmwg"
        ),
        "ERA5 FengWu ONNX checkpoint. The upstream SharePoint link may require browser cookies.",
        min_bytes=100_000_000,
    ),
    WeightSpec(
        "fengwu-v2",
        "fengwu",
        "fengwu/fengwu_v2.onnx",
        sharepoint_urls(
            "https://pjlab-my.sharepoint.cn/:u:/g/personal/chenkang_pjlab_org_cn/"
            "EZkFM7nQcEtBve6MsqlWaeIB_lmpa__hX0I8QYOPzf-X6A"
        ),
        "HRES/IFS FengWu ONNX checkpoint. The upstream SharePoint link may require browser cookies.",
        min_bytes=100_000_000,
    ),
    WeightSpec(
        "fuxi-ec",
        "fuxi",
        "fuxi/FuXi_EC.zip",
        ("https://zenodo.org/records/10401602/files/FuXi_EC.zip?download=1",),
        "FuXi ONNX archive from Zenodo; extracted and normalized to fuxi/{short,medium,long}.onnx.",
        archive=True,
        extract_dir="fuxi",
        expected_outputs=("fuxi/short.onnx", "fuxi/medium.onnx", "fuxi/long.onnx"),
        min_bytes=8_000_000_000,
    ),
    WeightSpec(
        "graphcast-era5-37",
        "graphcast",
        "graphcast/GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz",
        (
            hf_url(
                "shermansiu/dm_graphcast",
                "GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz",
            ),
        ),
        "GraphCast ERA5 37-level checkpoint used by src/graphcast/inference.py.",
        min_bytes=100_000_000,
    ),
    WeightSpec(
        "graphcast-operational-13",
        "graphcast",
        "graphcast/GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz",
        (),
        "src/graphcast/inference_operational.py expects this file, but the source repo checked in 2026-06 did not list it.",
        optional=True,
        env_url="GRAPHCAST_OPERATIONAL_URL",
        min_bytes=100_000_000,
    ),
    WeightSpec(
        "neuralgcm-deterministic-0.7",
        "neuralgcm",
        "neuralgcm/models_v1_deterministic_0_7_deg.pkl",
        ("https://storage.googleapis.com/neuralgcm/models/v1/deterministic_0_7_deg.pkl",),
        "Default NeuralGCM checkpoint used by src/models/neuralgcm_runner.py.",
        min_bytes=100_000_000,
    ),
    WeightSpec(
        "neuralgcm-deterministic-1.4",
        "neuralgcm",
        "neuralgcm/models_v1_deterministic_1_4_deg.pkl",
        ("https://storage.googleapis.com/neuralgcm/models/v1/deterministic_1_4_deg.pkl",),
        "NeuralGCM 1.4 degree deterministic checkpoint.",
        min_bytes=50_000_000,
    ),
    WeightSpec(
        "neuralgcm-deterministic-2.8",
        "neuralgcm",
        "neuralgcm/models_v1_deterministic_2_8_deg.pkl",
        ("https://storage.googleapis.com/neuralgcm/models/v1/deterministic_2_8_deg.pkl",),
        "NeuralGCM 2.8 degree deterministic checkpoint.",
        min_bytes=40_000_000,
    ),
    WeightSpec(
        "neuralgcm-stochastic-1.4",
        "neuralgcm",
        "neuralgcm/models_v1_stochastic_1_4_deg.pkl",
        ("https://storage.googleapis.com/neuralgcm/models/v1/stochastic_1_4_deg.pkl",),
        "Stochastic NeuralGCM checkpoint selected by src/neuralgcm/download_weights.py.",
        min_bytes=40_000_000,
    ),
    WeightSpec(
        "pangu-1h",
        "pangu",
        "pangu/pangu_weather_1.onnx",
        (hf_url("qq1990/Pangu", "pangu_weather_1.onnx"),),
        "Pangu-Weather 1h ONNX checkpoint.",
        min_bytes=100_000_000,
    ),
    WeightSpec(
        "pangu-3h",
        "pangu",
        "pangu/pangu_weather_3.onnx",
        (hf_url("qq1990/Pangu", "pangu_weather_3.onnx"),),
        "Pangu-Weather 3h ONNX checkpoint.",
        min_bytes=100_000_000,
    ),
    WeightSpec(
        "pangu-6h",
        "pangu",
        "pangu/pangu_weather_6.onnx",
        (hf_url("qq1990/Pangu", "pangu_weather_6.onnx"),),
        "Pangu-Weather 6h ONNX checkpoint.",
        min_bytes=100_000_000,
    ),
    WeightSpec(
        "pangu-24h",
        "pangu",
        "pangu/pangu_weather_24.onnx",
        (hf_url("qq1990/Pangu", "pangu_weather_24.onnx"),),
        "Pangu-Weather 24h ONNX checkpoint.",
        min_bytes=100_000_000,
    ),
    WeightSpec(
        "stormer-1.40625-patch2",
        "stormer",
        "stormer/stormer_1.40625_patch_size_2.ckpt",
        (hf_url("tungnd/stormer", "stormer_1.40625_patch_size_2.ckpt"),),
        "Stormer checkpoint loaded by src/stormer/inference.py and src/models/stormer_runner.py.",
        min_bytes=100_000_000,
    ),
)


def build_opener(timeout: int) -> urllib.request.OpenerDirector:
    # Cookie support is useful for SharePoint-style public links.
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(cookie_jar),
        urllib.request.HTTPRedirectHandler(),
    )
    opener.addheaders = [
        (
            "User-Agent",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/121.0 Safari/537.36",
        )
    ]
    opener.timeout = timeout  # type: ignore[attr-defined]
    return opener


def resolved_urls(spec: WeightSpec) -> List[str]:
    urls = []
    if spec.env_url and os.environ.get(spec.env_url):
        urls.append(os.environ[spec.env_url])
    urls.extend(spec.urls)
    return urls


def spec_outputs(spec: WeightSpec, root: Path) -> List[Path]:
    if spec.expected_outputs:
        return [root / item for item in spec.expected_outputs]
    return [root / spec.target]


def is_complete(spec: WeightSpec, root: Path) -> bool:
    outputs = spec_outputs(spec, root)
    threshold = spec.output_min_bytes if spec.expected_outputs else spec.min_bytes
    return all(path.exists() and path.stat().st_size >= threshold for path in outputs)


def byte_count(size: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{size} B"


def request_with_range(url: str, offset: int) -> urllib.request.Request:
    req = urllib.request.Request(url)
    if offset > 0:
        req.add_header("Range", f"bytes={offset}-")
    return req


def download_url(
    opener: urllib.request.OpenerDirector,
    url: str,
    dest: Path,
    *,
    force: bool,
    timeout: int,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    if force:
        for path in (dest, part):
            with contextlib.suppress(FileNotFoundError):
                path.unlink()

    offset = part.stat().st_size if part.exists() else 0
    req = request_with_range(url, offset)
    start = time.time()
    last_log = start
    mode = "ab" if offset else "wb"

    try:
        response = opener.open(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if offset and exc.code == 416:
            part.replace(dest)
            print(f"  server reports range complete; saved {dest}")
            return
        raise

    with response:
        status = getattr(response, "status", None)
        if offset and status != 206:
            print(f"  server ignored Range; restarting {dest.name}")
            offset = 0
            mode = "wb"
        total_header = response.headers.get("Content-Length")
        expected_total = (int(total_header) + offset) if total_header and status == 206 else None

        with part.open(mode) as handle:
            downloaded = offset
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_log >= 10:
                    if expected_total:
                        pct = downloaded / expected_total * 100
                        print(f"  {byte_count(downloaded)} / {byte_count(expected_total)} ({pct:.1f}%)")
                    else:
                        print(f"  {byte_count(downloaded)}")
                    last_log = now

    part.replace(dest)
    elapsed = max(time.time() - start, 0.001)
    print(f"  saved {dest} ({byte_count(dest.stat().st_size)}, {byte_count(int(dest.stat().st_size / elapsed))}/s)")


def extract_archive(spec: WeightSpec, root: Path, *, keep_archives: bool) -> None:
    archive_path = root / spec.target
    extract_dir = root / (spec.extract_dir or spec.model)
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"  extracting {archive_path.name} -> {extract_dir}")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(extract_dir)

    normalize_fuxi_layout(extract_dir)

    if not keep_archives and is_complete(spec, root):
        archive_path.unlink()
        print(f"  removed archive {archive_path}")


def normalize_fuxi_layout(fuxi_dir: Path) -> None:
    for stage in ("short", "medium", "long"):
        wanted = fuxi_dir / f"{stage}.onnx"
        if wanted.exists():
            continue
        matches = [p for p in fuxi_dir.rglob(f"{stage}.onnx") if p != wanted]
        if not matches:
            continue
        source = matches[0]
        try:
            wanted.symlink_to(os.path.relpath(source, wanted.parent))
            print(f"  linked {wanted.name} -> {source.relative_to(fuxi_dir)}")
        except OSError:
            shutil.copy2(source, wanted)
            print(f"  copied {source.relative_to(fuxi_dir)} -> {wanted.name}")


def selected_specs(args: argparse.Namespace) -> List[WeightSpec]:
    selectors = set(args.only or [])
    excludes = set(args.exclude or [])
    selected = []
    for spec in SPECS:
        if spec.optional and not args.include_optional:
            continue
        if selectors and spec.weight_id not in selectors and spec.model not in selectors:
            continue
        if spec.weight_id in excludes or spec.model in excludes:
            continue
        selected.append(spec)
    return selected


def print_manifest(specs: Iterable[WeightSpec]) -> None:
    for spec in specs:
        marker = "optional" if spec.optional else "required"
        urls = resolved_urls(spec)
        source = urls[0] if urls else f"<set {spec.env_url}>"
        print(f"{spec.weight_id:30s} {marker:8s} {spec.target}")
        print(f"  source: {source}")
        if spec.note:
            print(f"  note:   {spec.note}")


def download_spec(
    spec: WeightSpec,
    root: Path,
    opener: urllib.request.OpenerDirector,
    args: argparse.Namespace,
) -> bool:
    if is_complete(spec, root) and not args.force:
        print(f"[skip] {spec.weight_id}: already complete")
        return True

    urls = resolved_urls(spec)
    if not urls:
        print(f"[miss] {spec.weight_id}: no URL configured; set {spec.env_url}")
        return False

    target = root / spec.target
    print(f"[get]  {spec.weight_id} -> {target}")

    if spec.archive and target.exists() and target.stat().st_size >= spec.min_bytes and not args.force:
        try:
            print(f"  using existing archive {target}")
            extract_archive(spec, root, keep_archives=args.keep_archives)
            if is_complete(spec, root):
                return True
        except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
            print(f"  existing archive could not be reused: {exc}")

    errors = []
    for url in urls:
        try:
            print(f"  source {url}")
            download_url(opener, url, target, force=args.force, timeout=args.timeout)
            if target.stat().st_size < spec.min_bytes:
                raise RuntimeError(
                    f"downloaded file is too small: {byte_count(target.stat().st_size)} "
                    f"< {byte_count(spec.min_bytes)}"
                )
            if spec.archive:
                extract_archive(spec, root, keep_archives=args.keep_archives)
            if not is_complete(spec, root):
                missing = [str(p) for p in spec_outputs(spec, root) if not p.exists()]
                raise RuntimeError(f"expected outputs missing after download: {missing}")
            return True
        except (OSError, RuntimeError, urllib.error.URLError, urllib.error.HTTPError, zipfile.BadZipFile) as exc:
            errors.append(f"{url}: {exc}")
            print(f"  failed: {exc}")
    print(f"[fail] {spec.weight_id}")
    for item in errors:
        print(f"  {item}")
    return False


def verify(specs: Iterable[WeightSpec], root: Path) -> bool:
    ok = True
    for spec in specs:
        complete = is_complete(spec, root)
        status = "ok" if complete else "missing"
        print(f"{status:7s} {spec.weight_id:30s} {root / spec.target}")
        if not complete:
            for output in spec_outputs(spec, root):
                if not output.exists():
                    print(f"        missing output: {output}")
                else:
                    threshold = spec.output_min_bytes if spec.expected_outputs else spec.min_bytes
                    if output.stat().st_size < threshold:
                        print(
                            f"        too small: {output} "
                            f"({byte_count(output.stat().st_size)} < {byte_count(threshold)})"
                        )
            ok = False
    return ok


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--only", nargs="*", help="Download only these model names or weight ids.")
    parser.add_argument("--exclude", nargs="*", help="Skip these model names or weight ids.")
    parser.add_argument("--include-optional", action="store_true", help="Include entries with incomplete public source metadata.")
    parser.add_argument("--continue-on-error", action="store_true", help="Try remaining files after a failure.")
    parser.add_argument("--force", action="store_true", help="Redownload existing targets.")
    parser.add_argument("--keep-archives", action="store_true", help="Keep archive files after successful extraction.")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP socket timeout in seconds.")
    parser.add_argument("--list", action="store_true", help="Print the manifest and exit.")
    parser.add_argument("--verify-only", action="store_true", help="Only check whether expected outputs exist.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = args.weights_root.expanduser().resolve()
    specs = selected_specs(args)

    if args.list:
        print_manifest(specs)
        return 0

    print(f"weights root: {root}")
    print(f"HF endpoint:  {DEFAULT_HF_ENDPOINT}")
    print(f"selected:     {len(specs)} files/archive entries")

    if args.verify_only:
        return 0 if verify(specs, root) else 1

    opener = build_opener(args.timeout)
    failures = []
    for spec in specs:
        ok = download_spec(spec, root, opener, args)
        if not ok:
            failures.append(spec.weight_id)
            if not args.continue_on_error:
                break

    print("\nverification:")
    verify_ok = verify(specs, root)
    if failures:
        print("\nfailed entries:")
        for item in failures:
            print(f"  - {item}")
    return 0 if verify_ok and not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
