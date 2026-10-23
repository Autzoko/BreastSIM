#!/usr/bin/env python
"""Download the open-source resources BreastSIM needs into data/.

Large resources are stored under $BREASTSIM_BIG/<name> and symlinked into
data/<name>; small ones live directly in data/. Every step is resumable:
finished items are recorded in <dataset>/.fetch_state.json.

    python src/scripts/fetch_data.py                 # everything
    python src/scripts/fetch_data.py --only oa_breast,real_busbra
    python src/scripts/fetch_data.py --list

IDB-8164905 (1.18 TB of zips) is processed one zip at a time: download ->
extract only *_label.mat / *_acou.mat -> delete the zip -> next.
"""
import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("BREASTSIM_DATA", ROOT / "data"))
BIG = Path(os.environ.get("BREASTSIM_BIG", DATA))
HOST = os.environ.get("BREASTSIM_HOST", "local")

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
      "(KHTML, like Gecko) Version/17.0 Safari/605.1.15")
DATAVERSE = "https://dataverse.harvard.edu"
IDB = "https://databank.illinois.edu"
OA_BOX = "rqivtin0xcofjwlkz43acou8jknsbfx8"
OA_BOX_FILES = {  # Box file id -> name (shared folder listing, 2026-10-22)
    "149428308952": "README.txt",
    "227927118715": "Neg_47_Left.h5",
    "227922466397": "Neg_07_Left.h5",
    "227918635078": "Neg_35_Left.h5",
}
ABUS_TDSC_127 = Path("/home/nyuair/data-001/langtian/data/medical-image/ABUS_TDSC")

GB = 1e9


# --------------------------------------------------------------------- utils
def log(msg):
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + msg, flush=True)


def http_get(url, referer=None):
    req = urllib.request.Request(url, headers={"User-Agent": UA, **({"Referer": referer} if referer else {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


ARIA2 = shutil.which("aria2c") or shutil.which("aria2c", path=str(Path(sys.executable).parent))


def download(url, dest, referer=None, size=None, attempts=30, ua=UA):
    """Resumable download; aria2c with 8 connections if available, else curl.

    Single-connection S3 (Illinois Data Bank) runs at ~5 MB/s on lab127,
    8 connections at ~55 MB/s.
    """
    dest = Path(dest)
    if size and dest.exists() and dest.stat().st_size == size:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not ARIA2:
        return curl(url, dest, referer, size, attempts, ua)
    cmd = [ARIA2, "-c", "-x", "8", "-s", "8", "-k", "20M", "--file-allocation=none",
           "--max-tries=10", "--retry-wait=30", "--allow-overwrite=true", "--auto-file-renaming=false",
           "--summary-interval=0", "--show-console-readout=false", "--console-log-level=warn",
           "--download-result=hide", "-U", ua, "-d", str(dest.parent), "-o", dest.name]
    if referer:
        cmd += [f"--referer={referer}"]
    control = dest.with_name(dest.name + ".aria2")
    for i in range(1, attempts + 1):
        # keep SIGPIPE ignored: aria2c dies with -13 on dropped sockets otherwise
        rc = subprocess.call(cmd + [url], restore_signals=False)
        complete = dest.exists() and not control.exists()
        if complete and (dest.stat().st_size == size if size else rc == 0):
            return dest
        log(f"  aria2c rc={rc} ({dest.stat().st_size if dest.exists() else 0} B), retry {i}/{attempts}")
        time.sleep(min(30 * i, 300))
    raise RuntimeError(f"download failed: {url}")


def curl(url, dest, referer=None, size=None, attempts=30, ua=UA):
    """Resumable download with curl; retries until the file is complete."""
    dest = Path(dest)
    if size and dest.exists() and dest.stat().st_size == size:
        return dest
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-fL", "-sS", "-C", "-", "-A", ua, "--connect-timeout", "30",
           "--speed-time", "120", "--speed-limit", "10000", "-o", str(part)]
    if referer:
        cmd += ["-e", referer]
    for i in range(1, attempts + 1):
        rc = subprocess.call(cmd + [url])
        # 33: server does not support range on an already complete file
        if rc in (0, 33) and (size is None or part.stat().st_size == size):
            part.rename(dest)
            return dest
        log(f"  curl rc={rc} ({part.stat().st_size if part.exists() else 0} B), retry {i}/{attempts}")
        if size and part.exists() and part.stat().st_size > size:
            part.unlink()
        time.sleep(min(30 * i, 300))
    raise RuntimeError(f"download failed: {url}")


def free_gb(path):
    return shutil.disk_usage(path).free / GB


class State:
    """Set of finished item keys, persisted next to the dataset."""

    def __init__(self, root):
        self.path = Path(root) / ".fetch_state.json"
        self.done = set(json.loads(self.path.read_text())) if self.path.exists() else set()

    def __contains__(self, key):
        return key in self.done

    def add(self, key):
        self.done.add(key)
        self.path.write_text(json.dumps(sorted(self.done), indent=1))


def dataset_dir(name, big):
    """Create the storage dir; for big datasets symlink data/<name> to it."""
    link = DATA / name
    if not big or BIG.resolve() == DATA.resolve():
        link.mkdir(parents=True, exist_ok=True)
        return link
    target = BIG / name
    target.mkdir(parents=True, exist_ok=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() and link.resolve() != target.resolve():
        link.unlink()
    if not link.exists():
        link.symlink_to(target, target_is_directory=True)
    return target


def unzip(zpath, out, members=None):
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if members is None or members(n)]
        for n in names:
            z.extract(n, out)
    return names


# ----------------------------------------------------------------- fetchers
def dataverse(doi, out, keep=lambda f: True):
    meta = json.loads(http_get(f"{DATAVERSE}/api/datasets/:persistentId/?persistentId={doi}"))
    files = [f for f in meta["data"]["latestVersion"]["files"] if keep(f)]
    total = sum(f["dataFile"]["filesize"] for f in files) / GB
    log(f"  {doi}: {len(files)} files, {total:.2f} GB")
    st = State(out)
    for f in files:
        df = f["dataFile"]
        rel = Path(f.get("directoryLabel") or "") / df["filename"]
        if str(rel) in st:
            continue
        log(f"  {rel} ({df['filesize'] / 1e6:.1f} MB)")
        download(f"{DATAVERSE}/api/access/datafile/{df['id']}", out / rel, size=df["filesize"])
        st.add(str(rel))


def idb_files(dataset_id):
    """{filename: datafile_id} scraped from an Illinois Data Bank dataset page."""
    page = html.unescape(http_get(f"{IDB}/datasets/{dataset_id}").decode())
    files = {}
    for m in re.finditer(r"/datafiles/([a-z0-9]+)/download", page):
        names = re.findall(r"([A-Za-z0-9_.-]+\.(?:zip|txt|mat|h5))", page[max(0, m.start() - 3000):m.start()])
        if names:
            files.setdefault(names[-1], m.group(1))
    return files


def idb_download(dataset_id, file_id, dest):
    return download(f"{IDB}/datafiles/{file_id}/download", dest, referer=f"{IDB}/datasets/{dataset_id}")


# ----------------------------------------------------------------- datasets
def fetch_nbp3d_usct():
    """Li et al. 2021, 4 3D USCT phantoms (BI-RADS A-D): label/sos/density/aa .mat."""
    dataverse("doi:10.7910/DVN/KBYQQ7", dataset_dir("nbp3d_usct", big=True))


def fetch_nbp2d_usct():
    """1,089 2D phantom slices (IDB-5648161) + Dataverse CUFVKE slices, no USCT measurements."""
    out = dataset_dir("nbp2d_usct", big=True)
    dataverse("doi:10.7910/DVN/CUFVKE", out / "dataverse",
              keep=lambda f: not (f.get("directoryLabel") or "").startswith("measurements"))
    st = State(out)
    if "NBPslices2D.zip" not in st:
        files = idb_files("IDB-5648161")
        z = idb_download("IDB-5648161", files["NBPslices2D.zip"], out / "NBPslices2D.zip")
        unzip(z, out / "idb")
        z.unlink()
        st.add("NBPslices2D.zip")


def fetch_oa_breast():
    """OA-Breast (WUSTL MRI-derived) merged phantoms, HDF5 version only."""
    out = dataset_dir("oa_breast", big=True)
    st = State(out)
    for fid, name in OA_BOX_FILES.items():
        if name in st:
            continue
        log(f"  {name}")
        download(f"https://wustl.box.com/index.php?rm=box_download_shared_file"
             f"&shared_name={OA_BOX}&file_id=f_{fid}", out / name)
        st.add(name)


def fetch_nbp3d_oat64(min_free_gb=150):
    """IDB-8164905: 64 sets x (healthy, lesion). Keep *_label.mat and *_acou.mat only."""
    out = dataset_dir("nbp3d_oat64", big=True)
    st = State(out)
    files = idb_files("IDB-8164905")
    zips = sorted(n for n in files if n.endswith(".zip"))
    log(f"  {len(zips)} zips listed, {sum(z in st for z in zips)} already done")
    if "README.txt" not in st:
        idb_download("IDB-8164905", files["README.txt"], out / "README.txt")
        st.add("README.txt")
    keep = re.compile(r"_(label|acou)\.mat$")
    for k, name in enumerate(zips, 1):
        if name in st:
            continue
        while free_gb(out) < min_free_gb:
            log(f"  only {free_gb(out):.0f} GB free on {out}, waiting (need {min_free_gb})")
            time.sleep(600)
        t0 = time.time()
        log(f"  [{k}/{len(zips)}] {name}: downloading")
        z = idb_download("IDB-8164905", files[name], out / "_incoming" / name)
        kept = unzip(z, out / name[:-4], members=lambda n: bool(keep.search(n)))
        dropped = len(zipfile.ZipFile(z).namelist()) - len(kept)
        size = z.stat().st_size / GB
        z.unlink()
        kept_gb = sum(p.stat().st_size for p in (out / name[:-4]).rglob("*") if p.is_file()) / GB
        log(f"  [{k}/{len(zips)}] {name}: zip {size:.1f} GB -> kept {len(kept)} files "
            f"({kept_gb:.1f} GB), dropped {dropped}, {(time.time() - t0) / 60:.1f} min, "
            f"{free_gb(out):.0f} GB free")
        st.add(name)


def fetch_real_busbra():
    """BUS-BRA (Zenodo 8231412): 1,875 2D breast US images with masks."""
    out = dataset_dir("real/busbra", big=False)
    st = State(out)
    if "BUSBRA.zip" not in st:
        # Zenodo answers 403 to browser-like user agents
        z = download("https://zenodo.org/api/records/8231412/files/BUSBRA.zip/content", out / "BUSBRA.zip",
                     ua="BreastSIM-fetch/0.1")
        unzip(z, out)
        z.unlink()
        st.add("BUSBRA.zip")


def fetch_real_breast_lesions_usg():
    """BrEaST-Lesions_USG (TCIA): 256 2D breast US images with masks + clinical data."""
    out = dataset_dir("real/breast_lesions_usg", big=False)
    st = State(out)
    base = "https://www.cancerimagingarchive.net/wp-content/uploads/"
    for name in ["BrEaST-Lesions_USG-images_and_masks-Dec-15-2023.zip",
                 "BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx"]:
        if name in st:
            continue
        f = download(base + name, out / name)
        if name.endswith(".zip"):
            unzip(f, out)
            f.unlink()
        st.add(name)


def fetch_real_abus_tdsc():
    """TDSC-ABUS 2023: already on lab127, linked in place (not downloaded)."""
    link = DATA / "real" / "abus_tdsc"
    if link.exists():
        return
    if HOST == "lab127" and ABUS_TDSC_127.exists():
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(ABUS_TDSC_127, target_is_directory=True)
    else:
        log("  TDSC-ABUS: request from https://tdsc-abus2023.grand-challenge.org/ and place at data/real/abus_tdsc")


def fetch_real_busi():
    """BUSI (780 2D breast US images with masks). The official host (scholar.cu.edu.eg) sits behind a
    browser challenge, so the unmodified archive is taken from a Hugging Face mirror."""
    out = dataset_dir("real/busi", big=False)
    st = State(out)
    if "busi_mirror" not in st:
        z = download("https://huggingface.co/datasets/gymprathap/Breast-Cancer-Ultrasound-Images-Dataset/"
                     "resolve/main/Breast-Cancer-Ultrasound-Images-Dataset.zip", out / "BUSI_mirror.zip")
        unzip(z, out / "extracted")
        z.unlink()
        st.add("busi_mirror")


DATASETS = {
    "nbp3d_usct": fetch_nbp3d_usct,
    "nbp2d_usct": fetch_nbp2d_usct,
    "oa_breast": fetch_oa_breast,
    "real_busbra": fetch_real_busbra,
    "real_breast_lesions_usg": fetch_real_breast_lesions_usg,
    "real_abus_tdsc": fetch_real_abus_tdsc,
    "real_busi": fetch_real_busi,
    "nbp3d_oat64": fetch_nbp3d_oat64,  # last: ~1.2 TB streamed zip by zip
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma-separated dataset names")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for k, f in DATASETS.items():
            print(f"{k:28s} {f.__doc__.strip().splitlines()[0]}")
        return
    names = args.only.split(",") if args.only else list(DATASETS)
    log(f"host={HOST} data={DATA} big={BIG} ({free_gb(BIG if BIG.exists() else DATA):.0f} GB free)")
    failed = []
    for n in names:
        log(f"== {n}")
        try:
            DATASETS[n]()
            log(f"== {n}: done")
        except Exception as e:  # keep going with the other datasets
            log(f"== {n}: FAILED: {e!r}")
            failed.append(n)
    log("all done" if not failed else f"failed: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
