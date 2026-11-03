#!/usr/bin/env python3
"""
Accumulated rainfall map for Kanchanaburi from satellite-gauge data.

Default source: CHIRPS v2 daily (0.05 deg, free, no login).
Optional source: NASA GPM IMERG (30-min, 0.1 deg, needs a free Earthdata login).

Examples
  python kanchanaburi_rain_map.py --days 7
  python kanchanaburi_rain_map.py --end 2026-10-28 --days 7 --boundary kanchanaburi.geojson
  python kanchanaburi_rain_map.py --source imerg --days 3     # needs: pip install earthaccess h5py

Install:  pip install numpy requests rasterio matplotlib
Optional: pip install geopandas   (to draw a province boundary from --boundary)
"""
import argparse
import datetime as dt
import gzip
import sys
from pathlib import Path

import numpy as np
import requests
import rasterio
from rasterio.windows import from_bounds
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# lon_min, lat_min, lon_max, lat_max  (a little wider than Kanchanaburi)
BBOX = (98.2, 13.6, 100.0, 15.5)

# Approximate district centres (lon, lat) used to print point values
POINTS = {
    "Mueang Kanchanaburi": (99.55, 14.00),
    "Bo Phloi": (99.50, 14.34),
    "Lao Khwan": (99.75, 14.80),
    "Si Sawat": (99.10, 14.60),
    "Thong Pha Phum": (98.64, 14.74),
    "Sangkhla Buri": (98.47, 15.14),
}

CHIRPS_URL = ("https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/"
              "tifs/p05/{y}/chirps-v2.0.{y}.{m:02d}.{d:02d}.tif.gz")

LEVELS = [0, 1, 5, 10, 20, 30, 50, 75, 100, 125, 150]          # mm
COLORS = ["#ffffff", "#d6ecf7", "#9ed3ea", "#4f9fc2", "#2f8f4e",
          "#9bc53d", "#f4e04d", "#f5a623", "#e8542a", "#b3201a"]


def chirps_day(day, cache):
    path = cache / f"chirps-{day:%Y%m%d}.tif"
    if not path.exists():
        url = CHIRPS_URL.format(y=day.year, m=day.month, d=day.day)
        r = requests.get(url, timeout=180)
        if r.status_code == 404:
            sys.exit(f"CHIRPS has no file for {day} yet. Try an earlier --end date.")
        r.raise_for_status()
        path.write_bytes(gzip.decompress(r.content))
    with rasterio.open(path) as src:
        win = from_bounds(*BBOX, transform=src.transform)
        arr = src.read(1, window=win, masked=True).astype("float32")
        tr = src.window_transform(win)
    return np.ma.masked_less(arr, 0), tr


def imerg_total(start, end, cache):
    """Sum of IMERG Final daily precipitation (mm/day). Needs Earthdata login."""
    import earthaccess
    import h5py
    earthaccess.login()
    res = earthaccess.search_data(short_name="GPM_3IMERGDF", version="07",
                                  temporal=(f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}"),
                                  bounding_box=BBOX)
    if not res:
        sys.exit("No IMERG files found. Check the dates (Final run has ~3-4 months latency; "
                 "use GPM_3IMERGDL for the Late run).")
    files = earthaccess.download(res, str(cache))
    total = None
    for f in files:
        with h5py.File(f) as h:
            lon = h["Grid/lon"][:]
            lat = h["Grid/lat"][:]
            p = h["Grid/precipitation"][0]          # (lon, lat)
            xi = np.where((lon >= BBOX[0]) & (lon <= BBOX[2]))[0]
            yi = np.where((lat >= BBOX[1]) & (lat <= BBOX[3]))[0]
            sub = np.ma.masked_less(p[np.ix_(xi, yi)].T[::-1], 0)   # north-up
            total = sub if total is None else total + sub
    res_deg = 0.1
    left, top = lon[xi].min() - res_deg / 2, lat[yi].max() + res_deg / 2
    tr = rasterio.transform.from_origin(left, top, res_deg, res_deg)
    return total, tr


def sample(total, tr, lon, lat):
    col, row = ~tr * (lon, lat)
    r, c = int(row), int(col)
    if 0 <= r < total.shape[0] and 0 <= c < total.shape[1] and not np.ma.is_masked(total[r, c]):
        return float(total[r, c])
    return float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--end", type=lambda s: dt.date.fromisoformat(s),
                    default=dt.date.today() - dt.timedelta(days=3),
                    help="last day of the window (default: today - 3 days)")
    ap.add_argument("--days", type=int, default=7, help="window length in days")
    ap.add_argument("--source", choices=["chirps", "imerg"], default="chirps")
    ap.add_argument("--boundary", help="province boundary file (GeoJSON/Shapefile) to overlay")
    ap.add_argument("--out", default="kanchanaburi_rain.png")
    ap.add_argument("--cache", default="rain_cache")
    a = ap.parse_args()

    cache = Path(a.cache)
    cache.mkdir(exist_ok=True)
    start = a.end - dt.timedelta(days=a.days - 1)

    if a.source == "chirps":
        arrays, tr = [], None
        for i in range(a.days):
            arr, tr = chirps_day(start + dt.timedelta(days=i), cache)
            arrays.append(arr)
        total = np.ma.stack(arrays).sum(axis=0)
    else:
        total, tr = imerg_total(start, a.end, cache)

    h, w = total.shape
    left, top = tr.c, tr.f
    right, bottom = left + tr.a * w, top + tr.e * h

    cmap = ListedColormap(COLORS)
    cmap.set_over("#5c0000")
    cmap.set_bad("#eeeeee")
    norm = BoundaryNorm(LEVELS, cmap.N)

    fig, ax = plt.subplots(figsize=(6.5, 7.5), dpi=150)
    im = ax.imshow(total, extent=(left, right, bottom, top), cmap=cmap, norm=norm,
                   interpolation="nearest")
    if a.boundary:
        import geopandas as gpd
        gpd.read_file(a.boundary).to_crs(4326).boundary.plot(ax=ax, color="black", linewidth=0.8)
    for name, (lon, lat) in POINTS.items():
        ax.plot(lon, lat, "k.", ms=4)
        ax.annotate(name, (lon, lat), xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set_xlim(BBOX[0], BBOX[2])
    ax.set_ylim(BBOX[1], BBOX[3])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Total precipitation {a.days}D, {start} to {a.end}\n"
                 f"Kanchanaburi ({a.source.upper()})", fontsize=11, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, extend="max", shrink=0.7, label="Rain (mm)")
    cb.set_ticks(LEVELS)
    fig.tight_layout()
    fig.savefig(a.out)
    print(f"saved {a.out}")

    print(f"\nTotal rain {start} to {a.end} (mm)")
    for name, (lon, lat) in POINTS.items():
        print(f"  {name:22s} {sample(total, tr, lon, lat):7.1f}")
    print(f"  {'Area max':22s} {float(total.max()):7.1f}")


if __name__ == "__main__":
    main()
