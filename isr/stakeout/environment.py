"""What the sky was doing while the camera watched: sun position (computed) and weather history (fetched).

Both exist to explain movement that is not somebody: gusts shaking a hedge,
rain crossing the lens, the light changing at dawn and dusk.

The sun needs no network. Weather history comes from Open-Meteo, a free and
open weather service that needs no account. The only things sent to it are the
camera's position **rounded to about a kilometre** and the dates watched.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta

FORECAST = "https://api.open-meteo.com/v1/forecast"  # also serves roughly the last three months
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"  # reanalysis, from 1940 to about a week ago
FIELDS = "precipitation,snowfall,wind_speed_10m,wind_gusts_10m,cloud_cover"
GUSTY_KMH, WET_MM = 38.0, 0.6  # an hour counts as windy / wet from here up


class WeatherError(Exception):
    pass


def sun_elevation(lat: float, lon: float, when: datetime, utc_offset_h: float) -> float:
    """Degrees above the horizon at a local clock time (NOAA's low-accuracy formula, good to a fraction of a degree)."""
    n = when.timetuple().tm_yday
    hour = when.hour + when.minute / 60 + when.second / 3600
    g = 2 * math.pi / 365 * (n - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g) - 0.014615 * math.cos(2 * g)
                       - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    tst = hour * 60 + eqtime + 4 * lon - 60 * utc_offset_h  # true solar time, minutes
    ha = math.radians(tst / 4 - 180)
    la = math.radians(lat)
    cz = math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(ha)
    return 90 - math.degrees(math.acos(max(-1.0, min(1.0, cz))))


def sun_times(lat: float, lon: float, day: date, utc_offset_h: float):
    """(sunrise, sunset) as local clock times, or (None, None) in polar day or night."""
    t0 = datetime(day.year, day.month, day.day)
    up = [sun_elevation(lat, lon, t0 + timedelta(minutes=m), utc_offset_h) > -0.833 for m in range(0, 1440, 2)]
    if all(up) or not any(up):
        return None, None
    rise = next((i for i in range(len(up)) if up[i] and not up[i - 1]), None)
    sett = next((i for i in range(len(up)) if not up[i] and up[i - 1]), None)
    f = lambda i: None if i is None else (t0 + timedelta(minutes=2 * i)).strftime("%H:%M")  # noqa: E731
    return f(rise), f(sett)


def guess_utc_offset(lon: float) -> float:
    """Clock offset from longitude alone: right to within an hour or two, used only when weather was not fetched."""
    return float(round(lon / 15))


def fetch_weather(lat: float, lon: float, first: date, last: date, timeout: float = 20.0, opener=None) -> dict:
    """Hourly weather for the days watched, keyed by local hour ("2026-11-09T14").

    Returns {"hours": {key: {"gust", "wind", "rain", "snow", "cloud"}}, "utc_offset_h", "timezone", "source"}.
    """
    q = {"latitude": f"{lat:.2f}", "longitude": f"{lon:.2f}", "hourly": FIELDS, "start_date": first.isoformat(),
         "end_date": last.isoformat(), "timezone": "auto", "wind_speed_unit": "kmh"}
    recent = (date.today() - first).days <= 80
    errors = []
    for url in ((FORECAST, ARCHIVE) if recent else (ARCHIVE, FORECAST)):
        try:
            req = urllib.request.Request(url + "?" + urllib.parse.urlencode(q), headers={"User-Agent": "isrstudio"})
            with (opener or urllib.request.urlopen)(req, timeout=timeout) as r:
                d = json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                errors.append(json.loads(e.read()).get("reason") or str(e))
            except Exception:
                errors.append(str(e))
            continue
        except Exception as e:  # no network, DNS, timeout
            errors.append(str(getattr(e, "reason", e)))
            continue
        h = d.get("hourly") or {}
        if not h.get("time"):
            errors.append("no hourly data returned")
            continue
        hours = {}
        for i, t in enumerate(h["time"]):
            v = lambda k: (h.get(k) or [None] * (i + 1))[i]  # noqa: E731
            if v("wind_gusts_10m") is None and v("precipitation") is None:
                continue  # the archive has not caught up with this hour yet
            hours[t[:13]] = {"gust": v("wind_gusts_10m") or 0.0, "wind": v("wind_speed_10m") or 0.0,
                             "rain": v("precipitation") or 0.0, "snow": v("snowfall") or 0.0, "cloud": v("cloud_cover")}
        if hours:
            return {"hours": hours, "utc_offset_h": (d.get("utc_offset_seconds") or 0) / 3600,
                    "timezone": d.get("timezone") or "", "source": "Open-Meteo " + ("forecast archive" if url == FORECAST else "reanalysis (ERA5)")}
        errors.append("no weather for those dates yet")
    raise WeatherError("; ".join(dict.fromkeys(errors)) or "no answer")
