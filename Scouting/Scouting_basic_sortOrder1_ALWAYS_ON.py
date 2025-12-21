import datetime as _dt
import math

# MAE Scouting – patch13n19d
# Change from 13n19c: **Family unification rule**
# - Any event code that starts with "ILCMP" is mapped to the same family "ILCMP".
#   This unifies Israeli Championship divisions (e.g., ILCMPHDRO / ILCMPSOLR / ILCMPARC / ILCMPPCF) with ILCMP finals.
# Everything else identical to 13n19c.

import streamlit as st
import typing as t

from datetime import datetime


def _coerce_season(season):
    """
    Clamp season to API supported range: 2006..current active year.
    """
    try:
        y = int(season)
    except Exception:
        try:
            import streamlit as st
            from datetime import datetime as _dtnow
            y = int(st.session_state.get("season") or st.session_state.get("Season") or _dtnow.now().year)
        except Exception:
            from datetime import datetime as _dtnow
            y = _dtnow.now().year
    MIN_SEASON = 2006
    from datetime import datetime as _dtnow
    current_active = _dtnow.now().year
    if y < MIN_SEASON:
        y = MIN_SEASON
    if y > current_active:
        y = current_active
    return y
def _set_creds_state(u, t, meta=None):
    try:
        import streamlit as st
        st.session_state['creds_ok'] = bool(u and t)
        st.session_state['creds_source'] = (meta or {}).get('source') if isinstance(meta, dict) else None
        st.session_state['creds_path'] = (meta or {}).get('path') if isinstance(meta, dict) else None
    except Exception:
        pass


def has_creds():
    try:
        import streamlit as st, os
        if st.session_state.get('creds_ok'):
            return True
        return bool(os.getenv('FTC_API_USER') and os.getenv('FTC_API_TOKEN'))
    except Exception:
        return False

def _ftc_api_diagnostics(season: int):
    """Quick diagnostic call to FTC API for events endpoint."""
    try:
        import requests, os
        season = _coerce_season(season) if "_coerce_season" in globals() else season
        url = f"https://ftc-api.firstinspires.org/v2.0/{season}/events"
        u = globals().get("FTC_USER") or os.getenv("FTC_API_USER") or os.getenv("FTC_USER")
        k = globals().get("FTC_TOKEN") or os.getenv("FTC_API_TOKEN") or os.getenv("FTC_API_KEY")
        auth = (u, k) if (u and k) else None
        r = requests.get(url, auth=auth, timeout=20)
        sample = r.text[:600] if isinstance(r.text, str) else str(r.text)[:600]
        return {"ok": (200 <= r.status_code < 300), "status": r.status_code, "url": r.url, "sample": sample}
    except Exception as e:
        return {"ok": False, "status": None, "url": None, "sample": f"Exception: {e}"}

# --- HTTP Delta Cache (per endpoint+params) ---
from pathlib import Path as _Path
import json as _json, hashlib as _hashlib, os as _os, time as _time

_FTC_HTTP_DIR = _Path("./.ftc_cache/http")
_FTC_HTTP_DIR.mkdir(parents=True, exist_ok=True)
_FTC_META = _Path("./.ftc_cache/meta.json")
try:
    _META = _json.loads(_FTC_META.read_text(encoding="utf-8"))
except Exception:
    _META = {}


def _key_for(path, params):
    s = path + "|" + "&".join(f"{k}={params[k]}" for k in sorted(params.keys())) if params else path
    return _hashlib.sha1(s.encode("utf-8")).hexdigest()


def _read_body(key):
    p = _FTC_HTTP_DIR / f"{key}.json"
    if p.exists():
        try:
            return _json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _write_body(key, body):
    p = _FTC_HTTP_DIR / f"{key}.json"
    p.write_text(_json.dumps(body), encoding="utf-8")


def _save_meta():
    _FTC_META.parent.mkdir(parents=True, exist_ok=True)
    _FTC_META.write_text(_json.dumps(_META, indent=2), encoding="utf-8")


import numpy as np
import pandas as pd
import requests
import streamlit as st
import re




import os, json, glob
def load_ftc_credentials():

    import os, json, glob, streamlit as st, hashlib

    FIXED_PATH   = r"C:\Users\RoboMentor\Robot_stratagy\Scouting"
    ROAMING_PATH = r"C:\Users\RoboMentor\AppData\Roaming\mae_scout"

    def _try_read(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            user  = data.get("username") or data.get("user")
            token = data.get("token")    or data.get("password")
            if user and token:
                return user, token, path
        except Exception:
            pass
        return None, None, None

    def _sha1(path):
        try:
            h = hashlib.sha1()
            with open(path, "rb") as f:
                h.update(f.read())
            return h.hexdigest()[:12]
        except Exception:
            return None

    # אסוף מועמדים (לזיהוי כפילויות)
    candidates = []
    def _add_candidates(base):
        for pat in ["ftc_credentials.json", "FTC_credentials.json", "*.JASON"]:
            for f in glob.glob(os.path.join(base, pat)):
                if os.path.isfile(f):
                    candidates.append(os.path.abspath(f))

    _add_candidates(FIXED_PATH)
    _add_candidates(".")
    _add_candidates(ROAMING_PATH)

    # 1) חיפוש בנתיב הקבוע
    for f in glob.glob(os.path.join(FIXED_PATH, "ftc_credentials.json")) + \
             glob.glob(os.path.join(FIXED_PATH, "FTC_credentials.json")) + \
             glob.glob(os.path.join(FIXED_PATH, "*.JASON")):
        u, t, p = _try_read(f)
        if u and t:
            _report_credentials_source(p, candidates)
            return u, t

    # 2) חיפוש בתיקייה הנוכחית
    for f in glob.glob("ftc_credentials.json") + glob.glob("FTC_credentials.json") + glob.glob("*.JASON"):
        u, t, p = _try_read(f)
        if u and t:
            _report_credentials_source(os.path.abspath(p), candidates)
            return u, t

    # 3) משתני סביבה
    u = os.getenv("FTC_API_USER")
    t = os.getenv("FTC_API_TOKEN")
    if u and t:
        st.info("🔑 Credentials loaded from environment variables (FTC_API_USER / FTC_API_TOKEN).")
        if candidates:
            st.warning("זוהו גם קובצי אישורים בתיקיות, אך נטען מה-ENV. מומלץ לאחד לנתיב קבוע.")
        return u, t

    # 4) בחירת קובץ / העלאה (פופ-אפ)
    st.warning("⚠️ לא נמצאו אישורים. ניתן לבחור/להעלות קובץ JSON ונשמור אותו בנתיב הקבוע.")
    open_picker = st.button("📂 בחר/העלה קובץ Credentials")
    if open_picker:
        uploaded = st.file_uploader("בחר את קובץ ה-FTC credentials", type=["json","JASON","txt"], accept_multiple_files=False)
        if uploaded is not None:
            try:
                data  = json.loads(uploaded.getvalue().decode("utf-8", errors="ignore"))
                user  = data.get("username") or data.get("user")
                token = data.get("token")    or data.get("password")
                if not (user and token):
                    st.error("❌ הקובץ חסר שדות username/token.")
                    st.stop()
                os.makedirs(FIXED_PATH, exist_ok=True)
                out_path = os.path.join(FIXED_PATH, "ftc_credentials.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump({"username": user, "token": token}, f, ensure_ascii=False, indent=2)
                st.success(f"✅ נשמר בהצלחה: {out_path}")
                _report_credentials_source(out_path, candidates + [out_path])
                return user, token
            except Exception as e:
                st.error(f"שגיאה בקריאת/שמירת הקובץ: {e}")
                st.stop()

    st.error("""
    ❌ לא נמצאו ולא הועלו אישורים ל-FTC API.
    אפשר לשמור מראש קובץ:
    C:\\Users\\RoboMentor\\Robot_stratagy\\Scouting\\ftc_credentials.json
    בפורמט:
    {
      "username": "your_username",
      "token": "your_api_token"
    }
    או להגדיר משתני סביבה:
      FTC_API_USER   ו-  FTC_API_TOKEN
    """)
    st.stop()


def _report_credentials_source(used_path, all_candidates):
    """מציג מאיזה קובץ נטענו האישורים ומתריע על כפילויות עם תוכן שונה."""
    import os, streamlit as st, hashlib
    used_path = os.path.abspath(used_path)
    st.info(f"🔑 Credentials loaded from: `{used_path}`")

    def _sha1(p):
        try:
            h = hashlib.sha1()
            with open(p, "rb") as f:
                h.update(f.read())
            return h.hexdigest()[:12]
        except Exception:
            return None

    others = sorted(set(os.path.abspath(p) for p in all_candidates if os.path.abspath(p) != used_path))
    if not others:
        return

    # השוואת hash כדי לזהות תוכן שונה
    def _hash(p):
        try:
            h = hashlib.sha1()
            with open(p, "rb") as f:
                h.update(f.read())
            return h.hexdigest()[:12]
        except Exception:
            return None

    used_hash = _hash(used_path)
    conflicts = []
    for p in others:
        h = _hash(p)
        if h and used_hash and h != used_hash:
            conflicts.append((p, h))

    if conflicts:
        st.warning("⚠️ נמצאו קובצי אישורים נוספים עם תוכן שונה:")
        for p, h in conflicts:
            st.write(f"- `{p}` (sha1:{h})")
        st.caption("מומלץ להשאיר קובץ אחד בלבד בנתיב הקבוע ולאחד גרסאות.")


# הפוך את האישורים לגלובליים לשאר האפליקציה
FTC_USER, FTC_TOKEN = load_ftc_credentials()


def dedupe_teams_master(teams_master: "pd.DataFrame", ev_view: "pd.DataFrame") -> "pd.DataFrame":
    import pandas as pd
    if teams_master is None or not isinstance(teams_master, pd.DataFrame) or teams_master.empty:
        return pd.DataFrame(columns=["team", "team_name", "country", "ev", "ev_family"])
    ev_dates = ev_view.rename(columns={"event_code": "ev"})[["ev", "start_dt"]].drop_duplicates()
    tm = teams_master.merge(ev_dates, on="ev", how="left")
    if "start_dt" in tm.columns:
        tm = tm.sort_values(["team", "start_dt", "ev_family", "ev"])
    else:
        tm["__ord__"] = range(len(tm))
        tm = tm.sort_values(["team", "__ord__"])
    tm = tm.drop_duplicates(subset=["team"], keep="last")
    keep_cols = [c for c in ["team", "team_name", "country", "ev", "ev_family", "start_dt"] if c in tm.columns]
    return tm[keep_cols]


@st.cache_data(ttl=900, show_spinner=False)
def events_list_cached(season: int):
    """
    Cached wrapper for events_list(season).
    Ensures a DataFrame is returned and that a 'country' column exists.
    """
    try:
        df = events_list(season)  # uses existing function in this file
    except Exception as e:
        try:
            st.warning(f"טעינת רשימת אירועים נכשלה: {e}")
        except Exception:
            pass
        import pandas as pd
        return pd.DataFrame(columns=["event_code", "name", "country", "start_dt"])
    import pandas as pd
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(columns=["event_code", "name", "country", "start_dt"])
    if "country" not in df.columns:
        # fallback: try 'region' or fill empty
        df["country"] = df["region"] if "region" in df.columns else ""
    return df


@st.cache_data(ttl=900, show_spinner=True)
def _safe_dataset(season: int, country: str, max_workers: int):
    """
    Guaranteed to return a 3-tuple of DataFrames.
    Falls back to last_dataset in session_state or empty DFs.
    """
    try:
        res = fetch_season_data(int(season), country, int(max_workers))
    except Exception as e:
        st.warning(f"טעינת דאטה נכשלה (ננסה פולבאק): {e}")
        res = None

    if not isinstance(res, (tuple, list)) or len(res) != 3:
        if "last_dataset" in st.session_state:
            st.info("שימוש בדאטה האחרון שנשמר בזיכרון.")
            return st.session_state["last_dataset"]
        st.error("שרת החזיר מבנה לא תקין — מציג דאטה ריק.")
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    # Normalize to tuple of DFs and store snapshot
    ev_view, base_raw, teams_master = res[0], res[1], res[2]
    st.session_state["last_dataset"] = (ev_view, base_raw, teams_master)
    return (ev_view, base_raw, teams_master)


# ===== Cached dataset + signatures to avoid re-compute between tabs =====
@st.cache_data(ttl=900, show_spinner=False)
def _cached_dataset(season: int, country: str, max_workers: int):
    try:
        ev_view, base_raw, teams_master = _safe_dataset(int(season), str(country), int(max_workers))
    except Exception:
        if "last_dataset" in st.session_state:
            return st.session_state["last_dataset"]
        import pandas as pd
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    return (ev_view, base_raw, teams_master)


# ===== Helpers for Event (Live) tab =====
def _ev_col(df, candidates):
    if df is None: return None
    for c in candidates:
        if c in df.columns: return c
    # try case-insensitive match
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low: return low[c.lower()]
    return None


def _latest_event_code(ev_df):
    if ev_df is None or ev_df.empty:
        return None, None
    code_col = _ev_col(ev_df, ["eventCode", "code", "ev", "event_code"])
    name_col = _ev_col(ev_df, ["name", "eventName", "event_name"])
    end_col = _ev_col(ev_df, ["end_dt", "endDate", "end", "end_date"])
    start_col = _ev_col(ev_df, ["start_dt", "startDate", "start", "start_date"])
    df = ev_df.copy()
    if end_col and df[end_col].notna().any():
        df = df.sort_values(by=[end_col], ascending=[True])
    elif start_col and df[start_col].notna().any():
        df = df.sort_values(by=[start_col], ascending=[True])
    row = df.iloc[-1]
    return (row.get(code_col, None), row.get(name_col, None))


def _teams_in_event_from_base(base_df, ev_code):
    import re
    import pandas as pd
    if base_df is None or base_df.empty or ev_code is None:
        return []

    def _ev_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        low = [c for c in df.columns if "event" in c.lower() and "code" in c.lower()]
        return low[0] if low else None

    evcol = _ev_col(base_df, ["eventCode", "event_code", "ev", "code"])
    famcol = _ev_col(base_df, ["ev_family", "family", "event_family", "family_code"])
    b = base_df.copy()
    family_target = None
    if evcol is not None and famcol is not None:
        rows_for_code = b[b[evcol].astype(str) == str(ev_code)]
        if not rows_for_code.empty:
            v = rows_for_code.iloc[0][famcol] if famcol in rows_for_code.columns else None
            family_target = str(v) if (v is not None and pd.notna(v)) else None
    if family_target is None:
        ev_str = str(ev_code)
        m = re.match(r"^([A-Z]+CMP)", ev_str, flags=re.I)
        if m:
            family_target = m.group(1).upper()
    if family_target and famcol is not None:
        b = b[b[famcol].astype(str).str.upper() == str(family_target).upper()].copy()
    elif evcol is not None:
        b = b[b[evcol].astype(str) == str(ev_code)].copy()
    team_cols = [c for c in b.columns if re.search(r'(red|blue).*?(1|2|3)$', c, re.I) or 'team' in c.lower()]
    teams = set()
    for c in team_cols:
        try:
            teams |= set(pd.to_numeric(b[c], errors="coerce").dropna().astype(int).tolist())
        except Exception:
            pass
    return sorted(list(teams))


def _team_col(df):
    return _ev_col(df, ["team", "teamNumber", "team_number", "number"])


def _epa_col(df):
    return _ev_col(df, ["EPA", "epa", "epa_final", "epaMean", "opr", "OPR"])


def _adv_team_col(df):
    return _ev_col(df, ["team", "teamNumber", "team_number", "number"])


def _adv_points_col(df):
    return _ev_col(df, ["AdvancementPoints", "adv_points", "adv", "totalPoints", "total_points"])



def _teams_seen(teams_master, base_raw):
    import pandas as pd
    try:
        if getattr(teams_master, "empty", True) is False:
            for c in ["team","teamNumber","team_number","number","Team","TEAM"]:
                if c in teams_master.columns:
                    try:
                        return int(pd.to_numeric(teams_master[c], errors="coerce").dropna().nunique())
                    except Exception:
                        return int(teams_master[c].dropna().nunique())
    except Exception:
        pass
    try:
        if getattr(base_raw, "empty", True) is False:
            # try common per-match team columns
            for c in ["team","teamNumber","team_number","number","t1","t2","r1","r2","b1","b2","red1","red2","blue1","blue2"]:
                if c in base_raw.columns:
                    try:
                        return int(pd.to_numeric(base_raw[c], errors="coerce").dropna().nunique())
                    except Exception:
                        return int(base_raw[c].dropna().nunique())
    except Exception:
        pass
    return 0

def dataset_signature(ev_view, base_raw, teams_master):
    try:
        n_ev = len(ev_view) if ev_view is not None else 0
        n_base = len(base_raw) if base_raw is not None else 0
        n_team = len(teams_master) if teams_master is not None else 0
        last_dt = None
        if ev_view is not None and "start_dt" in ev_view.columns and not ev_view["start_dt"].dropna().empty:
            last_dt = str(ev_view["start_dt"].max())
        return (n_ev, n_base, n_team, last_dt)
    except Exception:
        return (0, 0, 0, "na")


@st.cache_data(ttl=600, show_spinner=False)
def build_rankings_cached(sig, base_raw, ev_view):
    try:
        if 'build_rankings' in globals():
            return globals()['build_rankings'](sig, base_raw, ev_view)
    except Exception:
        pass
    return None


@st.cache_data(ttl=600, show_spinner=False)
def build_advancement_cached(sig, base_raw, teams_master, ev_view, country, season):
    try:
        if 'build_advancement' in globals():
            return globals()['build_advancement'](sig, base_raw, teams_master, ev_view, country, season)
        if 'compute_advancement_table' in globals():
            return globals()['compute_advancement_table'](ev_view, base_raw, teams_master, country, season, None)
    except Exception:
        pass
    return None


from datetime import datetime

# ===== Snapshot cache of computed tables (per season,country) =====
from pathlib import Path as _Pth
import pandas as _pd
import time as _time

_SNAP_DIR = _Pth("./.ftc_cache/snapshots")
_SNAP_DIR.mkdir(parents=True, exist_ok=True)


def _snapshot_path(kind: str, season: int, country: str) -> _Pth:
    safe_country = (country or "NA").replace(" ", "_")
    return _SNAP_DIR / f"{kind}_season{season}_{safe_country}.parquet"


def _load_snapshot(kind: str, season: int, country: str, max_age_sec: int = 12 * 3600):
    p = _snapshot_path(kind, season, country)
    if p.exists():
        try:
            if (_time.time() - p.stat().st_mtime) <= max_age_sec:
                return _pd.read_parquet(p)
        except Exception:
            csvp = p.with_suffix(".csv")
            if csvp.exists() and (_time.time() - csvp.stat().st_mtime) <= max_age_sec:
                try:
                    return _pd.read_csv(csvp)
                except Exception:
                    pass
    return None


def _save_snapshot(kind: str, season: int, country: str, df):
    if df is None:
        return
    p = _snapshot_path(kind, season, country)
    try:
        df.to_parquet(p, index=False)
    except Exception:
        df.to_csv(p.with_suffix(".csv"), index=False, encoding="utf-8-sig")


USA_MAX_WORKERS = 3

def _coerce_season(season):
    try:
        return int(season)
    except Exception:
        try:
            import streamlit as st
            s = st.session_state.get("season") or st.session_state.get("Season") or DEFAULT_SEASON
            return int(s)
        except Exception:
            return DEFAULT_SEASON

USA_CHUNK_SIZE = 25

import time, requests


def _http_get(url, headers=None, params=None, timeout=20.0, retries=2, **kwargs):
    import requests, time
    last_exc = None
    for attempt in range(retries + 1):
        try:
            kwargs = dict(kwargs)
            kwargs.setdefault('auth', (FTC_USER, FTC_TOKEN))
            resp = requests.get(url, headers=headers, params=params, timeout=timeout, **kwargs)
            if resp.status_code >= 500:
                last_exc = RuntimeError(f"HTTP {resp.status_code} at {url}")
            else:
                resp.raise_for_status()
                return resp
        except Exception as e:
            last_exc = e
        time.sleep(min(1.5 * (attempt + 1), 5))
    raise last_exc if last_exc else RuntimeError("HTTP GET failed")


from pathlib import Path


# === AdvPoints: reuse the app's existing FTC API session & wrappers ===
def adv_api_events(season: int) -> pd.DataFrame:
    try:
        return events_list(int(season))
    except Exception:
        return pd.DataFrame()


def adv_api_rankings(season: int, code: str) -> pd.DataFrame:
    ok, s, js, url = api_rankings(int(season), str(code))
    rows = []
    items = (js.get("rankings") or js.get("Rankings") or []) if ok and isinstance(js, dict) else []
    for r in items:
        try:
            team = int(r.get("teamNumber") or r.get("team") or r.get("teamId") or r.get("number"))
        except Exception:
            team = None
        try:
            rank = int(r.get("rank") or r.get("ranking") or r.get("qualAverageRank") or r.get("qualRank"))
        except Exception:
            rank = None
        if team and rank:
            rows.append({"team": team, "rank": rank})
    return pd.DataFrame(rows)


def adv_api_alliances(season: int, code: str):
    """
    Return a list like: [{"number": 1, "teams": [1234, 5678, 9012]}, ].
    Primary source: /v2.0/{season}/alliances/{eventCode} (field 'alliances').
    Fallback:       /v2.0/{season}/alliances/{eventCode}/selection (field 'selections')
    """

    def to_int(x):
        try:
            return int(x)
        except:
            return None

    def unique(seq):
        seen = set();
        out = []
        for v in seq:
            if v not in seen:
                seen.add(v);
                out.append(v)
        return out

    # ---- primary: /alliances/{eventCode}
    ok, s, js, url = get_json_safe(f"/v2.0/{int(season)}/alliances/{code}")
    items = []
    if ok and isinstance(js, dict):
        for k in ["alliances", "Alliances", "items", "data", "results"]:
            if isinstance(js.get(k), list):
                items = js[k];
                break

    def team_ids_from(obj):
        ids = []
        if not isinstance(obj, dict):
            return ids
        # common containers / shapes
        buckets = [
            obj.get("teams"), obj.get("teamNumbers"), obj.get("members"), obj.get("allianceMembers"),
            obj.get("roster"), obj.get("picks"), obj.get("selectionResults"),
            [obj.get("captain")], [obj.get("Captain")],
            [obj.get("firstPick")], [obj.get("secondPick")], [obj.get("thirdPick")],
            [obj.get("pick1")], [obj.get("pick2")], [obj.get("pick3")]
        ]
        for r in ["round", "round1", "round2", "round3"]:
            if r in obj: buckets.append(obj.get(r))

        def extract(e):
            if e is None: return
            if isinstance(e, list):
                for x in e: extract(x)
            elif isinstance(e, dict):
                tn = e.get("teamNumber") or e.get("team") or e.get("teamId") or e.get("number")
                v = to_int(tn)
                if v is not None: ids.append(v)
                # nested
                for key in ["team", "member", "picked", "selection"]:
                    if key in e: extract(e.get(key))
            else:
                v = to_int(e)
                if v is not None: ids.append(v)

        for b in buckets: extract(b)
        return unique(ids)

    out = []
    for idx, a in enumerate(items if isinstance(items, list) else [], start=1):
        num = to_int((a or {}).get("number") or (a or {}).get("allianceNumber") or idx)
        teams = team_ids_from(a)
        out.append({"number": num if num is not None else idx, "teams": teams})

    # ---- fallback: /alliances/{eventCode}/selection
    if not out or all(len(x.get("teams") or []) == 0 for x in out):
        ok2, s2, js2, url2 = get_json_safe(f"/v2.0/{int(season)}/alliances/{code}/selection")
        selections = []
        if ok2 and isinstance(js2, dict):
            for k in ["selections", "Selections", "items", "data", "results"]:
                if isinstance(js2.get(k), list):
                    selections = js2[k];
                    break
        # reconstruct per alliance
        agg = {}
        for sel in selections if isinstance(selections, list) else []:
            an = sel.get("allianceNumber") or sel.get("alliance") or sel.get("number")
            an = to_int(an)
            if an is None:
                continue
            # possible team keys
            cand = [
                sel.get("teamNumber"), sel.get("team"), sel.get("teamId"),
                sel.get("captain"), sel.get("Captain"),
                sel.get("pickedTeam"), sel.get("picked"), sel.get("pick")
            ]
            # sometimes nested objects
            if isinstance(sel.get("team"), dict):
                cand.append(sel["team"].get("teamNumber"))
            teams = []
            for c in cand:
                v = to_int(c)
                if v is not None:
                    teams.append(v)
            if an not in agg:
                agg[an] = []
            agg[an].extend(teams)
        # finalize
        out = [{"number": k, "teams": unique(v)} for k, v in sorted(agg.items(), key=lambda kv: kv[0])]

    return out


def adv_api_awards(season: int, code: str):
    ok, s, js, url = get_json_safe(f"/v2.0/{int(season)}/awards/{code}")
    items = []
    if ok and isinstance(js, dict):
        items = js.get("Awards") or js.get("awards") or js.get("items") or []
        if items is None:
            items = []
    norm = []
    for aw in items if isinstance(items, list) else []:
        try:
            name = str(aw.get("name") or aw.get("award") or "").strip()
        except Exception:
            name = ""
        rk = aw.get("rank") if isinstance(aw, dict) else None
        if rk is None:
            rk = aw.get("place") if isinstance(aw, dict) else None
        if rk is None:
            rk = aw.get("position") if isinstance(aw, dict) else None
        tm = aw.get("teamNumber") if isinstance(aw, dict) else None
        if tm is None and isinstance(aw, dict):
            tm = aw.get("team") or aw.get("teamId")
        try:
            tm = int(tm) if tm is not None else None
        except Exception:
            tm = None
        try:
            rk = int(rk) if rk is not None else None
        except Exception:
            rk = None
        norm.append({"name": name, "rank": rk, "team": tm})
    return norm


# =======================
# Advancement Points (FTC 2025-2026) helpers
# =======================
ADV_PLAYOFF_POINTS = {"winner": 40, "finalist": 20, "3rd": 10, "4th": 5}
ADV_INSPIRE_POINTS = {1: 60, 2: 30, 3: 15}
ADV_OTHER_AWARDS_POINTS = {"1st": 12, "2nd": 6, "3rd": 3}


def adv_qual_points(rank, field_size):
    try:
        r = int(rank);
        n = max(2, int(field_size))
        if n <= 1: return 16
        val = 16 - (r - 1) * (14 / (n - 1))
        return max(2, min(16, int(round(val))))
    except Exception:
        return 0


def adv_alliance_points(alliance_no):
    if alliance_no is None or alliance_no == "": return 0
    try:
        return max(0, 21 - int(alliance_no))
    except Exception:
        return 0


def adv_playoff_points(finish):
    if not finish: return 0
    key = str(finish).strip().lower()
    if "winner" in key: return ADV_PLAYOFF_POINTS["winner"]
    if "finalist" in key: return ADV_PLAYOFF_POINTS["finalist"]
    if "3rd" in key or "third" in key: return ADV_PLAYOFF_POINTS["3rd"]
    if "4th" in key or "fourth" in key: return ADV_PLAYOFF_POINTS["4th"]
    return 0


def adv_awards_points(awards):
    import pandas as pd
    if awards is None or (isinstance(awards, float) and pd.isna(awards)): return 0
    if isinstance(awards, str):
        parts = [a.strip() for a in awards.split(';') if a.strip()]
    else:
        parts = [str(a).strip() for a in awards if str(a).strip()]
    total = 0
    for a in parts:
        low = a.lower()
        if "inspire" in low:
            if " 1" in low or "1st" in low: total += ADV_INSPIRE_POINTS[1]; continue
            if " 2" in low or "2nd" in low: total += ADV_INSPIRE_POINTS[2]; continue
            if " 3" in low or "3rd" in low: total += ADV_INSPIRE_POINTS[3]; continue
            total += ADV_INSPIRE_POINTS[1]
        else:
            if "1st" in low or "first" in low or "winner" in low: total += ADV_OTHER_AWARDS_POINTS["1st"]; continue
            if "2nd" in low or "second" in low: total += ADV_OTHER_AWARDS_POINTS["2nd"]; continue
            if "3rd" in low or "third" in low: total += ADV_OTHER_AWARDS_POINTS["3rd"]; continue
    return total


# =======================
# FTC-Events API helpers
# =======================
import os, requests
from typing import Dict, Any, Optional

FTC_API_BASE = "https://ftc-api.firstinspires.org"  # v2.0 endpoints; season path now uses 2025 for DECODE



def _ftc_get_auth():
    """Unified credentials resolver: globals -> secrets -> env -> UI -> file."""
    u = None; t = None
    try:
        u = globals().get("FTC_USER") or u
        t = globals().get("FTC_TOKEN") or t
    except Exception:
        pass
    try:
        import streamlit as st
        u = u or st.secrets.get("ftc_username", None)
        t = t or st.secrets.get("ftc_token", None)
        sec = st.secrets.get("ftc", {})
        if isinstance(sec, dict):
            u = u or sec.get("username", None)
            t = t or sec.get("token", None)
    except Exception:
        pass
    import os
    u = u or os.getenv("FTC_API_USER") or os.getenv("FTC_USER")
    t = t or os.getenv("FTC_API_TOKEN") or os.getenv("FTC_API_KEY") or os.getenv("FTC_TOKEN")
    try:
        import streamlit as st
        if (u is None or t is None) and st.session_state.get("ftc_auth_ui"):
            ui = st.session_state.get("ftc_auth_ui")
            u = u or ui.get("user")
            t = t or ui.get("token")
    except Exception:
        pass
    if u is None or t is None:
        try:
            gu, gt, _ = get_credentials()
            if gu and gt: u, t = gu, gt
        except Exception:
            pass
    return (u, t)
@st.cache_data(show_spinner=False, ttl=180)
def ftc_get(season: int, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    u, tok = _ftc_get_auth()
    headers = {"Accept": "application/json"}
    if "{season}" in path:
        url = f"{FTC_API_BASE}{path.format(season=season)}"
    elif path.startswith("/v2.0/"):
        url = f"{FTC_API_BASE}{path}"
    else:
        url = f"{FTC_API_BASE}/v2.0/{season}/{path.lstrip('/')}"
    try:
        resp = _http_get(url, headers=headers, params=params or {}, auth=(u, tok), timeout=15)
        if resp.status_code == 304:
            return {}
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}





@st.cache_data(show_spinner=False, ttl=300)
def ftc_rankings_df(season: int, event_code: str):
    """
    Returns rankings with RS (Ranking Score). If API lacks explicit RS,
    compute RS = total RP / qualification matches played.
    Columns: team, rank, RS, RP_total, qualMatches.
    """
    season = _coerce_season(season) if "_coerce_season" in globals() else season
    data = ftc_get(season, f"/v2.0/{season}/rankings/{event_code}")
    rows = []
    ranking_list = []
    try:
        ranking_list = data.get("Rankings") or data.get("rankings") or data.get("items") or []
    except Exception:
        ranking_list = []

    def _gi(obj, *keys):
        for k in keys:
            if k in obj and obj[k] is not None:
                return obj[k]
        return None

    for r in ranking_list:
        team     = _gi(r, "teamNumber", "team", "teamId", "number")
        rank     = _gi(r, "rank", "ranking", "qualAverageRank", "qualRank")
        rs       = _gi(r, "rankingScore", "RankingScore", "RS", "rankingPointsAverage", "avgRankingPoints", "rankScore")
        sort1    = _gi(r, "sortOrder1", "SortOrder1", "sort1")
        rp_total = _gi(r, "rankingPoints", "rp", "totalRankingPoints", "totalRP", "rankingPointsTotal")
        matches  = _gi(r, "qualMatchesPlayed", "matchesPlayed", "played", "qualMatches")

        try: team = int(team)
        except Exception: team = None
        try: rank = int(rank) if rank is not None else None
        except Exception: rank = None
        try: rs = float(rs) if rs is not None else None
        except Exception: rs = None
        try: sort1 = float(sort1) if sort1 is not None else None
        except Exception: sort1 = None
        try: rp_total = float(rp_total) if rp_total is not None else None
        except Exception: rp_total = None
        try: matches = int(matches) if matches is not None else None
        except Exception: matches = None

        if rs is None and rp_total is not None and matches and matches > 0:
            rs = rp_total / matches

        if team is not None:
            row = {"team": team}
            if rank is not None:     row["rank"] = rank
            if sort1 is not None:    row["sortOrder1"] = sort1
            if rs is not None:       row["RS"] = round(float(rs), 3)
            if rp_total is not None: row["RP_total"] = float(rp_total)
            if matches is not None:  row["qualMatches"] = int(matches)
            rows.append(row)

    import pandas as pd
    df = pd.DataFrame(rows)
    for col in ("rank","sortOrder1","RS","RP_total","qualMatches"):
        if col not in df.columns:
            df[col] = None

    if not df.empty:
        if "sortOrder1" in df.columns and df["sortOrder1"].notna().any():
            df = df.sort_values(["rank","sortOrder1","team"], ascending=[True, False, True])
        else:
            df = df.sort_values(["rank","team"], ascending=[True, True])

    return df[["team","rank","sortOrder1","RS","RP_total","qualMatches"]].copy() if not df.empty else df

@st.cache_data(show_spinner=False, ttl=300)
def ftc_alliances(season: int, event_code: str):
    data = ftc_get(season, f"/v2.0/{season}/alliances/{event_code}")
    out = []
    try:
        arr = data.get("Alliances") or data.get("alliances") or []
        for idx, a in enumerate(arr, start=1):
            num = a.get("number") or a.get("allianceNumber") or idx
            teams = []
            cap = a.get("captain") or a.get("Captain") or None
            if cap:
                tn = cap.get("teamNumber") or cap.get("team") or cap.get("teamId")
                if tn:
                    try:
                        teams.append(int(tn))
                    except:
                        pass
            picks = a.get("round") or a.get("picks") or a.get("teams") or []

            def _extract(t):
                if isinstance(t, dict):
                    return t.get("teamNumber") or t.get("team") or t.get("teamId")
                return t

            for t in picks:
                tn = _extract(t)
                if tn:
                    try:
                        teams.append(int(tn))
                    except:
                        pass
            out.append({"number": int(num), "teams": teams})
    except Exception:
        pass
    return out


@st.cache_data(show_spinner=False, ttl=300)
def ftc_awards(season: int, event_code: str):
    data = ftc_get(season, f"/v2.0/{season}/awards/{event_code}")
    items = data.get("Awards") or data.get("awards") or data.get("items") or []
    norm = []
    for aw in items:
        name = str(aw.get("name") or aw.get("award") or "").strip()
        rank = aw.get("rank") or aw.get("place") or aw.get("position")
        team = aw.get("teamNumber") or aw.get("team") or aw.get("teamId")
        try:
            team = int(team)
        except:
            team = None
        try:
            rank = int(rank)
        except:
            rank = None
        norm.append({"name": name, "rank": rank, "team": team})
    return norm


@st.cache_data(show_spinner=False, ttl=300)
def ftc_playoff_matches(season: int, event_code: str):
    data = ftc_get(season, f"/v2.0/{season}/matches/{event_code}",
                   params={"tournamentLevel": "Playoff", "start": 0, "end": 999})
    matches = data.get("Matches") or data.get("matches") or data.get("items") or []
    out = []
    for m in matches:
        wa = m.get("winner") or m.get("matchWinner") or m.get("winningAlliance") or ""
        red = m.get("red") or {}
        blue = m.get("blue") or {}

        def collect(side):
            teams = side.get("teams") or side.get("participant") or side.get("alliances") or []
            arr = []
            for t in teams:
                tn = t.get("teamNumber") if isinstance(t, dict) else t
                try:
                    arr.append(int(tn))
                except:
                    pass
            return arr

        out.append({"winner": str(wa).lower(), "red": collect(red), "blue": collect(blue)})
    return out


@st.cache_data(show_spinner=False, ttl=300)
def ftc_events(season: int):
    """Return list of events for the season with normalized keys: code, name, country_code, country_name."""
    data = ftc_get(season, f"/v2.0/{season}/events")
    items = data.get("Events") or data.get("events") or data.get("items") or []
    out = []
    for e in items:
        code = e.get("code") or e.get("eventCode") or e.get("event_code") or e.get("eventcode")
        name = e.get("name") or e.get("eventName") or ""
        cc = e.get("country") or e.get("countryCode") or e.get("country_code") or ""
        cn = e.get("countryName") or e.get("country_name") or ""
        out.append(
            {"code": str(code or "").strip(), "name": str(name or "").strip(), "country_code": str(cc or "").upper(),
             "country_name": str(cn or "")})
    return out


def _has_auth():
    u, t = _ftc_get_auth()
    return bool(u and t)


# =======================
# Compute Advancement
# =======================
def build_advancement_template(ev_view, teams_master, country, season):
    import pandas as pd
    teams_il = teams_master[teams_master["country"].apply(lambda c: _country_matches(c, country))].sort_values("team")
    teams_il = dedupe_teams_master(teams_il, ev_view)
    evs = ev_view[ev_view["season"] == season][
        ["ev", "ev_family", "ev_name"]].drop_duplicates() if "season" in ev_view.columns else ev_view[
        [c for c in ["ev", "ev_family", "ev_name"] if c in ev_view.columns]].drop_duplicates()
    rows = []
    for _, t in teams_il.iterrows():
        for _, e in evs.iterrows():
            rows.append({"team": int(t["team"]), "team_name": t.get("team_name", ""),
                         "event": e.get("ev_name") or e.get("ev", ""), "family": e.get("ev_family", ""),
                         "qual_rank": "", "field_size": "", "alliance_no": "", "playoff_finish": "", "awards": ""})
    return pd.DataFrame(rows)


def compute_advancement_table(ev_view, base, teams_master, country, season, adv_csv_df):
    import pandas as pd
    # teams filter using helper from the main app
    teams_il = teams_master[teams_master["country"].apply(lambda c: _country_matches(c, country))].sort_values("team")
    teams_il = dedupe_teams_master(teams_il, ev_view)
    per_team_detail = {}

    # events list + country
    evs_df = adv_api_events(int(season))
    if not evs_df.empty:
        evs_df = evs_df[evs_df["country"].apply(lambda c: _country_matches(c, country))].copy()

    # families actually present in base (season already filtered upstream)
    fams_base = []
    try:
        fams_base = base["ev_family"].dropna().astype(str).unique().tolist() if "ev_family" in base.columns else []
    except Exception:
        fams_base = []

    # map families -> codes & nice name (left-join to events_list)
    fam_to_codes = {};
    fam_to_name = {}
    if fams_base:
        fam_df = pd.DataFrame({"ev_family": [str(f) for f in fams_base]})
        if not evs_df.empty:
            tmp = fam_df.merge(evs_df.rename(columns={"family": "ev_family"}), how="left", on="ev_family")
            if tmp["event_code"].isna().all():  # case-insensitive retry
                evs_df["_fam_l"] = evs_df["family"].astype(str).str.lower()
                fam_df["_fam_l"] = fam_df["ev_family"].astype(str).str.lower()
                tmp = fam_df.merge(evs_df, left_on="_fam_l", right_on="_fam_l", how="left")
        else:
            tmp = fam_df.assign(event_code=None, name=fam_df["ev_family"])
        for fam, g in tmp.groupby("ev_family"):
            fam = str(fam)
            codes = g["event_code"].dropna().astype(str).unique().tolist()
            name = g["name"].dropna().astype(str).iloc[0] if "name" in g and not g["name"].dropna().empty else fam
            fam_to_codes[fam] = codes
            fam_to_name[fam] = name

    # helper functions
    def adv_qual_points(rank, field_size):
        try:
            r = int(rank);
            n = max(2, int(field_size))
            if n <= 1: return 16
            val = 16 - (r - 1) * (14 / (n - 1))
            return max(2, min(16, int(round(val))))
        except Exception:
            return 0

    def compute_points_for_code(tid: int, code: str):
        rk_df = adv_api_rankings(int(season), code)
        qrank = None;
        field = None
        if isinstance(rk_df, pd.DataFrame) and not rk_df.empty:
            field = int(len(rk_df))
            if tid in rk_df["team"].values:
                qrank = int(rk_df.loc[rk_df["team"] == tid, "rank"].iloc[0])
        qual_pts = adv_qual_points(qrank, field) if (qrank and field) else 0

        alliance_pts = 0;
        alliance_no = None
        allis = adv_api_alliances(int(season), code) or []
        for a in allis:
            if tid in (a.get("teams") or []):
                alliance_no = a.get("number", None)
                try:
                    alliance_pts = max(0, 21 - int(alliance_no))
                except Exception:
                    alliance_pts = 0
                break

        # awards
        awards_best = 0;
        awards_comp = []
        aws = adv_api_awards(int(season), code) or []
        for it in aws:
            if it.get("team") != tid:
                continue
            nm = (it.get("name") or "").lower()
            rk = it.get("rank")
            pts = ({1: 60, 2: 30, 3: 15}.get(rk, 60) if "inspire" in nm else {1: 12, 2: 6, 3: 3}.get(rk, 12))
            if pts > awards_best:
                awards_best = pts
            awards_comp.append({"name": it.get("name"), "rank": rk, "points": pts, "code": code})

        # playoff via awards text
        playoff_pts = 0;
        playoff_finish = ""
        text = " ".join([str(x.get("name") or "").lower() for x in aws if x.get("team") == tid])
        if "winning alliance" in text or "winner alliance" in text:
            playoff_pts = 40;
            playoff_finish = "winner"
        elif "finalist alliance" in text or "finalist" in text:
            playoff_pts = 20;
            playoff_finish = "finalist"
        elif "semifinalist" in text or "semi-finalist" in text:
            playoff_pts = 10;
            playoff_finish = "semifinalist"
        elif "quarterfinalist" in text or "quarter-finalist" in text:
            playoff_pts = 5;
            playoff_finish = "quarterfinalist"

        comp = {"qual_rank": qrank, "field_size": field, "alliance_no": alliance_no, "playoff_finish": playoff_finish,
                "awards": awards_comp, "code": code}
        return qual_pts, alliance_pts, playoff_pts, awards_best, (
                    qual_pts + alliance_pts + playoff_pts + awards_best), comp

    rows_summary = []
    for tid in teams_il["team"].astype(int).tolist():
        # families this team actually played in (from base)
        fams = []
        try:
            fams = base.loc[(base.get("t1", pd.Series(dtype=int)) == tid) | (
                        base.get("t2", pd.Series(dtype=int)) == tid), "ev_family"].dropna().astype(
                str).unique().tolist()
        except Exception:
            fams = []

        ev_rows = []
        for fam in fams:
            codes = fam_to_codes.get(str(fam), [])
            label = fam_to_name.get(str(fam), str(fam))

            best_qual = 0;
            best_qual_comp = {}
            best_alli = 0;
            best_alli_comp = {}
            best_play = 0;
            best_play_comp = {}
            awards_best = 0;
            awards_best_comp = {};
            awards_list = []

            for code in codes:
                qp, ap, pp, awp, tot, comp = compute_points_for_code(tid, code)
                if comp.get("qual_rank") is not None and qp > best_qual:
                    best_qual, best_qual_comp = qp, comp
                if ap > best_alli:
                    best_alli, best_alli_comp = ap, comp
                if pp > best_play:
                    best_play, best_play_comp = pp, comp
                if awp > awards_best:
                    awards_best, awards_best_comp = awp, comp
                awards_list.extend(comp.get("awards") or [])

            # Build one row per FAMILY (aggregated over codes)
            qual_rank = best_qual_comp.get("qual_rank") if best_qual_comp else None
            field_size = best_qual_comp.get("field_size") if best_qual_comp else None
            alliance_no = best_alli_comp.get("alliance_no") if best_alli_comp else None
            total = int((best_qual or 0) + (best_alli or 0) + (best_play or 0) + (awards_best or 0))

            ev_rows.append({
                "event": label,
                "qual_pts": int(best_qual or 0),
                "alliance_pts": int(best_alli or 0),
                "playoff_pts": int(best_play or 0),
                "awards_pts": int(awards_best or 0),
                "total": total,
                "qual_rank": qual_rank,
                "field_size": field_size,
                "alliance_no": alliance_no,
                "composition": {
                    "qual_source": best_qual_comp.get("code") if best_qual_comp else None,
                    "alliance_source": best_alli_comp.get("code") if best_alli_comp else None,
                    "playoff_source": best_play_comp.get("code") if best_play_comp else None,
                    "awards": awards_list,
                    "awards_source": awards_best_comp.get("code") if awards_best_comp else None
                }
            })

        df_ev = pd.DataFrame(ev_rows)
        per_team_detail[tid] = df_ev

    # summary table
    rows_summary = []
    for tid, df_ev in per_team_detail.items():
        if df_ev is None or df_ev.empty:
            rows_summary.append({"team": tid, "AdvPts": 0, "Qual": 0, "Alliance": 0, "Playoff": 0, "Awards": 0})
        else:
            rows_summary.append({
                "team": tid,
                "AdvPts": int(df_ev["total"].sum()),
                "Qual": int(df_ev["qual_pts"].sum()),
                "Alliance": int(df_ev["alliance_pts"].sum()),
                "Playoff": int(df_ev["playoff_pts"].sum()),
                "Awards": int(df_ev["awards_pts"].sum()),
            })

    summary = pd.DataFrame(rows_summary).merge(teams_il[["team", "team_name"]], on="team", how="left").sort_values(
        ["AdvPts", "Qual", "Awards", "Alliance", "Playoff", "team"],
        ascending=[False, False, False, False, False, True]).reset_index(drop=True)
    return summary, per_team_detail


APP_TITLE = "MISHMASH Scouting platform - Version 13n21 (V10i) (updated: 07.09.2025)"
BASE_URL = "https://ftc-api.firstinspires.org"
TIMEOUT = 25
TEAMS_PER_ALLIANCE = 2

# Keep explicit aliases for rare one-off codes if needed; prefix rule (ILCMP*) below covers most
EVENT_FAMILY_MAP = {
    "ILCMP": "ILCMP",
    "ILCMPARC": "ILCMP",
    "ILCMPPCF": "ILCMP",
    "ILCMPSOLR": "ILCMP",
    "ILCMPHDRO": "ILCMP",
}


def family_of(code: str) -> str:
    if not isinstance(code, str):
        return code
    c = code.strip().upper()
    # Prefix rule – any Israeli Championship code unifies to ILCMP family
    if c.startswith("ILCMP"):
        return "ILCMP"
    # fallback exact aliasing
    return EVENT_FAMILY_MAP.get(c, c)


# ---------- Credentials & HTTP ----------

def _cred_paths() -> list[str]:
    import os, pathlib
    here = pathlib.Path(__file__).resolve().parent
    p1 = str(here / "ftc_api_credentials.json")
    appdata = os.getenv("APPDATA") or ""
    p2 = str(pathlib.Path(appdata) / "mae_scout" / "ftc_api_credentials.json") if appdata else ""
    return [p for p in [p1, p2] if p]


# === JSON Credentials Loader (patched) ===
def _load_json_credentials(path: str) -> t.Tuple[t.Optional[str], t.Optional[str]]:
    """
    Loads FTC API credentials (username, token/authkey) from a JSON file.
    Returns (username, key), or (None, None) if missing/invalid.
    """
    try:
        import json, os
        if not path or not os.path.exists(path):
            return None, None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None, None
        user = data.get("username") or data.get("user") or data.get("FTC_USER")
        key  = data.get("token") or data.get("authkey") or data.get("FTC_TOKEN") or data.get("api_key") or data.get("FTC_API_KEY")
        if user and key:
            return str(user), str(key)
    except Exception:
        pass
    return None, None



def get_credentials() -> t.Tuple[t.Optional[str], t.Optional[str], list[str]]:
    tried = []
    try:
        gu = globals().get("FTC_USER"); gt = globals().get("FTC_TOKEN")
        if gu and gt: return gu, gt, tried
    except Exception:
        pass
    import os
    user = os.getenv("FTC_API_USER") or os.getenv("FTC_USER")
    key  = os.getenv("FTC_API_TOKEN") or os.getenv("FTC_API_KEY") or os.getenv("FTC_TOKEN")
    if user and key:
        return user, key, tried
    for p in _cred_paths():
        tried.append(p)
        u, k = _load_json_credentials(p)
        if u and k:
            return u, k, tried
    return None, None, tried



def clean(s):
    return (s or "").replace("\\u200f", "").replace("\\u200e", "").replace("\\xa0", " ").strip()


def auth_session():
    u, k = _ftc_get_auth()
    if not (u and k):
        raise RuntimeError("FTC API credentials missing (user/token). Configure ENV, secrets, or ftc_api_credentials.json")
    s = requests.Session()
    s.auth = (u, k)
    s.headers.update({"Accept": "application/json"})
    return s


def get_json(path: str):
    url = f"{BASE_URL}{path}"
    r = auth_session().get(url, timeout=TIMEOUT);
    r.raise_for_status()
    return r.status_code, r.json(), url


def get_json_safe(path: str):
    try:
        return True, *get_json(path)
    except Exception as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        return False, status or -1, str(e), f"{BASE_URL}{path}"


# ---------- API wrappers ----------



def events_list(season: int) -> pd.DataFrame:
    season = _coerce_season(season)
    ok, s, d, u = get_json_safe(f"/v2.0/{season}/events")

    items = []
    if ok and isinstance(d, dict):
        for k in ("Events","events","items","data","results"):
            v = d.get(k)
            if isinstance(v, list) and (len(v) == 0 or isinstance(v[0], dict)):
                items = v
                break
        if not items:
            for v in d.values():
                if isinstance(v, list) and (len(v) == 0 or isinstance(v[0], dict)):
                    if len(v) == 0:
                        items = v; break
                    cand = v[0]
                    if any(k in cand for k in ("code","eventCode","name","eventName","country","countryCode","startDate","dateStart")):
                        items = v; break

    rows = []
    for ev in (items or []):
        code_   = clean(str(ev.get("code") or ev.get("eventCode") or ""))
        name    = clean(str(ev.get("name") or ev.get("eventName") or ""))
        country = clean(str(ev.get("countryCode") or ev.get("country") or ""))
        start   = ev.get("dateStart") or ev.get("startDate") or ev.get("start")
        rows.append({"event_code": code_.upper(), "name": name, "country": country.upper(), "start": start,
                     "family": family_of(code_)})

    df = pd.DataFrame(rows)
    if df.empty:
        try:
            import streamlit as st
            total = 0 if not isinstance(items, list) else len(items)
            keys = list((items[0].keys())) if isinstance(items, list) and items else []
            st.warning(f"Failed to list events: empty events dataframe. Check credentials/network (server ok={ok}, items={total})")
            if isinstance(d, dict):
                st.caption("Top-level keys from API:"); st.code(", ".join(list(d.keys())[:40]))
            if keys:
                st.caption("First event keys:"); st.code(", ".join(list(keys)[:40]))
        except Exception:
            pass
        return pd.DataFrame(columns=["event_code","name","country","start","family"])

    try:
        import streamlit as st
        country_filter = (st.session_state.get("country_filter") or st.session_state.get("Country") or "").strip().upper()
        if country_filter:
            df = df[df["country"] == country_filter].copy()
    except Exception:
        pass

    df["start_dt"] = pd.to_datetime(df["start"], errors="coerce", utc=True)
    return df





def api_teams(season: int, code: str):    return get_json_safe(f"/v2.0/{season}/teams?eventCode={code}")


def api_matches(season: int, code: str):  return get_json_safe(f"/v2.0/{season}/matches/{code}")


def api_rankings(season: int, code: str): return get_json_safe(f"/v2.0/{season}/rankings/{code}")


# ---------- JSON helpers ----------
def extract_list(js, *keys):
    if isinstance(js, dict):
        for k in keys:
            v = js.get(k)
            if isinstance(v, list):
                return v
    return js if isinstance(js, list) else []


def extract_teams(js):   return extract_list(js, "teams", "Teams")


def extract_matches(js): return extract_list(js, "matches", "MatchScores", "Scores", "Matches")


def _to_int(x):
    try:
        if isinstance(x, dict):
            for k in ("teamNumber", "number", "team"):
                if k in x and x[k] is not None:
                    return int(x[k])
        return int(str(x))
    except Exception:
        return None


def _collect_participants(m: dict):
    red = {};
    blue = {}
    parts = m.get("participants") or m.get("teams") or []
    for p in parts or []:
        tn = _to_int(p.get("teamNumber") or p.get("team") or p.get("number"))
        station = str(p.get("station") or p.get("allianceStation") or "").lower().replace(" ", "")
        alias = {"red1": "red1", "red2": "red2", "r1": "red1", "r2": "red2", "blue1": "blue1", "blue2": "blue2",
                 "b1": "blue1", "b2": "blue2"}
        station = alias.get(station, station)
        if tn and station in ("red1", "red2"): red[station] = tn
        if tn and station in ("blue1", "blue2"): blue[station] = tn
    return red, blue


def _get_ci(d: dict, *names: str):
    if not isinstance(d, dict): return None
    keys = {str(k).lower(): k for k in d.keys()}
    for nm in names:
        k = keys.get(str(nm).lower())
        if k is not None: return d[k]
    for nm in names:
        ln = str(nm).lower()
        for lk, k in keys.items():
            if ln in lk: return d[k]
    return None


def _num(v):
    try:
        if v is None: return None
        if isinstance(v, (int, float)): return float(v)
        return float(str(v))
    except Exception:
        return None


def _match_stage(m: dict) -> str:
    lvl = _get_ci(m, "tournamentLevel", "matchLevel", "compLevel", "level", "phase", "description", "matchType")
    lvls = str(lvl or "").lower()
    flags = {str(k).lower(): _get_ci(m, k) for k in ["isPlayoff", "playoff", "elim", "isElim", "isElimination"]}
    if any(bool(v) for v in flags.values() if isinstance(v, bool)):
        if any(bool(v) for v in flags.values()): return "elim"
    if any(x in lvls for x in ["playoff", "elim", "final", "semi", "quarter", "qf", "sf", "f-"]): return "elim"
    if any(x in lvls for x in ["qual", "qualification", "qualifier", "league meet", "meet"]): return "qual"
    return "unknown"


# --- Helpers for Endgame via Robot Locations (FTC Events API) ---
_LEVEL_MAP = {
    None: 0, "": 0,
    "LEVEL 1": 3, "LEVEL1": 3, "L1": 3, 1: 3, "1": 3,
    "LEVEL 2": 15, "LEVEL2": 15, "L2": 15, 2: 15, "2": 15,
    "LEVEL 3": 30, "LEVEL3": 30, "L3": 30, 3: 30, "3": 30,
}


def _val_to_level_points(v):
    try:
        if isinstance(v, str):
            v = v.strip().upper()
        return float(_LEVEL_MAP.get(v, 0))
    except Exception:
        return 0.0


def _iter_kv(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v
            for kk, vv in _iter_kv(v):
                yield f"{k}.{kk}", vv
    elif isinstance(obj, list):
        for i, it in enumerate(obj):
            for kk, vv in _iter_kv(it):
                yield f"[{i}].{kk}", vv


def _robot_loc_endgame_points(match_dict, side: str):
    """Reconstruct Endgame points from 'Robot 1/2 Location' fields when the API
    doesn't expose an explicit endgame score. Returns float or None."""
    try:
        s = side.lower()
        vals = []
        for k, v in _iter_kv(match_dict):
            lk = str(k).lower()
            if "location" in lk and "robot" in lk and (
                    s in lk or any(tag in lk for tag in [f"{s}1", f"{s}2", "robot1", "robot2"])):
                if isinstance(v, (str, int, float)):
                    vals.append(v)
                elif isinstance(v, dict):
                    for cand in ("level", "value", "name", "loc"):
                        if cand in v:
                            vals.append(v[cand]);
                            break
        if not vals:
            return None
        pts = 0.0
        cnt = 0
        for val in vals:
            pts += _val_to_level_points(val)
            cnt += 1
            if cnt >= 2:
                break
        return pts if pts > 0 else None
    except Exception:
        return None


def alliances_rows_from_payload(payload) -> pd.DataFrame:
    items = extract_matches(payload);
    rows = [];
    idx = 0
    for m in items:
        if not isinstance(m, dict): continue
        idx += 1
        red, blue = _collect_participants(m)

        def pair(d, prefix):
            low = {k.lower(): v for k, v in d.items()};
            a = low.get(f"{prefix}1");
            b = low.get(f"{prefix}2")
            return [a, b] if a and b else None

        rpair = pair(red, "red");
        bpair = pair(blue, "blue")
        rtot = _num(_get_ci(m, "scoreRedFinal", "finalScoreRed"));
        btot = _num(_get_ci(m, "scoreBlueFinal", "finalScoreBlue"))
        rauto = _num(_get_ci(m, "scoreRedAuto"));
        bauto = _num(_get_ci(m, "scoreBlueAuto"))
        rtele = _num(_get_ci(m, "scoreRedTeleop", "scoreRedTeleOp"));
        btele = _num(_get_ci(m, "scoreBlueTeleop", "scoreBlueTeleOp"))
        rfoul = _num(_get_ci(m, "scoreRedFoul"));
        bfoul = _num(_get_ci(m, "scoreBlueFoul"))
        rend = _num(_get_ci(m, "scoreRedEndgame", "scoreRedEnd"));
        bend = _num(_get_ci(m, "scoreBlueEndgame", "scoreBlueEnd"))
        # Try to reconstruct Endgame from Robot 1/2 Location if missing/zero
        try:
            cand_r = _robot_loc_endgame_points(m, 'red')
            cand_b = _robot_loc_endgame_points(m, 'blue')
            if (rend is None or float(rend) == 0.0) and (cand_r is not None):
                rend = float(cand_r)
            if (bend is None or float(bend) == 0.0) and (cand_b is not None):
                bend = float(cand_b)
        except Exception:
            pass
        stage = _match_stage(m)

        def derive(total, auto, tele, foul, end, side):
            # rem = total - auto - foul (remainder allocated to Teleop+Endgame)
            if total is None:
                return tele, end
            rem = float(total) - (auto or 0.0) - (foul or 0.0)
            if rem < 0:
                rem = 0.0

            # Prefer explicit Endgame if present; otherwise infer from robot locations
            if (end is None) or (abs(float(end)) < 1e-9):
                try:
                    cand = _robot_loc_endgame_points(m, side)
                    if cand is not None and float(cand) > 0:
                        end = float(cand)
                except Exception:
                    pass

            # Fill missing component from the remainder
            if tele is None and end is None:
                tele, end = rem, 0.0
            elif tele is None:
                tele = rem - (end or 0.0)
            elif end is None:
                end = rem - (tele or 0.0)

            # Clamp to valid ranges
            if end is None: end = 0.0
            if tele is None: tele = 0.0
            if end < 0: end = 0.0
            if tele < 0: tele = 0.0
            if end > rem: end = rem
            if tele + end > rem:
                tele = max(0.0, rem - end)

            return tele, end

        rtele, rend = derive(rtot, rauto, rtele, rfoul, rend, 'red')
        btele, bend = derive(btot, bauto, btele, bfoul, bend, 'blue')

        def np_total(a, t, e):
            aa = 0.0 if a is None else float(a)
            tt = 0.0 if t is None else float(t)
            ee = 0.0 if e is None else float(e)
            return aa + tt + ee

        if rpair and rtot is not None:
            rows.append({"idx": idx, "side": "red", "t1": rpair[0], "t2": rpair[1],
                         "score": rtot, "np_score": np_total(rauto, rtele, rend),
                         "auto": rauto, "teleop": rtele, "endgame": rend, "foul": rfoul, "stage": stage})
        if bpair and btot is not None:
            rows.append({"idx": idx, "side": "blue", "t1": bpair[0], "t2": bpair[1],
                         "score": btot, "np_score": np_total(bauto, btele, bend),
                         "auto": bauto, "teleop": btele, "endgame": bend, "foul": bfoul, "stage": stage})
    df = pd.DataFrame(rows)
    for c in ["score", "np_score", "auto", "teleop", "endgame", "foul"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------- K/M schedules ----------
def K_schedule(n_qual: int) -> float:
    if n_qual <= 6: return 0.5
    if n_qual <= 12:
        frac = (n_qual - 6) / 6.0
        return 0.5 + (0.3 - 0.5) * frac
    return 0.3


def M_schedule(n_qual: int) -> float:
    if n_qual <= 12: return 0.0
    if n_qual <= 36: return (n_qual - 12) / 24.0
    return 1.0


# ---------- FIX: Definition of _build_score_component_map ----------
@st.cache_data(ttl=600, show_spinner=False)
def _build_score_component_map(season: int, event_code: str) -> dict:
    """
    Fetches match data for an event and returns a map of (idx, side) -> (auto, teleop, endgame, score).
    """
    ok, _, payload, _ = api_matches(season, event_code)
    if not ok:
        return {}

    match_df = alliances_rows_from_payload(payload)
    if match_df.empty:
        return {}

    comp_map = {}
    for _, row in match_df.iterrows():
        key = (int(row["idx"]), str(row["side"]).lower())
        # Ensure all components are float, default to 0.0 if missing
        auto = float(row.get("auto", 0.0) or 0.0)
        teleop = float(row.get("teleop", 0.0) or 0.0)
        endgame = float(row.get("endgame", 0.0) or 0.0)
        score = float(row.get("score", 0.0) or 0.0)
        comp_map[key] = (auto, teleop, endgame, score)
    return comp_map


# ---------- END FIX ----------


# ---------- Statbotics EPA with history (team-share fix) ----------
def compute_epa_statbotics_with_history(base: pd.DataFrame, ev_view: pd.DataFrame):
    if base is None or base.empty:
        final_df = pd.DataFrame(columns=["team", "EPA", "EPA_Auto", "EPA_Endgame", "EPA_Teleop", "N"])
        hist_df = pd.DataFrame(
            columns=["team", "ev", "ev_family", "idx", "start_dt", "EPA", "EPA_Auto", "EPA_Endgame", "EPA_Teleop"])
        return final_df, hist_df

    ev_dates = ev_view[["event_code", "start_dt"]].rename(columns={"event_code": "ev"})
    b = base.merge(ev_dates, on="ev", how="left").sort_values(by=["start_dt", "ev", "idx", "side"], kind="mergesort")

    # --- Fill component columns from FTC scores (override to ensure non-zero endgame) ---
    try:
        _season = int(st.session_state.get("season") or st.session_state.get("Season") or season)
    except Exception:
        try:
            _season = int(season)
        except Exception:
            _season = None
    if _season is not None:
        comp_cache = {}
        b["__idx__"] = pd.to_numeric(b.get("idx"), errors="coerce").astype(int)
        b["__side__"] = b.get("side").astype(str).str.lower()
        b["__ev__"] = b.get("ev").astype(str)
        for ev_code in b["__ev__"].dropna().unique().tolist():
            try:
                comp_cache[ev_code] = _build_score_component_map(_season, str(ev_code))
            except Exception:
                comp_cache[ev_code] = {}

        def _fill_from_map(row, which: int):
            mp = comp_cache.get(row["__ev__"], {})
            tup = mp.get((int(row["__idx__"]), row["__side__"]), None)
            return None if not tup else float(tup[which])

        # override when mapped present, keep original otherwise
        b["auto"] = b.apply(lambda r: (_fill_from_map(r, 0) if _fill_from_map(r, 0) is not None else r.get("auto")),
                            axis=1)
        b["endgame"] = b.apply(
            lambda r: (_fill_from_map(r, 2) if _fill_from_map(r, 2) is not None else r.get("endgame")), axis=1)
        b["score"] = b.apply(lambda r: (_fill_from_map(r, 3) if _fill_from_map(r, 3) is not None else r.get("score")),
                             axis=1)
        b.drop(columns=["__idx__", "__side__", "__ev__"], inplace=True, errors="ignore")
    # --- end component enrichment ---
    # --- numeric guards to avoid NaN leaking to EPA updates ---
    for col in ["auto", "endgame", "score"]:
        if col not in b.columns:
            b[col] = 0.0
    b[["auto", "endgame", "score"]] = b[["auto", "endgame", "score"]].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    # --- end numeric guards ---

    # --- numeric guards to avoid NaN leaking to EPA updates ---
    for col in ["auto", "endgame", "score"]:
        if col not in b.columns:
            b[col] = 0.0
    b[["auto", "endgame", "score"]] = b[["auto", "endgame", "score"]].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    # --- end numeric guards ---

    mean_total = b["score"].dropna().mean()
    mean_auto = b["auto"].dropna().mean()
    mean_end = b["endgame"].dropna().mean()
    init_total = (mean_total or 0.0) / TEAMS_PER_ALLIANCE
    init_auto = (mean_auto or 0.0) / TEAMS_PER_ALLIANCE
    init_end = (mean_end or 0.0) / TEAMS_PER_ALLIANCE

    teams = sorted(set(b["t1"]).union(set(b["t2"])))
    EPA = {int(t): float(init_total) for t in teams}
    EPA_A = {int(t): float(init_auto) for t in teams}
    EPA_E = {int(t): float(init_end) for t in teams}
    N_qual = {int(t): 0 for t in teams}

    hist_rows = []
    for (ev, idx), g in b.groupby(["ev", "idx"], sort=False):
        try:
            r = g[g["side"] == "red"].iloc[0];
            bl = g[g["side"] == "blue"].iloc[0]
        except Exception:
            continue
        red = [int(r["t1"]), int(r["t2"])]
        blue = [int(bl["t1"]), int(bl["t2"])]
        stage = (r.get("stage") or bl.get("stage") or "").lower()

        red_score = float(r["score"] or 0.0);
        blue_score = float(bl["score"] or 0.0)
        red_auto = float(r.get("auto") or 0.0);
        blue_auto = float(bl.get("auto") or 0.0)
        red_end = float(r.get("endgame") or 0.0);
        blue_end = float(bl.get("endgame") or 0.0)

        redEPA = sum(EPA.get(t, init_total) for t in red)
        blueEPA = sum(EPA.get(t, init_total) for t in blue)

        for t in red:
            n = N_qual.get(t, 0);
            K = K_schedule(n);
            M = M_schedule(n)
            delta = (K * ((red_score - redEPA) - M * (blue_score - blueEPA))) / TEAMS_PER_ALLIANCE
            EPA[t] = EPA.get(t, init_total) + float(delta)
            EPA_A[t] = EPA_A.get(t, init_auto) + (K * (red_auto - EPA_A.get(t, init_auto))) / TEAMS_PER_ALLIANCE
            EPA_E[t] = EPA_E.get(t, init_end) + (K * (red_end - EPA_E.get(t, init_end))) / TEAMS_PER_ALLIANCE
            if stage == "qual": N_qual[t] = n + 1

        for t in blue:
            n = N_qual.get(t, 0);
            K = K_schedule(n);
            M = M_schedule(n)
            delta = (K * ((blue_score - blueEPA) - M * (red_score - redEPA))) / TEAMS_PER_ALLIANCE
            EPA[t] = EPA.get(t, init_total) + float(delta)
            EPA_A[t] = EPA_A.get(t, init_auto) + (K * (blue_auto - EPA_A.get(t, init_auto))) / TEAMS_PER_ALLIANCE
            EPA_E[t] = EPA_E.get(t, init_end) + (K * (blue_end - EPA_E.get(t, init_end))) / TEAMS_PER_ALLIANCE
            if stage == "qual": N_qual[t] = n + 1

        for t in red + blue:
            hist_rows.append({
                "team": int(t), "ev": ev, "ev_family": g["ev_family"].iloc[0],
                "idx": int(idx), "start_dt": g["start_dt"].iloc[0],
                "EPA": EPA[t], "EPA_Auto": EPA_A[t], "EPA_Endgame": EPA_E[t],
                "EPA_Teleop": EPA[t] - EPA_A[t] - EPA_E[t],
            })

    final_df = pd.DataFrame({"team": list(EPA.keys())})
    final_df["EPA"] = pd.to_numeric(final_df["team"].map(EPA), errors="coerce")
    final_df["EPA_Auto"] = pd.to_numeric(final_df["team"].map(EPA_A), errors="coerce")
    final_df["EPA_Endgame"] = pd.to_numeric(final_df["team"].map(EPA_E), errors="coerce")
    final_df[["EPA", "EPA_Auto", "EPA_Endgame"]] = final_df[["EPA", "EPA_Auto", "EPA_Endgame"]].fillna(0.0)
    final_df["EPA_Teleop"] = final_df["EPA"] - final_df["EPA_Auto"] - final_df["EPA_Endgame"]
    final_df["N"] = final_df["team"].map(lambda t: int(N_qual.get(t, 0)))
    hist_df = pd.DataFrame(hist_rows)
    return final_df, hist_df


def compute_epa_statbotics(base: pd.DataFrame, ev_view: pd.DataFrame) -> pd.DataFrame:
    final_df, _ = compute_epa_statbotics_with_history(base, ev_view)
    return final_df


def opr_two_team(df, ycol="np_score", lam: float = 1e-6) -> pd.DataFrame:
    if df.empty or ycol not in df.columns:
        return pd.DataFrame(columns=["team", ycol])
    df2 = df.copy()
    df2[ycol] = pd.to_numeric(df2[ycol], errors="coerce")
    df2 = df2[np.isfinite(df2[ycol])]
    if df2.empty:
        return pd.DataFrame(columns=["team", ycol])
    teams = sorted(set(df2["t1"]).union(set(df2["t2"])))
    idx = {t: i for i, t in enumerate(teams)}
    X = np.zeros((len(df2), len(teams)), dtype=float)
    y = df2[ycol].to_numpy(dtype=float)
    for i, row in df2.reset_index(drop=True).iterrows():
        X[i, idx[int(row["t1"])]] = 1.0
        X[i, idx[int(row["t2"])]] = 1.0
    XtX = X.T @ X + float(lam) * np.eye(len(teams))
    Xty = X.T @ y
    try:
        w = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        w, *_ = np.linalg.lstsq(X, y, rcond=None)
    return pd.DataFrame({"team": teams, ycol: w})


def _country_matches(s: str, country: str) -> bool:
    """
    Returns True iff string s denotes the same country as 'country' (with common aliases).
    If 'country' is empty/None or equals 'WORLD', always returns True.
    """
    if not country or (isinstance(country, str) and country.strip().upper() in {"", "WORLD"}):
        return True
    if not s:
        return False
    aliases = {
        "IL": {"IL", "ISR", "ISRAEL", "ישראל"},
        "ISR": {"IL", "ISR", "ISRAEL", "ישראל"},
        "US": {"US", "USA", "UNITED STATES", "UNITED-STATES", "UNITED STATES OF AMERICA"},
        "USA": {"US", "USA", "UNITED STATES", "UNITED-STATES", "UNITED STATES OF AMERICA"},
    }
    a = s.strip().upper()
    b = (country or "").strip().upper()
    S = aliases.get(b, {b})
    return a in S or a.lower() in {x.lower() for x in S}


@st.cache_data(ttl=600, show_spinner=False)
def rankings_for_event(season: int, code: str) -> pd.DataFrame:
    ok, _, d, _ = api_rankings(season, code)
    items = extract_list(d, "rankings", "Rankings")
    rows = []
    for r in items:
        t = r.get("teamNumber") or r.get("team") or r.get("number")
        if t is None: continue

        def _get_int(*names):
            for n in names:
                v = r.get(n)
                if v is not None:
                    try:
                        return int(v)
                    except:
                        pass
            return None

        rows.append({"team": int(t), "rank": _get_int("rank", "Rank"),
                     "wins": _get_int("wins", "W"), "losses": _get_int("losses", "L"), "ties": _get_int("ties", "T")})
    return pd.DataFrame(rows)


@st.cache_data(ttl=1800, show_spinner=True)
def fetch_season_data(season: int, country: str, max_workers: int):
    """
    Root-cause fixed loader: returns (ev_view, base_raw, teams_master).
    - Filters events by country using _country_matches unless country is WORLD/empty.
    - Fetches matches/teams concurrently with sane throttling for USA.
    - Never returns None; raises with clear message only on truly fatal issues.
    """
    # Throttle for USA
    _mw = int(max_workers)
    if (country or "").strip().upper() in {"USA", "US"}:
        try:
            _mw = min(_mw, int(USA_MAX_WORKERS))
        except Exception:
            _mw = max(4, _mw // 2)

    # List all events
    ok_ev = False
    ev_df = events_list_cached(season)
    if isinstance(ev_df, pd.DataFrame) and not ev_df.empty:
        ok_ev = True
    if not ok_ev:
        raise RuntimeError("Failed to list events: empty events dataframe. Check credentials/network.")

    # Filter by country (unless WORLD)
    ev_f = ev_df[ev_df["country"].apply(lambda s: _country_matches(s, country))].copy()
    if ev_f.empty:
        ev_df["family"] = ev_df["event_code"].apply(family_of)
        return ev_df[["event_code", "name", "country", "start_dt", "family"]], pd.DataFrame(
            columns=["idx", "side", "t1", "t2", "score", "np_score", "auto", "teleop", "endgame", "foul", "stage", "ev",
                     "ev_family"]), pd.DataFrame(columns=["team", "team_name", "country", "ev", "ev_family"])
    cand = ev_f["event_code"].dropna().astype(str).str.upper().unique().tolist()

    # --- Augment: include global/CMP events where selected-country teams participated ---
    home_cc = (country or "").strip().upper()
    all_events_df = events_list_cached(season)

    def _is_global_event(row):
        n = str(row.get("name", "")).upper()
        c = str(row.get("event_code", "")).upper()
        return ("WORLD" in n) or ("CHAMP" in n) or ("CMP" in c) or ("HOUSTON" in n)

    glob = all_events_df[all_events_df.apply(_is_global_event, axis=1)].copy()
    extra = []
    for _, er in glob.iterrows():
        ec = str(er.get("event_code", "")).upper()
        if ec in cand:
            continue
        try:
            ok, _, d, _ = api_teams(season, ec)
            found = False
            for t in extract_teams(d):
                tn = t.get("teamNumber") or t.get("team") or t.get("number")
                cc = str(t.get("countryCode") or t.get("country") or "").strip().upper()
                if tn is not None and (cc == home_cc):
                    found = True;
                    break
            if found:
                extra.append(ec)
                ev_f = pd.concat([ev_f, er.to_frame().T], ignore_index=True)
        except Exception:
            continue
    if extra:
        cand = sorted(set(cand) | set(extra))

    # --- end augment ---

    # Builders for matches/teams
    def _matches(code):
        ok, _, d, _ = api_matches(season, code)
        b = alliances_rows_from_payload(d)
        if not b.empty:
            b = b.assign(ev=code, ev_family=family_of(code))
        return b

    def _teams_full(code):
        ok, _, d, _ = api_teams(season, code)
        rows = []
        for t in extract_teams(d):
            if isinstance(t, dict):
                tn = t.get("teamNumber") or t.get("team") or t.get("number")
                nm = t.get("nameShort") or t.get("name") or ""
                cc = clean(str(t.get("countryCode") or t.get("country") or ""))
                if tn:
                    rows.append({"team": int(tn), "team_name": nm, "country": cc.upper(), "ev": code,
                                 "ev_family": family_of(code)})
        return pd.DataFrame(rows)

    base_all, teams_all = [], []
    diag_counts = {"matches_ok": 0, "matches_err": 0, "teams_ok": 0, "teams_err": 0}

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=_mw) as ex:
        fut_m = {ex.submit(_matches, c): c for c in cand}
        fut_t = {ex.submit(_teams_full, c): c for c in cand}
        for f in as_completed(fut_m):
            try:
                b = f.result()
                if b is not None and not b.empty:
                    base_all.append(b)
                    diag_counts["matches_ok"] += 1
            except Exception as e:
                diag_counts["matches_err"] += 1
        for f in as_completed(fut_t):
            try:
                t = f.result()
                if t is not None and not t.empty:
                    teams_all.append(t)
                    diag_counts["teams_ok"] += 1
            except Exception as e:
                diag_counts["teams_err"] += 1

    # Assemble tables
    if base_all:
        base = pd.concat(base_all, ignore_index=True)
    else:
        base = pd.DataFrame(
            columns=["idx", "side", "t1", "t2", "score", "np_score", "auto", "teleop", "endgame", "foul", "stage", "ev",
                     "ev_family"])
    if teams_all:
        teams_master = pd.concat(teams_all, ignore_index=True).drop_duplicates(subset=["team", "ev"], keep="last")
    else:
        teams_master = pd.DataFrame(columns=["team", "team_name", "country", "ev", "ev_family"])

    ev_f = ev_f.copy()
    ev_f["family"] = ev_f["event_code"].apply(family_of)

    try:
        st.caption(
            f"📦 Loaded: matches_ok={diag_counts['matches_ok']}, teams_ok={diag_counts['teams_ok']} (errors: m={diag_counts['matches_err']}, t={diag_counts['teams_err']})")
    except Exception:
        pass

    return ev_f[["event_code", "name", "country", "start_dt", "family"]], base, teams_master


def _int0(x):
    try:
        if pd.isna(x): return 0
        return int(x)
    except Exception:
        return 0


# ---------- Rankings (OPR = last family) ----------
def build_ranking_table(ev_view: pd.DataFrame, base: pd.DataFrame, teams_master: pd.DataFrame,
                        country: str) -> pd.DataFrame:
    teams_master_f = teams_master[teams_master["country"].apply(lambda c: _country_matches(c, country))].copy()
    if teams_master_f.empty: return pd.DataFrame()

    epa_final, _ = compute_epa_statbotics_with_history(base, ev_view)
    # ensure single row per team
    if isinstance(epa_final, pd.DataFrame) and not epa_final.empty:
        epa_final = epa_final.sort_values('team').drop_duplicates(subset=['team'], keep='last')
    base_np = base.copy()
    base_np["np_score"] = pd.to_numeric(base_np["np_score"], errors="coerce")
    base_np = base_np[np.isfinite(base_np["np_score"])]
    opr_fam_list = []
    for fam, fam_df in base_np.groupby("ev_family"):
        if fam_df.empty:
            continue
        fam_opr = opr_two_team(fam_df, "np_score", lam=1e-6).rename(columns={"np_score": "OPR"})
        fam_opr["ev_family"] = fam
        opr_fam_list.append(fam_opr)
    opr_by_family = pd.concat(opr_fam_list, ignore_index=True) if opr_fam_list else pd.DataFrame(
        columns=["team", "OPR", "ev_family"])
    opr_by_family["team"] = pd.to_numeric(opr_by_family["team"], errors="coerce").astype("Int64")

    ev_dates = ev_view[["event_code", "start_dt", "family"]].rename(columns={"event_code": "ev"})
    bb = base.merge(ev_dates, on="ev", how="left")
    long_rows = []
    for _, r in bb.iterrows():
        for tcol in ("t1", "t2"):
            t = r.get(tcol)
            if pd.notna(t):
                long_rows.append({"team": int(t), "ev_family": r.get("ev_family"), "start_dt": r.get("start_dt")})
    long_df = pd.DataFrame(long_rows)
    if not long_df.empty:
        last_idx = long_df.sort_values("start_dt").groupby("team", as_index=False).tail(1).reset_index(drop=True)
        last_fam_per_team = last_idx[["team", "ev_family"]].rename(columns={"ev_family": "last_family"})
    else:
        last_fam_per_team = pd.DataFrame(columns=["team", "last_family"])

    last_opr = last_fam_per_team.merge(opr_by_family, left_on=["team", "last_family"],
                                       right_on=["team", "ev_family"], how="left").drop(columns=["ev_family"])
    last_opr = last_opr.rename(columns={"OPR": "OPR_last_family"})[["team", "OPR_last_family"]]

    rec = {}
    for gid, rows in base.groupby(["ev", "idx"]):
        try:
            r = rows[rows["side"] == "red"].iloc[0];
            b = rows[rows["side"] == "blue"].iloc[0]
        except Exception:
            continue
        rs = float(r["score"]);
        bs = float(b["score"])
        red = [int(r["t1"]), int(r["t2"])];
        blue = [int(b["t1"]), int(b["t2"])]
        for t in red + blue: rec.setdefault(t, {"W": 0, "L": 0, "T": 0})
        if abs(rs - bs) < 1e-9:
            for t in red + blue: rec[t]["T"] += 1
        elif rs > bs:
            for t in red: rec[t]["W"] += 1
            for t in blue: rec[t]["L"] += 1
        else:
            for t in blue: rec[t]["W"] += 1
            for t in red:  rec[t]["L"] += 1
    rec_df = pd.DataFrame([{"team": t, **v} for t, v in rec.items()])

    df = teams_master_f.merge(epa_final.rename(
        columns={"EPA": "EPA", "EPA_Auto": "EPA_Auto", "EPA_Teleop": "EPA_Teleop", "EPA_Endgame": "EPA_Endgame"}),
                              on="team", how="left") \
        .merge(last_opr, on="team", how="left").merge(rec_df, on="team", how="left")
    for c in ["W", "L", "T"]:
        if c not in df.columns: df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    df["RECORD"] = df.apply(
        lambda r: f"{_int0(r['W'])}-{_int0(r['L'])}" + (f"-{_int0(r.get('T', 0))}" if _int0(r.get('T', 0)) > 0 else ""),
        axis=1)
    df = df.rename(columns={"OPR_last_family": "OPR"})
    show = ["team", "team_name", "country", "EPA", "EPA_Auto", "EPA_Teleop", "EPA_Endgame", "OPR", "RECORD"]
    df = df.drop_duplicates(subset=["team"], keep="first")
    return df[show].sort_values(by=["EPA", "OPR"], ascending=[False, False], na_position="last")


# ---------- Diagnostics ----------
def build_diagnostics(ev_view: pd.DataFrame, base: pd.DataFrame, teams_master: pd.DataFrame, country: str):
    teams_master_f = teams_master[teams_master["country"].apply(lambda c: _country_matches(c, country))].copy()
    if teams_master_f.empty: return pd.DataFrame()

    # dedupe to one row per team (latest event)
    teams_master_f = dedupe_teams_master(teams_master_f, ev_view)
    if teams_master_f.empty: return pd.DataFrame(), 0, 0

    epa_final, _ = compute_epa_statbotics_with_history(base, ev_view)
    # ensure single row per team
    if isinstance(epa_final, pd.DataFrame) and not epa_final.empty:
        epa_final = epa_final.sort_values('team').drop_duplicates(subset=['team'], keep='last')
    base_np = base.copy()
    base_np["np_score"] = pd.to_numeric(base_np["np_score"], errors="coerce")
    base_np = base_np[np.isfinite(base_np["np_score"])]
    opr_df = opr_two_team(base_np, "np_score", lam=1e-6).rename(columns={"np_score": "OPR"})
    n_matches = len(base_np);
    n_teams = len(set(base_np["t1"]).union(set(base_np["t2"])))

    cnt = {}
    for _, r in base.iterrows():
        for t in [r["t1"], r["t2"]]: cnt[t] = cnt.get(t, 1) + 1 if t in cnt else 1
    cnt_df = pd.DataFrame([{"team": int(k), "matches": int(v)} for k, v in cnt.items()])

    epa_map = dict(zip(epa_final["team"], epa_final["EPA"]))
    rows = []
    for gid, g in base.groupby(["ev", "idx"]):
        try:
            r = g[g["side"] == "red"].iloc[0];
            b = g[g["side"] == "blue"].iloc[0]
        except Exception:
            continue
        rs = [int(r["t1"]), int(r["t2"])];
        bs = [int(b["t1"]), int(b["t2"])]
        for i, t in enumerate(rs):
            ally = rs[1 - i];
            opp = bs
            rows.append({"team": t, "ally_epa": epa_map.get(ally, np.nan),
                         "opp_epa": np.nanmean([epa_map.get(x, np.nan) for x in opp])})
        for i, t in enumerate(bs):
            ally = bs[1 - i];
            opp = rs
            rows.append({"team": t, "ally_epa": epa_map.get(ally, np.nan),
                         "opp_epa": np.nanmean([epa_map.get(x, np.nan) for x in opp])})
    str_df = pd.DataFrame(rows).groupby("team").mean(numeric_only=True).reset_index(names="team")

    dtab = teams_master_f.merge(epa_final, on="team", how="left") \
        .merge(opr_df, on="team", how="left").merge(cnt_df, on="team", how="left").merge(str_df, on="team", how="left")
    dtab["OPR"] = pd.to_numeric(dtab.get("OPR", np.nan), errors="coerce")
    dtab["delta(OPR-EPA)"] = dtab["OPR"] - dtab["EPA"]
    dtab["N_EPA_updates"] = dtab["N"].fillna(0).astype(int) if "N" in dtab.columns else 0
    return dtab, n_matches, n_teams


# ---------- Team drilldown ----------
def team_event_breakdown(season: int, team: int, base: pd.DataFrame, ev_view: pd.DataFrame) -> pd.DataFrame:
    if base.empty:
        return pd.DataFrame(
            columns=["event", "date", "rank", "record", "EPA", "EPA Auto", "EPA Teleop", "EPA Endgame", "OPR",
                     "matches"])

    epa_final, hist_df = compute_epa_statbotics_with_history(base, ev_view)

    fams = base[(base["t1"] == team) | (base["t2"] == team)]["ev_family"].dropna().unique().tolist()
    rows = []
    for fam in fams:
        fam_hist = hist_df[(hist_df["team"] == team) & (hist_df["ev_family"] == fam)].copy()
        if fam_hist.empty:
            continue
        last = fam_hist.sort_values(["start_dt", "ev", "idx"]).iloc[-1]
        EPA = float(last["EPA"]);
        EPA_A = float(last["EPA_Auto"]);
        EPA_E = float(last["EPA_Endgame"])
        EPA_T = float(last["EPA_Teleop"])

        fam_base = base[base["ev_family"] == fam].copy()
        fam_base["np_score"] = pd.to_numeric(fam_base["np_score"], errors="coerce")
        fam_base = fam_base[np.isfinite(fam_base["np_score"])]
        opr_ev = opr_two_team(fam_base, "np_score", lam=1e-6).rename(columns={"np_score": "OPR"})
        opr_row = opr_ev[opr_ev["team"] == team]
        opr = float(opr_row["OPR"].iloc[0]) if not opr_row.empty else np.nan

        W = L = T = 0;
        match_cnt = 0
        for gid, rows_g in fam_base.groupby(["ev", "idx"]):
            try:
                r = rows_g[rows_g["side"] == "red"].iloc[0];
                b = rows_g[rows_g["side"] == "blue"].iloc[0]
            except Exception:
                continue
            players = [int(r["t1"]), int(r["t2"]), int(b["t1"]), int(b["t2"])]
            if team not in players:
                continue
            match_cnt += 1
            rs = float(r["score"]);
            bs = float(b["score"])
            if abs(rs - bs) < 1e-9:
                T += 1
            elif team in (int(r["t1"]), int(r["t2"])):
                if rs > bs:
                    W += 1
                else:
                    L += 1
            else:
                if bs > rs:
                    W += 1
                else:
                    L += 1

        rk = None
        codes_in_family = ev_view.loc[ev_view["family"] == fam, "event_code"].tolist()
        for code in codes_in_family:
            rk_df = rankings_for_event(season, code)
            if not rk_df.empty and (rk_df["team"] == team).any():
                rk = int(rk_df.loc[rk_df["team"] == team, "rank"].iloc[0]);
                break

        evs_meta = ev_view[ev_view["family"] == fam].copy().sort_values("start_dt")
        fam_name = fam;
        date_str = ""
        if not evs_meta.empty:
            unique_names = [n for n in evs_meta["name"].dropna().unique().tolist() if n]
            fam_name = f"{fam} — " + (" / ".join(unique_names[:3]) + (" …" if len(unique_names) > 3 else ""))
            date = evs_meta["start_dt"].min()
            date_str = date.date().isoformat() if pd.notna(date) else ""

        rows.append({"event": fam_name, "date": date_str, "rank": rk,
                     "record": f"{W}-{L}" + (f"-{T}" if T > 0 else ""),
                     "EPA": round(EPA, 4), "EPA Auto": round(EPA_A, 4), "EPA Teleop": round(EPA_T, 4),
                     "EPA Endgame": round(EPA_E, 4),
                     "OPR": round(opr, 4) if pd.notna(opr) else np.nan,
                     "matches": match_cnt})

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date_sort"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values(by=["date_sort", "event"]).drop(columns=["date_sort"])
    return df


# ---------- UI ----------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.caption("Build: **V8** — single-event ranking-only, cached Adv tab")

# ---- Manual refresh for cached computations ----
if st.sidebar.button("🔄 רענון נתונים"):
    try:
        st.cache_data.clear()
    except Exception:
        pass
    st.rerun()

st.title(APP_TITLE)

# --- Session-state for same-tab navigation ---
if "team" not in st.session_state:
    st.session_state["team"] = None


def _clear_team():
    st.session_state["team"] = None


def _select_team(team_id: int | str):
    try:
        st.session_state["team"] = int(team_id)
    except Exception:
        st.session_state["team"] = str(team_id)
    st.rerun()


user, key, tried = get_credentials()
def ensure_ftc_creds_interactive(default_season=None):
    """
    If credentials are missing or API rejects them (401/403), prompt user to upload
    ftc_api_credentials.json and save it to %APPDATA%\mae_scout\ftc_api_credentials.json.
    """
    import os, json, pathlib
    import streamlit as st

    status = None
    try:
        season_probe = default_season or (st.session_state.get("season") or st.session_state.get("Season"))
        if "_ftc_api_diagnostics" in globals() and season_probe:
            diag = _ftc_api_diagnostics(_coerce_season(season_probe))
            status = (diag or {}).get("status")
    except Exception:
        pass

    need_upload = (not has_creds()) or (status in (401,403))
    if not need_upload:
        return False

    st.warning("FTC API credentials missing or invalid. Please upload ftc_api_credentials.json")
    uploaded = st.file_uploader("Upload ftc_api_credentials.json", type=["json"], key="ftc_json_upl")
    if uploaded is not None:
        try:
            data = json.load(uploaded)
            user = data.get("username") or data.get("user") or data.get("FTC_USER")
            key  = data.get("authkey")  or data.get("token") or data.get("FTC_TOKEN") or data.get("api_key")
            if user and key:
                appdata = os.getenv("APPDATA") or ""
                target = pathlib.Path(appdata) / "mae_scout" / "ftc_api_credentials.json" if appdata else pathlib.Path("ftc_api_credentials.json")
                target.parent.mkdir(parents=True, exist_ok=True)
                with open(target, "w", encoding="utf-8") as f:
                    json.dump({"user": user, "token": key}, f)
                st.success(f"Credentials saved to: {target}")
                os.environ["FTC_API_USER"] = str(user)
                os.environ["FTC_API_TOKEN"] = str(key)
                _set_creds_state(user, key, {"source":"file","path":str(target)})
                return True
            else:
                st.error("Uploaded JSON does not contain username/token fields.")
        except Exception as e:
            st.error(f"Failed to parse uploaded file: {e}")
    return False
if 'st' in globals():
    try:
        ensure_ftc_creds_interactive(
            default_season=(st.session_state.get('season') if hasattr(st, 'session_state') and 'season' in st.session_state else None)
        )
    except Exception as _e:
        try:
            st.warning(f"Credential check skipped: {_e}")
        except Exception:
            pass
if not (user and key):
    if not has_creds():
        st.error("FTC API credentials missing")
    else:
        st.empty()
    st.code("\\n".join(tried) or "(no paths)")

with st.sidebar:
    st.caption("EPA per Statbotics (moving-average in points). "
               "OPR בדירוגים = OPR של *המשפחה האחרונה* של הקבוצה (np, כולל פלייאוף). "
               "ILCMP* מאוחד למשפחה אחת כדי לאחד שלבי בתים וגמר.")

# --- Season default (Asia/Jerusalem): before Oct -> last year, from Oct -> current ---
try:
    from zoneinfo import ZoneInfo

    _tz = ZoneInfo("Asia/Jerusalem")
except Exception:
    _tz = None
_now = __import__("datetime").datetime.now(_tz) if _tz else __import__("datetime").datetime.now()
DEFAULT_SEASON = _now.year if _now.month >= 10 else (_now.year - 1)

c1, c2 = st.columns([1, 1])
with c1:
    season = st.number_input("Season", min_value=2006, value=DEFAULT_SEASON, step=1, key="season_input", on_change=_clear_team)
with c2: country = st.text_input("Country filter (e.g., IL / ISR / ישראל)", "IL", key="country_input",
                                 on_change=_clear_team)

MAX_WORKERS = 16
with st.spinner("Loading season data..."):
    MAXW = MAX_WORKERS
    if country.strip().upper() in {"USA", "US"}:
        MAXW = min(MAXW, 3)
    ev_view, base_raw, teams_master = _safe_dataset(int(season), country, int(MAXW))
st.session_state["last_dataset"] = (ev_view, base_raw, teams_master)

st.session_state["last_dataset"] = (ev_view, base_raw, teams_master)

sig = dataset_signature(ev_view, base_raw, teams_master)
try:
    st.caption("Last refresh (client): " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
except Exception:
    pass

# === Prefetch heavy computations for all tabs (parallel) ===
from concurrent.futures import ThreadPoolExecutor

_prefetch_key = (int(season), str(country))
_need_prefetch = (
        "prefetch_key" not in st.session_state
        or st.session_state.get("prefetch_key") != _prefetch_key
        or any(k not in st.session_state for k in ["rank_df", "adv_summary", "per_team_detail", "diag_df", "diag_meta"])
)

if _need_prefetch and not base_raw.empty:
    with st.spinner("Precomputing rankings, advancements, and diagnostics"):
        with ThreadPoolExecutor(max_workers=3) as ex:
            fut_rank = ex.submit(build_ranking_table, ev_view, base_raw, teams_master, country)
            fut_adv = ex.submit(compute_advancement_table, ev_view, base_raw, teams_master, country, int(season),
                                pd.DataFrame())
            fut_diag = ex.submit(build_diagnostics, ev_view, base_raw, teams_master, country)
        rank_df = fut_rank.result()
        adv_summary, per_team_detail = fut_adv.result()
        diag_df, n_matches_opr, n_teams_opr = fut_diag.result()
        st.session_state.update({
            "prefetch_key": _prefetch_key,
            "rank_df": rank_df,
            "adv_summary": adv_summary,
            "per_team_detail": per_team_detail,
            "diag_df": diag_df,
            "diag_meta": (n_matches_opr, n_teams_opr),
        })
else:
    rank_df = st.session_state.get("rank_df")
    adv_summary = st.session_state.get("adv_summary")
    per_team_detail = st.session_state.get("per_team_detail")
    diag_df = st.session_state.get("diag_df")
    diag_meta = st.session_state.get("diag_meta", (0, 0))
# === end prefetch ===
print("Columns in teams_master:", teams_master.columns)
print(teams_master.head())

st.caption(f"Events aggregated: {base_raw['ev'].nunique() if 'ev' in base_raw.columns else 0} "
           f"(families: {base_raw['ev_family'].nunique() if 'ev_family' in base_raw.columns else 0}) • "
           f"Teams seen: {_teams_seen(teams_master, base_raw)}")

if base_raw.empty:
    st.warning("No match data found for the selected filters.")
    st.stop()

tab_rank, tab_adv, tab_single, tab_diag = st.tabs(
    ["🏆 Rankings", "🏅 Ranking: Advancements Points", "📅 Ranking: Single Event", "🔎 Diagnostics"])


# ---- helper: compact Event code for Rankings inline (overwrite, drop noise) ----
def tidy_epa_inline_df(df):
    import pandas as pd
    if df is None or getattr(df, 'empty', True):
        return pd.DataFrame()
    # Normalize column names to lower for mapping
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    # Build tidy frame with best-available columns
    out = pd.DataFrame()
    ev = pick('event', 'event_name', 'ev', 'ev_name', 'event code', 'event_code')
    if ev: out['Event'] = df[ev]
    epa = pick('epa')
    if epa: out['EPA'] = df[epa]
    npor = pick('npopr', 'np_opr', 'npo', 'opr_np', 'opr')
    if npor: out['npOPR'] = df[npor]
    auto = pick('epa auto', 'auto epa', 'auto_epa', 'epa_auto')
    if auto: out['Auto EPA'] = df[auto]
    tele = pick('epa teleop', 'teleop epa', 'teleop_epa', 'epa_teleop')
    if tele: out['Teleop EPA'] = df[tele]
    endg = pick('epa endgame', 'endgame epa', 'endgame_epa', 'epa_endgame')
    if endg: out['Endgame EPA'] = df[endg]
    rec = pick('record', 'w-l-t', 'wl', 'wlt', 'record_str')
    if rec: out['Record'] = df[rec]
    # If mapping missed, just return original df as fallback
    if out.empty:
        return df
    return out


def _header_cell(txt, shift=0, align='center'):
    a = str(align).lower()
    just = 'flex-start' if a == 'left' else ('flex-end' if a == 'right' else 'center')
    return (
        f"<div style='width:100%; direction:ltr; display:flex; justify-content:{just}; align-items:center; "
        f"transform: translateX({shift}px); font-weight:600; letter-spacing:.3px'>{txt}</div>"
    )

    if df is None or getattr(df, "empty", False):
        return df
    df = df.copy()

    # Compute code from existing code columns or parse from name
    def _event_code_from_row(row):
        for k in ("event_code", "code", "ev", "ev_code", "eventCode", "event_code_x", "event_code_y"):
            if k in row and not _pd.isna(row[k]):
                s = str(row[k]).strip()
                if s:
                    return s
        # Try to parse from textual name
        for k in ("event", "event_name", "ev_name", "name", "Name"):
            if k in row and isinstance(row[k], str):
                m = _re.findall(r"[A-Z]{2,}[A-Z0-9]{2,}", row[k])
                if m:
                    return m[-1]
        return ""

    # Build Event code as first column
    try:
        code_series = df.apply(_event_code_from_row, axis=1)
    except Exception:
        code_series = _pd.Series([""] * len(df))

    # Overwrite/insert Event column
    if "Event" in df.columns:
        df.drop(columns=["Event"], inplace=True, errors="ignore")
    df.insert(0, "Event", code_series)

    # Columns to drop (noise / redundant)
    drop_exact = {
        "event", "event_name", "ev_name", "name", "Name", "date", "matches", "match_count",
        "level", "Level", "type", "Type", "EventCode", "eventCode"
    }

    # Also drop any column that is entirely empty strings/NaN (keep numerics)
    def _is_all_empty(s):
        try:
            if s.dtype.kind in "biufc":
                return False
            return s.fillna("").astype(str).str.strip().eq("").all()
        except Exception:
            return False

    # Remove noisy columns if exist
    to_drop = [c for c in df.columns if c in drop_exact]
    if to_drop:
        df = df.drop(columns=to_drop)

    # Finally, prune fully-empty text columns (but keep numerics even if zeros)
    empty_cols = [c for c in df.columns if c != "Event" and _is_all_empty(df[c])]
    if empty_cols:
        df = df.drop(columns=empty_cols)

    return df


with tab_rank:
    # --- UI polish: chips + compact layout (Rankings only) ---
    st.markdown("""
    <style>
    .mae-chip {display:inline-block;padding:4px 10px;border-radius:999px;border:1px solid rgba(255,255,255,.08);font-size:16px;line-height:22px;background:rgba(255,255,255,0.05)line-height:18px;background:rgba(255,255,255,0.05)}
    .mae-chip.epa   {background: rgba(212, 175, 55, 0.28);}
    .mae-chip.npopr {background: rgba(176, 176, 190, 0.28);}
    .mae-chip.auto  {background:#14233d40;}
    .mae-chip.tele  {background:#1a1f3d40;}
    .mae-chip.end   {background:#2a1f3d40;}
    .mae-chip.rec   {background: rgba(24, 160, 85, 0.30);}
    .mae-chip.adv   {background: rgba(96, 165, 250, 0.28);}
    .mae-chip.qual  {background: rgba(176, 176, 190, 0.28);}
    .mae-chip.alli  {background: rgba(147, 112, 219, 0.28);}
    .mae-chip.play  {background: rgba(16, 185, 129, 0.28);}
    .mae-chip.awd   {background: rgba(245, 158, 11, 0.28);}
    /* layout utilities for rankings */
    .mae-left, .mae-center, .mae-right { display:flex; align-items:center; width:100%; direction:ltr; }
    .mae-left  { justify-content:flex-start; }
    .mae-center{ justify-content:center; }
    .mae-right { justify-content:flex-end; }
    .mae-rankchip { display:inline-flex; align-items:center; justify-content:center; padding:4px 10px;
                     border-radius:999px; font-weight:600; background:rgba(255,255,255,0.06); }
        </style>
    """, unsafe_allow_html=True)


    def _chip(val, cls):
        s = "" if val is None else str(val)
        return f"<span class='mae-chip {cls}'>{s}</span>"


    with st.spinner("Computing rankings"):
        rank_df = rank_df if isinstance(rank_df, pd.DataFrame) else build_rankings_cached(sig, ev_view, base_raw,
                                                                                          teams_master, country)
    st.caption(
        "EPA per Statbotics (moving-average). בעמודת **npOPR** מוצג ה‑OPR של המשפחה האחרונה בעונה (np, כולל פלייאוף). אירועי ILCMP* מאוחדים למשפחה אחת.")
    if rank_df.empty:
        st.info("No teams found for this country filter.")
    else:
        COLS = [0.5, 4.6, 0.85, 0.9, 0.85, 0.85, 0.85, 0.9, 1.1]  # added RS col
        hc = st.columns(COLS, gap='small')


        def _header_cell(txt, shift=0, align='center', cls=''):
            a = str(align).lower()
            just = 'flex-start' if a == 'left' else ('flex-end' if a == 'right' else 'center')
            padl = 'padding-left: 18px;' if ('team' in str(cls).lower()) else ''
            nudge = f"transform: translateX({shift}px);" if isinstance(shift, (int, float)) and shift != 0 else ''
            return (
                f"<div style='width:100%; direction:ltr; display:flex; justify-content:{just}; align-items:center; "
                f"{padl}{nudge}font-weight:600; letter-spacing:.3px'>{txt}</div>"
            )


        hc[0].markdown(_header_cell('Rank', 0, 'center'), unsafe_allow_html=True)
        hc[1].markdown(_header_cell('Team', 0, 'left', 'team'), unsafe_allow_html=True)
        hc[2].markdown(_header_cell('EPA', -18, 'right'), unsafe_allow_html=True)
        hc[3].markdown(_header_cell('npOPR', 0, 'right'), unsafe_allow_html=True)
        hc[4].markdown(_header_cell('Auto EPA', 0, 'right'), unsafe_allow_html=True)
        hc[5].markdown(_header_cell('Teleop EPA', 0, 'right'), unsafe_allow_html=True)
        hc[6].markdown(_header_cell('Endgame EPA', 36, 'right'), unsafe_allow_html=True)
        hc[7].markdown(_header_cell('Record', 0, 'right'), unsafe_allow_html=True)
        for idx, row in enumerate(rank_df.itertuples(index=False), start=1):
            c = st.columns(COLS, gap="small")
            t_id = getattr(row, "team", None)
            t_name = getattr(row, "team_name", "") or str(t_id)
            label = f"{t_id} – {t_name}" if t_name and str(t_id) not in str(t_name) else str(t_name or t_id)
            c[0].markdown(f"<div class=\'mae-center\'><div class='mae-rankchip'>{idx}</div></div>",
                          unsafe_allow_html=True)
            with c[1]:
                exp = st.expander(label)
                with exp:
                    with st.spinner("Loading team breakdown"):
                        try:
                            df_ev = team_event_breakdown(int(season), int(t_id), base_raw, ev_view)
                        except Exception:
                            df_ev = None
                    if df_ev is not None and not df_ev.empty:
                        df_ev = tidy_epa_inline_df(df_ev)

                        st.dataframe(df_ev, use_container_width=True, hide_index=True)
                    else:
                        st.info("No games detected for this team in the selected season/country filter.")


            def _fmt(x):
                try:
                    return f"{float(x):.1f}"
                except Exception:
                    return ""


            c[2].markdown(f"<div class='mae-right'>{_fmt(getattr(row, 'EPA', ''))}</div>", unsafe_allow_html=True)
            c[3].markdown(f"<div class='mae-right'>{_fmt(getattr(row, 'OPR', ''))}</div>", unsafe_allow_html=True)
            c[4].markdown(f"<div class='mae-right'>{_fmt(getattr(row, 'EPA_Auto', ''))}</div>", unsafe_allow_html=True)
            c[5].markdown(f"<div class='mae-right'>{_fmt(getattr(row, 'EPA_Teleop', ''))}</div>",
                          unsafe_allow_html=True)
            c[6].markdown(f"<div class='mae-right'>{_fmt(getattr(row, 'EPA_Endgame', ''))}</div>",
                          unsafe_allow_html=True)
            c[7].markdown(f"<div class='mae-right'>{getattr(row, 'RECORD', '') or getattr(row, 'Record', '')}</div>",
                          unsafe_allow_html=True)
    csv = rank_df.rename(columns={"OPR": "npOPR"}).to_csv(index=False).encode("utf-8-sig")
    st.download_button("Download CSV", data=csv, file_name=f"mae_scout_rankings_{country}_{season}.csv",
                       mime="text/csv")

with tab_adv:
    # Diagnostics: families detected (from base_raw) & mapped (after compute)
    try:
        fams_detected = sorted(
            list(set(base_raw["ev_family"].dropna().astype(str).tolist()))) if "ev_family" in base_raw.columns else []
    except Exception:
        fams_detected = []
    st.caption(f"Families detected (from base): {len(fams_detected)}")

    # Diagnostics
    families = sorted(list(set(ev_view.get("ev_family", pd.Series(dtype=str)).dropna().astype(
        str).tolist()))) if "ev_family" in ev_view.columns else []
    mapped = 0
    try:
        # quick call matching the compute function's logic
        tmp_evn = ev_view if isinstance(ev_view, pd.DataFrame) else None
        mapped = len(families)
    except Exception:
        pass
    st.caption(f"API auth: {'✅ detected' if _has_auth() else '⚠️ missing'}  |  Families detected: {len(families)}")

    st.subheader("Ranking: Advancements Points")
    with st.expander("FTC-Events API (already configured in app) — optional override"):
        c1, c2 = st.columns(2)
        user_in = c1.text_input("FTC API Username (or Email)", value=st.secrets.get("ftc_auth_ui", {}).get("user", ""),
                                type="default", key="ftc_user_input")
        token_in = c2.text_input("FTC API Token", value=st.secrets.get("ftc_auth_ui", {}).get("token", ""),
                                 type="password", key="ftc_token_input")
        if st.button("Use these credentials", use_container_width=False):
            st.session_state["ftc_auth_ui"] = {"user": user_in.strip(), "token": token_in.strip()}
            st.success("Credentials saved for this session")
    st.caption(
        "חישוב נקודות עפ״י מודל ההעפלה החדש של FTC (2025‑2026): Qualification (2‑16), Alliance (21−מס׳ הברית), Playoff (40/20/10/5), Inspire (60/30/15), פרסים אחרים (12/6/3).")
    up = st.file_uploader(
        "(אופציונלי) העלה קובץ CSV עם נתוני Advancement (team,event/family,qual_rank,field_size,alliance_no,playoff_finish,awards)",
        type=["csv"], key="adv_csv_upload")
    adv_csv_df = None
    if up is not None:
        try:
            adv_csv_df = pd.read_csv(up)
            st.success(f"נטען קובץ Advancement עם {len(adv_csv_df)} שורות (יעדכן/ישלים מעל נתוני ה-API)")
        except Exception as e:
            st.error(f"שגיאה בקריאת הקובץ: {e}")
    adv_summary = st.session_state.get('adv_summary')
    adv_per_team = st.session_state.get('per_team_detail')
    if not isinstance(adv_summary, pd.DataFrame) or adv_summary.empty:
        with st.spinner("מחשב Advancement Points"):
            adv_summary, adv_per_team = compute_advancement_table(ev_view, base_raw, teams_master, country, int(season),
                                                                  adv_csv_df)
        st.session_state['adv_summary'] = adv_summary
        st.session_state['per_team_detail'] = adv_per_team
    if adv_summary.empty:
        st.info("אין נתונים לחישוב Advancement לעונה זו (בדוק אישורים ל-FTC API/קודי אירוע).")
    else:
        COLS = [0.5, 4.6, 1.0, 1.0, 1.0, 1.0, 1.1]
        hc = st.columns(COLS, gap='small')


        def _header_cell(txt, shift=0, align='center', cls=''):
            a = str(align).lower()
            just = 'flex-start' if a == 'left' else ('flex-end' if a == 'right' else 'center')
            padl = 'padding-left: 18px;' if ('team' in str(cls).lower()) else ''
            nudge = f"transform: translateX({shift}px);" if isinstance(shift, (int, float)) and shift != 0 else ''
            return (
                f"<div style='width:100%; direction:ltr; display:flex; justify-content:{just}; align-items:center; "
                f"{padl}{nudge}font-weight:600; letter-spacing:.3px'>{txt}</div>"
            )


        hc[0].markdown(_header_cell('Rank', 0, 'center'), unsafe_allow_html=True)
        hc[1].markdown(_header_cell('Team', 0, 'left', 'team'), unsafe_allow_html=True)
        hc[2].markdown(_header_cell('AdvPts', 0, 'right'), unsafe_allow_html=True)
        hc[3].markdown(_header_cell('Qual', 0, 'right'), unsafe_allow_html=True)
        hc[4].markdown(_header_cell('Alliance', 0, 'right'), unsafe_allow_html=True)
        hc[5].markdown(_header_cell('Playoff', 0, 'right'), unsafe_allow_html=True)
        hc[6].markdown(_header_cell('Awards', 0, 'right'), unsafe_allow_html=True)
        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
        for idx, row in enumerate(adv_summary.itertuples(index=False), start=1):
            c = st.columns(COLS, gap='small')
            c[0].markdown(f"<div class='mae-center'><div class='mae-rankchip'>{idx}</div></div>",
                          unsafe_allow_html=True)
            tid = int(getattr(row, "team"));
            tnm = getattr(row, "team_name")
            exp = c[1].expander(f"{tid} – {tnm}" if tnm else str(tid), expanded=False)
            c[2].markdown(f"<div class='mae-right'>{_chip(int(getattr(row, 'AdvPts')), 'adv')}</div>",
                          unsafe_allow_html=True)
            c[3].markdown(f"<div class='mae-right'>{_chip(int(getattr(row, 'Qual')), 'qual')}</div>",
                          unsafe_allow_html=True)
            c[4].markdown(f"<div class='mae-right'>{_chip(int(getattr(row, 'Alliance')), 'alli')}</div>",
                          unsafe_allow_html=True)
            c[5].markdown(f"<div class='mae-right'>{_chip(int(getattr(row, 'Playoff')), 'play')}</div>",
                          unsafe_allow_html=True)
            c[6].markdown(f"<div class='mae-right'>{_chip(int(getattr(row, 'Awards')), 'awd')}</div>",
                          unsafe_allow_html=True)

            df_ev = adv_per_team.get(tid)
            with exp:
                if df_ev is not None and not df_ev.empty:
                    show = df_ev.copy()
                    if "composition" in show.columns:
                        comp = show["composition"].apply(lambda d: d if isinstance(d, dict) else {})
                        for k in ["qual_rank", "field_size", "alliance_no", "playoff_finish", "awards"]:
                            show[k] = comp.apply(lambda x: x.get(k, ""))
                        show = show.drop(columns=["composition"], errors="ignore")
                    for _col in ["qual_rank", "field_size", "alliance_no", "playoff_finish", "awards"]:
                        if _col not in show.columns:
                            show[_col] = ""
                    for _col in ["event", "qual_pts", "alliance_pts", "playoff_pts", "awards_pts", "total"]:
                        if _col not in show.columns:
                            show[_col] = 0 if _col != "event" else ""
                    _show_cols = ["event", "qual_pts", "alliance_pts", "playoff_pts", "awards_pts", "total"]
                    show = show.reindex(columns=_show_cols, fill_value="")
                    st.dataframe(show, use_container_width=True, hide_index=True)
                else:
                    st.caption("אין פירוט אירועים עבור קבוצה זו.")

with tab_diag:
    with st.spinner("Computing diagnostics"):
        # Guarded diagnostics reuse: only use diag_df/diag_meta if they exist
        _use_cached_diag = False
        try:
            _use_cached_diag = isinstance(diag_df, pd.DataFrame) and isinstance(diag_meta, tuple) and len(
                diag_meta) == 2
        except NameError:
            _use_cached_diag = False

        if _use_cached_diag:
            dtab, n_matches_opr, n_teams_opr = diag_df, diag_meta[0], diag_meta[1]
        else:
            dtab, n_matches_opr, n_teams_opr = build_diagnostics(ev_view, base_raw, teams_master, country)
    st.caption(f"npOPR computed on ALL matches: {n_matches_opr} matches across {n_teams_opr} teams.")
    if not dtab.empty:
        df_corr = dtab[["EPA", "OPR"]].dropna()
        r = float(df_corr["EPA"].corr(df_corr["OPR"])) if len(df_corr) >= 3 else float("nan")
        st.subheader(f"EPA vs npOPR correlation: r = {r:.3f}" if not math.isnan(r) else "EPA vs npOPR correlation: n/a")
        k = st.slider("Show top-K outliers", 5, 50, 15, 1)
        outliers = dtab.assign(abs_delta=lambda d: d["delta(OPR-EPA)"].abs()).sort_values("abs_delta",
                                                                                          ascending=False).head(k)
        cols = ["team", "team_name", "country", "EPA", "EPA_Auto", "EPA_Teleop", "EPA_Endgame", "OPR", "delta(OPR-EPA)",
                "matches", "N_EPA_updates", "ally_epa_avg", "opp_epa_avg"]
        cols = [c for c in cols if c in outliers.columns]
        st.dataframe(outliers[cols], use_container_width=True, height=420)
    else:
        st.info("No teams found in selected country.")


#
# --- Single-event Ranking tab (clean, fast, uses cached ranking df) ---


# ---------- FIX: Definition of endgame_points_by_team ----------
@st.cache_data(ttl=600, show_spinner=False)
def endgame_points_by_team(season: int, event_code: str, user: str, token: str):
    """
    Fetches match details for an event and computes the average endgame and teleop
    points per team. Returns two dictionaries (endgame_map, teleop_map).
    """
    ok, _, payload, _ = api_matches(season, event_code)
    if not ok:
        return {}, {}

    df = alliances_rows_from_payload(payload)
    if df.empty:
        return {}, {}

    team_scores = []
    for _, row in df.iterrows():
        for team_col in ["t1", "t2"]:
            team_id = row.get(team_col)
            if pd.notna(team_id):
                team_scores.append({
                    "team": int(team_id),
                    "endgame": float(row.get("endgame", 0.0)),
                    "teleop": float(row.get("teleop", 0.0))
                })

    if not team_scores:
        return {}, {}

    agg_df = pd.DataFrame(team_scores).groupby("team").mean()
    eg_map = agg_df["endgame"].to_dict()
    tl_map = agg_df["teleop"].to_dict()
    return eg_map, tl_map


# ---------- END FIX ----------

# ---------- FIX: Definition of _ensure_endgame_epa ----------
def _ensure_endgame_epa(df: pd.DataFrame, endgame_override=None, teleop_override=None) -> pd.DataFrame:
    """
    Overrides EPA component columns in a DataFrame if override series are provided.
    This ensures the display uses more precise, event-specific averages when available.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df_out = df.copy()

    if endgame_override is not None and not endgame_override.empty:
        df_out["EPA_Endgame"] = df_out["team"].map(endgame_override).fillna(df_out["EPA_Endgame"])

    if teleop_override is not None and not teleop_override.empty:
        df_out["EPA_Teleop"] = df_out["team"].map(teleop_override).fillna(df_out["EPA_Teleop"])

    return df_out


# ---------- END FIX ----------


with tab_single:
    # --- Single-event Ranking (uses EXACT same layout as Rankings tab) ---
    st.markdown("### 🧭 אירוע בודד — EPA (מתבסס על הנתונים שכבר נטענו)")

    # same CSS & helpers as Rankings
    COLS = [0.5, 4.6, 0.85, 0.9, 0.85, 0.85, 0.85, 0.9, 1.1]  # added RS col


    def _header_cell(txt, shift=0, align='center', cls=''):
        a = str(align).lower()
        just = 'flex-start' if a == 'left' else ('flex-end' if a == 'right' else 'center')
        padl = 'padding-left: 18px;' if ('team' in str(cls).lower()) else ''
        nudge = f"transform: translateX({shift}px);" if isinstance(shift, (int, float)) and shift != 0 else ''
        return (
            f"<div style='width:100%; direction:ltr; display:flex; justify-content:{just}; align-items:center; "
            f"{padl}{nudge}font-weight:600; letter-spacing:.3px'>{txt}</div>"
        )


    # Build events list & collapse ILCMP*
    evs = ev_view.copy()


    def _country_matches(c, target):
        cc = str(c or '').strip().upper();
        tt = str(target or '').strip().upper()
        if tt in ('', 'WORLD'): return True
        return cc == tt or (cc.startswith('IS') and tt in ('IL', 'ISR', 'ישראל'))


    if "country" in evs.columns and str(country).strip():
        evs = evs[evs["country"].apply(lambda x: _country_matches(x, country))]
    evs["fam"] = evs["event_code"].astype(str).map(lambda x: family_of(x))
    codes_non_family = sorted([c for c, f in zip(evs["event_code"].astype(str), evs["fam"]) if f != "ILCMP"])
    has_ilcmp = any(f == "ILCMP" for f in evs["fam"])
    selector_choices = codes_non_family + (["ILCMP"] if has_ilcmp else [])

    if not selector_choices:
        st.info("לא נמצאו אירועים עבור המדינה שנבחרה.")
    else:
        default_idx = (
            selector_choices.index("ILCMP") if "ILCMP" in selector_choices else max(len(selector_choices) - 1, 0))
        ev_code = st.selectbox("בחר אירוע", selector_choices, index=default_idx, key="single_event_ev_code")

        # Mask matches from base_raw
        if ev_code == "ILCMP":
            mask = base_raw["ev_family"].astype(str).str.upper() == "ILCMP"
        else:
            mask = base_raw["ev"].astype(str).str.upper() == str(ev_code).upper()
        b = base_raw[mask].copy()

        import pandas as pd

        if b.empty:
            st.info("לא נמצאו משחקים/קבוצות לאירוע.")
        else:
            # event teams
            teams = set()
            for ccol in ("t1", "t2"):
                if ccol in b.columns:
                    teams |= set(pd.to_numeric(b[ccol], errors="coerce").dropna().astype(int).tolist())

            # ranking df from cache (same as Rankings)
            rank_df_src = st.session_state.get("rank_df", st.session_state.get("rank_df_cached", None))
            rank_df2 = rank_df_src.copy() if isinstance(rank_df_src, pd.DataFrame) else pd.DataFrame()
            if rank_df2.empty:
                sig = f"rnk|{int(season)}|{str(country).upper()}"
                try:
                    rank_df2 = build_rankings_cached(sig, ev_view, base_raw, teams_master, country)
                except Exception:
                    rank_df2 = pd.DataFrame()


                # build event-only Record
                rec = {}
                try:
                    for (evv, idxv), rows in b.groupby(["ev", "idx"]):
                        try:
                            r = rows[rows["side"] == "red"].iloc[0];
                            bl = rows[rows["side"] == "blue"].iloc[0]
                            rs = float(r.get("score", 0));
                            bs = float(bl.get("score", 0))
                        except Exception:
                            continue
                        red = [];
                        blue = []
                        for col in ("t1", "t2"):
                            try:
                                red.append(int(r.get(col))); blue.append(int(bl.get(col)))
                            except Exception:
                                pass
                        for t in red + blue: rec.setdefault(t, {"W": 0, "L": 0, "T": 0})
                        if abs(rs - bs) < 1e-9:
                            for t in red + blue: rec[t]["T"] += 1
                        elif rs > bs:
                            for t in red:  rec[t]["W"] += 1
                            for t in blue: rec[t]["L"] += 1
                        else:
                            for t in blue: rec[t]["W"] += 1
                            for t in red:  rec[t]["L"] += 1
                except Exception:
                    rec = {}
                rec_df = pd.DataFrame([{"team": t, **v} for t, v in rec.items()]) if rec else pd.DataFrame()
                if not rec_df.empty:
                    for c in ["W", "L", "T"]:
                        if c not in rec_df.columns: rec_df[c] = 0
                        rec_df[c] = pd.to_numeric(rec_df[c], errors="coerce").fillna(0).astype(int)
                    rec_df["RECORD"] = rec_df.apply(
                        lambda r: f"{int(r['W'])}-{int(r['L'])}" + (f"-{int(r['T'])}" if int(r['T']) > 0 else ""),
                        axis=1)


                # --- Normalize RECORD column (handle merges) ---
                cols = set(rank_df2.columns.astype(str))
                # Coalesce merged columns if present
                if {'RECORD_x', 'RECORD_y'} <= cols:
                    rank_df2['RECORD'] = rank_df2['RECORD_x'].fillna('').astype(str)
                    msk = rank_df2['RECORD'].eq('') | rank_df2['RECORD'].isna()
                    rank_df2.loc[msk, 'RECORD'] = rank_df2.loc[msk, 'RECORD_y'].fillna('').astype(str)
                    rank_df2.drop(columns=[c for c in ['RECORD_x', 'RECORD_y'] if c in rank_df2.columns], inplace=True)
                elif 'RECORD' not in cols and 'Record' in cols:
                    rank_df2.rename(columns={'Record': 'RECORD'}, inplace=True)
                # Ensure exists and string type
                if 'RECORD' not in rank_df2.columns:
                    rank_df2['RECORD'] = ''
                rank_df2['RECORD'] = rank_df2['RECORD'].fillna('').astype(str)

                # Normalize 'Record' -> 'RECORD' and ensure string format
                if 'RECORD' not in rank_df2.columns and 'Record' in rank_df2.columns:
                    rank_df2.rename(columns={'Record': 'RECORD'}, inplace=True)
                if 'RECORD' not in rank_df2.columns:
                    rank_df2['RECORD'] = ''
                rank_df2['RECORD'] = rank_df2['RECORD'].fillna('').astype(str)

                # Normalize possible 'Record' legacy column name
                if 'RECORD' not in rank_df2.columns and 'Record' in rank_df2.columns:
                    rank_df2.rename(columns={'Record': 'RECORD'}, inplace=True)
                else:
                    if 'RECORD' not in rank_df2.columns:  # FIX: Check existence before assigning
                        rank_df2["RECORD"] = ""

                # sort like Rankings
                try:
                    # --- RECORD fallback: copy exactly from Rankings tab if event-specific record is empty ---
                    try:
                        need_fallback = ('RECORD' not in rank_df2.columns) or rank_df2['RECORD'].fillna('').eq('').all()
                    except Exception:
                        need_fallback = True
                    if need_fallback:
                        try:
                            if isinstance(rank_df_src, pd.DataFrame) and 'team' in rank_df_src.columns:
                                rec_col = 'Record' if 'Record' in rank_df_src.columns else (
                                    'RECORD' if 'RECORD' in rank_df_src.columns else None)
                                if rec_col:
                                    rec_map = rank_df_src.set_index('team')[rec_col]
                                    rank_df2['RECORD'] = rank_df2['team'].map(rec_map).fillna('').astype(str)
                        except Exception:
                            pass
                    # --- Merge Endgame & Teleop clean (from FTC API) ---
                    try:
                        _user = st.session_state.get("ftc_user_input") or st.secrets.get("ftc_api_user", "aviad")
                        _token = st.session_state.get("ftc_token_input") or st.secrets.get("ftc_api_token",
                                                                                           "90A41204-F536-41DE-B35D-8A128719ED23")
                    except Exception:
                        _user, _token = "aviad", "90A41204-F536-41DE-B35D-8A128719ED23"
                    # ---------- FIX: Replaced 'selected_event_code' with 'ev_code' ----------
                    evt_code = st.session_state.get("single_event_code") or ev_code
                    # ---------- END FIX ----------
                    eg_map, tl_map = endgame_points_by_team(int(season), str(evt_code), _user, _token)
                    import pandas as _pd

                    eg_series = _pd.Series(eg_map, name="EndgameFromAPI")
                    # === RS merge from FTC rankings ===
                    try:
                        _evt_code = st.session_state.get("single_event_code") or evt_code
                        _rsdf = ftc_rankings_df(int(season), str(_evt_code))
                        if _rsdf is not None and not _rsdf.empty and 'team' in rank_df2.columns:
                            import pandas as pd
                            _rs_map = _rsdf.set_index('team')['RS']
                            rank_df2['RS'] = pd.to_numeric(rank_df2['team'], errors='coerce').map(_rs_map)
                    except Exception:
                        pass

                    tl_series = _pd.Series(tl_map, name="TeleopCleanFromAPI")
                    if 'team' in rank_df2.columns:
                        end_override = rank_df2['team'].map(eg_series)
                        tele_override = rank_df2['team'].map(tl_series)
                    else:
                        _parse_team = rank_df2.get('Team', '').astype(str).str.extract(r'^(\d+)').iloc[:, 0].astype(
                            float)
                        end_override = _parse_team.map(eg_series)
                        tele_override = _parse_team.map(tl_series)
                    rank_df2 = _ensure_endgame_epa(rank_df2, endgame_override=end_override,
                                                   teleop_override=tele_override)

                except Exception:
                    pass

                # Header
                hc = st.columns(COLS, gap='small')
                hc[0].markdown(_header_cell('Rank', 0, 'center'), unsafe_allow_html=True)
                hc[1].markdown(_header_cell('Team', 0, 'left', 'team'), unsafe_allow_html=True)
                hc[2].markdown(_header_cell('EPA', -18, 'right'), unsafe_allow_html=True)
                hc[3].markdown(_header_cell('npOPR', 0, 'right'), unsafe_allow_html=True)
                hc[4].markdown(_header_cell('Auto EPA', 0, 'right'), unsafe_allow_html=True)
                hc[5].markdown(_header_cell('Teleop EPA', 0, 'right'), unsafe_allow_html=True)
                hc[6].markdown(_header_cell('Endgame EPA', 36, 'right'), unsafe_allow_html=True)
                hc[7].markdown(_header_cell('Record', 0, 'right'), unsafe_allow_html=True)

                # Rows (use same attribute names as Rankings)
                for idx, row in enumerate(rank_df2.itertuples(index=False), start=1):
                    c = st.columns(COLS, gap="small")
                    t_id = getattr(row, "team", None)
                    t_name = getattr(row, "team_name", "") or str(t_id)
                    label = f"{t_id} – {t_name}" if t_name and str(t_id) not in str(t_name) else str(t_name or t_id)
                    c[0].markdown(f"<div class='mae-center'><div class='mae-rankchip'>{idx}</div></div>",
                                  unsafe_allow_html=True)
                    with c[1]:
                        with st.expander(label):
                            import pandas as pd

                            # Ensure per_team_detail exists (compute from cache if missing)
                            if not isinstance(st.session_state.get('per_team_detail'), dict):
                                sig_adv = f"adv|{int(season)}|{str(country).upper()}"
                                try:
                                    adv_summary, per_team_detail = build_advancement_cached(sig_adv, base_raw,
                                                                                            teams_master, ev_view,
                                                                                            country, season)
                                    st.session_state['adv_summary'] = adv_summary
                                    st.session_state['per_team_detail'] = per_team_detail
                                except Exception:
                                    st.session_state['per_team_detail'] = {}
                            ptd = st.session_state.get('per_team_detail') or {}
                            best = {'qual': 0, 'alli': 0, 'play': 0, 'aw': 0}
                            df_ev = ptd.get(int(t_id)) if isinstance(ptd, dict) else None
                            if isinstance(df_ev, pd.DataFrame) and not df_ev.empty:
                                fam = family_of(str(ev_code))
                                sub = pd.DataFrame()
                                # Prefer using family/code column if present
                                if fam == 'ILCMP' and 'family' in df_ev.columns:
                                    sub = df_ev[df_ev['family'].astype(str).str.upper() == 'ILCMP']
                                elif fam != 'ILCMP' and 'code' in df_ev.columns:
                                    sub = df_ev[df_ev['code'].astype(str) == str(ev_code)]
                                # If still empty, filter via 'composition' sources (qual/alliance/playoff/awards)
                                if sub.empty and 'composition' in df_ev.columns:
                                    def _match_comp(comp):
                                        c = comp if isinstance(comp, dict) else {}
                                        srcs = [c.get('qual_source'), c.get('alliance_source'), c.get('playoff_source'),
                                                c.get('awards_source')]
                                        if fam == 'ILCMP':
                                            return any(
                                                isinstance(s, str) and s.upper().startswith('ILCMP') for s in srcs)
                                        else:
                                            return any(
                                                isinstance(s, str) and s.upper() == str(ev_code).upper() for s in srcs)


                                    try:
                                        sub = df_ev[df_ev['composition'].apply(_match_comp)]
                                    except Exception:
                                        sub = df_ev.copy()
                                if not sub.empty:
                                    for src_col, dst in [('qual_pts', 'qual'), ('alliance_pts', 'alli'),
                                                         ('playoff_pts', 'play')]:
                                        if src_col in sub.columns:
                                            try:
                                                best[dst] = int(float(sub[src_col].max()))
                                            except Exception:
                                                best[dst] = 0
                                    # Awards: prefer exact awards_source
                                    aw_val = 0
                                    if 'composition' in sub.columns:
                                        def _aw_exact(comp):
                                            c = comp if isinstance(comp, dict) else {}
                                            src = c.get('awards_source')
                                            if not isinstance(src, str): return False
                                            return (src.upper() == 'ILCMP') if fam == 'ILCMP' else (
                                                        src.upper() == str(ev_code).upper())


                                        try:
                                            exact = sub[sub['composition'].apply(_aw_exact)]
                                            if not exact.empty and 'awards_pts' in exact.columns:
                                                aw_val = int(float(exact['awards_pts'].max()))
                                            elif 'awards_pts' in sub.columns:
                                                aw_val = int(float(sub['awards_pts'].max()))
                                        except Exception:
                                            if 'awards_pts' in sub.columns:
                                                try:
                                                    aw_val = int(float(sub['awards_pts'].max()))
                                                except Exception:
                                                    aw_val = 0
                                    best['aw'] = aw_val
                            # Summary
                            ap_tot = int(best['qual'] + best['alli'] + best['play'] + best['aw'])
                            st.markdown(f"**ניקוד העפלה (סה\"כ):** {ap_tot}")
                            st.markdown(
                                f"• Qual: {best['qual']} | Alliance: {best['alli']} | Playoff: {best['play']} | Awards: {best['aw']}")


                    def _fmt(x):
                        try:
                            return f"{float(x):.2f}"
                        except Exception:
                            return ""



                    c[2].markdown(f"<div class='mae-right'>{_fmt(getattr(row, 'EPA', ''))}</div>",
                                  unsafe_allow_html=True)
                    c[3].markdown(f"<div class='mae-right'>{_fmt(getattr(row, 'OPR', ''))}</div>",
                                  unsafe_allow_html=True)
                    c[4].markdown(f"<div class='mae-right'>{_fmt(getattr(row, 'EPA_Auto', ''))}</div>",
                                  unsafe_allow_html=True)
                    c[5].markdown(f"<div class='mae-right'>{_fmt(getattr(row, 'EPA_Teleop', ''))}</div>",
                                  unsafe_allow_html=True)
                    c[6].markdown(f"<div class='mae-right'>{_fmt(getattr(row, 'EPA_Endgame', ''))}</div>",
                                  unsafe_allow_html=True)
                    c[7].markdown( f"<div class='mae-right'>{getattr(row, 'RECORD', '') or getattr(row, 'Record', '')}</div>",
                        unsafe_allow_html=True)

# ===== Rankings UI helper =====

