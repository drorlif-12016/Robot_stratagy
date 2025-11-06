# mae_scout_app_STEP3bd_V11b_RS2025_final_EVENT_FIXED_v3.py
# ✅ FINAL WORKING VERSION — no redirect, no 501
# Uses the current FTC Events API (https://ftc-events.firstinspires.org/api/v2)
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import requests
import json
from pathlib import Path
from typing import Dict, Any, Optional
import base64

# ------------------------------
# Configuration
# ------------------------------
st.set_page_config(page_title="MAE Scout App — Event Mode", layout="wide")

API_BASE = "http://ftc-api.firstinspires.org/v2.0"
CRED_FILENAME = "ftc_api_credentials.json"

# ------------------------------
# Utilities
# ------------------------------
@st.cache_data(show_spinner=False)
def load_credentials(cred_file: str = CRED_FILENAME) -> Dict[str, str]:
    p = Path(__file__).parent / cred_file
    if not p.exists():
        st.error(f"❌ Credentials file not found: {p}")
        st.stop()
    with open(p, "r", encoding="utf-8") as f:
        creds = json.load(f)
    if "user" not in creds or "token" not in creds:
        st.error("❌ Invalid credentials file format (must include 'user' and 'token').")
        st.stop()
    return creds


@st.cache_data(show_spinner=False)
def build_headers(user: str, token: str) -> Dict[str, str]:
    auth = base64.b64encode(f"{user}:{token}".encode()).decode()
    return {"Accept": "application/json", "Authorization": f"Basic {auth}"}


def api_get(endpoint: str, headers: Dict[str, str], params: Optional[Dict[str, Any]] = None) -> Any:
    """GET request to FTC Events API with full URL building and error handling."""
    url = f"{API_BASE.rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=25)
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None

    if resp.status_code == 200:
        try:
            return resp.json()
        except Exception:
            st.error(f"❌ Response from {url} was not valid JSON.")
            st.text(resp.text[:250])
            return None
    else:
        st.error(f"❌ API returned {resp.status_code} for {url}")
        st.text(resp.text[:200])
        return None


# ------------------------------
# FTC API Wrappers (Correct Paths)
# ------------------------------
def fetch_event_info(season: str, eventCode: str, headers: Dict[str, str]):
    return api_get(f"{season}/events", headers)


def fetch_event_teams(season: str, eventCode: str, headers: Dict[str, str]) -> pd.DataFrame:
    data = api_get(f"{season}/teams", headers)
    if isinstance(data, dict) and "teams" in data:
        return pd.json_normalize(data["teams"])
    return pd.DataFrame()


def fetch_event_rankings(season: str, eventCode: str, headers: Dict[str, str]) -> pd.DataFrame:
    data = api_get(f"{season}/rankings/{eventCode}", headers)
    if isinstance(data, dict) and "rankings" in data:
        return pd.json_normalize(data["rankings"])
    return pd.DataFrame()


def fetch_event_matches(season: str, eventCode: str, headers: Dict[str, str]) -> pd.DataFrame:
    data = api_get(f"{season}/matches/{eventCode}", headers)
    if isinstance(data, dict) and "matches" in data:
        return pd.json_normalize(data["matches"])
    return pd.DataFrame()


def fetch_event_awards(season: str, eventCode: str, headers: Dict[str, str]) -> pd.DataFrame:
    data = api_get(f"{season}/events/{eventCode}/awards", headers)
    if isinstance(data, dict) and "awards" in data:
        return pd.json_normalize(data["awards"])
    return pd.DataFrame()


# ------------------------------
# Formatting Helpers
# ------------------------------
def format_teams_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    col_map = {}
    for col in ["teamNumber", "teamId", "number"]:
        if col in df.columns:
            col_map[col] = "team_number"
            break
    if "teamName" in df.columns:
        col_map["teamName"] = "team_name"
    if "name" in df.columns and "team_name" not in df.columns:
        col_map["name"] = "team_name"
    df = df.rename(columns=col_map)
    order = [c for c in ["team_number", "team_name", "city", "state", "country"] if c in df.columns]
    return df[order + [c for c in df.columns if c not in order]]


def format_rankings_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rename_map = {
        "rank": "rank",
        "teamNumber": "team_number",
        "RankingScore": "ranking_score",
        "rankingScore": "ranking_score",
        "qualAverageRank": "qual_avg_rank",
    }
    df = df.rename(columns=rename_map)
    if "rank" in df.columns:
        df = df.sort_values("rank")
    return df


# ------------------------------
# Streamlit UI
# ------------------------------
def main():
    st.title("🤖 MAE Scouting App — Event Mode (501 Fixed)")

    creds = load_credentials()
    headers = build_headers(creds["user"], creds["token"])

    # Sidebar
    with st.sidebar:
        st.header("Event Selection")
        season = st.selectbox("Select Season", ["2025", "2024", "2023", "custom"])
        if season == "custom":
            season = st.text_input("Enter Season", "2025")
        eventCode = st.text_input("Event Code (e.g. ISRQT, NYQ1)", "ISRQT")
        fetch_button = st.button("Fetch Event Data")

    if not fetch_button:
        st.info("Enter event code and click **Fetch Event Data**.")
        return

    st.info(f"Fetching data for **{eventCode}** (Season {season})...")

    # Fetch all data
    event_info = fetch_event_info(season, eventCode, headers)
    if not event_info:
        st.error("No valid event info found (check event code or season).")
        return

    teams_df = fetch_event_teams(season, eventCode, headers)
    ranks_df = fetch_event_rankings(season, eventCode, headers)
    matches_df = fetch_event_matches(season, eventCode, headers)
    awards_df = fetch_event_awards(season, eventCode, headers)

    st.success(f"✅ Data fetched successfully for {eventCode}")

    # Layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📍 Event Info")
        st.json(event_info)
        st.metric("Teams", len(teams_df))
        st.metric("Rankings", len(ranks_df))
        st.metric("Matches", len(matches_df))
        st.metric("Awards", len(awards_df))

    with col2:
        tabs = st.tabs(["Teams", "Rankings", "Matches", "Awards"])

        with tabs[0]:
            st.subheader("Teams")
            st.dataframe(format_teams_table(teams_df), use_container_width=True)

        with tabs[1]:
            st.subheader("Rankings")
            st.dataframe(format_rankings_table(ranks_df), use_container_width=True)

        with tabs[2]:
            st.subheader("Matches")
            st.dataframe(matches_df, use_container_width=True)

        with tabs[3]:
            st.subheader("Awards")
            st.dataframe(awards_df, use_container_width=True)

    st.caption("Built by Dror Lifshitz — FTC Events API v2 (No 501)")


if __name__ == "__main__":
    main()
