import streamlit as st
import requests
import json

st.title("FTC API – SortOrder1 Fetcher")

# Load credentials
def load_credentials():
    try:
        with open("ftc_api_credentials.json", "r") as f:
            data = json.load(f)
        return data["user"], data["token"]
    except:
        st.error("Could not load ftc_api_credentials.json")
        st.stop()

USER, TOKEN = load_credentials()

# Get SortOrder1 from FTC API
def get_sortorder1(season: int, eventCode: str):
    url = f"http://ftc-api.firstinspires.org/v2.0/{season}/rankings/{eventCode}"
    r = requests.get(url, auth=(USER, TOKEN))

    if r.status_code != 200:
        st.error(f"API Error {r.status_code}")
        return None

    data = r.json()
    rankings = data.get("Rankings") or data.get("rankings") or []

    rows = []
    for r in rankings:
        rows.append({
            "team": r.get("teamNumber"),
            "rank": r.get("rank"),
            "SortOrder1": r.get("sortOrder1") or r.get("SortOrder1")
        })

    return rows


# UI
season = st.number_input("Season", value=2025)
event_code = st.text_input("Event Code", value="USTXFMS1")

if st.button("Fetch SortOrder1"):
    rows = get_sortorder1(season, event_code)
    if rows:
        st.success("Done!")
        st.dataframe(rows)
