import streamlit as st
import pandas as pd
import requests
import json

# -------------------------------
# CONFIGURATION
# -------------------------------
API_BASE = "http://ftc-api.firstinspires.org/v2.0"
USER = "drorl"
TOKEN = "6DB10CF7-BF8C-4415-BA0B-BE3786462A67"

# -------------------------------
# STREAMLIT PAGE SETUP
# -------------------------------
st.set_page_config(page_title="FTC Matches Viewer", page_icon="🤖", layout="wide")
st.title("🤖 FTC Matches Viewer")

st.write(
    "Enter a **season** and **event code** to fetch Match data from the official FTC API. "
    "If you see errors, check your credentials, season, or event code."
)

# -------------------------------
# USER INPUTS
# -------------------------------
season = st.text_input("Season (e.g. 2025):", value="2025")
event_code = st.text_input("Event Code (e.g. ISRAEL-Q):", value="ISRAEL-Q")

# -------------------------------
# FETCH BUTTON
# -------------------------------
if st.button("Fetch Matches"):
    with st.spinner("Fetching data from FTC API..."):
        url = f"{API_BASE}/{season}/matches/{event_code}"

        headers = {
            "Accept": "application/json",
            "User-Agent": "FTC-Scouting-App/1.0",
        }

        try:
            response = requests.get(url, auth=(USER, TOKEN), headers=headers)

            # Debug info
            st.write("**Status Code:**", response.status_code)
            st.write("**Request URL:**", response.url)
            st.text_area("Raw Response (first 500 chars):", response.text[:500])

            if response.status_code == 401:
                st.error("❌ Unauthorized (401): Check your USER or TOKEN in the script.")
                st.stop()
            elif response.status_code == 404:
                st.error("❌ Event not found (404). Check your season and event code.")
                st.stop()
            elif response.status_code == 302:
                st.warning("⚠️ Redirect (302): The API may have redirected you. Try again later.")
                st.stop()
            elif response.status_code != 200:
                st.error(f"❌ Error {response.status_code}: {response.reason}")
                st.stop()

            # Try to parse JSON
            try:
                data = response.json()
            except json.JSONDecodeError:
                st.error("❌ Response was not valid JSON. The API might have returned HTML or nothing.")
                st.text(response.text)
                st.stop()

            # Display data
            if "Matches" in data and len(data["Matches"]) > 0:
                df = pd.DataFrame(data["Matches"])
                st.success(f"✅ Successfully loaded {len(df)} Matches.")
                st.dataframe(df)
            else:
                st.warning("⚠️ No Match data found for this event.")
                st.json(data)

        except requests.RequestException as e:
            st.error(f"❌ Request failed: {e}")
