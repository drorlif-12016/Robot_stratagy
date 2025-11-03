import streamlit as st
import pandas as pd
import requests
import json

# -------------------------------
# CONFIG
# -------------------------------
API_BASE = "https://ftc-events.firstinspires.org/api/v2"
USER = "drorl"
TOKEN = "6DB10CF7-BF8C-4415-BA0B-BE3786462A67"

# -------------------------------
# STREAMLIT PAGE SETUP
# -------------------------------
st.set_page_config(page_title="FTC Rankings Viewer", page_icon="🤖", layout="wide")
st.title("🤖 FTC Rankings Viewer")

st.write("Enter a **season** and **event code** to fetch ranking data from the FTC API.")

# -------------------------------
# USER INPUTS
# -------------------------------
season = st.text_input("Season (e.g. 2025):", value="2025")
event_code = st.text_input("Event Code (e.g. ISRAEL-Q):", value="ISRAEL-Q")

# -------------------------------
# FETCH BUTTON
# -------------------------------
if st.button("Fetch Rankings"):
    with st.spinner("Fetching data from FTC API..."):
        url = f"{API_BASE}/{season}/events/{event_code}/rankings"

        headers = {
            "Accept": "application/json",
            "User-Agent": "FTC-Scouting-App/1.0",
        }

        try:
            response = requests.get(url, auth=(USER, TOKEN), headers=headers)
            status = response.status_code

            if status != 200:
                st.error(f"❌ Error {status}: Could not fetch data.")
                st.text(response.text)
            else:
                try:
                    data = response.json()
                except json.JSONDecodeError:
                    st.error("❌ Response was not valid JSON.")
                    st.text(response.text)
                    st.stop()

                if "rankings" in data and len(data["rankings"]) > 0:
                    df = pd.DataFrame(data["rankings"])
                    st.success(f"✅ Successfully loaded {len(df)} rankings.")
                    st.dataframe(df)
                else:
                    st.warning("⚠️ No ranking data found for this event.")
                    st.json(data)

        except requests.RequestException as e:
            st.error(f"❌ Request failed: {e}")
