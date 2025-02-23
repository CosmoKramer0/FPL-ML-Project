import streamlit as st
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Constants
PLAYER_API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURE_API_URL = "https://fantasy.premierleague.com/api/fixtures/"

# Helper function to fetch and process data
def fetch_data():
    player_response = requests.get(PLAYER_API_URL)
    fixture_response = requests.get(FIXTURE_API_URL)

    if player_response.status_code == 200 and fixture_response.status_code == 200:
        data = player_response.json()
        fixtures = fixture_response.json()

        players = data["elements"]
        teams = {team["id"]: team["name"] for team in data["teams"]}
        positions = {1: "Goalkeeper", 2: "Defender", 3: "Midfielder", 4: "Forward"}

        # Exclude Managers
        players = [player for player in players if player["element_type"] != 5]

        # Extract fixture difficulty for each team
        fixture_difficulty = {}
        for fixture in fixtures:
            home_team = fixture["team_h"]
            away_team = fixture["team_a"]
            fixture_difficulty.setdefault(home_team, []).append(fixture["team_h_difficulty"])
            fixture_difficulty.setdefault(away_team, []).append(fixture["team_a_difficulty"])

        # Compute average fixture difficulty for each team (next 3 matches)
        avg_fixture_difficulty = {
            team_id: round(sum(difficulties[:3]) / len(difficulties[:3]), 2)
            if difficulties else None
            for team_id, difficulties in fixture_difficulty.items()
        }

        # Prepare player data
        player_data = []
        for player in players:
            team_id = player["team"]
            position_id = player["element_type"]

            price = player["now_cost"] / 10
            total_points = player["total_points"]
            form = float(player["form"])
            xG = float(player.get("expected_goals", 0))
            xA = float(player.get("expected_assists", 0))
            xGI = xG + xA
            ppm = round(total_points / price, 2) if price > 0 else 0
            minutes_per_game = round(player["minutes"] / (player["total_points"] / form) if form > 0 else 1, 2)
            future_points = player.get("event_points", 0)

            # Differential score
            diff_score = round((form * 2 + xGI * 1.5) / (float(player["selected_by_percent"]) + 1), 2)

            player_data.append({
                "Name": player["web_name"],
                "Team": teams.get(team_id, "Unknown"),
                "Position": positions.get(position_id, "Unknown"),
                "Price": price,
                "Ownership %": float(player["selected_by_percent"]),
                "Expected Goals": xG,
                "Expected Assists": xA,
                "Expected Goal Involvement (xGI)": xGI,
                "Total Points": total_points,
                "Form": form,
                "Points per Million (PPM)": ppm,
                "Minutes per Game": minutes_per_game,
                "Fixture Difficulty (Next 3)": avg_fixture_difficulty.get(team_id, None),
                "Differential Score": diff_score,
                "Future Points": future_points
            })

        df = pd.DataFrame(player_data)
        return df
    else:
        st.error("Failed to fetch FPL data. Check your internet connection!")
        return pd.DataFrame()

# Train Random Forest model for future points prediction
def train_model(df):
    features = ["Form", "Expected Goals", "Expected Assists", "Total Points", "Ownership %", "Price"]
    target = "Future Points"

    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    # st.sidebar.write(f"Model Mean Absolute Error: {mae:.2f}")

    df["Predicted Future Points"] = model.predict(df[features])
    return df

# Streamlit UI
st.title("⚽ FPL Player Predictions Dashboard")

# Fetch data
df = fetch_data()

if not df.empty:
    # Train model and add predictions
    df = train_model(df)

    # Filters
    st.sidebar.header("Filter Players")
    position_filter = st.sidebar.multiselect("Select Position:", df["Position"].unique(), default=df["Position"].unique())
    price_range = st.sidebar.slider("Price Range:", float(df["Price"].min()), float(df["Price"].max()), (0.0, 20.0))
    ownership_range = st.sidebar.slider("Ownership %:", float(df["Ownership %"].min()), float(df["Ownership %"].max()), (0.0, 100.0))

    # Apply filters
    df_filtered = df[(df["Position"].isin(position_filter)) &
                     (df["Price"] >= price_range[0]) &
                     (df["Price"] <= price_range[1]) &
                     (df["Ownership %"] >= ownership_range[0]) &
                     (df["Ownership %"] <= ownership_range[1])]

    # Display top 10 players by default
    st.subheader("📊 Top 10 Players Based on Predictions")
    if "Predicted Future Points" in df_filtered.columns:
        top_10_players = df_filtered.sort_values(by="Predicted Future Points", ascending=False).head(10)
        st.dataframe(top_10_players)
    else:
        st.warning("Column 'Predicted Future Points' not found in dataset.")

    # Button to show top 10 differentials
    if st.button("Show Top 10 Differentials"):
        st.subheader("🔍 Top 10 Differentials (<5% Ownership)")
        df_differentials = df_filtered[df_filtered["Ownership %"] < 5].sort_values(by="Differential Score", ascending=False).head(10)
        st.dataframe(df_differentials)

    # Button to show predicted points visualization
    if st.button("Show Predicted Points Visualization"):
        st.subheader("📈 Predicted Points of Top 10 Players")
        if "Predicted Future Points" in df_filtered.columns:
            top_10_predicted = df_filtered.sort_values(by="Predicted Future Points", ascending=False).head(10)

            # Visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=top_10_predicted["Name"], y=top_10_predicted["Predicted Future Points"], palette="viridis", ax=ax)
            plt.xticks(rotation=45)
            plt.xlabel("Player")
            plt.ylabel("Predicted Future Points")
            plt.title("Top 10 Predicted Players")
            st.pyplot(fig)
        else:
            st.warning("Column 'Predicted Future Points' not found in dataset.")
else:
    st.warning("No data available. Please check your internet connection.")