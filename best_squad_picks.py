import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Constants
POSITION_LIMITS = {
    "Goalkeeper": 2,
    "Defender": 5,
    "Midfielder": 5,
    "Forward": 3
}

def fetch_data():
    # Fetch player data from the API
    player_response = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
    fixture_response = requests.get("https://fantasy.premierleague.com/api/fixtures/")

    if player_response.status_code == 200 and fixture_response.status_code == 200:
        data = player_response.json()
        fixtures = fixture_response.json()

        # Extract player and team data
        players = data["elements"]
        teams = {team["id"]: team["name"] for team in data["teams"]}
        positions = {1: "Goalkeeper", 2: "Defender", 3: "Midfielder", 4: "Forward"}

        # Exclude Managers (element_type == 5)
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
        raise Exception("Failed to fetch FPL data. Check your internet connection!")
    
def train_model(df):
    # Features and target
    features = ["Form", "Expected Goals", "Expected Assists", "Total Points", "Ownership %", "Price"]
    target = "Future Points"

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model MAE: {mae}")  # Log the model's performance

    # Add predictions to the DataFrame
    df["Predicted Future Points"] = model.predict(df[features])
    return df

def select_best_11(selected_players):
    # Separate players by position
    gks = selected_players[selected_players["Position"] == "Goalkeeper"]
    defs = selected_players[selected_players["Position"] == "Defender"]
    mids = selected_players[selected_players["Position"] == "Midfielder"]
    fwds = selected_players[selected_players["Position"] == "Forward"]

    # Sort players by predicted points
    gks = gks.sort_values(by="Predicted Future Points", ascending=False)
    defs = defs.sort_values(by="Predicted Future Points", ascending=False)
    mids = mids.sort_values(by="Predicted Future Points", ascending=False)
    fwds = fwds.sort_values(by="Predicted Future Points", ascending=False)

    # Ensure at least 1 goalkeeper is selected
    if len(gks) == 0:
        raise Exception("No goalkeeper selected. Please select at least 1 goalkeeper.")
        return pd.DataFrame()

    # Select the top goalkeeper
    best_11 = gks.head(1)

    # Select at least 3 defenders, 3 midfielders, and 1 forward
    min_defs = 3
    min_mids = 3
    min_fwds = 1

    # Ensure there are enough players in each position
    if len(defs) < min_defs or len(mids) < min_mids or len(fwds) < min_fwds:
        raise Exception("Not enough players in one or more positions to meet the minimum requirements.")
        return pd.DataFrame()

    # Select the required minimum players from each position
    selected_defs = defs.head(min_defs)
    selected_mids = mids.head(min_mids)
    selected_fwds = fwds.head(min_fwds)

    # Combine the selected players
    best_11 = pd.concat([best_11, selected_defs, selected_mids, selected_fwds])

    # Calculate remaining spots to fill (total 11 players)
    remaining_spots = 11 - len(best_11)

    # Select the remaining players from the highest predicted points
    remaining_players = pd.concat([defs, mids, fwds])
    remaining_players = remaining_players[~remaining_players.index.isin(best_11.index)]  # Exclude already selected players
    remaining_players = remaining_players.sort_values(by="Predicted Future Points", ascending=False).head(remaining_spots)

    # Combine to form the best 11
    best_11 = pd.concat([best_11, remaining_players])

    # Ensure total players is 11
    if len(best_11) != 11:
        raise Exception("Failed to form a team of 11 players.")
        return pd.DataFrame()

    return best_11

def select_best_11_with_captain(selected_players):
    # Select the best 11
    best_11 = select_best_11(selected_players)
    
    if best_11.empty:
        return pd.DataFrame(), None

    # Select the captain (player with the highest predicted points)
    captain = best_11.loc[best_11["Predicted Future Points"].idxmax()]
    
    return best_11, captain

def determine_formation(best_11):
    # Count the number of players in each position
    num_defs = len(best_11[best_11["Position"] == "Defender"])
    num_mids = len(best_11[best_11["Position"] == "Midfielder"])
    num_fwds = len(best_11[best_11["Position"] == "Forward"])

    # Determine the formation based on the counts
    if num_defs == 4 and num_mids == 4 and num_fwds == 2:
        return "4-4-2"
    elif num_defs == 3 and num_mids == 5 and num_fwds == 2:
        return "3-5-2"
    elif num_defs == 5 and num_mids == 3 and num_fwds == 2:
        return "5-3-2"
    elif num_defs == 4 and num_mids == 3 and num_fwds == 3:
        return "4-3-3"
    elif num_defs == 3 and num_mids == 4 and num_fwds == 3:
        return "3-4-3"
    elif num_defs == 4 and num_mids == 5 and num_fwds == 1:
        return "4-5-1"
    elif num_defs == 5 and num_mids == 4 and num_fwds == 1:
        return "5-4-1"
    else:
        return f"Custom ({num_defs}-{num_mids}-{num_fwds})"
