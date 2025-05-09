import time
import os
import pandas as pd
import tracemalloc
import re
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium import webdriver
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from fuzzywuzzy import fuzz, process

def clean_name(name):
    """Standardize player names for better matching"""
    if not isinstance(name, str):
        return ""
    name = re.sub(r'[^a-zA-Z\s]', '', name)  # Remove special characters
    name = name.lower().strip()
    return name

def estimate_player_value(file_path):
    try:
        # Load data
        df = pd.read_csv(file_path)
        
        # Filter out players without transfer values
        df = df[df['Transfer values'].notna() & (df['Transfer values'] != '')]
        
        if df.empty:
            raise ValueError("No players with valid transfer values found in the dataset")
        
        # Clean minutes column if necessary
        if df['Playing Time: minutes'].dtype == object:
            df['Playing Time: minutes'] = df['Playing Time: minutes'].str.replace(',', '').astype(int)
        
        # Fix Transfer values format to int
        df['Transfer values'] = (
            df['Transfer values']
            .str.replace('€', '', regex=False)
            .str.replace('m', '', regex=False)
            .astype(float) * 1_000_000
        )
        df['Transfer values'] = df['Transfer values'].astype(int)

        # Select features
        features = [
            'Age',
            'Position',
            'Playing Time: minutes',
            'Performance: goals',
            'Performance: assists',
            'GCA: GCA',
            'Progression: PrgR',
            'Tackles: Tkl'
        ]

        X = df[features]
        y = df['Transfer values']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Preprocessing: OneHot for categorical 'Position'
        categorical_features = ['Position']
        numeric_features = [col for col in features if col not in categorical_features]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'  # Numeric features stay unchanged
        )
        
        # Model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Mean Absolute Error: {mae:,.0f} €")
        
        return pipeline

    except Exception as e:
        print(f"Error in estimate_player_value: {str(e)}")
        return None

def get_data():
    try:
        # Load the CSV file
        file_path = os.path.join('P1_RES', 'results.csv')
        df = pd.read_csv(file_path)

        # Clean the 'Playing Time: minutes' column
        df['Playing Time: minutes'] = df['Playing Time: minutes'].str.replace(',', '').astype(int)

        # Filter players with more than 900 minutes
        filtered_df = df[df['Playing Time: minutes'] > 900]

        return filtered_df
    except Exception as e:
        print(f"Error in get_data: {str(e)}")
        return pd.DataFrame()

def update_data(filtered_df):
    try:
        # Initialize Chrome
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')  # Run in background
        driver = webdriver.Chrome(options=options)
        url = 'https://www.footballtransfers.com/us/players/uk-premier-league/'
        driver.get(url)
        
        # Get all players from Transfermarkt
        transfermarkt_players = {}
        cnt = 0
        max_pages = 22  # Adjust as needed
        
        while cnt < max_pages:
            time.sleep(2)  # Respectful delay
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            table = soup.find('table', class_='table table-hover no-cursor table-striped leaguetable mvp-table similar-players-table mb-0')
            if not table:
                print("No table found.")
                break

            # Extract names and values
            for row in table.find_all('tr')[1:]:  # Skip header
                try:
                    name_tag = row.find('div', class_='text')
                    value_tag = row.find('span', class_='player-tag')
                    
                    if name_tag and value_tag:
                        name = clean_name(name_tag.find('a').get('title'))
                        value = value_tag.text.strip()
                        if name and value:
                            transfermarkt_players[name] = value
                except Exception as e:
                    continue

            print(f"Collected {len(transfermarkt_players)} players from Transfermarkt")

            # Pagination
            try:
                cnt += 1
                next_button = driver.find_element(By.CLASS_NAME, 'pagination_next_button')
                driver.execute_script("arguments[0].click();", next_button)
            except:
                print("Reached last page or pagination failed")
                break

        driver.quit()

        # Match FBref players to Transfermarkt data
        matched_data = {}
        unmatched_players = []
        
        for fbref_name in filtered_df['Name']:
            clean_fbref = clean_name(fbref_name)
            
            # Exact match first
            if clean_fbref in transfermarkt_players:
                matched_data[fbref_name] = transfermarkt_players[clean_fbref]
                continue
                
            # Fuzzy matching if no exact match
            best_match, score = process.extractOne(
                clean_fbref, 
                transfermarkt_players.keys(),
                scorer=fuzz.token_sort_ratio
            )
            
            if score > 85:  # Adjust threshold as needed
                matched_data[fbref_name] = transfermarkt_players[best_match]
            else:
                unmatched_players.append(fbref_name)
                matched_data[fbref_name] = 'UNMATCHED'

        # Log unmatched players
        if unmatched_players:
            print(f"\nCould not find transfer values for {len(unmatched_players)} players:")
            for player in unmatched_players[:10]:  # Show first 10 for brevity
                print(f"- {player}")
            if len(unmatched_players) > 10:
                print(f"...and {len(unmatched_players)-10} more")
                
        return matched_data

    except Exception as e:
        print(f"Error in update_data: {str(e)}")
        if 'driver' in locals():
            driver.quit()
        return {}

def save_result(filtered_df, player_dic):
    try:
        filtered_df['Transfer values'] = filtered_df['Name'].map(player_dic)
        
        # Save main data
        os.makedirs('P4_RES', exist_ok=True)
        main_path = os.path.join('P4_RES', 'MoreThan900mins.csv')
        filtered_df.to_csv(main_path, index=False, encoding='utf-8-sig')
        
        # Save unmatched players
        unmatched = filtered_df[filtered_df['Transfer values'] == 'UNMATCHED']
        if not unmatched.empty:
            unmatched_path = os.path.join('P4_RES', 'UnmatchedPlayers.csv')
            unmatched.to_csv(unmatched_path, index=False)
            print(f"\nSaved {len(unmatched)} unmatched players to {unmatched_path}")
        
        print(f"\nSaved main data to {main_path}")
        return True
    except Exception as e:
        print(f"Error in save_result: {str(e)}")
        return False

def Task_1():
    filtered_df = get_data()
    if filtered_df.empty:
        print("Error: No data loaded from FBref")
        return False
    
    player_dic = update_data(filtered_df)
    if not player_dic:
        print("Error: Failed to get transfer values")
        return False
    
    return save_result(filtered_df, player_dic)

def Task_2():
    try:
        file_path = os.path.join('P4_RES', 'MoreThan900mins.csv')
        if not os.path.exists(file_path):
            print("Error: Data file not found. Run Task_1 first.")
            return

        model = estimate_player_value(file_path)
        if model is None:
            return

        # Example prediction
        new_player = pd.DataFrame({
            'Age': [26],
            'Position': ['GK'],
            'Playing Time: minutes': [2250],
            'Performance: goals': [0],
            'Performance: assists': [0],
            'GCA: GCA': [0],
            'Progression: PrgR': [0],
            'Tackles: Tkl': [0]
        })

        predicted_value = model.predict(new_player)
        print(f"\nEstimated player value: €{predicted_value[0]:,.0f}")

    except Exception as e:
        print(f"Error in Task_2: {str(e)}")

def Problem_4():
    print("\n=== Starting Task 1 ===")
    if not Task_1():
        return
    
    print("\n=== Starting Task 2 ===")
    Task_2()

if __name__ == '__main__':
    tracemalloc.start()
    start_time = time.time()
    
    try:
        Problem_4()
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
    finally:
        # Performance metrics
        end_time = time.time()   
        print(f"\nExecution time: {end_time - start_time:.2f} seconds")
        
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage: {current / 1024:.2f} KB")
        print(f"Peak memory usage: {peak / 1024 ** 2:.2f} MB")
        tracemalloc.stop()