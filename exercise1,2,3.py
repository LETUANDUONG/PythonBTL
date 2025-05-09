import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import warnings
import time
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
warnings.filterwarnings('ignore')

# List of URLs for different statistical tables
tables = [
    ('https://fbref.com/en/comps/9/2024-2025/stats/2024-2025-Premier-League-Stats#stats_standard', 'stats_standard'),
    ('https://fbref.com/en/comps/9/2024-2025/keepers/2024-2025-Premier-League-Stats#stats_keeper', 'stats_keeper'),
    ('https://fbref.com/en/comps/9/2024-2025/shooting/2024-2025-Premier-League-Stats#stats_shooting', 'stats_shooting'),
    ('https://fbref.com/en/comps/9/2024-2025/passing/2024-2025-Premier-League-Stats#stats_passing', 'stats_passing'),
    ('https://fbref.com/en/comps/9/2024-2025/gca/2024-2025-Premier-League-Stats#stats_gca', 'stats_gca'),
    ('https://fbref.com/en/comps/9/2024-2025/defense/2024-2025-Premier-League-Stats#stats_defense', 'stats_defense'),
    ('https://fbref.com/en/comps/9/2024-2025/possession/2024-2025-Premier-League-Stats#stats_possession', 'stats_possession'),
    ('https://fbref.com/en/comps/9/2024-2025/misc/2024-2025-Premier-League-Stats#stats_misc', 'stats_misc')
]

# Required statistics mapping to fbref column names
required_stats = {
    'Nation': 'Nation',
    'Team': 'Squad',
    'Position': 'Pos',
    'Age': 'Age',
    'Matches Played': 'MP',
    'Starts': 'Starts',
    'Minutes': 'Min',
    'Goals': 'Gls',
    'Assists': 'Ast',
    'Yellow Cards': 'CrdY',
    'Red Cards': 'CrdR',
    'xG': 'xG',
    'xAG': 'xAG',
    'PrgC': 'PrgC',
    'PrgP': 'PrgP',
    'PrgR': 'PrgR',
    'Gls/90': 'Gls_2',
    'Ast/90': 'Ast_2',
    'xG/90': 'xG_2',
    'xAG/90': 'xAG_2',
    'GA90': 'GA90',
    'Save%': 'Save%',
    'CS%': 'CS%',
    'PK Save%': 'Save%_2',
    'SoT%': 'SoT%',
    'SoT/90': 'SoT/90',
    'G/Sh': 'G/Sh',
    'Dist': 'Dist',
    'Cmp': 'Cmp',
    'Cmp%': 'Cmp%',
    'TotDist': 'TotDist',
    'Short Cmp%': 'Cmp%_2',
    'Medium Cmp%': 'Cmp%_3',
    'Long Cmp%': 'Cmp%_4',
    'KP': 'KP',
    '1/3': '1/3',
    'PPA': 'PPA',
    'CrsPA': 'CrsPA',
    'PrgP (Passing)': 'PrgP',
    'SCA': 'SCA',
    'SCA90': 'SCA90',
    'GCA': 'GCA',
    'GCA90': 'GCA90',
    'Tkl': 'Tkl',
    'TklW': 'TklW',
    'Att (Challenges)': 'Att',
    'Lost (Challenges)': 'Lost',
    'Blocks': 'Blocks',
    'Sh (Blocks)': 'Sh',
    'Pass (Blocks)': 'Pass',
    'Int': 'Int',
    'Touches': 'Touches',
    'Def Pen': 'Def Pen',
    'Def 3rd': 'Def 3rd',
    'Mid 3rd': 'Mid 3rd',
    'Att 3rd': 'Att 3rd',
    'Att Pen': 'Att Pen',
    'Att (Take-Ons)': 'Att',
    'Succ% (Take-Ons)': 'Succ%',
    'Tkld%': 'Tkld%',
    'Carries': 'Carries',
    'ProDist': 'PrgDist',
    'ProgC (Carries)': 'PrgC',
    '1/3 (Carries)': '1/3',
    'CPA': 'CPA',
    'Mis': 'Mis',
    'Dis': 'Dis',
    'Rec': 'Rec',
    'PrgR (Receiving)': 'PrgR',
    'Fls': 'Fls',
    'Fld': 'Fld',
    'Off': 'Off',
    'Crs': 'Crs',
    'Recov': 'Recov',
    'Won (Aerial)': 'Won',
    'Lost (Aerial)': 'Lost',
    'Won% (Aerial)': 'Won%'
}

def clean_player_name(name):
    """Clean player name by removing special characters and extra spaces."""
    return re.sub(r'[^\w\s]', '', name.strip()) if isinstance(name, str) else ''

def extract_first_name(name):
    """Extract the first name from the player name."""
    parts = name.split()
    return parts[0] if parts else name

def scrape_table(url, table_id, retries=7):
    """Scrape the specified table from the given URL with retries."""
    print(f"Scraping {url} for table ID {table_id}")
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--ignore-certificate-errors')
    
    for attempt in range(retries):
        driver = webdriver.Chrome(options=options)
        try:
            driver.get(url)
            wait = WebDriverWait(driver, 15)
            table = wait.until(EC.presence_of_element_located((By.ID, table_id)))
            table_html = table.get_attribute('outerHTML')
            
            soup = BeautifulSoup(table_html, 'html.parser')
            
            rows = []
            for i in range(0, 1000):
                row = soup.find('tr', {'data-row': str(i)})
                if row:
                    rows.append(row)
            
            data = []
            for row in rows:
                cells = row.find_all(['td', 'th'])
                row_data = [cell.get_text(strip=True) if cell.get_text(strip=True) else "N/a" for cell in cells]
                if row_data and len(row_data) > 1:
                    data.append(row_data)
            
            header_row = soup.find('tr', class_='thead')
            if not header_row:
                header_row = soup.find_all('tr')[0]
            headers = [th.get_text(strip=True) if th.get_text(strip=True) else f"Col_{i}" 
                       for i, th in enumerate(header_row.find_all(['th', 'td']))]
            
            if data:
                max_cols = max(len(row) for row in data)
                if len(headers) > max_cols:
                    headers = headers[:max_cols]
                elif len(headers) < max_cols:
                    headers = headers + [f'Col_{i}' for i in range(len(headers), max_cols)]
                
                df = pd.DataFrame(data, columns=headers)
                
                col_count = {}
                new_columns = []
                for col in df.columns:
                    if col in col_count:
                        col_count[col] += 1
                        new_columns.append(f"{col}_{col_count[col]}")
                    else:
                        col_count[col] = 1
                        new_columns.append(col)
                df.columns = new_columns
                
                df = df[df['Player'].notna() & (df['Player'] != 'Player') & (df['Player'] != '')].copy()
                
                duplicates = df[df['Player'].duplicated(keep=False)][['Player', 'Squad', 'Pos']].drop_duplicates().values.tolist()
                if duplicates:
                    print(f"Duplicate players in {table_id}: {df[df['Player'].duplicated()]['Player'].tolist()}")
                    print(f"Duplicate details: {duplicates}")
                    df = df.groupby('Player').first().reset_index()
                
                print(f"Columns in scraped table {table_id}: {df.columns.tolist()}")
                df.to_csv(f'debug_{table_id}.csv', index=False)
                return df
            else:
                print(f"No data found in table at {url}")
                return None
        
        except TimeoutException:
            print(f"Timeout waiting for table at {url} (attempt {attempt + 1}/{retries})")
        except NoSuchElementException:
            print(f"Table with ID {table_id} not found at {url} (attempt {attempt + 1}/{retries})")
        except Exception as e:
            print(f"Error scraping {url}: {e} (attempt {attempt + 1}/{retries})")
        finally:
            driver.quit()
        time.sleep(5)
    return None

def compute_top_bottom_players(df):
    """Compute top 3 and bottom 3 players for each numeric statistic, including zero scores, and save to top_3.txt."""
    print("Computing top and bottom 3 players for each statistic...")
    
    gk_columns = ['GA90', 'Save%', 'CS%', 'PK Save%']
    
    output_lines = []
    
    numeric_columns = []
    for col in df.columns:
        if col in ['Player', 'Team', 'Position', 'Nation', 'First_Name']:
            continue
        df[col] = pd.to_numeric(df[col].replace('N/a', pd.NA), errors='coerce')
        if df[col].dtype in ['int64', 'float64']:
            numeric_columns.append(col)
    
    for col in numeric_columns:
        output_lines.append(f"\nStatistic: {col}")
        output_lines.append("-" * (len(f"Statistic: {col}") + 2))
        
        if col in gk_columns:
            temp_df = df[df['Position'].str.contains('GK', na=False)].copy()
        else:
            temp_df = df[~df['Position'].str.contains('GK', na=False)].copy()
        
        temp_df = temp_df.dropna(subset=[col])
        
        if temp_df.empty:
            output_lines.append("Top 3: No data available")
            output_lines.append("Bottom 3: No data available")
            continue
        
        top_3 = temp_df[['Player', 'Team', col]].sort_values(by=col, ascending=False).head(3)
        output_lines.append("Top 3:")
        for _, row in top_3.iterrows():
            output_lines.append(f"  {row['Player']} ({row['Team']}): {row[col]:.2f}")
        
        bottom_3 = temp_df[['Player', 'Team', col]].sort_values(by=col, ascending=True).head(3)
        output_lines.append("Bottom 3:")
        for _, row in bottom_3.iterrows():
            output_lines.append(f"  {row['Player']} ({row['Team']}): {row[col]:.2f}")
    
    with open('top_3.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines))
    print("Top and bottom 3 players saved to 'top_3.txt'")

def compute_statistics_and_histograms(df):
    """Compute median, mean, std for each statistic, generate histograms, and identify top teams."""
    print("Computing statistics and generating histograms...")
    
    gk_columns = ['GA90', 'Save%', 'CS%', 'PK Save%']
    attacking_stats = ['Goals', 'Assists', 'xG']
    defensive_stats = ['Tkl', 'Int', 'Recov']
    teams = sorted(df['Team'].unique())
    
    # Identify numeric columns
    numeric_columns = []
    for col in df.columns:
        if col in ['Player', 'Team', 'Position', 'Nation', 'First_Name']:
            continue
        df[col] = pd.to_numeric(df[col].replace('N/a', pd.NA), errors='coerce')
        if df[col].dtype in ['int64', 'float64']:
            numeric_columns.append(col)
    
    # Prepare results DataFrame
    columns = ['Team']
    for col in numeric_columns:
        columns.extend([f'Median of {col}', f'Mean of {col}', f'Std of {col}'])
    results_df = pd.DataFrame(columns=columns)
    
    # Compute stats for 'all' and each team
    rows = []
    # All players
    all_row = {'Team': 'all'}
    for col in numeric_columns:
        if col in gk_columns:
            temp_df = df[df['Position'].str.contains('GK', na=False)].copy()
        else:
            temp_df = df[~df['Position'].str.contains('GK', na=False)].copy()
        
        temp_df = temp_df.dropna(subset=[col])
        if temp_df.empty:
            all_row[f'Median of {col}'] = np.nan
            all_row[f'Mean of {col}'] = np.nan
            all_row[f'Std of {col}'] = np.nan
        else:
            all_row[f'Median of {col}'] = temp_df[col].median()
            all_row[f'Mean of {col}'] = temp_df[col].mean()
            all_row[f'Std of {col}'] = temp_df[col].std()
    rows.append(all_row)
    
    # Per team
    for team in teams:
        team_row = {'Team': team}
        team_df = df[df['Team'] == team]
        for col in numeric_columns:
            if col in gk_columns:
                temp_df = team_df[team_df['Position'].str.contains('GK', na=False)].copy()
            else:
                temp_df = team_df[~team_df['Position'].str.contains('GK', na=False)].copy()
            
            temp_df = temp_df.dropna(subset=[col])
            if temp_df.empty:
                team_row[f'Median of {col}'] = np.nan
                team_row[f'Mean of {col}'] = np.nan
                team_row[f'Std of {col}'] = np.nan
            else:
                team_row[f'Median of {col}'] = temp_df[col].median()
                team_row[f'Mean of {col}'] = temp_df[col].mean()
                team_row[f'Std of {col}'] = temp_df[col].std() if len(temp_df) > 1 else 0
        rows.append(team_row)
    
    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(rows)
    results_df.to_csv('results2.csv', index=False)
    print("Statistics saved to 'results2.csv'")
    
    # Generate histograms
    os.makedirs('histograms', exist_ok=True)
    
    # All players
    plt.figure(figsize=(15, 10))
    for i, stat in enumerate(attacking_stats + defensive_stats, 1):
        if stat not in numeric_columns:
            print(f"Skipping histogram for {stat}: not found in data")
            continue
        plt.subplot(2, 3, i)
        temp_df = df.dropna(subset=[stat]) if stat not in gk_columns else df[df['Position'].str.contains('GK', na=False)].dropna(subset=[stat])
        plt.hist(temp_df[stat], bins=20, edgecolor='black')
        plt.title(f'{stat} (All Players)')
        plt.xlabel(stat)
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('histograms/hist_all_players.png')
    plt.close()
    
    # Per team
    for team in teams:
        plt.figure(figsize=(15, 10))
        team_df = df[df['Team'] == team]
        for i, stat in enumerate(attacking_stats + defensive_stats, 1):
            if stat not in numeric_columns:
                continue
            plt.subplot(2, 3, i)
            temp_df = team_df.dropna(subset=[stat]) if stat not in gk_columns else team_df[team_df['Position'].str.contains('GK', na=False)].dropna(subset=[stat])
            if not temp_df.empty:
                plt.hist(temp_df[stat], bins=20, edgecolor='black')
                plt.title(f'{stat} ({team})')
                plt.xlabel(stat)
                plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(f'histograms/hist_{team.replace(" ", "_")}.png')
        plt.close()
    
    print("Histograms saved in 'histograms' folder")
    
    # Compute top teams for each statistic
    top_teams = {}
    for col in numeric_columns:
        if col in gk_columns:
            temp_df = df[df['Position'].str.contains('GK', na=False)].copy()
        else:
            temp_df = df[~df['Position'].str.contains('GK', na=False)].copy()
        
        temp_df = temp_df.dropna(subset=[col])
        if temp_df.empty:
            continue
        
        team_stats = temp_df.groupby('Team')[col].mean().reset_index()
        if not team_stats.empty:
            top_team = team_stats.loc[team_stats[col].idxmax()]
            top_teams[col] = (top_team['Team'], top_team[col])
    
    print("\nTeams with highest average for each statistic:")
    for stat, (team, value) in top_teams.items():
        print(f"{stat}: {team} ({value:.2f})")
    
    # Analyze best-performing team
    key_stats = ['Goals', 'Assists', 'xG', 'Tkl', 'Int', 'Recov']
    team_scores = {}
    for team in teams:
        score = 0
        team_df = df[df['Team'] == team]
        for stat in key_stats:
            if stat in numeric_columns:
                temp_df = team_df.dropna(subset=[stat]) if stat not in gk_columns else team_df[team_df['Position'].str.contains('GK', na=False)].dropna(subset=[stat])
                if not temp_df.empty:
                    mean_val = temp_df[stat].mean()
                    if stat in top_teams and top_teams[stat][0] == team:
                        score += 2  # Bonus for leading in a stat
                    score += mean_val / df[stat].mean() if df[stat].mean() != 0 else 0  # Normalize by league average
        team_scores[team] = score
    
    best_team = max(team_scores, key=team_scores.get)
    print(f"\nBest-performing team: {best_team} (Score: {team_scores[best_team]:.2f})")
    print("Based on high averages in key attacking (Goals, Assists, xG) and defensive (Tkl, Int, Recov) stats.")

def perform_clustering_and_pca(df):
    """Apply K-means clustering, determine optimal clusters, apply PCA, and plot 2D clusters."""
    print("Performing K-means clustering and PCA visualization...")
    
    gk_columns = ['GA90', 'Save%', 'CS%', 'PK Save%']
    
    # Identify numeric columns
    numeric_columns = []
    for col in df.columns:
        if col in ['Player', 'Team', 'Position', 'Nation', 'First_Name']:
            continue
        df[col] = pd.to_numeric(df[col].replace('N/a', pd.NA), errors='coerce')
        if df[col].dtype in ['int64', 'float64']:
            numeric_columns.append(col)
    
    # Prepare data for clustering
    data_for_clustering = df[numeric_columns].copy()
    
    # Drop columns that are entirely NaN
    initial_cols = data_for_clustering.columns.tolist()
    data_for_clustering = data_for_clustering.dropna(axis=1, how='all')
    dropped_cols = set(initial_cols) - set(data_for_clustering.columns)
    if dropped_cols:
        print(f"Dropped columns with all NaN values: {dropped_cols}")
    
    # Check if any numeric columns remain
    if data_for_clustering.empty:
        raise ValueError("No numeric columns available for clustering after dropping NaN columns. Check data scraping and 'results.csv'.")
    
    # Impute remaining missing values with column means
    data_for_clustering = data_for_clustering.fillna(data_for_clustering.mean())
    
    # Verify no NaNs remain
    if data_for_clustering.isna().any().any():
        nan_cols = data_for_clustering.columns[data_for_clustering.isna().any()].tolist()
        raise ValueError(f"Data still contains NaN values after imputation in columns: {nan_cols}. Check 'results.csv' for data issues.")
    
    print(f"Columns used for clustering: {data_for_clustering.columns.tolist()}")
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_for_clustering)
    
    # Determine optimal number of clusters using Elbow Method and Silhouette Score
    wcss = []
    silhouette_scores = []
    k_range = range(2, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)
        labels = kmeans.labels_
        silhouette_scores.append(silhouette_score(scaled_data, labels))
    
    # Plot Elbow Method
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, wcss, 'bo-')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.savefig('elbow_plot.png')
    plt.close()
    
    # Plot Silhouette Scores
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, silhouette_scores, 'bo-')
    plt.title('Silhouette Score vs. Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.savefig('silhouette_plot.png')
    plt.close()
    
    # Choose optimal k (based on elbow and silhouette score)
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters chosen: {optimal_k}")
    print("Reason: Highest silhouette score, indicating well-separated clusters. Elbow method plot also considered.")
    
    # Apply K-means with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    # Add cluster labels to DataFrame
    df['Cluster'] = cluster_labels
    
    # Save clustering results
    df[['Player', 'Team', 'Position', 'Cluster']].to_csv('clusters.csv', index=False)
    print("Clustering results saved to 'clusters.csv'")
    
    # Analyze clusters
    print("\nCluster Analysis:")
    for cluster in range(optimal_k):
        cluster_df = df[df['Cluster'] == cluster]
        print(f"\nCluster {cluster} (Size: {len(cluster_df)} players):")
        print(f"Average Goals: {cluster_df['Goals'].mean():.2f}")
        print(f"Average Assists: {cluster_df['Assists'].mean():.2f}")
        print(f"Average Tkl: {cluster_df['Tkl'].mean():.2f}")
        print(f"Average Recov: {cluster_df['Recov'].mean():.2f}")
        print(f"Position Distribution:\n{cluster_df['Position'].value_counts()}")
        print(f"Sample Players:\n{cluster_df[['Player', 'Team', 'Position']].head(3)}")
    
    # Apply PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    explained_variance = pca.explained_variance_ratio_.sum()
    print(f"\nPCA: Explained variance ratio (2 components): {explained_variance:.2f}")
    
    # Plot 2D clusters
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Cluster')
    plt.title('2D PCA Clustering of Premier League Players')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    
    # Label a few representative players
    for cluster in range(optimal_k):
        cluster_points = pca_data[cluster_labels == cluster]
        cluster_df = df[cluster_labels == cluster]
        # Select a representative player (e.g., highest Goals in cluster)
        if not cluster_df.empty:
            top_player = cluster_df.sort_values(by='Goals', ascending=False).iloc[0]
            idx = cluster_df.index[cluster_df['Player'] == top_player['Player']].tolist()[0]
            idx = df.index.get_loc(idx)
            plt.annotate(top_player['Player'], (pca_data[idx, 0], pca_data[idx, 1]), fontsize=8)
    
    plt.savefig('clusters_2d.png')
    plt.close()
    print("2D PCA cluster plot saved to 'clusters_2d.png'")

def main():
    print("Tables list before validation:")
    for i, item in enumerate(tables):
        print(f"Index {i}: {item}")
    
    print("\nValidating tables list...")
    for i, item in enumerate(tables):
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(f"Invalid table entry at index {i}: {item}. Expected (url, table_id) tuple.")
    print("Tables list validated successfully.")

    standard_url, standard_table_id = tables[0]
    df_main = scrape_table(standard_url, standard_table_id)
    if df_main is None:
        print("Failed to scrape standard stats. Exiting.")
        return

    if 'Player' not in df_main.columns:
        print("No 'Player' column found in standard stats table. Exiting.")
        return

    df_main['Player'] = df_main['Player'].apply(clean_player_name)
    df_main['Min'] = pd.to_numeric(df_main['Min'].str.replace(',', ''), errors='coerce').fillna(0)
    df_main = df_main[df_main['Min'] > 90].copy()
    df_main['First_Name'] = df_main['Player'].apply(extract_first_name)

    result_df = pd.DataFrame({'Player': df_main['Player'], 'First_Name': df_main['First_Name']})

    unmapped_columns = []
    for display_name, fbref_col in required_stats.items():
        if fbref_col in df_main.columns:
            result_df[display_name] = df_main[fbref_col].fillna('N/a')
            print(f"Mapped {display_name} to {fbref_col} from standard stats")
        else:
            result_df[display_name] = 'N/a'
            unmapped_columns.append((display_name, fbref_col, 'stats_standard'))
            print(f"Column {fbref_col} not found in standard stats for {display_name}")

    for url, table_id in tables[1:]:
        df = scrape_table(url, table_id)
        if df is None or 'Player' not in df.columns:
            print(f"Skipping {url} due to missing table or 'Player' column")
            continue
        df['Player'] = df['Player'].apply(clean_player_name)
        for display_name, fbref_col in required_stats.items():
            if fbref_col in df.columns:
                temp_df = df[['Player', fbref_col]].drop_duplicates('Player')
                merged_data = result_df[['Player']].merge(
                    temp_df,
                    on='Player',
                    how='left'
                )[fbref_col].fillna('N/a')
                if result_df[display_name].eq('N/a').all():
                    result_df[display_name] = merged_data
                    print(f"Mapped {display_name} to {fbref_col} from {table_id}")
                else:
                    print(f"Skipping {display_name} as it was already mapped")
            else:
                if result_df[display_name].eq('N/a').all():
                    unmapped_columns.append((display_name, fbref_col, table_id))
                print(f"Column {fbref_col} not found in {table_id} for {display_name}")

    if unmapped_columns:
        print("\nUnmapped columns:")
        for display_name, fbref_col, table_id in unmapped_columns:
            print(f"{display_name} ({fbref_col}) not found in {table_id}")

    result_df.to_csv('results.csv', index=False)
    print("Data saved to 'results.csv'")

    compute_top_bottom_players(result_df)
    compute_statistics_and_histograms(result_df)
    perform_clustering_and_pca(result_df)

    na_summary = result_df.eq('N/a').sum()
    print("\nSummary of 'N/a' columns in results.csv:")
    for col, count in na_summary.items():
        if count > 0:
            print(f"{col}: {count} 'N/a' values")

if __name__ == "__main__":
    main()