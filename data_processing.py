import pandas as pd

input_data = 'C:/Dev/ELTE_AI/mat_mod/data/l4d2_player_stats_final.parquet'

df = pd.read_parquet(input_data)

df.rename(columns={'Playtime_(Hours)': 'Playtime'}, inplace=True)

cols = df.columns

df.to_parquet('C:/Dev/ELTE_AI/mat_mod/data/l4d2_player_stats_final.parquet')

print(df.info())


