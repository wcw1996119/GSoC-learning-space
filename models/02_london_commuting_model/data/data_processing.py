import os
import pandas as pd
import geopandas as gpd
import numpy as np
import re
import json

base_dir = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(base_dir, 'raw')
processed_dir = os.path.join(base_dir, 'processed')

os.makedirs(processed_dir, exist_ok=True)

# ----------------------
# 1. OD Data
# ----------------------
od_raw = pd.read_csv(os.path.join(raw_dir, 'msoa_OD_travel2work.csv'))
london_od = od_raw[(od_raw['county_home'] == 'GREATER_LONDON_AUTHORITY') &
                   (od_raw['county_work'] == 'GREATER_LONDON_AUTHORITY')].copy()
london_od.to_csv(os.path.join(processed_dir, 'london_OD_travel2work.csv'), index=False)
print(f"OD pairs saved: {len(london_od)}")

# ----------------------
# 2. Boundaries
# ----------------------
gdf = gpd.read_file(os.path.join(raw_dir, 'msoa_boundaries_2021.geojson'))
london_msoa_codes = set(london_od['MSOA21CD_home']) | set(london_od['MSOA21CD_work'])
london_gdf = gdf[gdf['MSOA21CD'].isin(london_msoa_codes)].copy()
london_gdf.to_file(os.path.join(processed_dir, 'london_msoa_boundaries.geojson'), driver='GeoJSON')
print(f"London MSOAs saved: {len(london_gdf)}")

# ----------------------
# 3. Commute Mode (Census 2011)
# ----------------------
commute_raw = pd.read_csv(os.path.join(raw_dir, 'census2011_commute_mode_msoa.csv'))
commute_raw = commute_raw[commute_raw['Rural Urban'] == 'Total'].copy()

commute_raw = commute_raw.rename(columns={
    'geography code': 'MSOA11CD',
    'Method of Travel to Work: All categories: Method of travel to work; measures: Value': 'total',
    'Method of Travel to Work: Work mainly at or from home; measures: Value': 'wfh',
    'Method of Travel to Work: Underground, metro, light rail, tram; measures: Value': 'underground',
    'Method of Travel to Work: Train; measures: Value': 'train',
    'Method of Travel to Work: Bus, minibus or coach; measures: Value': 'bus',
    'Method of Travel to Work: Taxi; measures: Value': 'taxi',
    'Method of Travel to Work: Motorcycle, scooter or moped; measures: Value': 'motorcycle',
    'Method of Travel to Work: Driving a car or van; measures: Value': 'car',
    'Method of Travel to Work: Passenger in a car or van; measures: Value': 'car_passenger',
    'Method of Travel to Work: Bicycle; measures: Value': 'bicycle',
    'Method of Travel to Work: On foot; measures: Value': 'walk',
    'Method of Travel to Work: Other method of travel to work; measures: Value': 'other',
    'Method of Travel to Work: Not in employment; measures: Value': 'not_employed',
})

# Filter London MSOAs
london_codes_2021 = set(london_od['MSOA21CD_home']) | set(london_od['MSOA21CD_work'])
df_london = commute_raw[commute_raw['MSOA11CD'].isin(london_codes_2021)].copy()

# Active commuters
df_london['active_commuters'] = df_london['total'] - df_london['wfh'] - df_london['not_employed']

# Group into 3 modes
df_london['mode_car'] = df_london['car'] + df_london['car_passenger'] + df_london['motorcycle'] + df_london['taxi']
df_london['mode_pt'] = df_london['underground'] + df_london['train'] + df_london['bus']
df_london['mode_active'] = df_london['bicycle'] + df_london['walk']

# Proportions
df_london['prop_car'] = df_london['mode_car'] / df_london['active_commuters']
df_london['prop_pt'] = df_london['mode_pt'] / df_london['active_commuters']
df_london['prop_active'] = df_london['mode_active'] / df_london['active_commuters']

# Keep relevant columns
commute_out = df_london[['MSOA11CD', 'active_commuters', 'prop_car', 'prop_pt', 'prop_active']]
commute_out.to_csv(os.path.join(processed_dir, 'london_commute_mode_msoa.csv'), index=False)
print(f"Commute mode data saved: {len(commute_out)} MSOAs")

# ----------------------
# 4. TomTom Congestion
# ----------------------
import pandas as pd
import numpy as np
import geopandas as gpd

df = pd.read_csv(os.path.join(raw_dir, 'London_tomtom.csv'))
df['Time'] = pd.to_datetime(df['Time'])
df['hour'] = df['Time'].dt.hour
df['weekday'] = df['Time'].dt.weekday  # 0=Monday, 6=Sunday

# 只用工作日
weekday_df = df[df['weekday'] < 5].copy()

# 计算每个Borough每小时的平均拥堵比率
# congestion_ratio = free_flow_speed / actual_speed
# ratio > 1 意味着拥堵，ratio=1意味着畅通
weekday_df['congestion_ratio'] = weekday_df['Free flow speed [kmh]'] / weekday_df['Speed [kmh]']
hourly_congestion = weekday_df.groupby(['Region label', 'hour'])['congestion_ratio'].mean().reset_index()
hourly_congestion.columns = ['borough', 'hour', 'congestion_ratio']
peak = hourly_congestion[hourly_congestion['hour'] == 8].sort_values('congestion_ratio', ascending=False)
offpeak = hourly_congestion[hourly_congestion['hour'] == 12].sort_values('congestion_ratio', ascending=False)

# 保存hourly congestion ratio
output_path = 'd:/GIT/mesa Gsoc/GSoC-learning-space/models/02_london_commuting_model/data/processed/tomtom_hourly_congestion.csv'
hourly_congestion.to_csv(output_path, index=False)


gdf = gpd.read_file(os.path.join(processed_dir, 'london_msoa_boundaries.geojson'))
congestion = pd.read_csv(os.path.join(processed_dir, 'tomtom_hourly_congestion.csv'))

# Extract borough from MSOA name
gdf['borough_name'] = gdf['MSOA21NM'].apply(lambda x: re.sub(r'\s+\d+$', '', x).strip())

# Pivot congestion to wide format
congestion_wide = congestion.pivot(index='borough', columns='hour', values='congestion_ratio')
congestion_wide.columns = [f'hour_{h}' for h in congestion_wide.columns]
congestion_wide = congestion_wide.reset_index()

# Merge MSOA with congestion
msoa_congestion = gdf[['MSOA21CD', 'MSOA21NM', 'borough_name']].merge(
    congestion_wide, left_on='borough_name', right_on='borough', how='left'
)

# Fill missing boroughs with default
hour_cols = [f'hour_{h}' for h in range(24)]
for col in hour_cols:
    msoa_congestion[col] = msoa_congestion[col].fillna(1.2)

msoa_congestion[['MSOA21CD', 'MSOA21NM', 'borough_name'] + hour_cols].to_csv(
    os.path.join(processed_dir, 'msoa_hourly_congestion.csv'), index=False
)
print(f"TomTom congestion data saved: {len(msoa_congestion)} MSOAs")

# # ----------------------
# # 5. BRES Employment
# # ----------------------
# bres_raw = pd.read_csv(os.path.join(raw_dir, 'bres_msoa_2024.csv'), skiprows=8)
# bres_raw = bres_raw.dropna(axis=1, how='all')
# bres_raw = bres_raw.loc[:, ~bres_raw.columns.str.contains('^Unnamed')]
# bres_raw['MSOA21CD'] = bres_raw['2021 super output area - middle layer'].str.extract(r'(E\d+)')
# bres_raw = bres_raw.dropna(subset=['MSOA21CD'])

# # Replace * with 0
# industry_cols = [c for c in bres_raw.columns if c[0].isdigit()]
# for col in industry_cols:
#     bres_raw[col] = pd.to_numeric(bres_raw[col], errors='coerce').fillna(0)

# # Rename industry columns
# col_map = {
#     '1 : Agriculture, forestry & fishing (A)': 'agri',
#     '2 : Mining, quarrying & utilities (B,D and E)': 'mining',
#     '3 : Manufacturing (C)': 'mfg',
#     '4 : Construction (F)': 'construction',
#     '5 : Motor trades (Part G)': 'motor',
#     '6 : Wholesale (Part G)': 'wholesale',
#     '7 : Retail (Part G)': 'retail',
#     '8 : Transport & storage (inc postal) (H)': 'transport',
#     '9 : Accommodation & food services (I)': 'food',
#     '10 : Information & communication (J)': 'ict',
#     '11 : Financial & insurance (K)': 'finance',
#     '12 : Property (L)': 'property',
#     '13 : Professional, scientific & technical (M)': 'prof',
#     '14 : Business administration & support services (N)': 'admin',
#     '15 : Public administration & defence (O)': 'public_admin',
#     '16 : Education (P)': 'education',
#     '17 : Health (Q)': 'health',
#     '18 : Arts, entertainment, recreation & other services (R,S,T and U)': 'arts',
# }
# bres_raw = bres_raw.rename(columns=col_map)
# bres_cols = list(col_map.values())

# london_codes = set(london_od['MSOA21CD_home']) | set(london_od['MSOA21CD_work'])
# bres_london = bres_raw[bres_raw['MSOA21CD'].isin(london_codes)][['MSOA21CD'] + bres_cols].copy()
# bres_london['total_employment'] = bres_london[bres_cols].sum(axis=1)
# bres_london = bres_london.sort_values('total_employment', ascending=False).drop_duplicates(subset='MSOA21CD', keep='first')
# bres_london = bres_london.reset_index(drop=True)
# bres_london.to_csv(os.path.join(processed_dir, 'london_bres_msoa.csv'), index=False)
# print(f"BRES saved: {len(bres_london)} MSOAs")

# # ----------------------
# # 6. TS063 Occupation
# # ----------------------
# occ_raw = pd.read_csv(os.path.join(raw_dir, 'census2021_occupation_msoa.csv'))
# occ_raw = occ_raw.rename(columns={
#     'geography code': 'MSOA21CD',
#     'Occupation (current): Total: All usual residents aged 16 years and over in employment the week before the census': 'total',
#     'Occupation (current): 1. Managers, directors and senior officials': 'soc1',
#     'Occupation (current): 2. Professional occupations': 'soc2',
#     'Occupation (current): 3. Associate professional and technical occupations': 'soc3',
#     'Occupation (current): 4. Administrative and secretarial occupations': 'soc4',
#     'Occupation (current): 5. Skilled trades occupations': 'soc5',
#     'Occupation (current): 6. Caring, leisure and other service occupations': 'soc6',
#     'Occupation (current): 7. Sales and customer service occupations': 'soc7',
#     'Occupation (current): 8. Process, plant and machine operatives': 'soc8',
#     'Occupation (current): 9. Elementary occupations': 'soc9',
# })
# occ_cols = ['soc1','soc2','soc3','soc4','soc5','soc6','soc7','soc8','soc9']
# occ_london = occ_raw[occ_raw['MSOA21CD'].isin(london_codes)][['MSOA21CD', 'total'] + occ_cols].copy()
# for col in occ_cols:
#     occ_london[f'prop_{col}'] = occ_london[col] / occ_london['total'].replace(0, np.nan)
# occ_london = occ_london.fillna(0)
# occ_london.to_csv(os.path.join(processed_dir, 'london_occupation_msoa.csv'), index=False)
# print(f"Occupation data saved: {len(occ_london)} MSOAs")