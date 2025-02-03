# -*- coding: utf-8 -*-
"""
Module for working with skillcorner tracking data synced to Wyscout event data.


Data courtesy of Skillcorner and SRC | FTBL provided as part of the research competition
as part of the 2025 American Soccer Insights Summit held at Rice University.

The conference website is here: https://americansoccerinsights.com/

@author: Rob Oakley
"""

import pandas as pd
import numpy as np
import requests
import json

#if you have SKillcorner API access, you can replace this with your credentials
username = 'username'
password = 'password'

from skillcorner.client import SkillcornerClient
client = SkillcornerClient(username=username, password=password)

#Tracking data for match_id is stored locally here:
#{match_id}_tracking = f'{match_id}_tracking_raw.jsonl'

#event data is stored locally here:
#events = wyscout_events_updated.parquet

#load in your desired EPV grid
epv = np.loadtxt('EPV_grid.csv', delimiter = ',')

def get_tracking_data(match_id):
    """
    read tracking data from local skillcorner raw tracking for a specific match
    return as dataframe
    """
    extract = pd.read_json(f'{match_id}_tracking_raw.jsonl')

    step1 = pd.json_normalize(extract.to_dict('records'))

    tracking = pd.json_normalize(step1.to_dict('records'), 'player_data', ['frame','timestamp','period','possession.group','possession.player_id'])    
    
    return tracking

def get_player_data(match_id):
    """
    Parameters
    ----------
    match_id : skillcorner match_id.

    Returns
    -------
    player data frame
    """
    player_info_request=requests.get(f'https://skillcorner.com/api/match/{match_id}/?lang=en',
                                 auth=(username,password), timeout = 120)

    raw_players=pd.json_normalize(json.loads(player_info_request.text),max_level=2)
    players=pd.json_normalize(raw_players.to_dict('records'),'players',['home_team.short_name','home_team.id',
                                                            'away_team.short_name','away_team.id','ball.trackable_object',
                                                                   'home_team_side','home_team_kit.jersey_color','away_team_kit.jersey_color'])
    return players


def wy_to_sk_coords(data, field_dim = (106.0,68.0) ):
    """
    convert wyscout coords to meters to align with skillcorner coordinates
    -------

    """
    data_mod = data.copy()
    data_mod['location_x'] = ( data_mod['location_x'] - 50.0 ) * field_dim[0]/100
    data_mod['location_y'] = -1 * ( data_mod['location_y'] - 50.0) * field_dim[1]/100
    
    data_mod['pass_endlocation_x'] = ( data_mod['pass_endlocation_x'] - 50.0 ) * field_dim[0]/100
    data_mod['pass_endlocation_y'] = -1 * ( data_mod['pass_endlocation_y'] - 50.0) * field_dim[1]/100
    
    return data_mod


def get_attackers(data, player):
    """
    input: skillcorner player dataframe for a match and skillcorner player id for the attacking player
    result: list of attacking player ids and list of defending player ids
    """
    att_team_id = data.loc[data['id'] == player, 'team_id'].values[0]
    attackers = data.loc[data['team_id'] == att_team_id, 'id'].to_list()
    defenders = data.loc[data['team_id'] != att_team_id, 'id'].to_list()
    
    return attackers, defenders
    
def attacking_direction(data, match):
    """
    input: frame number and match id
    output: attacking direction is 1 for L to R; -1 for R to L
    """
    period = data['period'].iloc[0]
    
    #get match data
    #match = client.get_match(match_id=match_id)
    
    att_dir = match['home_team_side'][0]
    period = data['period'].values[0]
    poss_team = data['possession.group'].values[0]
    
    info_tuple = (att_dir, period, poss_team)
    
    if info_tuple == ('left_to_right', 1.0, 'home team') or info_tuple == ('left_to_right', 2.0, 'away team') or info_tuple == ('right_to_left', 1.0, 'away team') or info_tuple == ('right_to_left', 2.0, 'home team'):
        direction = 1.0
    else:
        direction = -1.0
        
    return direction

def get_event_sequences(match_id, events, EPV, field_dimen = (106., 68.)):
    """
    Parameters
    ----------
    match_id : skillcorner match_id
    events : dataframe of wyscout event data
    EPV : EPV grid

    Returns
    -------
    dataframe with locations and EPV added for pass sequences

    """
    
    #get events from match
    match = events.loc[events['sk_match_id'] == match_id]
    #change coordinates to skillcorner coords
    match = wy_to_sk_coords(match)
    #reset index for the match to be in order of frame in the tracking data
    match = match.sort_values(by = 'frame').reset_index()
    #get passes and shots
    data = match.loc[match['type_primary'].isin(['pass', 'shot'])]
    data = data[['sk_match_id', 'frame', 'wy_event_id', 'type_primary', 'pass_accurate', 'sk_team_id', 'sk_player_id', 'location_x', 'location_y', 'pass_endlocation_x', 'pass_endlocation_y', 'shot_xg']]
    
    #calculate the EPV added by each action
    #extract bins from EPV grid
    ny,nx = EPV.shape
    dx = field_dimen[0]/float(nx)
    dy = field_dimen[1]/float(ny)
    
    #get start and end points as series for each successful pass
    pass_ix0 = ((data.loc[(data['type_primary'] == 'pass') & (data['pass_accurate']),'location_x'] + field_dimen[0] / 2.0 - 0.0001) / dx).astype(int)
    pass_iy0 = ((data.loc[(data['type_primary'] == 'pass') & (data['pass_accurate']),'location_y'] + field_dimen[1] / 2.0 - 0.0001) / dy).astype(int)
    
    ix1 = ((data.loc[(data['type_primary'] == 'pass') & (data['pass_accurate']), 'pass_endlocation_x'] + field_dimen[0] / 2.0 - 0.0001) / dx).astype(int)
    iy1 = ((data.loc[(data['type_primary'] == 'pass') & (data['pass_accurate']), 'pass_endlocation_y'] + field_dimen[1] / 2.0 - 0.0001) / dy).astype(int)
    
    #get shot locations and their xG values as series objects
    shot_ix0 = ((data.loc[data['type_primary'] == 'shot','location_x'] + field_dimen[0] / 2.0 - 0.0001) / dx).astype(int)
    shot_iy0 = ((data.loc[data['type_primary'] == 'shot','location_y'] + field_dimen[1] / 2.0 - 0.0001) / dy).astype(int)
    
    shot_xg = data.loc[data['type_primary'] == 'shot', 'shot_xg']
    
    #get unsuccessful pass locations as series objects
    bad_pass_ix0 = ((data.loc[(data['type_primary'] == 'pass') & (data['pass_accurate'] == False),'location_x'] + field_dimen[0] / 2.0 - 0.0001) / dx).astype(int)
    bad_pass_iy0 = ((data.loc[(data['type_primary'] == 'pass') & (data['pass_accurate'] == False),'location_y'] + field_dimen[1] / 2.0 - 0.0001) / dy).astype(int)
    
    bad_pass_ix1 = ((data.loc[(data['type_primary'] == 'pass') & (data['pass_accurate'] == False),'pass_endlocation_x'] + field_dimen[0] / 2.0 - 0.0001) / dx).astype(int)
    bad_pass_iy1 = ((data.loc[(data['type_primary'] == 'pass') & (data['pass_accurate'] == False),'pass_endlocation_y'] + field_dimen[1] / 2.0 - 0.0001) / dy).astype(int)
    
    #calculate EPV added for each action and add it as a column to the dataframe
    data.loc[(data['type_primary'] == 'pass') & (data['pass_accurate']), 'EPV_added'] = EPV[iy1, ix1] - EPV[pass_iy0, pass_ix0]
    
    data.loc[(data['type_primary'] == 'pass') & (data['pass_accurate'] == False), 'EPV_added'] = EPV[bad_pass_iy0, bad_pass_ix0] - EPV[bad_pass_iy1 * -1, bad_pass_ix1 * -1]

    data.loc[data['type_primary'] == 'shot', 'EPV_added'] = shot_xg - EPV[shot_iy0, shot_ix0]
    
    #now pull out the sequences:
    #get event_id of previous two events in the dataframe
    data['prev_ev_id'] = data.wy_event_id.shift(1)
    data.fillna({'prev_ev_id':0}, inplace = True)
    data['prev2_ev_id'] = data.wy_event_id.shift(2)
    data.fillna({'prev2_ev_id':0}, inplace = True)
    
    #get difference between event ids to check if they are in the same possession sequence
    data['id_diff_1'] = data['wy_event_id'] - data['prev_ev_id']
    data['id_diff_2'] = data['wy_event_id'] - data['prev2_ev_id']
    
    #set previous values if in same pass sequence
    data.loc[data['id_diff_1'] == 1,'EPV_2'] = data.EPV_added.shift(1)
    data.loc[data['id_diff_1'] == 1,'x_2'] = data.location_x.shift(1)
    data.loc[data['id_diff_1'] == 1,'y_2'] = data.location_y.shift(1)
    data.loc[data['id_diff_2'] == 2,'EPV_1'] = data.EPV_added.shift(2)
    data.loc[data['id_diff_2'] == 2,'x_1'] = data.location_x.shift(2)
    data.loc[data['id_diff_2'] == 2,'y_1'] = data.location_y.shift(2)
    
    return data[['sk_match_id', 'wy_event_id', 'sk_player_id', 'sk_team_id', 'frame', 'location_x', 'location_y', 'pass_endlocation_x', 'pass_endlocation_y', 'EPV_added', 'x_1', 'y_1', 'EPV_1', 'x_2', 'y_2', 'EPV_2']]

def calc_player_distances(data, location):
    """
    Parameters
    ----------
    data : frame of tracking data
    location : location in the form of (x,y)

    Returns
    -------
    dataframe with relative distances of defenders
    """
    player_ids = data['player_id'].unique()
    
    data['dx'] = np.nan
    data['dy'] = np.nan
    data['dist'] = np.nan
    
    for p in player_ids: # cycle through players individually
       # difference player positions in timestep dt to get unsmoothed estimate of velicity
       dx = data.loc[data['player_id'] == p, 'x'].values[0] - location[0]
       dy = data.loc[data['player_id'] == p, 'y'].values[0] - location[1]
       
       data.loc[data['player_id'] == p, 'dx'] = dx
       data.loc[data['player_id'] == p, 'dy'] = dy
       data.loc[data['player_id'] == p, 'dist'] = np.sqrt( dx**2 + dy**2 )
       
    return data

def three_closest_def(data, defenders):
    """
    takes in the output of distance function and list of defenders
    outputs the three closest defenders and their locations, distances, and velocities
    """
    #get defenders
    def_dist = data[data['player_id'].isin(defenders)]
    
    #get closest 3
    close_3 = def_dist.sort_values(by='dist').head(3)
    close_3 = close_3[['player_id', 'dx', 'dy', 'dist']]
    return close_3
    
