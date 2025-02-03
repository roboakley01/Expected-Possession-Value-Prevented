# -*- coding: utf-8 -*-
"""
Module to get the 1v1 situations from the data that was created and saved after training the model

1v1 situations here are defined as events where, at the time of the event:
    1. the closest defender is closer than 4 meters
    AND
    2. the second closest defender is further than 8 meters
    
This code is written in blocks delimited by #%%

@author: Rob Oakley
"""

from skillcorner.client import SkillcornerClient
username = 'username'
password = 'password'

client = SkillcornerClient(username=username, password=password)

import pandas as pd
import numpy as np

import Skillcorner_IO_pub as skio

#import data that was saved after model training
data = pd.read_csv('data_w_pred_epv.csv')

#%%
#This block labels events as 1v1 situations if they fit the criteria

#get list of matches
matches_list = data['sk_match_id'].unique().tolist()
#instantiate columns:
data['is_1v1'] = False #binary flag for whether the event is classified as a 1v1
data['def_id'] = np.nan #skillcorner player id of the defender if it is a 1v1

#for every match, decide whether each event in the match is a 1v1. If so label it as such and label the defender
for mid in matches_list:
    tracking = skio.get_tracking_data(mid)
    df = data[data['sk_match_id'] == mid] #get sequences from the selected match
    frames_list = df['frame'].unique().tolist() #get list of frames for that match to go through
    players = skio.get_player_data(match_id = mid) #get player data for the match
    match = client.get_match(match_id= mid) #get match data
    
    for f in frames_list:
        frame = tracking[tracking['frame'] == f] #get frame of tracking data
        if not frame.empty: #check if the frame is empty
            datum = df[df['frame'] == f] #get the event for that frame
            #get id number of the attacker performing the pass or shot at that frame
            player = datum['sk_player_id'].iloc[0]
            att_player = players.loc[players['id'] == player] #get player data of that player
            if not att_player.empty: #check if we have a player assigned for this event
                att_team_id = att_player['team_id'].iloc[0]
                defenders = players.loc[players['team_id'] != att_team_id, 'id'].to_list() #get list of defenders
                #get attacking direction
                att_dir = skio.attacking_direction(frame, match)
                x = df.loc[df['frame'] == f, 'location_x'].values[0]
                y = df.loc[df['frame'] == f, 'location_y'].values[0]
                #flip coordinates so the event data aligns with the tracking data for that event
                x = x * att_dir
                y = y * att_dir
                #calculate distances and get three closest defenders to the attacker
                f_dist = skio.calc_player_distances(frame, (x,y))
                closest = skio.three_closest_def(f_dist, defenders)
        
                #label 1v1 if appropriate
                if closest['dist'].iloc[0] < 4 and closest['dist'].iloc[1] > 8:
            
                    data.loc[(data['sk_match_id'] == mid) & (data['frame'] == f), 'is_1v1'] = True
                    data.loc[(data['sk_match_id'] == mid) & (data['frame'] == f), 'def_id'] = closest['player_id'].iloc[0]
    print(mid, 'done')
#%%
#save the fully labeled data
data.to_csv('fully_labeled_data.csv', index = False)