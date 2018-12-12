import sys
import spotipy
import spotipy.util as util
import json
import pandas as pd
from math import ceil

scope = 'user-library-read'
LIMIT = 50
PLAYLIST_LEN = 33

def get_track_features_offset(playlist_id, offset):
    results = sp.user_playlist_tracks(
        'UroAv2poQoWSvUOfch8wmg', 
        playlist_id=playlist_id,
        limit=LIMIT,
        offset=offset,
    )
    track_infos = []
    for i in results['items']:
        track_infos.append({
            'id': i['track']['id'],
            'name': i['track']['name'],
            'popularity': i['track']['popularity'],
            'artist': i['track']['artists'][0]['name'] if len(i['track']['artists']) > 0 else None,
        })
    track_ids = [i['id'] for i in track_infos]

    try:
        tracks_features = sp.audio_features(track_ids)
    except:
        return []
    for idx, track in enumerate(tracks_features):
        track['name'] = track_infos[idx]['name']
        track['popularity'] = track_infos[idx]['popularity']
        track['artist'] = track_infos[idx]['artist']
    return tracks_features

def get_track_features(playlist_id, num_iters):
    track_features = []
    for i in range(num_iters):
        track_features.extend(
            get_track_features_offset(playlist_id, i * LIMIT)
        )
    return track_features


# Setup
if len(sys.argv) > 1:
    username = sys.argv[1]
else:
    print("Usage: %s username" % (sys.argv[0],))
    sys.exit()
token = util.prompt_for_user_token(username, scope)
if not token: 
    print("Can't get token for", username)

sp = spotipy.Spotify(auth=token)

# Get track features
n_playlist = ceil(PLAYLIST_LEN / LIMIT)

tracks_features = get_track_features('4ZEqmKFVTW5R2wSCDnqWlV', n_playlist)

with open('spotify-test.csv', mode='w') as f:
    df = pd.read_json(json.dumps(tracks_features))
    f.write(df.to_csv())

# with open('spotify.json', mode='w') as f:
#     f.write(json.dumps(tracks_features, indent=2))

# with open('spotify-0.json', mode='w') as f:
#     f.write(json.dumps(tracks_features0, indent=2))

# with open('spotify-1.json', mode='w') as f:
#     f.write(json.dumps(tracks_features1, indent=2))