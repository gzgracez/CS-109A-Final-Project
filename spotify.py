import sys
import spotipy
import spotipy.util as util
import json
from math import ceil

scope = 'user-library-read'
LIMIT = 50
PLAYLIST_1_LEN = 163
PLAYLIST_0_LEN = 897

def get_track_features_offset(playlist_id, offset, in_playlist):
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

	tracks_features = sp.audio_features(track_ids)
	for idx, track in enumerate(tracks_features):
		track['name'] = track_infos[idx]['name']
		track['popularity'] = track_infos[idx]['popularity']
		track['artist'] = track_infos[idx]['artist']
		track['in_playlist'] = in_playlist
	return tracks_features

def get_track_features(playlist_id, num_iters, in_playlist):
	track_features = []
	for i in range(num_iters):
		track_features.extend(
			get_track_features_offset(playlist_id, i * LIMIT, in_playlist)
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
n_playlist0 = ceil(PLAYLIST_0_LEN / LIMIT)
n_playlist1 = ceil(PLAYLIST_1_LEN / LIMIT)

tracks_features0 = get_track_features('4B3qR5p6PD8nXXeq4C0Gz7', n_playlist0, 0)
tracks_features1 = get_track_features('6Jpt5r9KD8FEUDioBFV0r0', n_playlist1, 1)
tracks_features = tracks_features0 + tracks_features1

with open('spotify.json', mode='w') as f:
    f.write(json.dumps(tracks_features, indent=2))

# with open('spotify-0.json', mode='w') as f:
#     f.write(json.dumps(tracks_features0, indent=2))

# with open('spotify-1.json', mode='w') as f:
#     f.write(json.dumps(tracks_features1, indent=2))