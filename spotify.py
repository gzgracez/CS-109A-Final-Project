import sys
import spotipy
import spotipy.util as util
import json

scope = 'user-library-read'
LIMIT = 50

if len(sys.argv) > 1:
    username = sys.argv[1]
else:
    print("Usage: %s username" % (sys.argv[0],))
    sys.exit()
# os.environ['DEBUSSY']
# username = 'gzgracez2'

token = util.prompt_for_user_token(username, scope)

if not token: 
    print("Can't get token for", username)
sp = spotipy.Spotify(auth=token)
results = sp.user_playlist_tracks(
	'UroAv2poQoWSvUOfch8wmg', 
	playlist_id='6Jpt5r9KD8FEUDioBFV0r0',
	limit=LIMIT,
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
print(len(tracks_features))

with open('spotify-1.json', mode='w') as f:
    f.write(json.dumps(tracks_features, indent=2))
     
# import spotipy
# from spotipy.oauth2 import SpotifyClientCredentials

# client_credentials_manager = SpotifyClientCredentials()
# sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# playlists = sp.current_user_playlists('spotify')
# # sp.audio_features([])
# print([i['name'] for i in playlists['items']])
# # while playlists:
# #     for i, playlist in enumerate(playlists['items']):
# #         print("%4d %s %s" % (i + 1 + playlists['offset'], playlist['uri'],  playlist['name']))
# #     if playlists['next']:
# #         playlists = sp.next(playlists)
# #     else:
# #         playlists = None