import sys
import spotipy
import spotipy.util as util
import json

scope = 'user-library-read'

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
results = sp.user_playlist_tracks('UroAv2poQoWSvUOfch8wmg', playlist_id='6JIq8OVNcyuYk4vDDRqflZ')
tracks = [i['track']['id'] for i in results['items']]
tracks_features = sp.audio_features(tracks)


with open('spotify.csv', mode='w') as f:
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