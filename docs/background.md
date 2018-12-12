---
title: Background
layout: default
---

## Motivation

Spotify is a popular music streaming platform that allows users to curate their own playlists. 
Music recommender systems help users find songs based on what they might like. 
We wanted to help Grace curate a Spotify playlist of her favorite songs using the classification techniques we had learned in class.
Grace struggles to classify with words what makes a song fit into her vibe. "I'd jam to Madison Thompson's 'Lonely Together,' but definitely not to Avicii's."
When words fail, data science might help. 
We decided to solve the problem of finding the best classification model for Grace's playlist, 
and then use that model to find her more songs for her playlist.
The Spotify API allows us to download playlists with feature data for each song in the playlist.
We asked Grace to label a set of songs as included in her playlist and another set of songs as songs she would not want to listen to.
We then constructed and analyzed classification models based on this data.

## Playlists
### Grace's Favorites Playlist
<iframe src="https://open.spotify.com/embed/user/gzgracez2/playlist/6Jpt5r9KD8FEUDioBFV0r0" width="300" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>
### Grace's Not-So-Favorites Playlist
<iframe src="https://open.spotify.com/embed/user/gzgracez2/playlist/4B3qR5p6PD8nXXeq4C0Gz7" width="300" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>

## Literature Review 
[Current Challenges and Visions in Music Recommender Systems Research](https://arxiv.org/pdf/1710.03208.pdf)
+ Biggest current issues in MRS: cold start, automatic playlist generation, and evaluation.
+ State-of-the-art techniques in playlist extension include collaborative filtering and Markov chain models
	- limitations include ordering of songs within playlists and incorporating situational characteristics that affect listeners
+ Future work includes incorporating personality, current emotional state, political situation, and cultural situation into music recommendations.

[An Analysis of Approaches Taken in the ACM RecSys Challenge 2018](https://arxiv.org/pdf/1810.01520.pdf)
+ In 2018, Spotify sponsored a challenge involving addressing the automatic music playlist continuation problem
+ Most accurate classifiers involved:
	- Larger training sets produced better-performing models
		- If training sets were subset, using a random subset of the playlist rather than the sequentially first songs in a playlist was most accurate in training a model
	- Excluding "title" as meta-data for the playlist produced better models across the board
	- Solutions using the descriptors from the Spotify API were more efficient

[TrailMix: An Ensemble Recommender System for Playlist Curation and Continuation](people.tamu.edu/~zhaoxing623/publications/XZ_TrailMix.pdf)
+ RecSys feature TrailMix compared 3 recommender models:
	- song clustering purely based on title
	- decorated neural collaborative filtering
	- decision tree
+ The title model performed very poorly compared to the other two models
	- The authors acknowledge that using analysis beyond the literal words, ex. incorporating NLP methods, could help
+ An ensemble of all three models performed best

>here is a quote



Here is *emph* and **bold**.

Here is some inline math $\alpha = \frac{\beta}{\gamma}$ and, of-course, E rules:

$$ G_{\mu\nu} + \Lambda g_{\mu\nu}  = 8 \pi T_{\mu\nu} . $$
