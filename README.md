# StacksFX - The emotion-based music player

### [Blog Post](https://mixstersite.wordpress.com/2018/08/20/stacksfx-at-hackiiit-my-1-hackathon/) | [Slides](https://bit.ly/vemotions)

A revolutionary music player that recommends songs by detecting your emotions through the camera and constantly learning from your actions through machine learning with a constant feedback loop.

The winning project in the hackathon **HackIIITD** of Indraprastha Institute of Information Technology, Delhi's techfest **ESYA**


## Introduction 

- Emotions are powerful feelings that grow from different circumstances and situations.
- StacksFX harnesses those emotions and detects your mood in real-time. 
- And from that, it recommends music to you. Getting better each time, by learning from you.

## Methodology behind StacksFX 

StacksFX uses the camera installed in the laptops to click about 16 frames of the user in one go. These images are sent to the server and get classified into different moods with the use of Computer Vision. On the basis of the mood classified, songs are recommended to the user.

With every recommendation, there is a feedback loop in place; that learns preferences song by song, constantly building a custom profile of the user. With each song played, the recommendation gets better and better. Inching closer to that perfect song for our users.  

## Under the hood 

- StacksFX uses WebRTC to click images
- Cascade classifier in OpenCV for emotion detection 
- Spotify API for song recommendations
- Django for the backend 

## License
The source code is under MIT license.
