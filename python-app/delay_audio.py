import moviepy.editor as mp

def delay_audio_in_video(input_video_path, output_video_path, delay_seconds):
    # Load the video file
    video = mp.VideoFileClip(input_video_path)
    
    # Extract the audio from the video
    audio = video.audio
    
    # Create a new audio clip with the delay
    delayed_audio = mp.CompositeAudioClip([audio.set_start(delay_seconds)])
    
    # Set the new audio to the video
    new_video = video.set_audio(delayed_audio)
    
    # Write the result to a new file
    new_video.write_videofile(output_video_path, codec='libx264', audio_codec='aac')

# Example usage
input_video_path = "recording1.mov"
output_video_path = "delayed_r1.mp4"
delay_seconds = 0.9

delay_audio_in_video(input_video_path, output_video_path, delay_seconds)
