from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import re

def fetch_youtube_transcript(video_id, preferred_language='en'):
    """
    Fetches an auto-generated transcript of a YouTube video and returns its translation
    to the specified preferred language.

    Args:
        video_id (str): The ID of the YouTube video.
        preferred_language (str, optional): The language code to translate the transcript to. Defaults to 'en'.

    Returns:
        str: The translated transcript text, or an error message.
    """
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.list(video_id)

        generated_transcript = None
        for transcript in transcript_list:
            if transcript.is_generated:
                generated_transcript = transcript
                break

        if generated_transcript and generated_transcript.is_translatable:
            try:
                translated_transcript = generated_transcript.translate(preferred_language)
                fetched_transcript = translated_transcript.fetch()
                return "\n".join([snippet.text for snippet in fetched_transcript])
            except Exception as e:
                return f"Error during translation: {e}"
        elif generated_transcript:
            return "Auto-generated transcript is not translatable."
        else:
            return "No auto-generated transcript found for this video."

    except TranscriptsDisabled:
        return "Transcripts are disabled for this video."
    except NoTranscriptFound:
        return "No transcript found for this video."
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def extract_video_id(url):
    patterns = [
        r"(?:https?:\/\/)?(?:www\.)?(?:m\.)?(?:youtu\.be\/|youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=))([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None