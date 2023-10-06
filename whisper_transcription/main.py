import whisper
import time
import hashlib
import pickle
import os

# Load cache from disk if it exists
cache_file_path = 'transcription_cache.pkl'
if os.path.exists(cache_file_path):
    with open(cache_file_path, 'rb') as f:
        transcription_cache = pickle.load(f)


def get_cache_key(file_path, model_name):
    """Generate a unique key for the cache based on file path and model name."""
    combined = file_path + model_name
    return hashlib.sha256(combined.encode()).hexdigest()


def get_transcription(file_path, model_name, verbose=False, language="en"):
    """Get transcription either from cache or by running the model."""
    key = get_cache_key(file_path, model_name)

    # Check if the result is in the cache
    if key in transcription_cache:
        print("Cache hit")
        return transcription_cache[key]

    # If not, load the model and get the result
    print("Cache miss")
    model = whisper.load_model(model_name)
    result = model.transcribe(file_path, verbose=verbose, language=language)

    # Store the result in the cache for future use
    transcription_cache[key] = result

    # Store the cache on disk as well
    with open(cache_file_path, 'wb') as f:
        pickle.dump(transcription_cache, f)

    return result


# Example usage
start_time = time.time()

# Get the transcription, using the cache
result = get_transcription('../recordings/flow_like_water.mp3', "medium.en", verbose=True, language="en")
line_by_line = result["text"].split(".")
for line in line_by_line:
    print(line)
end_time = time.time()
print(f"Time elapsed: {end_time - start_time} seconds")
