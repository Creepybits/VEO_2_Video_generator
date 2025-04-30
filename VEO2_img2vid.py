import time
import os
import json
import urllib.request
import cv2
import base64
import io
import numpy as np
from PIL import Image
from google import genai
from google.genai import types
from google.genai.types import GenerateVideosConfig

# Only run this block for Gemini Developer API
client = genai.Client(api_key='YOUR_API_KEY')

client = genai.Client()

veo_model_name = "veo-2.0-generate-001"
prompt_text = "A 3d stylized cartoon character,  bright blue fur, large expressive ears, rainbow colored hair,  wearing oversized headphones in pastel colors, happily and playfully dancing,  vibrant colors,  impressionistic background of colorful paint splatters and confetti in various shades, hyperrealistic fur details,  soft lighting,  close-up perspective, fun and energetic, detailed, high resolution, cute, adorable, digital art. Dynamic motions."
image_path = r"PATH_TO_IMAGE.png"


image_part = None # Initialize image part
try:
    print(f"Loading and encoding image from: {image_path}")
    with Image.open(image_path) as img:
        # Convert image to RGB (common format, drops alpha)
        # You might need 'RGBA' if alpha is relevant, but RGB is standard for many models
        img = img.convert("RGB")

        # Save the PIL Image to a BytesIO object in PNG format
        # PNG is lossless and commonly supported for inline data
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        image_bytes = buffered.getvalue()

        # Encode the bytes to Base64
        image_base64_str = base64.b64encode(image_bytes).decode('utf-8')

        # Construct the image part dictionary as expected by the API
        image_part = {
            "inlineData": {
                "mimeType": "image/png", # Specify the MIME type
                "data": image_base64_str # The Base64 encoded data
            }
        }
        print("Image successfully loaded and encoded to Base64.")

except FileNotFoundError:
    print(f"Error: Image file not found at {image_path}. Proceeding without image.")
    image_part = None # Ensure image_part is None if file not found
except Exception as e:
    print(f"Error processing and encoding image {image_path}: {e}. Proceeding without image.")
    image_part = None # Ensure image_part is None if processing fails

contents_payload_parts = []

# Add the text part(s)
# If you have a system prompt and user instructions, you would combine them here
combined_prompt_text = prompt_text # For this script, the prompt is the combined text

contents_payload_parts.append({"text": combined_prompt_text})

if image_part is not None:
    contents_payload_parts.append(image_part)

    contents_payload = [{
    "parts": contents_payload_parts,
    "role": "user" # Explicitly set role as 'user' for the input turn
}]

if not contents_payload_parts:
     print("Error: No content parts were created (text prompt and image were missing or failed).")
     exit()


# --- Construct the VEO Config payload ---
# Using a dictionary, which the library should handle
veo_config = {
    # "person_generation": Not allowed in img2vid,
    "aspect_ratio": "16:9",
    "number_of_videos": "1",    # Map string to API value

}
print(f"VEO Config: {veo_config}")


# --- Create operation ---
operation = None
try:
    print(f"Initiating VEO video generation for prompt: '{prompt_text}' with model '{veo_model_name}'...")
    if client is None:
         raise RuntimeError("Gemini client was not initialized.")
    if not hasattr(client, 'models') or not hasattr(client.models, 'generate_videos'):
         raise AttributeError(f"Client object or its 'models' attribute does not have 'generate_videos'. Library version: {genai.__version__ if hasattr(genai, '__version__') else 'unknown'}")


    # Pass the prompt text directly, and the config dictionary
    # The API endpoint for VEO generate_videos expects prompt and config directly, not a 'contents' payload like generate_content
    operation = client.models.generate_videos(
        model=veo_model_name,
        prompt=prompt_text, # Pass the prompt text directly
        config=veo_config, # Pass the dictionary config
    )
    print(f"VEO operation initiated. Operation name: {operation.name}")

except AttributeError as e:
    print(f"Error initiating VEO generation: {e}. This likely means your installed 'google-generativeai' library version ({genai.__version__ if hasattr(genai, '__version__') else 'unknown'}) does not support VEO generation methods on the Client object.")
    print("Try upgrading the library in your environment: `python_embeded\python.exe -m pip install --upgrade google-generativeai`")
    exit()
except Exception as e:
    print(f"Error initiating VEO generation API call: {e}")
    error_message = str(e)
    if hasattr(e, 'response') and e.response:
         if hasattr(e.response, 'text'):
              try:
                   error_json = json.loads(e.response.text)
                   if 'error' in error_json and 'message' in error_json['error']:
                        api_error_message = error_json['error']['message']
                        print(f"API Error Message from response: {api_error_message}")
                        error_message = f"API Error: {api_error_message}"
                   elif hasattr(e.response, 'status_code') and hasattr(e.response, 'reason'):
                        error_message = f"HTTP Error: {e.response.status_code} {e.response.reason}"
              except (json.JSONDecodeError, AttributeError):
                   if hasattr(e.response, 'status_code') and hasattr(e.response, 'reason'):
                        error_message = f"HTTP Error: {e.response.status_code} {e.response.reason}"
    print(f"Error initiating VEO generation: {error_message}")
    exit()


# --- Poll for Operation Completion ---
start_time = time.time()
polling_interval_seconds = 15
polling_timeout_minutes = 10
timeout_seconds = polling_timeout_minutes * 60
print(f"Polling for VEO operation completion (up to {polling_timeout_minutes} minutes)...")

try:
    if operation is None:
         raise RuntimeError("VEO operation was not successfully initiated.")

    # Store the initial operation name for consistent retrieval
    initial_operation_name = operation.name if hasattr(operation, 'name') else None
    if not initial_operation_name:
         raise RuntimeError("Initial operation object has no name.")


    while not operation.done:
        if (time.time() - start_time) > timeout_seconds:
            print(f"Error: VEO operation timed out after {polling_timeout_minutes} minutes.")
            exit()

        print(f"Operation '{operation.name if hasattr(operation, 'name') else 'unknown'}' still processing... Waiting {polling_interval_seconds} seconds.")
        time.sleep(polling_interval_seconds)

        if client is None or not hasattr(client, 'operations') or not hasattr(client.operations, 'get'):
             raise AttributeError(f"Client object or its 'operations' attribute does not have 'get'. Library version: {genai.__version__ if hasattr(genai, '__version__') else 'unknown'}")

        # --- Fetch the updated operation status and check its type ---
        try:
             # Fetch using the initial operation name in case the 'operation' variable gets corrupted
             updated_operation = client.operations.get(initial_operation_name)

             # --- CHECK THE TYPE IMMEDIATELY ---
             # Check if the returned object looks like a valid operation object
             # At minimum, check for .done and .name attributes.
             if not hasattr(updated_operation, 'done') or not hasattr(updated_operation, 'name'):
                  print(f"Error during polling: client.operations.get() returned unexpected type ({type(updated_operation)}) instead of an operation object.")
                  # Print the unexpected value for debugging
                  print(f"DEBUG: Unexpected value returned by client.operations.get(): {updated_operation}")
                  # If the unexpected value is a string, it might be an error message from the API/library
                  if isinstance(updated_operation, str):
                       raise RuntimeError(f"Polling stopped: client.operations.get() returned a string error message: {updated_operation}")
                  else:
                       raise TypeError("client.operations.get() returned unexpected non-operation type during polling.")


             # If the type seems valid, update the operation variable
             operation = updated_operation

        except (TypeError, RuntimeError) as e:
             # Catch the specific TypeErrors/RuntimeError raised above for unexpected return type
             print(f"Polling stopped due to unexpected return from client.operations.get(): {e}")
             exit()
        except Exception as e:
             # Catch other potential errors during the get() call (e.g., network issues, API errors)
             print(f"Error fetching operation status '{initial_operation_name}': {e}")
             # Re-raise the exception to stop polling
             exit()


    print("VEO operation completed.")

except AttributeError as e:
    print(f"Error during VEO polling: {e}. This likely means your installed 'google-generativeai' library version ({genai.__version__ if hasattr(genai, '__version__') else 'unknown'}) does not support getting operations via this method.")
    print("Try upgrading the library in your environment: `python_embeded\python.exe -m pip install --upgrade google-generativeai`")
    exit()
except Exception as e:
    # Catch any errors not caught by the specific TypeError/RuntimeError
    print(f"Error during VEO polling for operation {operation.name if hasattr(operation, 'name') else 'unknown'}: {e}")
    print(f"An unexpected error occurred during polling: {e}")
    exit()


# --- Process Response and Save Video(s) ---
generated_video_paths = []
save_dir = os.path.dirname(os.path.abspath(__file__)) # Save in the same directory as the script

if not save_dir:
    print("Error: Save directory could not be determined.")
    exit()

try:
    os.makedirs(save_dir, exist_ok=True)
    print(f"Ensured save directory exists: {save_dir}")
except Exception as e:
    print(f"Error creating save directory '{save_dir}': {e}")
    exit()


print("Processing VEO operation response...")


# Try to access generated_videos list using both attribute and key access, based on type check
generated_videos_list = None
if isinstance(operation.response, dict) and 'generated_videos' in operation.response and isinstance(operation.response['generated_videos'], list):
    generated_videos_list = operation.response['generated_videos'] # Access as dict key
elif hasattr(operation.response, 'generated_videos') and isinstance(operation.response.generated_videos, list):
    generated_videos_list = operation.response.generated_videos # Access as attribute

# Process the list if we successfully found it
if generated_videos_list:
    print(f"Processing {len(generated_videos_list)} generated video item(s) found in response.")
    for n, generated_video_item in enumerate(generated_videos_list):

        video_object_for_download = None
        # Try accessing the 'video' part using attribute or key access
        if hasattr(generated_video_item, 'video'): # Try dot notation (if item is an object)
             video_object_for_download = generated_video_item.video
             print(f"DEBUG: Accessed video object for item {n} using dot notation.")
        elif isinstance(generated_video_item, dict) and 'video' in generated_video_item: # Try key notation (if item is a dictionary)
             video_object_for_download = generated_video_item['video']
             print(f"DEBUG: Accessed video object for item {n} using key notation.")
        else:
             print(f"Warning: Generated video item {n} has no 'video' attribute or key.")
             continue # Skip this item if video data is missing


        # Check if the obtained video_object is valid for download
        if video_object_for_download:
            # Construct the filename
            safe_prompt = "".join([c for c in prompt_text if c.isalnum() or c in (' ', '-', '_')]).replace(' ', '_')[:50].strip('_') # Use prompt text from the script
            if not safe_prompt: safe_prompt = "veo_video"
            timestamp_str = str(int(time.time()))
            filename = f"{safe_prompt}_{timestamp_str}_{n}.mp4"
            full_save_path = os.path.join(save_dir, filename) # Use save_dir from the script

            print(f"Attempting to download video {n} to {full_save_path} using client.files.download()...")
            try:
                # Use client.files.download() to download the remote video object
                # We previously confirmed client.files.download(file=video_object) returns bytes.

                print(f"DEBUG: Calling client.files.download(file=video_object_for_download)...")
                if client is None or not hasattr(client, 'files') or not hasattr(client.files, 'download'):
                     raise AttributeError(f"Client object or its 'files' attribute does not have 'download'. Library version: {genai.__version__ if hasattr(genai, '__version__') else 'unknown'}")

                # Call download - assume it returns bytes
                video_content_bytes = client.files.download(file=video_object_for_download)
                print(f"DEBUG: client.files.download returned type: {type(video_content_bytes)}")
                print(f"DEBUG: Length of returned content: {len(video_content_bytes) if hasattr(video_content_bytes, '__len__') else 'N/A'}")

                # Check if returned content is bytes and non-empty
                if isinstance(video_content_bytes, bytes) and video_content_bytes:
                    # Write the bytes to the local file
                    print(f"DEBUG: Writing {len(video_content_bytes)} bytes to local file {full_save_path}...")
                    with open(full_save_path, 'wb') as f:
                        f.write(video_content_bytes)
                    print(f"Successfully saved video {n} to {full_save_path}.")
                    generated_video_paths.append(full_save_path)
                else:
                     # This handles cases where download might return None, an empty bytes object, or something else unexpected
                     print(f"Error: client.files.download() returned unexpected content type ({type(video_content_bytes)}) or empty content for video {n}.")
                     print(f"DEBUG: Start of content: {video_content_bytes[:100]}")
                     print(f"DEBUG: End of content: {video_content_bytes[-100:]}")


            except AttributeError as e:
                 error_message = f"Error downloading video {n}: {e}. This likely means your installed 'google-generativeai' library version ({genai.__version__ if hasattr(genai, '__version__') else 'unknown'}) does not support file downloads via this method."
                 print(error_message)
                 print("Check your library version and installation.")
                 exit()
            except Exception as e:
                print(f"Error downloading and saving video {n} to {full_save_path}: {e}")
                exit()

        else:
             print(f"Warning: Generated video item {n} has no downloadable video object after checking attributes/keys.")


    if not generated_video_paths:
         print("Warning: Operation completed, but no video files were successfully saved.")
         if hasattr(operation, 'response') and operation.response and hasattr(operation.response, 'prompt_feedback') and operation.response.prompt_feedback:
              if hasattr(operation.response.prompt_feedback, 'block_reason') and operation.response.prompt_feedback.block_reason and hasattr(operation.response.prompt_feedback.block_reason, 'name') and operation.response.prompt_feedback.block_reason.name != 'UNASSIGNED':
                   block_reason = operation.response.prompt_feedback.block_reason.name
                   print(f"Prompt Block Reason in Response: {block_reason}")

         print("Error: No videos were successfully generated or saved after processing response.")


else:
     print("Error: VEO operation completed, but the 'generated_videos' list was not found in the response structure.")
     print(f"Full operation response object: {operation.response}")


if generated_video_paths:
    print("\nSuccessfully generated and saved video(s) at:")
    for p in generated_video_paths:
        print(p)
else:
    print("\nNo videos were successfully generated or saved.")
