from flask import Flask, request, jsonify
from langchain_core.globals import set_llm_cache
from langchain_redis import RedisSemanticCache
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.outputs import Generation
import os
import logging
import time
from typing import Optional, Any, List, Tuple, Union

# --- Configuration & Setup ---

# Setting up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# NOTE: Using a valid model name for reliable API calls
MODEL_NAME = "gpt-5-nano"
LLM_TEMPERATURE = 0 
# Define the expected LLM_STRING used as a metadata key in RedisSemanticCache
# This string must match what LangChain generates internally for the given model/config.
LLM_STRING = f"openai/{MODEL_NAME}/{LLM_TEMPERATURE}" 

# --- Global Initialization for LangChain Cache ---

def initialize_llm_cache(redis_url: str = "redis://localhost:6379", distance_threshold: float = 0.8) -> Optional[RedisSemanticCache]:
    """Initializes and sets the global LLM semantic cache."""
    # Retrieve the API key from environment variables
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        logging.error("OPENAI_API_KEY environment variable not set. Cannot initialize embeddings/LLM.")
        return None

    try:
        # 1. Initialize embeddings model
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)

        # 2. Initialize RedisSemanticCache
        semantic_cache = RedisSemanticCache(
            redis_url=redis_url,
            embeddings=embeddings,
            distance_threshold=distance_threshold
        )
        
        # 3. Set it as the global LLM cache (for /query endpoint)
        set_llm_cache(semantic_cache)
        logging.info(f"LangChain RedisSemanticCache initialized and set globally. LLM_STRING used: {LLM_STRING}")
        return semantic_cache
    except Exception as e:
        logging.error(f"Error initializing LangChain cache. Ensure Redis is running: {e}")
        return None

# Use the provided API key (for testing purposes, ideally use env vars)
OPENAI_API_KEY = "sk-proj-....." 
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY 

# Initialize cache and LLM globally
cache = initialize_llm_cache()

if cache:
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=LLM_TEMPERATURE)
else:
    llm = None 

# --- Flask Application Setup ---
app = Flask(__name__)

# Type hint for the error response tuple
ErrorResponse = Tuple[Any, int]
DataResponse = Tuple[Optional[dict], Optional[ErrorResponse]]

# Helper function for JSON request body validation and error handling
def _handle_json_error() -> DataResponse:
    """
    Parses JSON request body.
    Returns: A tuple (data, error_response), where error_response is None on success, 
             or a tuple (response_content, status_code) on failure.
    """
    try:
        data = request.get_json()
        if data is None:
             return None, (jsonify({"error": "Request body must be valid JSON."}), 400)
        return data, None
    except Exception:
        # Returns (response_content, status_code) on JSON parsing error
        return None, (jsonify({"error": "Invalid JSON format."}), 400)

def _check_cache_ready() -> Optional[ErrorResponse]:
    """Checks if the cache object was successfully initialized."""
    if not cache:
        return (jsonify({"error": "Cache is not initialized. Check Redis connection."}), 503)
    return None

# --- Health Check Endpoint ---
@app.route("/health", methods=["GET"])
def health():
    """Simple health check endpoint."""
    return "ok", 200

# --- LLM Query Endpoint (Uses Global Cache) ---

@app.route('/query', methods=['POST'])
def handle_query():
    """
    Endpoint to process a prompt using the LLM, automatically leveraging global cache.
    Expects: {"prompt": "Your question here"}
    """
    start_time = time.time() # Start timing

    if not llm:
        return jsonify({"error": "LLM initialization failed. Check API key."}), 503

    data, error_response = _handle_json_error()
    if error_response: return error_response
    
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "Missing 'prompt' in request body."}), 400

    logging.info(f"LLM Query: '{prompt}'")

    try:
        response = llm.invoke(prompt)
        
        duration = time.time() - start_time # End timing
        
        return jsonify({
            "prompt": prompt,
            "response": response.content,
            "model": llm.model_name,
            "llm_string": LLM_STRING,
            "cache_note": "Cache look-up/update was handled automatically by LangChain framework.",
            "response_time_seconds": round(duration, 4)
        })

    except Exception as e:
        duration = time.time() - start_time
        logging.error(f"Error during LLM invocation: {e}")
        return jsonify({
            "error": f"An error occurred while processing the request: {str(e)}",
            "response_time_seconds": round(duration, 4)
        }), 500

# --- Cache Management Endpoints ---

@app.route('/cache/lookup', methods=['POST'])
def cache_lookup():
    """
    Manually look up a prompt in the semantic cache.
    Expects: {"prompt": "What is the capital of France?"}
    """
    start_time = time.time() # Start timing

    error_response = _check_cache_ready()
    if error_response: return error_response

    data, error_response = _handle_json_error()
    if error_response: return error_response

    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "Missing 'prompt' in request body."}), 400

    logging.info(f"Cache Lookup for: '{prompt}' using LLM_STRING: {LLM_STRING}")

    try:
        # cache.lookup returns Optional[RETURN_VAL_TYPE] (List[Generation])
        cached_result: Optional[List[Generation]] = cache.lookup(prompt, LLM_STRING)
        duration = time.time() - start_time # End timing

        if cached_result:
            # Extract the text content from the Generation object
            response_text = cached_result[0].text if cached_result and cached_result[0] else None
            return jsonify({
                "status": "Hit",
                "prompt": prompt,
                "llm_string": LLM_STRING,
                "cached_response": response_text,
                "response_time_seconds": round(duration, 4)
            })
        else:
            return jsonify({
                "status": "Miss",
                "prompt": prompt,
                "llm_string": LLM_STRING,
                "message": "No semantically similar response found within the distance threshold.",
                "response_time_seconds": round(duration, 4)
            })
    except Exception as e:
        duration = time.time() - start_time
        logging.error(f"Cache Lookup Error: {e}")
        return jsonify({
            "error": f"An error occurred during cache lookup: {str(e)}",
            "response_time_seconds": round(duration, 4)
        }), 500


@app.route('/cache/update', methods=['POST'])
def cache_update():
    """
    Manually update the semantic cache with a prompt and a custom response.
    Expects: {"prompt": "...", "response_text": "..."}
    """
    start_time = time.time() # Start timing

    error_response = _check_cache_ready()
    if error_response: return error_response
    
    data, error_response = _handle_json_error()
    if error_response: return error_response

    prompt = data.get("prompt")
    response_text = data.get("response_text")
    
    if not prompt or not response_text:
        return jsonify({"error": "Missing 'prompt' or 'response_text' in request body."}), 400

    logging.info(f"Cache Update for: '{prompt}'")

    try:
        # Create the necessary LangChain Generation object structure
        return_val = [Generation(text=response_text)]
        cache.update(prompt, LLM_STRING, return_val)
        
        duration = time.time() - start_time # End timing

        return jsonify({
            "status": "Success",
            "message": "Cache updated successfully.",
            "prompt": prompt,
            "llm_string": LLM_STRING,
            "response_time_seconds": round(duration, 4)
        })
    except Exception as e:
        duration = time.time() - start_time
        logging.error(f"Cache Update Error: {e}")
        return jsonify({
            "error": f"An error occurred during cache update: {str(e)}",
            "response_time_seconds": round(duration, 4)
        }), 500


@app.route('/cache/clear', methods=['POST', 'DELETE'])
def cache_clear():
    """Clear all entries in the entire semantic cache."""
    start_time = time.time() # Start timing

    error_response = _check_cache_ready()
    if error_response: return error_response

    logging.warning("Clearing entire semantic cache!")
    
    try:
        cache.clear()
        duration = time.time() - start_time # End timing
        return jsonify({
            "status": "Success",
            "message": "All entries in the semantic cache have been cleared.",
            "response_time_seconds": round(duration, 4)
        })
    except Exception as e:
        duration = time.time() - start_time
        logging.error(f"Cache Clear Error: {e}")
        return jsonify({
            "error": f"An error occurred during cache clear: {str(e)}",
            "response_time_seconds": round(duration, 4)
        }), 500

# --- Main Execution Block ---
if __name__ == '__main__':
    PORT = 5001 
    logging.info(f"Starting Flask server on http://127.0.0.1:{PORT}/")
    app.run(host='0.0.0.0', port=PORT, debug=True)