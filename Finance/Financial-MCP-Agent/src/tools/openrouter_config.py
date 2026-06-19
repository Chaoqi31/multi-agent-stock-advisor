import os
import time
from google import genai
from dotenv import load_dotenv
from dataclasses import dataclass
import backoff
from src.utils.logging_config import setup_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON
from src.utils.llm_clients import LLMClientFactory

# Set up logging
logger = setup_logger('api_calls')


@dataclass
class ChatMessage:
    content: str


@dataclass
class ChatChoice:
    message: ChatMessage


@dataclass
class ChatCompletion:
    choices: list[ChatChoice]


# Get the project root directory
project_root = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(project_root, '.env')

# Load environment variables
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)
    logger.info(f"{SUCCESS_ICON} Loaded environment variables: {env_path}")
else:
    logger.warning(f"{ERROR_ICON} Environment variable file not found: {env_path}")

# Validate environment variables
api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL")

if not api_key:
    logger.error(f"{ERROR_ICON} GEMINI_API_KEY environment variable not found")
    raise ValueError("GEMINI_API_KEY not found in environment variables")
if not model:
    model = "gemini-1.5-flash"
    logger.info(f"{WAIT_ICON} Using default model: {model}")

# Initialize the Gemini client
client = genai.Client(api_key=api_key)
logger.info(f"{SUCCESS_ICON} Gemini client initialized successfully")


@backoff.on_exception(
    backoff.expo,
    (Exception),
    max_tries=5,
    max_time=300,
    giveup=lambda e: "AFC is enabled" not in str(e)
)
def generate_content_with_retry(model, contents, config=None):
    """Content generation function with a retry mechanism."""
    try:
        logger.info(f"{WAIT_ICON} Calling the Gemini API...")
        logger.debug(f"Request content: {contents}")
        logger.debug(f"Request config: {config}")

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )

        logger.info(f"{SUCCESS_ICON} API call succeeded")
        logger.debug(f"Response content: {response.text[:500]}...")
        return response
    except Exception as e:
        error_msg = str(e)
        if "location" in error_msg.lower():
            # Use a red exclamation mark and red text for the prompt
            logger.info(f"\033[91m❗ Gemini API geographic restriction error: please retry using a US-based VPN node\033[0m")
            logger.error(f"Detailed error: {error_msg}")
        elif "AFC is enabled" in error_msg:
            logger.warning(f"{ERROR_ICON} API rate limit triggered, waiting to retry... Error: {error_msg}")
            time.sleep(5)
        else:
            logger.error(f"{ERROR_ICON} API call failed: {error_msg}")
        raise e


def get_chat_completion(messages, model=None, max_retries=3, initial_retry_delay=1,
                        client_type="auto", api_key=None, base_url=None):
    """
    Get the chat completion result, including retry logic.

    Args:
        messages: the message list, in OpenAI format
        model: the model name (optional)
        max_retries: the maximum number of retries
        initial_retry_delay: the initial retry delay (in seconds)
        client_type: the client type ("auto", "gemini", "openai_compatible")
        api_key: the API key (optional, only for the OpenAI Compatible API)
        base_url: the API base URL (optional, only for the OpenAI Compatible API)

    Returns:
        str: the model's response content, or None (if an error occurs)
    """
    try:
        # Create the client
        client = LLMClientFactory.create_client(
            client_type=client_type,
            api_key=api_key,
            base_url=base_url,
            model=model
        )

        # Get the response
        response = client.get_completion(
            messages=messages,
            max_retries=max_retries,
            initial_retry_delay=initial_retry_delay
        )

        # Check the response format and handle different types of return values
        if isinstance(response, dict):
            # The OpenAI-compatible API may return a dict format
            if 'choices' in response and len(response['choices']) > 0:
                if 'message' in response['choices'][0] and 'content' in response['choices'][0]['message']:
                    return response['choices'][0]['message']['content']
                elif 'text' in response['choices'][0]:
                    return response['choices'][0]['text']

        # If it is a string, return it directly
        if isinstance(response, str):
            return response

        # For other response types, try to extract the text
        logger.warning(f"{WAIT_ICON} Unknown response format, attempting to extract text: {type(response)}")
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'content'):
            return response.content
        elif hasattr(response, 'message') and hasattr(response.message, 'content'):
            return response.message.content

        # Unhandleable response format
        logger.error(f"{ERROR_ICON} Could not extract text from the response: {response}")
        return str(response)
    except Exception as e:
        logger.error(f"{ERROR_ICON} An error occurred in get_chat_completion: {str(e)}")
        return None
