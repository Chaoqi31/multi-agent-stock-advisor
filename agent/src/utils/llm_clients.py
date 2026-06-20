import os
import time
import backoff
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from src.utils.logging_config import setup_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON

# Set up logging
logger = setup_logger('llm_clients')


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def get_completion(self, messages, **kwargs):
        """Get the model's response."""
        pass


class GeminiClient(LLMClient):
    """Google Gemini API client."""

    def __init__(self, api_key=None, model=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

        if not self.api_key:
            logger.error(f"{ERROR_ICON} GEMINI_API_KEY environment variable not found")
            raise ValueError(
                "GEMINI_API_KEY not found in environment variables")

        # Initialize the Gemini client
        self.client = genai.Client(api_key=self.api_key)
        logger.info(f"{SUCCESS_ICON} Gemini client initialized successfully")

    @backoff.on_exception(
        backoff.expo,
        (Exception),
        max_tries=5,
        max_time=300,
        giveup=lambda e: "AFC is enabled" not in str(e)
    )
    def generate_content_with_retry(self, contents, config=None):
        """Content generation function with a retry mechanism."""
        try:
            logger.info(f"{WAIT_ICON} Calling the Gemini API...")
            logger.debug(f"Request content: {contents}")
            logger.debug(f"Request config: {config}")

            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config
            )

            logger.info(f"{SUCCESS_ICON} API call succeeded")
            logger.debug(f"Response content: {response.text[:500]}...")
            return response
        except Exception as e:
            error_msg = str(e)
            if "location" in error_msg.lower():
                logger.info(
                    f"\033[91m❗ Gemini API geographic restriction error: please retry using a US-based VPN node\033[0m")
                logger.error(f"Detailed error: {error_msg}")
            elif "AFC is enabled" in error_msg:
                logger.warning(
                    f"{ERROR_ICON} API rate limit triggered, waiting to retry... Error: {error_msg}")
                time.sleep(5)
            else:
                logger.error(f"{ERROR_ICON} API call failed: {error_msg}")
            raise e

    def get_completion(self, messages, max_retries=3, initial_retry_delay=1, **kwargs):
        """Get the chat completion result, including retry logic."""
        try:
            logger.info(f"{WAIT_ICON} Using Gemini model: {self.model}")
            logger.debug(f"Message content: {messages}")

            for attempt in range(max_retries):
                try:
                    # Convert the message format
                    prompt = ""
                    system_instruction = None

                    for message in messages:
                        role = message["role"]
                        content = message["content"]
                        if role == "system":
                            system_instruction = content
                        elif role == "user":
                            prompt += f"User: {content}\n"
                        elif role == "assistant":
                            prompt += f"Assistant: {content}\n"

                    # Prepare the config
                    config = {}
                    if system_instruction:
                        config['system_instruction'] = system_instruction

                    # Call the API
                    response = self.generate_content_with_retry(
                        contents=prompt.strip(),
                        config=config
                    )

                    if response is None:
                        logger.warning(
                            f"{ERROR_ICON} Attempt {attempt + 1}/{max_retries}: API returned an empty value")
                        if attempt < max_retries - 1:
                            retry_delay = initial_retry_delay * (2 ** attempt)
                            logger.info(
                                f"{WAIT_ICON} Waiting {retry_delay} seconds before retrying...")
                            time.sleep(retry_delay)
                            continue
                        return None

                    logger.debug(f"Raw API response: {response.text}")
                    logger.info(f"{SUCCESS_ICON} Successfully obtained Gemini response")

                    # Return the text content directly
                    return response.text

                except Exception as e:
                    logger.error(
                        f"{ERROR_ICON} Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        retry_delay = initial_retry_delay * (2 ** attempt)
                        logger.info(f"{WAIT_ICON} Waiting {retry_delay} seconds before retrying...")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"{ERROR_ICON} Final error: {str(e)}")
                        return None

        except Exception as e:
            logger.error(f"{ERROR_ICON} An error occurred in get_completion: {str(e)}")
            return None


class OpenAICompatibleClient(LLMClient):
    """OpenAI-compatible API client."""

    def __init__(self, api_key=None, base_url=None, model=None):
        self.api_key = api_key or os.getenv("OPENAI_COMPATIBLE_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_COMPATIBLE_BASE_URL")
        self.model = model or os.getenv("OPENAI_COMPATIBLE_MODEL")

        if not self.api_key:
            logger.error(f"{ERROR_ICON} OPENAI_COMPATIBLE_API_KEY environment variable not found")
            raise ValueError(
                "OPENAI_COMPATIBLE_API_KEY not found in environment variables")

        if not self.base_url:
            logger.error(f"{ERROR_ICON} OPENAI_COMPATIBLE_BASE_URL environment variable not found")
            raise ValueError(
                "OPENAI_COMPATIBLE_BASE_URL not found in environment variables")

        if not self.model:
            logger.error(f"{ERROR_ICON} OPENAI_COMPATIBLE_MODEL environment variable not found")
            raise ValueError(
                "OPENAI_COMPATIBLE_MODEL not found in environment variables")

        # Initialize the OpenAI client
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        logger.info(f"{SUCCESS_ICON} OpenAI Compatible client initialized successfully")

    @backoff.on_exception(
        backoff.expo,
        (Exception),
        max_tries=5,
        max_time=300
    )
    def call_api_with_retry(self, messages, stream=False):
        """API call function with a retry mechanism."""
        try:
            logger.info(f"{WAIT_ICON} Calling the OpenAI Compatible API...")
            logger.debug(f"Request content: {messages}")
            logger.debug(f"Model: {self.model}, streaming: {stream}")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=stream
            )

            logger.info(f"{SUCCESS_ICON} API call succeeded")
            return response
        except Exception as e:
            error_msg = str(e)
            logger.error(f"{ERROR_ICON} API call failed: {error_msg}")
            raise e

    def get_completion(self, messages, max_retries=3, initial_retry_delay=1, **kwargs):
        """Get the chat completion result, including retry logic."""
        try:
            logger.info(f"{WAIT_ICON} Using OpenAI Compatible model: {self.model}")
            logger.debug(f"Message content: {messages}")

            for attempt in range(max_retries):
                try:
                    # Call the API
                    response = self.call_api_with_retry(messages)

                    if response is None:
                        logger.warning(
                            f"{ERROR_ICON} Attempt {attempt + 1}/{max_retries}: API returned an empty value")
                        if attempt < max_retries - 1:
                            retry_delay = initial_retry_delay * (2 ** attempt)
                            logger.info(
                                f"{WAIT_ICON} Waiting {retry_delay} seconds before retrying...")
                            time.sleep(retry_delay)
                            continue
                        return None

                    # Handle different types of responses
                    content = None

                    # If the response is a dict (some compatible APIs may return a dict directly)
                    if isinstance(response, dict):
                        if 'choices' in response and len(response['choices']) > 0:
                            if 'message' in response['choices'][0] and 'content' in response['choices'][0]['message']:
                                content = response['choices'][0]['message']['content']
                            elif 'text' in response['choices'][0]:
                                content = response['choices'][0]['text']
                    # If the response is a standard OpenAI object
                    elif hasattr(response, 'choices') and len(response.choices) > 0:
                        if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                            content = response.choices[0].message.content

                    # If the content cannot be extracted, try other methods
                    if content is None:
                        if hasattr(response, 'text'):
                            content = response.text
                        elif hasattr(response, 'content'):
                            content = response.content
                        else:
                            # As a last resort, stringify the entire response
                            content = str(response)
                            logger.warning(f"{WAIT_ICON} Could not extract response content directly, using the stringified response")

                    if content:
                        logger.debug(f"API response content: {content[:500]}...")
                        logger.info(
                            f"{SUCCESS_ICON} Successfully obtained OpenAI Compatible response")
                        return content
                    else:
                        logger.warning(f"{ERROR_ICON} Could not extract content from the response")
                        if attempt < max_retries - 1:
                            retry_delay = initial_retry_delay * (2 ** attempt)
                            logger.info(
                                f"{WAIT_ICON} Waiting {retry_delay} seconds before retrying...")
                            time.sleep(retry_delay)
                            continue
                        return "Could not extract content from the response"

                except Exception as e:
                    logger.error(
                        f"{ERROR_ICON} Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        retry_delay = initial_retry_delay * (2 ** attempt)
                        logger.info(f"{WAIT_ICON} Waiting {retry_delay} seconds before retrying...")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"{ERROR_ICON} Final error: {str(e)}")
                        return None

        except Exception as e:
            logger.error(f"{ERROR_ICON} An error occurred in get_completion: {str(e)}")
            return None


class LLMClientFactory:
    """LLM client factory class."""

    @staticmethod
    def create_client(client_type="auto", **kwargs):
        """
        Create an LLM client.

        Args:
            client_type: the client type ("auto", "gemini", "openai_compatible")
            **kwargs: configuration parameters for the specific client

        Returns:
            LLMClient: the instantiated LLM client
        """
        # If set to auto, automatically detect the available client
        if client_type == "auto":
            # Check whether the OpenAI Compatible API configuration is provided
            if (kwargs.get("api_key") and kwargs.get("base_url") and kwargs.get("model")) or \
               (os.getenv("OPENAI_COMPATIBLE_API_KEY") and os.getenv("OPENAI_COMPATIBLE_BASE_URL") and os.getenv("OPENAI_COMPATIBLE_MODEL")):
                client_type = "openai_compatible"
                logger.info(f"{WAIT_ICON} Automatically selected OpenAI Compatible API")
            else:
                client_type = "gemini"
                logger.info(f"{WAIT_ICON} Automatically selected Gemini API")

        if client_type == "gemini":
            return GeminiClient(
                api_key=kwargs.get("api_key"),
                model=kwargs.get("model")
            )
        elif client_type == "openai_compatible":
            return OpenAICompatibleClient(
                api_key=kwargs.get("api_key"),
                base_url=kwargs.get("base_url"),
                model=kwargs.get("model")
            )
        else:
            raise ValueError(f"Unsupported client type: {client_type}")
