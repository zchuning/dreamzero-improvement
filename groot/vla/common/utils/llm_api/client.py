import logging
import time

import openai
from openai import AzureOpenAI
import requests

logger = logging.getLogger(__name__)


def get_oauth_token(p_token_url, p_client_id, p_client_secret, p_scope):
    try:
        response = requests.post(
            p_token_url,
            auth=(p_client_id, p_client_secret),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "client_credentials",
                "scope": p_scope,
            },
        )
        response.raise_for_status()
        token = response.json()
        authToken = token["access_token"]
        return authToken
    except Exception as e:
        if len(p_client_id) > 10:
            p_client_id = "..." + p_client_id[-4:]
        logger.info(f"Error occurred while getting OAuth token: {e}")
        logger.info(f"Client ID: {p_client_id}")
        return None


class LLMClient:
    def __init__(
        self,
        client_id,
        client_secret,
        token_url="https://5kbfxgaqc3xgz8nhid1x1r8cfestoypn-trofuum-oc.ssa.nvidia.com/token",
        scope="awsanthropic-readwrite azureopenai-readwrite",  # googlegemini-readwrite
        api_version="2025-03-01-preview",
        openai_api_base="https://prod.api.nvidia.com/llm/v1/azure/",
        claude_api_base="https://prod.api.nvidia.com/llm/v1/aws/model",
        max_token_generation_attempts=1,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.scope = scope
        self.api_version = api_version
        self.openai_api_base = openai_api_base
        self.claude_api_base = claude_api_base
        self.max_token_generation_attempts = max_token_generation_attempts
        token = self.generate_token()
        self.openai_client = AzureOpenAI(
            api_key=token, api_version=api_version, azure_endpoint=openai_api_base
        )
        self.claude_token = token

    def generate_token(self):
        token_generation_attempts = 0
        while True:
            logger.info("Attempting to get a new token...")
            token = get_oauth_token(
                self.token_url,
                self.client_id,
                self.client_secret,
                self.scope,
            )
            if token is not None:
                logger.info("New token generated!")
                return token
            token_generation_attempts += 1
            if token_generation_attempts >= self.max_token_generation_attempts:
                logger.info("Max token generation attempts reached.")
                return None
            else:
                logger.info("Failed to get a new token. Retrying...")
                time.sleep(5)

    def _is_claude_model(self, model):
        return "claude" in model.lower() or "sonnet" in model.lower() or "opus" in model.lower()

    def _is_openai_model(self, model):
        return "gpt" in model.lower() or "o3" in model.lower() or "o4" in model.lower()

    def _convert_messages_for_claude(self, messages):
        """Convert OpenAI-style messages to Claude format."""
        claude_messages = []
        system_prompt = None

        for message in messages:
            if message["role"] == "system":
                system_prompt = message["content"]
            else:
                content = self._claude_convert_message_content(message["content"])
                claude_messages.append({"role": message["role"], "content": content})

        return claude_messages, system_prompt

    def _claude_convert_message_content(self, content):
        """Convert message content to Claude format."""
        if isinstance(content, list):
            return self._claude_convert_multimodal_content(content)
        else:
            return content

    def _claude_convert_multimodal_content(self, content_parts):
        """Convert multimodal content parts to Claude format."""
        converted_parts = []

        for part in content_parts:
            if part["type"] == "text":
                converted_parts.append({"type": "text", "text": part["text"]})
            elif part["type"] == "image_url":
                converted_parts.append(self._claude_convert_image_url(part["image_url"]["url"]))
            else:
                # Claude only supports text and image content types
                raise ValueError(
                    f"Unsupported content type: {part['type']}. Supported types: text, image_url"
                )

        return converted_parts

    def _claude_convert_image_url(self, image_url):
        """Convert image URL to Claude image format."""
        if image_url.startswith("data:"):
            header, data = image_url.split(",", 1)
            media_type = header.split(";")[0].split(":")[1]
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data,
                },
            }
        else:
            # Handle non-data URLs if needed
            return {"type": "text", "text": f"[Image URL: {image_url}]"}

    def __call__(
        self,
        messages,
        max_tokens=4096,
        temperature=0,
        max_api_call_attempts=2,
        model="gpt-4o",
        n=1,
        **generation_kwargs,
    ):
        if self._is_claude_model(model):
            return self._call_claude(
                messages,
                max_tokens,
                temperature,
                max_api_call_attempts,
                model,
                n,
                **generation_kwargs,
            )
        elif self._is_openai_model(model):
            return self._call_openai(
                messages,
                max_tokens,
                temperature,
                max_api_call_attempts,
                model,
                n,
                **generation_kwargs,
            )
        else:
            raise NotImplementedError(f"Provider for {model} is not yet supported")

    def _call_claude(
        self,
        messages,
        max_tokens,
        temperature,
        max_api_call_attempts,
        model,
        n=1,
        **generation_kwargs,
    ):
        claude_messages, system_prompt = self._convert_messages_for_claude(messages)
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": claude_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            # "n": n,
            **generation_kwargs,
        }
        if system_prompt:
            payload["system"] = system_prompt
        headers = {
            "Authorization": f"Bearer {self.claude_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        base_url = self.claude_api_base.rstrip("/")
        endpoint = f"{base_url}/{model}/invoke"
        logger.info(f"Claude API Endpoint: {endpoint}")
        logger.info(f"Claude API Model: {model}")
        logger.info(f"Claude API Headers: {headers}")
        api_call_attempts = 0
        while True:
            try:
                logger.info(f"Sending request to {endpoint}")
                response = requests.post(endpoint, headers=headers, json=payload, timeout=240)
                logger.info(f"Claude API Response Status: {response.status_code}")
                logger.info(f"Claude API Response Headers: {dict(response.headers)}")
                if response.status_code == 401:
                    logger.error(f"Authentication error occurred - Response: {response.text}")
                    # Attempt to generate a new token
                    token = self.generate_token()
                    if token is None:
                        logger.error("Failed to generate new token")
                        return None
                    self.claude_token = token
                    headers["Authorization"] = f"Bearer {token}"
                elif response.status_code == 200:
                    # Success, return the response
                    try:
                        response_data = response.json()
                        return self._convert_claude_response_from_dict(response_data)
                    except ValueError:
                        logger.error("Invalid JSON response received")
                        return None
                else:
                    logger.error(f"Claude API error: {response.status_code} - {response.text}")
                    if response.status_code >= 500:
                        # Server side errors, retry
                        pass
                    else:
                        # Client side errors, quit without retrying
                        return None
            except requests.exceptions.Timeout:
                logger.error("Request timed out")
            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed: {e}")
            except Exception as e:
                logger.error(f"Claude API error occurred: {e}")
                return None
            api_call_attempts += 1
            if api_call_attempts >= max_api_call_attempts:
                logger.error("Max api call attempts reached.")
                return None
            else:
                logger.error("API call failed. Retrying...")
                time.sleep(5)

    def _call_openai(
        self,
        messages,
        max_tokens,
        temperature,
        max_api_call_attempts,
        model,
        n=1,
        **generation_kwargs,
    ):
        params = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "n": n,
            **generation_kwargs,
        }
        # Reasoning models don't support temperature
        if not model.startswith(("o3", "o4")):
            params["temperature"] = temperature

        api_call_attempts = 0
        while True:
            try:
                response = self.openai_client.chat.completions.create(**params)
                return response
            except Exception as e:
                if isinstance(e, openai.AuthenticationError):
                    # Authentication error, generate a new token
                    logger.error(f"{type(e).__name__} occurred: {e}")
                    token = self.generate_token()
                    self.openai_client = AzureOpenAI(
                        api_key=token,
                        api_version=self.api_version,
                        azure_endpoint=self.openai_api_base,
                    )
                elif isinstance(
                    e,
                    (
                        openai.BadRequestError,
                        openai.InternalServerError,
                        openai.PermissionDeniedError,
                    ),
                ):
                    # Client side errors, quit without retrying
                    logger.error(f"{type(e).__name__} occurred: {e}")
                    return None
                elif isinstance(
                    e,
                    (
                        openai.RateLimitError,
                        openai.APITimeoutError,
                        openai.APIConnectionError,
                        openai.UnprocessableEntityError,
                    ),
                ):
                    # Server side errors, retry
                    logger.error(f"{type(e).__name__} occurred: {e}")
                else:
                    # Other errors, quit without retrying
                    logger.error(f"Other error occurred: {e}")
                    return None
            api_call_attempts += 1
            if api_call_attempts >= max_api_call_attempts:
                # Max attempts reached, quit without retrying
                logger.error("Max api call attempts reached.")
                return None
            else:
                # Retry
                logger.error("API call failed. Retrying...")
                time.sleep(5)

    def _convert_claude_response_from_dict(self, response_data):
        class Choice:
            def __init__(self, message):
                self.message = message

        class Message:
            def __init__(self, content):
                self.content = content

        class Response:
            def __init__(self, choices):
                self.choices = choices

        content = ""
        if "content" in response_data and response_data["content"]:
            for content_block in response_data["content"]:
                if content_block.get("type") == "text" and "text" in content_block:
                    content += content_block["text"]
        message = Message(content)
        choice = Choice(message)
        return Response([choice])
