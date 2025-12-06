import google.generativeai as genai
from typing import Generator


class Summarizer:
    """Generate summaries using Gemini 2.0 Flash-Lite."""

    def __init__(self, api_key: str):
        """
        Initialize Summarizer with Gemini API key.

        Args:
            api_key: Google Gemini API key
        """
        genai.configure(api_key=api_key)
        # Using Gemini 2.0 Flash (experimental) - fastest model for summarization
        self.model = genai.GenerativeModel('gemini-2.0-flash-lite')

    def summarize(self, topic_title: str, content: str, stream: bool = False) -> str:
        """
        Generate detailed summary for an entire chapter/topic.

        Args:
            topic_title: Title of the topic/chapter
            content: Full text content of the chapter to summarize
            stream: Whether to stream the response

        Returns:
            Detailed summary text
        """
        prompt = f"""Provide a DETAILED and COMPREHENSIVE summary of the entire chapter below.

Chapter/Topic: {topic_title}

Instructions:
- This is a FULL CHAPTER summary - be thorough and detailed
- Capture ALL major points, key concepts, and important details
- Include main ideas, arguments, examples, and conclusions
- Use clear, professional language
- Structure with bullet points and paragraphs for readability
- Length: Aim for  to cover the full chapter comprehensively
- Make sure the summary is self-contained and complete

Full Chapter Content:
{content[:40000]}

DETAILED SUMMARY:"""

        try:
            if stream:
                return self._summarize_stream(prompt)
            else:
                response = self.model.generate_content(prompt)
                return response.text.strip()

        except Exception as e:
            print(f"Error generating summary: {e}")
            raise

    def _summarize_stream(self, prompt: str) -> Generator[str, None, None]:
        """
        Generate streaming summary.

        Args:
            prompt: Prompt for summarization

        Yields:
            Chunks of summary text
        """
        try:
            response = self.model.generate_content(prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            print(f"Error in streaming summary: {e}")
            yield f"Error: {str(e)}"
