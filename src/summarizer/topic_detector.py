import google.generativeai as genai
import json
from typing import List, Dict
import os
import re


class TopicDetector:
    """Detect topics and chapters in documents using Gemini 2.0 Flash-Lite."""

    def __init__(self, api_key: str):
        """
        Initialize TopicDetector with Gemini API key.

        Args:
            api_key: Google Gemini API key
        """
        genai.configure(api_key=api_key)
        # Using Gemini 2.0 Flash (experimental) - fastest model for topic detection
        self.model = genai.GenerativeModel('gemini-2.0-flash-lite')
        self.chunk_size = 40000  # ~10000 tokens (10000 * 4 chars per token)

    def chunk_text(self, text: str, chunk_size: int = None) -> List[str]:
        """
        Split text into chunks of approximately chunk_size characters.
        Splits at paragraph boundaries to maintain context.

        Args:
            text: Full document text
            chunk_size: Target size for each chunk in characters

        Returns:
            List of text chunks
        """
        if chunk_size is None:
            chunk_size = self.chunk_size

        # Split by paragraphs (double newlines or single newlines)
        paragraphs = re.split(r'\n\s*\n|\n', text)

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            # If adding this paragraph exceeds chunk size and current chunk is not empty
            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
            else:
                current_chunk += para + "\n\n"

        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def detect_topics(self, document_text: str) -> List[Dict[str, str]]:
        """
        Analyze document and detect major topics/chapters using chunked processing.

        Args:
            document_text: Full text of the document

        Returns:
            List of dictionaries containing topic information
        """
        # Split document into chunks
        chunks = self.chunk_text(document_text)
        print(f"Split document into {len(chunks)} chunks for processing")

        all_topics = []

        # Process each chunk
        for i, chunk in enumerate(chunks):
            chunk_num = i + 1
            print(f"Processing chunk {chunk_num}/{len(chunks)}...")

            prompt = f"""Analyze the following text segment (Part {chunk_num} of {len(chunks)}) and identify ONLY the MAJOR topics or chapters.

IMPORTANT: Focus on MAJOR topics only (like chapters, main sections, key concepts). Ignore minor subsections or small details.

For each MAJOR topic, provide:
1. A clear, concise title (chapter/section name)
2. A brief description (1-2 sentences about what this chapter covers)
3. The exact starting text (first 30-50 words of the chapter/section)

Return the results as a JSON array with this structure:
[
  {{
    "title": "Chapter/Topic title",
    "description": "Brief description of what this major section covers",
    "start_marker": "First 30-50 words of this section..."
  }}
]

Text segment:
{chunk}

CRITICAL RULES:
- Identify ONLY MAJOR topics/chapters (not subsections or minor points)
- Look for chapter headings, major section breaks, or significant topic shifts
- Return ONLY valid JSON, no additional text
- Aim for 2-5 major topics per chunk maximum
"""

            try:
                response = self.model.generate_content(prompt)
                response_text = response.text.strip()

                # Remove markdown code blocks if present
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.startswith("```"):
                    response_text = response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]

                response_text = response_text.strip()

                chunk_topics = json.loads(response_text)
                all_topics.extend(chunk_topics)
                print(f"Found {len(chunk_topics)} topics in chunk {chunk_num}")

            except json.JSONDecodeError as e:
                print(f"JSON parsing error in chunk {chunk_num}: {e}")
                print(f"Response: {response_text[:200]}")
                continue
            except Exception as e:
                print(f"Error processing chunk {chunk_num}: {e}")
                continue

        # Deduplicate similar topics
        deduplicated_topics = self._deduplicate_topics(all_topics)
        print(f"Total topics after deduplication: {len(deduplicated_topics)}")

        # Filter and merge small topics (< 300 words)
        filtered_topics = self._filter_small_topics(deduplicated_topics, document_text)
        print(f"Total topics after filtering small topics: {len(filtered_topics)}")

        if not filtered_topics:
            # Fallback if no topics detected
            return [{
                "title": "Full Document",
                "description": "Unable to detect specific topics. Showing full document.",
                "start_marker": document_text[:50]
            }]

        return filtered_topics

    def _deduplicate_topics(self, topics: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Remove duplicate topics based on title similarity.

        Args:
            topics: List of topic dictionaries

        Returns:
            Deduplicated list of topics
        """
        if not topics:
            return []

        unique_topics = []
        seen_titles = set()

        for topic in topics:
            title_lower = topic['title'].lower().strip()

            # Check if similar title already exists
            is_duplicate = False
            for seen_title in seen_titles:
                # Check for high similarity (simple word overlap)
                if self._calculate_similarity(title_lower, seen_title) > 0.7:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_topics.append(topic)
                seen_titles.add(title_lower)

        return unique_topics

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate simple word overlap similarity between two strings.

        Args:
            str1: First string
            str2: Second string

        Returns:
            Similarity score between 0 and 1
        """
        words1 = set(str1.split())
        words2 = set(str2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _filter_small_topics(self, topics: List[Dict[str, str]], full_text: str, min_words: int = 300) -> List[Dict[str, str]]:
        """
        Filter out small topics by merging them with adjacent topics.
        Topics with less than min_words are merged with the next topic.

        Args:
            topics: List of topic dictionaries
            full_text: Full document text to extract content
            min_words: Minimum word count for a topic to be kept separate

        Returns:
            Filtered list of topics with small topics merged
        """
        if len(topics) <= 1:
            return topics

        filtered_topics = []
        i = 0

        while i < len(topics):
            current_topic = topics[i]

            # Get next topic marker if exists
            next_marker = topics[i + 1]['start_marker'] if i + 1 < len(topics) else None

            # Extract content for current topic
            topic_content = self.extract_topic_content(full_text, current_topic['start_marker'], next_marker)
            word_count = len(topic_content.split())

            print(f"Topic '{current_topic['title']}': {word_count} words")

            # If topic is too small and not the last one, merge with next
            if word_count < min_words and i + 1 < len(topics):
                next_topic = topics[i + 1]
                merged_title = f"{current_topic['title']} & {next_topic['title']}"
                merged_description = f"{current_topic['description']} {next_topic['description']}"

                print(f"  -> Merging '{current_topic['title']}' ({word_count} words) with '{next_topic['title']}'")

                # Create merged topic
                merged_topic = {
                    "title": merged_title,
                    "description": merged_description,
                    "start_marker": current_topic['start_marker']  # Keep first start marker
                }

                filtered_topics.append(merged_topic)
                i += 2  # Skip both current and next topic
            else:
                # Topic is large enough or is the last one, keep it
                filtered_topics.append(current_topic)
                i += 1

        return filtered_topics

    def extract_topic_content(self, full_text: str, current_marker: str, next_marker: str = None) -> str:
        """
        Extract FULL content for a specific topic/chapter.

        Args:
            full_text: Full document text
            current_marker: Starting marker for current topic
            next_marker: Starting marker for next topic (optional)

        Returns:
            Complete text content for the entire chapter/topic
        """
        # Try to find the exact marker
        start_idx = full_text.find(current_marker)

        if start_idx == -1:
            # Try partial match with first 10 words
            words = current_marker.split()[:10]
            search_term = " ".join(words)
            start_idx = full_text.find(search_term)

        if start_idx == -1:
            # Try even shorter match
            words = current_marker.split()[:5]
            search_term = " ".join(words)
            start_idx = full_text.find(search_term)

        if start_idx == -1:
            return full_text[:10000]  # Return first part as fallback

        # Extract content up to next marker or end of document
        if next_marker:
            # Try to find next marker
            end_idx = full_text.find(next_marker, start_idx + len(current_marker))

            if end_idx == -1:
                # Try partial match for next marker
                next_words = next_marker.split()[:10]
                next_search = " ".join(next_words)
                end_idx = full_text.find(next_search, start_idx + len(current_marker))

            if end_idx != -1:
                # Return FULL chapter content between current and next marker
                return full_text[start_idx:end_idx].strip()

        # If no next marker, return from start to end (or large chunk)
        # Extract up to 50,000 chars to capture full chapter
        return full_text[start_idx:start_idx + 50000].strip()
