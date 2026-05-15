"""
YouTube Channel Intelligence Dashboard
Flask backend — handles YouTube API calls and Claude AI analysis
"""

import os
import re
import json
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import anthropic

# Load API keys from .env file
load_dotenv()

app = Flask(__name__)

# Support both ANTHROPIC_API_KEY and YT_ANTHROPIC_API_KEY variable names
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY', '').strip()
ANTHROPIC_API_KEY = (os.getenv('ANTHROPIC_API_KEY') or os.getenv('YT_ANTHROPIC_API_KEY', '')).strip()

# Create Anthropic client once at startup
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
MODEL = "claude-sonnet-4-6"


# ─── Helper functions ────────────────────────────────────────────────────────

def extract_video_id(url):
    """Pull the 11-character video ID out of any YouTube URL format."""
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([0-9A-Za-z_-]{11})',
        r'[?&]v=([0-9A-Za-z_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def fetch_video_data(video_id):
    """Call YouTube Data API v3 and return cleaned video metadata."""
    resp = requests.get(
        'https://www.googleapis.com/youtube/v3/videos',
        params={'key': YOUTUBE_API_KEY, 'id': video_id, 'part': 'snippet,statistics'},
        timeout=10
    )
    resp.raise_for_status()
    data = resp.json()

    if not data.get('items'):
        return None

    item = data['items'][0]
    snippet = item['snippet']
    stats = item.get('statistics', {})

    return {
        'title':       snippet.get('title', ''),
        'description': snippet.get('description', '')[:600],
        'tags':        snippet.get('tags', []),
        'views':       int(stats.get('viewCount', 0)),
        'likes':       int(stats.get('likeCount', 0)),
        'channel':     snippet.get('channelTitle', ''),
        'thumbnail':   snippet.get('thumbnails', {}).get('high', {}).get('url', ''),
    }


def parse_claude_json(text):
    """Extract JSON from Claude's response — handles markdown code fences."""
    match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', text)
    if match:
        return json.loads(match.group(1))
    return json.loads(text.strip())


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/debug')
def debug():
    """Shows whether API keys are loaded — never exposes the actual values."""
    return jsonify({
        'YOUTUBE_API_KEY':    'loaded' if YOUTUBE_API_KEY    else 'MISSING',
        'YT_ANTHROPIC_API_KEY': 'loaded' if ANTHROPIC_API_KEY else 'MISSING',
    })


@app.route('/api/analyze-video', methods=['POST'])
def analyze_video():
    """
    Section 1 — Video Analyser
    Accepts a YouTube URL, fetches video metadata, asks Claude for SEO analysis.
    """
    try:
        url = request.json.get('url', '').strip()
        video_id = extract_video_id(url)
        if not video_id:
            return jsonify({'error': 'Invalid YouTube URL. Please paste a valid link.'}), 400

        video_data = fetch_video_data(video_id)
        if not video_data:
            return jsonify({'error': 'Video not found — it may be private or deleted.'}), 404

        prompt = f"""You are an expert YouTube SEO analyst. Analyse this video and return ONLY valid JSON — no markdown, no explanation.

Video Details:
- Title: {video_data['title']}
- Channel: {video_data['channel']}
- Views: {video_data['views']:,}
- Likes: {video_data['likes']:,}
- Tags: {', '.join(video_data['tags'][:20]) or 'None'}
- Description: {video_data['description']}

Return this exact JSON structure:
{{
  "seo_score": <integer 1-10>,
  "seo_explanation": "<2-3 sentences explaining the score>",
  "title_suggestions": ["<title 1>", "<title 2>", "<title 3>"],
  "better_tags": ["<tag1>", "<tag2>", "<tag3>", "<tag4>", "<tag5>"],
  "thumbnail_concept": "<specific, actionable thumbnail design description>",
  "reach_potential": "<honest assessment with clear reasoning>",
  "summary": "<one compelling sentence summarising the video>"
}}"""

        message = claude.messages.create(
            model=MODEL,
            max_tokens=1500,
            messages=[{'role': 'user', 'content': prompt}]
        )
        analysis = parse_claude_json(message.content[0].text)

        return jsonify({'video_data': video_data, 'analysis': analysis})

    except json.JSONDecodeError:
        return jsonify({'error': 'AI returned unexpected format. Please try again.'}), 500
    except requests.RequestException as e:
        return jsonify({'error': f'YouTube API error: {e}'}), 500
    except anthropic.APIError as e:
        return jsonify({'error': f'Claude API error: {e}'}), 500
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {e}'}), 500


@app.route('/api/trending', methods=['GET'])
def get_trending():
    """
    Section 2 — Trending Video Ideas
    Fetches top trending education videos from YouTube India,
    then asks Claude to suggest short-form and long-form video ideas.
    """
    try:
        videos = []

        # Search 1: English speaking / spoken English videos popular in India
        for query in [
            'spoken English course India',
            'English learning tips India viral',
            'how to speak English fluently India',
        ]:
            resp = requests.get(
                'https://www.googleapis.com/youtube/v3/search',
                params={
                    'key': YOUTUBE_API_KEY,
                    'q': query,
                    'type': 'video',
                    'regionCode': 'IN',
                    'maxResults': 8,
                    'order': 'viewCount',
                    'part': 'snippet',
                    'relevanceLanguage': 'en',
                },
                timeout=10
            )
            resp.raise_for_status()
            for item in resp.json().get('items', []):
                s = item['snippet']
                videos.append({
                    'title':   s.get('title', ''),
                    'channel': s.get('channelTitle', ''),
                    'views':   0,
                    'tags':    [],
                })

        prompt = f"""You are a viral content strategist for an English learning YouTube channel targeting young Indians (18-35).

These are the currently trending education/English learning videos on YouTube India:
{json.dumps(videos[:18], indent=2)}

Based on these trends, suggest creative viral ideas. Return ONLY valid JSON — no markdown:
{{
  "short_form_ideas": [
    {{
      "title": "<viral clickbait-worthy title, max 60 chars>",
      "tagline": "<one-line promise of what viewers will learn>",
      "duration": "1-2 mins",
      "why_trending": "<one short phrase>"
    }}
  ],
  "long_form_ideas": [
    {{
      "title": "<detailed informative title>",
      "tagline": "<one-line value proposition>",
      "duration": "7-10 mins",
      "why_trending": "<one short phrase>"
    }}
  ]
}}

Generate EXACTLY 7 short_form_ideas and EXACTLY 3 long_form_ideas. Titles must be emotionally compelling Indian English style."""

        message = claude.messages.create(
            model=MODEL,
            max_tokens=2500,
            messages=[{'role': 'user', 'content': prompt}]
        )
        ideas = parse_claude_json(message.content[0].text)

        return jsonify(ideas)

    except json.JSONDecodeError:
        return jsonify({'error': 'AI returned unexpected format. Please try again.'}), 500
    except requests.RequestException as e:
        return jsonify({'error': f'YouTube API error: {e}'}), 500
    except anthropic.APIError as e:
        return jsonify({'error': f'Claude API error: {e}'}), 500
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {e}'}), 500


@app.route('/api/generate-script', methods=['POST'])
def generate_script():
    """
    Section 3 — Script Generator
    Takes a video idea and asks Claude to write a full production script.
    """
    try:
        data = request.json
        title    = data.get('title', '')
        tagline  = data.get('tagline', '')
        duration = data.get('duration', '5-10 mins')

        prompt = f"""You are an expert YouTube scriptwriter for English learning content targeting Indian audience.

Write a complete, energetic script for:
Title: {title}
Tagline: {tagline}
Target Duration: {duration}

Return ONLY valid JSON — no markdown:
{{
  "hook": {{
    "text": "<15-20 second hook that grabs attention immediately>",
    "direction": "<camera or presentation direction>"
  }},
  "intro": {{
    "text": "<30-second intro that promises value and builds excitement>",
    "direction": "<direction>"
  }},
  "body": [
    {{
      "section": "<Section Name>",
      "text": "<detailed script for this section>",
      "direction": "<direction>"
    }}
  ],
  "cta": {{
    "text": "<strong call-to-action for likes, subscribe, comment>",
    "direction": "<direction>"
  }},
  "outro": {{
    "text": "<warm closing with teaser for next video>",
    "direction": "<direction>"
  }}
}}

Include 3-4 body sections. Keep it conversational and perfect for an Indian English learning audience."""

        message = claude.messages.create(
            model=MODEL,
            max_tokens=3500,
            messages=[{'role': 'user', 'content': prompt}]
        )
        script = parse_claude_json(message.content[0].text)

        return jsonify({'script': script, 'title': title})

    except json.JSONDecodeError:
        return jsonify({'error': 'AI returned unexpected format. Please try again.'}), 500
    except anthropic.APIError as e:
        return jsonify({'error': f'Claude API error: {e}'}), 500
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {e}'}), 500


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') != 'production'
    app.run(host='0.0.0.0', port=port, debug=debug)
