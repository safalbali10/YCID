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

# Load API keys from .env file (only used locally — Railway reads env vars directly)
load_dotenv()

app = Flask(__name__)

HEYGEN_BASE = 'https://api.heygen.com'
MODEL       = "claude-sonnet-4-6"


def _load_key(env_var):
    """
    Read an environment variable and return it only if it looks like a real value.
    Placeholders like 'your_*_here' or empty strings are treated as missing (returns '').
    This prevents placeholder text from leaking into API URLs.
    """
    value = os.getenv(env_var, '').strip()
    if not value:
        return ''
    # Reject obvious placeholder patterns from .env.example
    lower = value.lower()
    if lower.startswith('your_') or lower.endswith('_here') or lower in ('changeme', 'placeholder', 'xxx'):
        return ''
    return value


YOUTUBE_API_KEY = _load_key('YOUTUBE_API_KEY')
HEYGEN_API_KEY  = _load_key('HEYGEN_API_KEY')

# Lazy-initialized — created on the first request, not at import time.
# This ensures env vars are fully settled before we read them.
_claude = None

def get_claude():
    """Return the shared Anthropic client, creating it on first call."""
    global _claude
    if _claude is None:
        # _load_key rejects placeholder values like 'your_anthropic_api_key_here'
        api_key = _load_key('ANTHROPIC_API_KEY') or _load_key('YT_ANTHROPIC_API_KEY')
        _claude = anthropic.Anthropic(api_key=api_key)
    return _claude


# ─── Helper functions ────────────────────────────────────────────────────────

def extract_video_id(url):
    """Pull the 11-character video ID out of any YouTube URL format, including Shorts."""
    patterns = [
        r'(?:youtube\.com/(?:watch\?v=|shorts/|embed/)|youtu\.be/)([0-9A-Za-z_-]{11})',
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


def parse_channel_input(user_input):
    """
    Parse a YouTube channel URL or @handle into (type, value).
    Returns ('id', 'UCxxx...') for channel IDs, or ('handle', 'name') for handles.
    """
    user_input = user_input.strip()

    # Direct channel ID (starts with UC, 24 chars total)
    if re.match(r'^UC[0-9A-Za-z_-]{22}$', user_input):
        return 'id', user_input

    # URL with /channel/UCxxxx
    m = re.search(r'youtube\.com/channel/(UC[0-9A-Za-z_-]{22})', user_input)
    if m:
        return 'id', m.group(1)

    # URL with /@handle
    m = re.search(r'youtube\.com/@([0-9A-Za-z._-]+)', user_input)
    if m:
        return 'handle', m.group(1)

    # URL with /c/name or /user/name (legacy formats)
    m = re.search(r'youtube\.com/(?:c|user)/([0-9A-Za-z._-]+)', user_input)
    if m:
        return 'handle', m.group(1)

    # @handle (strip the @ for the API call)
    if user_input.startswith('@'):
        return 'handle', user_input[1:]

    # Assume plain handle
    return 'handle', user_input


def fetch_channel_data(channel_type, channel_value):
    """
    Fetch channel metadata and recent 10 uploads from YouTube Data API v3.
    channel_type is 'id' or 'handle'; channel_value is the raw value without @.
    """
    params = {
        'key':  YOUTUBE_API_KEY,
        'part': 'snippet,statistics,contentDetails',
    }
    if channel_type == 'id':
        params['id'] = channel_value
    else:
        params['forHandle'] = channel_value  # YouTube API expects handle without @

    resp = requests.get(
        'https://www.googleapis.com/youtube/v3/channels',
        params=params, timeout=10
    )
    resp.raise_for_status()
    data = resp.json()

    if not data.get('items'):
        return None

    item    = data['items'][0]
    snippet = item['snippet']
    stats   = item.get('statistics', {})
    uploads_playlist_id = (
        item.get('contentDetails', {})
            .get('relatedPlaylists', {})
            .get('uploads', '')
    )

    # Fetch up to 10 most recent uploads for Claude to analyse patterns
    recent_videos = []
    if uploads_playlist_id:
        pl_resp = requests.get(
            'https://www.googleapis.com/youtube/v3/playlistItems',
            params={
                'key':        YOUTUBE_API_KEY,
                'playlistId': uploads_playlist_id,
                'part':       'snippet',
                'maxResults': 10,
            },
            timeout=10
        )
        pl_resp.raise_for_status()
        for pl_item in pl_resp.json().get('items', []):
            s = pl_item['snippet']
            if s.get('title') not in ('Private video', 'Deleted video'):
                recent_videos.append({
                    'title':        s.get('title', ''),
                    'published_at': s.get('publishedAt', '')[:10],
                })

    return {
        'name':          snippet.get('title', ''),
        'description':   snippet.get('description', '')[:500],
        'subscribers':   int(stats.get('subscriberCount', 0)),
        'total_views':   int(stats.get('viewCount', 0)),
        'video_count':   int(stats.get('videoCount', 0)),
        'country':       snippet.get('country', 'N/A'),
        'created_at':    snippet.get('publishedAt', '')[:10],
        'thumbnail':     snippet.get('thumbnails', {}).get('high', {}).get('url', ''),
        'recent_videos': recent_videos,
    }


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/debug')
def debug():
    """Shows whether API keys are loaded — never exposes the actual values."""
    # Use _load_key so placeholder values are also flagged as missing
    yt = _load_key('YOUTUBE_API_KEY')
    ai = _load_key('ANTHROPIC_API_KEY') or _load_key('YT_ANTHROPIC_API_KEY')
    return jsonify({
        'YOUTUBE_API_KEY':      'loaded' if yt else 'MISSING',
        'YT_ANTHROPIC_API_KEY': 'loaded' if ai else 'MISSING',
    })


@app.route('/api/analyze-video', methods=['POST'])
def analyze_video():
    """
    Section 1 — Video Analyser
    Accepts a YouTube URL, fetches video metadata, asks Claude for SEO analysis.
    """
    if not YOUTUBE_API_KEY:
        return jsonify({'error': 'YOUTUBE_API_KEY is not set. Add it to your .env file (locally) or Railway environment variables (production).'}), 500
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
  "summary": "<one compelling sentence summarising the video>",
  "detected_language": "<primary language detected from title and description, e.g. English, Hindi, Hinglish, Tamil, Telugu>"
}}"""

        message = get_claude().messages.create(
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
    if not YOUTUBE_API_KEY:
        return jsonify({'error': 'YOUTUBE_API_KEY is not set. Add it to your .env file (locally) or Railway environment variables (production).'}), 500
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

        message = get_claude().messages.create(
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
        language = data.get('language', 'English')

        # Map language choice to a specific writing instruction for Claude
        lang_instructions = {
            'Hindi':    'Write the ENTIRE script in Hindi using Devanagari script. Keep it natural and conversational.',
            'Hinglish': 'Write in Hinglish — blend Hindi and English naturally. Use Hindi for explanations and context, English for key terms and trending phrases.',
            'English':  'Write in clear, conversational Indian English. Keep it energetic and relatable for young Indians.',
        }
        lang_note = lang_instructions.get(language, lang_instructions['English'])

        prompt = f"""You are an expert YouTube scriptwriter for English learning content targeting Indian audience.

Write a complete, energetic script for:
Title: {title}
Tagline: {tagline}
Target Duration: {duration}
Script Language: {language} — {lang_note}

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

        message = get_claude().messages.create(
            model=MODEL,
            max_tokens=2500,
            messages=[{'role': 'user', 'content': prompt}]
        )
        script = parse_claude_json(message.content[0].text)

        return jsonify({'script': script, 'title': title, 'language': language})

    except json.JSONDecodeError:
        return jsonify({'error': 'AI returned unexpected format. Please try again.'}), 500
    except anthropic.APIError as e:
        return jsonify({'error': f'Claude API error: {e}'}), 500
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {e}'}), 500


@app.route('/api/analyze-channel', methods=['POST'])
def analyze_channel():
    """
    Section 4 — Channel Analyser
    Accepts a YouTube channel URL or @handle, fetches channel stats + recent uploads,
    then asks Claude for a full channel health analysis.
    """
    if not YOUTUBE_API_KEY:
        return jsonify({'error': 'YOUTUBE_API_KEY is not set. Add it to your .env file (locally) or Railway environment variables (production).'}), 500
    try:
        user_input = request.json.get('channel', '').strip()
        if not user_input:
            return jsonify({'error': 'Please enter a channel URL or @handle.'}), 400

        channel_type, channel_value = parse_channel_input(user_input)
        channel_data = fetch_channel_data(channel_type, channel_value)

        if not channel_data:
            return jsonify({'error': 'Channel not found. Please check the URL or handle and try again.'}), 404

        recent_titles = [v['title'] for v in channel_data['recent_videos']]

        prompt = f"""You are an expert YouTube channel strategist. Analyse this channel and return ONLY valid JSON — no markdown, no explanation.

Channel Details:
- Name: {channel_data['name']}
- Subscribers: {channel_data['subscribers']:,}
- Total Views: {channel_data['total_views']:,}
- Videos Published: {channel_data['video_count']}
- Country: {channel_data['country']}
- Channel Created: {channel_data['created_at']}
- Description: {channel_data['description']}
- Recent Video Titles: {json.dumps(recent_titles[:10])}

Return this exact JSON structure:
{{
  "health_score": <integer 1-10>,
  "health_explanation": "<2-3 sentences explaining the health score based on subscribers, views-to-sub ratio, upload consistency, and niche clarity>",
  "niche_analysis": "<2-3 sentences describing the content niche, target audience, and how well the channel is positioned>",
  "posting_frequency_recommendation": "<specific actionable recommendation, e.g. 3 videos per week with 2 Shorts and 1 long-form>",
  "top_performing_patterns": ["<pattern 1>", "<pattern 2>", "<pattern 3>"],
  "growth_opportunities": ["<opportunity 1>", "<opportunity 2>", "<opportunity 3>"]
}}"""

        message = get_claude().messages.create(
            model=MODEL,
            max_tokens=1500,
            messages=[{'role': 'user', 'content': prompt}]
        )
        analysis = parse_claude_json(message.content[0].text)

        return jsonify({'channel_data': channel_data, 'analysis': analysis})

    except json.JSONDecodeError:
        return jsonify({'error': 'AI returned unexpected format. Please try again.'}), 500
    except requests.RequestException as e:
        return jsonify({'error': f'YouTube API error: {e}'}), 500
    except anthropic.APIError as e:
        return jsonify({'error': f'Claude API error: {e}'}), 500
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {e}'}), 500


@app.route('/api/growth-hacks', methods=['GET'])
def get_growth_hacks():
    """
    Section 5 — Growth Hacks
    Ask Claude to generate 10 specific, actionable growth hacks for an English learning
    YouTube channel targeting Indians.
    """
    try:
        prompt = """You are an expert YouTube growth strategist specialising in English learning content for the Indian market.

Generate 10 highly specific, actionable growth hacks for an English learning YouTube channel targeting young Indians (18-35).

Cover a diverse mix of: posting strategies, thumbnail tactics, title formulas, engagement tricks, collaboration ideas, and monetisation tips.

Return ONLY valid JSON — no markdown:
{
  "hacks": [
    {
      "title": "<short punchy hack title, max 8 words>",
      "description": "<specific, actionable 2-3 sentence description with concrete examples — not generic advice>",
      "category": "<one of: Posting Strategy | Thumbnail | Title Formula | Engagement | Collaboration | Monetisation>"
    }
  ]
}

Generate EXACTLY 10 hacks. Every hack must be niche-specific to Indian English learning — no generic YouTube tips."""

        message = get_claude().messages.create(
            model=MODEL,
            max_tokens=2500,
            messages=[{'role': 'user', 'content': prompt}]
        )
        hacks = parse_claude_json(message.content[0].text)

        return jsonify(hacks)

    except json.JSONDecodeError:
        return jsonify({'error': 'AI returned unexpected format. Please try again.'}), 500
    except anthropic.APIError as e:
        return jsonify({'error': f'Claude API error: {e}'}), 500
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {e}'}), 500


# ─── HeyGen routes ───────────────────────────────────────────────────────────

def _heygen_headers():
    """Return auth headers for all HeyGen API calls."""
    return {'X-Api-Key': HEYGEN_API_KEY, 'Content-Type': 'application/json'}


@app.route('/api/heygen/avatars', methods=['GET'])
def get_heygen_avatars():
    """
    Fetch avatars from HeyGen and return them in two separate lists:
      - custom_avatars: the user's own Digital Twin / Instant Avatars
      - stock_avatars:  HeyGen's built-in library avatars

    We call two endpoints in parallel:
      - /v1/instant_avatar.list  → only the user's custom avatars (most reliable)
      - /v2/avatars              → everything (custom + stock); we strip custom ones
                                   from this list to avoid duplicates
    """
    try:
        # Fetch both lists at the same time for speed
        custom_resp, all_resp = (
            requests.get(f'{HEYGEN_BASE}/v1/instant_avatar.list',
                         headers=_heygen_headers(), timeout=15),
            requests.get(f'{HEYGEN_BASE}/v2/avatars',
                         headers=_heygen_headers(), timeout=15),
        )
        custom_resp.raise_for_status()
        all_resp.raise_for_status()

        custom_data = custom_resp.json()
        all_data     = all_resp.json()

        # Extract the user's custom avatars (instant avatars + talking photos)
        custom_avatars = (
            custom_data.get('data', {}).get('avatars', []) +
            custom_data.get('data', {}).get('talking_photos', [])
        )

        # Build a set of custom avatar_ids so we can exclude them from the stock list
        custom_ids = {av['avatar_id'] for av in custom_avatars}

        # Stock avatars = everything from /v2/avatars that isn't a custom avatar
        stock_avatars = [
            av for av in all_data.get('data', {}).get('avatars', [])
            if av['avatar_id'] not in custom_ids
        ]

        return jsonify({
            'custom_avatars': custom_avatars,
            'stock_avatars':  stock_avatars,
        })

    except requests.RequestException as e:
        return jsonify({'error': f'HeyGen API error: {e}'}), 500
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {e}'}), 500


@app.route('/api/heygen/voices', methods=['GET'])
def get_heygen_voices():
    """Fetch available TTS voices from HeyGen."""
    try:
        resp = requests.get(
            f'{HEYGEN_BASE}/v2/voices',
            headers=_heygen_headers(), timeout=15
        )
        resp.raise_for_status()
        return jsonify(resp.json())
    except requests.RequestException as e:
        return jsonify({'error': f'HeyGen API error: {e}'}), 500
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {e}'}), 500


@app.route('/api/heygen/generate', methods=['POST'])
def generate_heygen_video():
    """
    Start a HeyGen video generation job.
    Expects: avatar_id, voice_id, script_text, format ('shorts' or 'longform').
    Returns: the HeyGen job response including video_id.
    """
    try:
        data        = request.json or {}
        avatar_id   = data.get('avatar_id', '')
        voice_id    = data.get('voice_id', '')
        script_text = data.get('script_text', '')
        fmt         = data.get('format', 'shorts')

        # Validate required fields before calling HeyGen
        if not avatar_id:
            return jsonify({'error': 'avatar_id is missing from request.'}), 400
        if not voice_id:
            return jsonify({'error': 'voice_id is missing from request.'}), 400
        if not script_text:
            return jsonify({'error': 'script_text is empty.'}), 400

        # Shorts = vertical 1080×1920, long-form = horizontal 1920×1080
        width, height = (1080, 1920) if fmt == 'shorts' else (1920, 1080)

        payload = {
            "video_inputs": [{
                "character": {
                    "type": "avatar",
                    "avatar_id": avatar_id,
                    "avatar_style": "normal"
                },
                "voice": {
                    "type": "text",
                    "input_text": script_text,
                    "voice_id": voice_id
                },
                "background": {
                    "type": "color",
                    "value": "#1a1a2e"
                }
            }],
            "dimension": {"width": width, "height": height}
        }

        print(f"[HeyGen] POST /v2/video/generate — avatar={avatar_id} voice={voice_id} "
              f"format={fmt} ({width}x{height}) script_len={len(script_text)}", flush=True)

        resp = requests.post(
            f'{HEYGEN_BASE}/v2/video/generate',
            headers=_heygen_headers(),
            json=payload, timeout=30
        )

        # Log HeyGen's raw response regardless of status code
        print(f"[HeyGen] response HTTP {resp.status_code}: {resp.text[:500]}", flush=True)

        # Don't use raise_for_status — instead extract HeyGen's own error message
        # so the frontend can display exactly what HeyGen said went wrong
        heygen_data = resp.json()
        heygen_error = heygen_data.get('error') or heygen_data.get('message') or heygen_data.get('msg')
        if resp.status_code != 200 or heygen_error:
            error_text = str(heygen_error) if heygen_error else f'HeyGen returned HTTP {resp.status_code}'
            print(f"[HeyGen] ERROR: {error_text}", flush=True)
            return jsonify({'error': error_text}), 500

        return jsonify(heygen_data)

    except requests.RequestException as e:
        print(f"[HeyGen] RequestException: {e}", flush=True)
        return jsonify({'error': f'Network error reaching HeyGen: {e}'}), 500
    except Exception as e:
        print(f"[HeyGen] Unexpected error: {e}", flush=True)
        return jsonify({'error': f'Unexpected error: {e}'}), 500


@app.route('/api/heygen/status/<video_id>', methods=['GET'])
def get_heygen_status(video_id):
    """Poll HeyGen for the status of a video generation job."""
    try:
        resp = requests.get(
            f'{HEYGEN_BASE}/v1/video_status.get',
            params={'video_id': video_id},
            headers=_heygen_headers(), timeout=15
        )
        resp.raise_for_status()
        return jsonify(resp.json())
    except requests.RequestException as e:
        return jsonify({'error': f'HeyGen API error: {e}'}), 500
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {e}'}), 500


@app.route('/api/review-script', methods=['POST'])
def review_script():
    """
    Claude reviews a video script and returns:
    engagement score, hook strength, SEO title, description, thumbnail text,
    best posting time for Indian audience, and estimated reach.
    """
    try:
        data        = request.json
        script_text = data.get('script_text', '')
        language    = data.get('language', 'English')

        prompt = f"""You are an expert YouTube content strategist specialising in Indian digital content.

Review this video script and return ONLY valid JSON — no markdown, no explanation.

Language: {language}

Script:
{script_text[:3000]}

Return this exact JSON:
{{
  "engagement_score": <integer 1-10>,
  "hook_strength": "<one of: Weak | Average | Strong | Viral>",
  "hook_explanation": "<one sentence explaining the hook rating>",
  "suggested_title": "<SEO-optimised YouTube title, max 60 characters>",
  "suggested_description": "<3-4 line YouTube description with 5-8 relevant hashtags at the end>",
  "thumbnail_text": "<3-5 punchy words for thumbnail overlay — high contrast and compelling>",
  "recommended_posting_time": "<best day and time for Indian audience e.g. Tuesday 7-9 PM IST, with one-line reason>",
  "estimated_reach": "<realistic view estimate in first 48 hours with brief reasoning>"
}}"""

        message = get_claude().messages.create(
            model=MODEL,
            max_tokens=1500,
            messages=[{'role': 'user', 'content': prompt}]
        )
        review = parse_claude_json(message.content[0].text)
        return jsonify({'review': review})

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
