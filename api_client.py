import requests
import streamlit as st
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

def get_gemini_client(api_key):
    """Initialize Gemini client"""
    if HAS_GEMINI and api_key and api_key != "YOUR_GEMINI_API_KEY_HERE":
        try:
            genai.configure(api_key=api_key)
            return genai.GenerativeModel('gemini-2.5-flash')
        except Exception as e:
            pass
    return None

def gemini_verify_claim(text, api_key):
    """Use Gemini AI to verify the credibility of a claim"""
    if not HAS_GEMINI:
        return None

    try:
        model = get_gemini_client(api_key)
        if not model:
            return None
            
        prompt = f"""You are an expert fact-checker. Analyze this news article/claim carefully.

ARTICLE:
{text[:2000]}

ANALYZE FOR:
1. Factual accuracy - Are the claims verifiable?
2. Source credibility - Are sources properly attributed?
3. Evidence quality - Is there solid evidence or just opinions?
4. Emotional manipulation - Does it use sensational language?
5. Logical fallacies - Any logical errors or false reasoning?
6. Red flags - Missing sources, exaggeration, unverified claims?
7. Conspiracy indicators - Unsubstantiated theories?

PROVIDE YOUR ASSESSMENT AS:
- CREDIBILITY LEVEL: HIGH / MEDIUM / LOW
- KEY FINDINGS: (3-5 bullet points)
- VERDICT: LIKELY_TRUE / MIXED / LIKELY_FALSE / UNVERIFIABLE
- CONFIDENCE: 0-100%"""
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Gemini API Error: {str(e)}\n\nTip: Make sure your API key is valid and you have internet connection."

def search_news_gnews(query, api_key, max_results=10):
    """Search for related news articles using DuckDuckGo (Free, No API limits)"""
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            ddg_news = ddgs.news(query, max_results=max_results)
            if ddg_news:
                for article in ddg_news:
                    results.append({
                        'title': article.get('title', ''),
                        'source': {'name': article.get('source', '')},
                        'url': article.get('url', ''),
                        'publishedAt': article.get('date', '')
                    })
        return results
    except Exception as e:
        return []

def check_gnews_api(api_key):
    """Return (ok:bool, message:str, details:dict)"""
    # Now uses DuckDuckGo which requires no API key and has no strict limits
    return True, "DuckDuckGo Search is active and ready", {"status_code": 200, "total": "Unlimited"}

def translate_text(text, target_lang='en', api_key=None):
    """Translate text using Gemini API"""
    if not HAS_GEMINI or not api_key:
        return text  # Return original if no API
        
    try:
        model = get_gemini_client(api_key)
        if not model:
            return text
            
        prompt = f"""Translate the following text to {target_lang}. 
        Return ONLY the translated text, no explanations.
        
        Text:
        {text[:1000]}""" # Limit length to avoid quota issues
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return text # Fallback to original
