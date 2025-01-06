import gradio as gr
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain_google_community import GoogleSearchAPIWrapper
from firecrawl import FirecrawlApp
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
from typing import List, Dict, Any, Optional
import logging
import sys
import json
import plotly.graph_objects as go
import hashlib
import time

# Configure Langsmith < set all these with environment variables, this is just an example, wont work like this
LANGCHAIN_TRACING_V2=True
LANGCHAIN_ENDPOINT="https://eu.api.smith.langchain.com"
LANGCHAIN_API_KEY=os.getenv("LANGSMITH_API_KEY")
LANGCHAIN_PROJECT="EnhancedFinancialScanner"

# Initialize APIs
firecrawl = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
claude = ChatAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), model='claude-3-5-sonnet-20241022', verbose=True)
google = GoogleSearchAPIWrapper(google_api_key=os.getenv("GOOGLE_API_KEY"), google_cse_id=os.getenv("GOOGLE_CSE_ID"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('EnhancedFinancialScanner')

class EnhancedFinancialScanner:
    def __init__(self):
        self.firecrawl = firecrawl
        self.claude = claude
        self.search = google
        logger.info("Initialized EnhancedFinancialScanner")
        
        # Templates voor verschillende analyses
        self.event_analysis_template = """
        Analyze this financial event for {ticker}:
        Event: {event_details}
        Context: {market_context}
        
        Provide only:
        1. Impact Level (High/Medium/Low)
        2. Brief explanation of relevance and potential market impact
        
        Keep the analysis focused on how this event might affect {ticker} and overall market sentiment.
        """
        
        self.competitor_analysis_template = """
        For the company with ticker {ticker}, provide a detailed competitive analysis:
        1. Top 3-5 direct competitors (with tickers)
        2. Main industry and sub-industry
        3. Market position and competitive advantages
        4. Key strengths and weaknesses vs competitors
        
        Return as JSON with the following structure:
        {
            "competitors": ["TICK1", "TICK2", ...],
            "industry": "Industry name",
            "sub_industry": "Sub-industry name",
            "market_position": "Description",
            "competitive_advantages": ["adv1", "adv2", ...],
            "strengths": ["str1", "str2", ...],
            "weaknesses": ["weak1", "weak2", ...]
        }
        """
        
        self.component_template = """
        As a financial markets expert, identify and provide detailed component information based on any user input.
        
        User Input: {query}
        
        Return a JSON object with the following structure ONLY, no other text:
        {
            "ticker": "official ticker symbol",
            "name": "full official name",
            "type": "stock/etf/index/crypto/forex/commodity",
            "components": [] # For indices/ETFs, include top components as ticker symbols
        }
        """
        
        # Initialize prompt templates
        self.event_prompt = PromptTemplate(
            input_variables=["ticker", "event_details", "market_context"],
            template=self.event_analysis_template
        )
        
        self.competitor_prompt = PromptTemplate(
            input_variables=["ticker"],
            template=self.competitor_analysis_template
        )
        
        self.component_prompt = PromptTemplate(
            input_variables=["query"],
            template=self.component_template
        )

    def create_summary(self, events, competitor_analysis, news):
        """Create a comprehensive summary of all analyzed data."""
        try:
            summary_prompt = f"""
            Based on the following financial data, provide a concise but comprehensive market summary:

            Events:
            {json.dumps(events, indent=2)}

            Competitor Analysis:
            {json.dumps(competitor_analysis, indent=2)}

            Recent News:
            {json.dumps(news, indent=2)}

            Please provide:
            1. Most significant events and their potential impact
            2. Key competitive dynamics and market position
            3. Overall market sentiment based on news
            4. Main opportunities and risks
            5. Summary of management actions and strategic direction

            Keep the summary focused, analytical, and actionable for investors.
            """
            response = self.claude.with_config(run_name="create_summary").invoke(summary_prompt)
            return response.content
        except Exception as e:
            logger.error(f"Error creating summary: {e}")
            return "Error generating summary. Please check the detailed data tabs for information."
            
    def get_news_sources(self, ticker: str, timeframe: str) -> List[str]:
        """Get a focused list of high-quality news sources using Google Search and LLM evaluation."""
        try:
            # Calculate date range based on timeframe
            end_date = datetime.now()
            days = self.timeframe_to_days(timeframe)
            start_date = end_date - timedelta(days=days)
            
            # Prepare search queries for different source types
            search_queries = [
                f"site:finance.yahoo.com {ticker} stock news",
                f"site:reuters.com {ticker} company news",
                f"(site:seekingalpha.com OR site:marketwatch.com OR site:bloomberg.com) {ticker} analysis"
            ]
            
            # Sort by date parameters for Google Search API
            sort_params = {
                "dateRestrict": f"d{days}"  # Restrict to last X days
            }
            
            # Collect potential URLs from search
            potential_urls = []
            for query in search_queries:
                try:
                    time.sleep(1)  # Rate limiting
                    search_results = self.search.results(query, num_results=2)
                    
                    for result in search_results:
                        url = result.get('link', '')
                        if url and any(x in url.lower() for x in ['/news/', '/article/', '/analysis/']):
                            potential_urls.append({
                                'url': url,
                                'title': result.get('title', ''),
                                'snippet': result.get('snippet', '')
                            })
                except Exception as e:
                    logger.error(f"Error in search query '{query}': {e}")
                    continue

            # Prepare the prompt with found URLs
            news_sources_prompt = f"""
            For the financial instrument {ticker}, analyze these potential news sources and return exactly 3 high-quality URLs.
            
            Found URLs:
            {json.dumps(potential_urls, indent=2)}
            
            Return a JSON object with exactly 3 news URLs in this exact format (no newlines or extra spaces):
            {{"sources":[{{"url":"url1","name":"name1","type":"type1","reliability":0.9}},{{"url":"url2","name":"name2","type":"type2","reliability":0.8}},{{"url":"url3","name":"name3","type":"type3","reliability":0.7}}]}}
            
            Rules:
            1. Prefer recent articles from the found URLs if they're high quality
            2. Use this format for Yahoo Finance if needed: https://finance.yahoo.com/quote/{ticker}/news?p={ticker}
            3. Use this format for Reuters if needed: https://www.reuters.com/markets/companies/{ticker}
            4. Third source should be specialized for the instrument type (e.g., SeekingAlpha, MarketWatch, etc.)
            5. No extra text or explanations, just the JSON
            6. The URL should be the direct link to the page that displays the most possible news article
            """

            # Get response from Claude
            response = self.claude.with_config(run_name="get_news_sources").invoke(
                news_sources_prompt
            )
            
            # Clean and parse JSON response
            cleaned_response = response.content.strip()
            # Remove any potential markdown code block indicators
            cleaned_response = cleaned_response.replace('```json', '').replace('```', '')
            sources_data = json.loads(cleaned_response)
            
            # Extract URLs and log source info
            urls = []
            for source in sources_data['sources'][:3]:  # Limit to top 3
                urls.append(source['url'])
                logger.info(
                    f"Selected news source: {source['name']} "
                    f"(Type: {source['type']}, Reliability: {source['reliability']})"
                )

            logger.info(f"Using {len(urls)} curated news sources for {ticker}")
            return urls

        except Exception as e:
            logger.error(f"Error getting news sources for {ticker}: {e}")
            # Fallback URLs that point to news archives
            fallback_urls = [
                f"https://finance.yahoo.com/quote/{ticker}/news?p={ticker}",
                f"https://www.reuters.com/markets/companies/{ticker}",
                f"https://seekingalpha.com/symbol/{ticker}"
            ]
            logger.warning(f"Falling back to default sources: {fallback_urls}")
            return fallback_urls
    
    def scrape_financial_news(self, ticker: str, timeframe: str) -> List[Dict[str, Any]]:
        """Scrape financial news from dynamically determined sources."""
        news_sources = self.get_news_sources(ticker, timeframe)
        logger.info(f"Scraping {len(news_sources)} news sources for {ticker}")
        
        news_items = []
        seen_content_hashes = set()
        
        for source in news_sources:
            try:
                # Add rate limiting
                time.sleep(1)  # Basic rate limiting
                
                news_data = self.firecrawl.scrape_url(
                    source,
                    params={
                        'formats': ['markdown', 'extract'],
                        'extract': {
                            'schema': {
                                'type': 'object',
                                'properties': {
                                    'articles': {
                                        'type': 'array',
                                        'items': {
                                            'type': 'object',
                                            'properties': {
                                                'title': {'type': 'string'},
                                                'date': {'type': 'string'},
                                                'summary': {'type': 'string'},
                                                'url': {'type': 'string'},
                                                'source': {'type': 'string'}
                                            },
                                            'required': ['title']
                                        }
                                    }
                                }
                            }
                        }
                    }
                )
                
                if news_data.get('extract', {}).get('articles'):
                    for article in news_data['extract']['articles']:
                        # Create content hash for deduplication
                        content_hash = hashlib.md5(
                            f"{article['title'].lower().strip()}".encode()
                        ).hexdigest()
                        
                        if content_hash not in seen_content_hashes:
                            seen_content_hashes.add(content_hash)
                            
                            # Only analyze sentiment for unique articles
                            sentiment = self.analyze_sentiment(article['title'])
                            
                            article.update({
                                'sentiment': sentiment,
                                'source_domain': source.split('/')[2],
                                'processed_date': datetime.now().isoformat()
                            })
                            
                            news_items.append(article)
                            
                else:
                    logger.warning(f"No articles found in response from {source}")
                    
            except Exception as e:
                logger.error(f"Error scraping news from {source}: {e}")
                continue
        
        # Sort by date and limit total items
        sorted_news = sorted(
            news_items, 
            key=lambda x: x.get('date', ''), 
            reverse=True
        )[:20]  # Limit to 20 most recent items
        
        logger.info(f"Found {len(sorted_news)} unique news items for {ticker}")
        return sorted_news

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text using Claude."""
        try:
            sentiment_prompt = """
            Analyze the sentiment of this financial news text and return ONLY a JSON object with sentiment scores.
            The scores should reflect both the emotional tone and financial implications.
            
            Text to analyze: {text}
            
            Rules for scoring:
            1. positive: Indicates positive news/developments (e.g., growth, profits, expansion)
            2. negative: Indicates negative news/developments (e.g., losses, layoffs, risks)
            3. neutral: Indicates neutral or balanced information
            4. confidence: How confident are you in this assessment

            Each score should be a float between 0.0 and 1.0
            The sum of positive, negative, and neutral should equal 1.0

            Return ONLY a JSON object in this format (no newlines or extra spaces):
            {{"positive":0.0,"negative":0.0,"neutral":0.0,"confidence":0.0}}
            """
            
            response = self.claude.with_config(run_name="analyze_sentiment").invoke(
                sentiment_prompt.format(text=text)
            )
            
            # Clean and parse JSON response
            cleaned_response = response.content.strip()
            # Remove any potential markdown code block indicators
            cleaned_response = cleaned_response.replace('```json', '').replace('```', '')
            sentiment_scores = json.loads(cleaned_response)
            
            # Validate scores
            for key in ['positive', 'negative', 'neutral', 'confidence']:
                if key not in sentiment_scores:
                    sentiment_scores[key] = 0.0
                sentiment_scores[key] = float(sentiment_scores[key])
                
            # Normalize sentiment scores to sum to 1.0
            total = sentiment_scores['positive'] + sentiment_scores['negative'] + sentiment_scores['neutral']
            if total != 0:
                sentiment_scores['positive'] /= total
                sentiment_scores['negative'] /= total
                sentiment_scores['neutral'] /= total
            
            logger.debug(f"Sentiment analysis for text: {text[:100]}...")
            logger.debug(f"Scores: {sentiment_scores}")
            
            return sentiment_scores
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            # Return balanced default scores in case of error
            return {
                "positive": 0.33,
                "negative": 0.33,
                "neutral": 0.34,
                "confidence": 0.5
            }

    def get_instrument_type(self, ticker: str) -> str:
        """Determine the type of financial instrument."""
        try:
            # Simpelere prompt die direct het type vraagt
            type_prompt = f"""What type of financial instrument is {ticker}?
            Return ONLY ONE of these words: stock, etf, index, crypto, forex, commodity
            No explanation, no JSON, just the single word."""
            
            response = self.claude.with_config(run_name="get_instrument_type").invoke(type_prompt)
            
            # Clean en valideer de response
            instrument_type = response.content.strip().lower()
            
            valid_types = {'stock', 'etf', 'index', 'crypto', 'forex', 'commodity'}
            if instrument_type not in valid_types:
                logger.warning(f"Unexpected instrument type '{instrument_type}' for {ticker}, defaulting to 'stock'")
                instrument_type = 'stock'  # Default to stock if invalid response
            
            logger.info(f"Determined {ticker} is type: {instrument_type}")
            return instrument_type
            
        except Exception as e:
            logger.error(f"Error determining instrument type for {ticker}: {e}")
            return "stock"  # Default fallback

    def analyze_management_changes(self, ticker: str, timeframe: str) -> List[Dict[str, Any]]:
        """Track management changes from news articles."""
        try:
            # Use the events we already have instead of making new API calls
            events = self.scrape_financial_events(ticker, timeframe)
            
            # Filter for management change events
            management_changes = [
                event for event in events 
                if event.get('type') == 'Management Change'
            ]
            
            # Sort by date and limit to most recent
            sorted_changes = sorted(
                management_changes,
                key=lambda x: x.get('parsed_date', ''),
                reverse=True
            )[:10]  # Limit to 10 most recent changes
            
            logger.info(f"Found {len(sorted_changes)} management changes for {ticker}")
            return sorted_changes
            
        except Exception as e:
            logger.error(f"Error analyzing management changes: {e}")
            return []

    def get_ticker_components(self, ticker: str) -> List[str]:
            """Get main components for any ticker using LLM."""
            logger.info(f"Getting components for ticker/query: {ticker}")
            try:
                # Simplified prompt that asks for minimal JSON format
                component_prompt = f"""For the ticker {ticker}, return a JSON object in this exact format (no whitespace, newlines, or explanation):
    {{"ticker":"{ticker}","name":"Company Name","type":"stock/etf/index/crypto/forex/commodity","components":[]}}"""
                
                # Get structured response from Claude
                response = self.claude.with_config(run_name="get_ticker_components").invoke(component_prompt)
                
                # Clean response and parse JSON
                cleaned_response = response.content.strip()
                # Remove any potential markdown code block indicators
                cleaned_response = cleaned_response.replace('```json', '').replace('```', '')
                
                result = json.loads(cleaned_response)
                
                logger.info(f"Identified input as {result['name']} ({result['ticker']}) - Type: {result['type']}")
                
                # If it's an ETF/index with components, return those
                if result['components']:
                    logger.info(f"Found {len(result['components'])} components")
                    return result['components']
                
                # Otherwise return just the identified ticker
                return [result['ticker']]
                
            except Exception as e:
                logger.error(f"Error processing ticker/query '{ticker}': {e}")
                logger.warning(f"Falling back to original ticker: {ticker}")
                return [ticker]

    def get_component_weight(self, index: str, component: str) -> float:
            """Get component weight in an index."""
            try:
                if index == component:
                    return 100.0
                    
                # For indices, get actual weights from yfinance if possible
                index_ticker = yf.Ticker(index)
                if hasattr(index_ticker, 'info') and 'holdings' in index_ticker.info:
                    holdings = index_ticker.info['holdings']
                    for holding in holdings:
                        if holding['symbol'] == component:
                            return holding['weight']
                
                # Fallback weights for S&P 500
                if index == "SPY":
                    weights = {
                        "AAPL": 7.2, "MSFT": 6.8, "AMZN": 3.5,
                        "NVDA": 4.5, "GOOGL": 1.8, "META": 2.1,
                        "BRK.B": 1.7, "GOOG": 1.8, "TSLA": 1.6,
                        "UNH": 1.4
                    }
                    return weights.get(component, 0.1)
                
                return 0.1  # Default weight for unknown components
                
            except Exception as e:
                logger.error(f"Error getting component weight: {e}")
                return 0.1
            
    def scrape_financial_events(self, ticker: str, timeframe: str) -> List[Dict[str, Any]]:
        """Scrape financial events for a ticker and its components."""
        logger.info(f"Starting financial event scraping for {ticker} with timeframe {timeframe}")
        events = []
        
        # Get all components to analyze
        components = self.get_ticker_components(ticker)
        logger.info(f"Processing {len(components)} components")
        
        # Keep track of processed components to prevent loops
        processed_components = set()
        
        for component in components:
            if component in processed_components:
                logger.info(f"Skipping already processed component: {component}")
                continue
                
            processed_components.add(component)
            logger.info(f"\n==== Processing component: {component} ====")
            
            try:
                # Get news articles directly without recursive calls
                news_sources = self.get_news_sources(component, timeframe)
                news_items = []
                
                for source in news_sources:
                    try:
                        time.sleep(1)  # Rate limiting
                        
                        news_data = self.firecrawl.scrape_url(
                            source,
                            params={
                                'formats': ['markdown', 'extract'],
                                'extract': {
                                    'schema': {
                                        'type': 'object',
                                        'properties': {
                                            'articles': {
                                                'type': 'array',
                                                'items': {
                                                    'type': 'object',
                                                    'properties': {
                                                        'title': {'type': 'string'},
                                                        'date': {'type': 'string'},
                                                        'summary': {'type': 'string'},
                                                        'url': {'type': 'string'},
                                                        'source': {'type': 'string'}
                                                    },
                                                    'required': ['title']
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        )
                        
                        if news_data.get('extract', {}).get('articles'):
                            news_items.extend(news_data['extract']['articles'])
                        else:
                            logger.warning(f"No articles found in response from {source}")
                            
                    except Exception as e:
                        logger.error(f"Error scraping news from {source}: {e}")
                        continue
                
                # Process articles into events
                for article in news_items:
                    try:
                        parsed_date = self.parse_date(article.get('date'))
                        if parsed_date:
                            formatted_date = parsed_date.strftime('%Y-%m-%d')
                            
                            # Basic event
                            event = {
                                'date': formatted_date,
                                'title': article.get('title', ''),
                                'description': article.get('summary', 'No summary available'),
                                'company': component,
                                'type': 'News',
                                'url': article.get('url', ''),
                                'sentiment': self.analyze_sentiment(article.get('title', '')),
                                'parsed_date': formatted_date
                            }
                            
                            # Check for management changes
                            mgmt_keywords = ["ceo", "cfo", "president", "chairman", "chief executive", "chief financial"]
                            if any(keyword in article.get('title', '').lower() for keyword in mgmt_keywords):
                                event.update({
                                    'type': 'Management Change',
                                    'importance': 'High'
                                })
                            
                            events.append(event)
                            
                    except Exception as e:
                        logger.error(f"Error processing article: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error processing component {component}: {e}")
                continue
        
        # Filter events by timeframe
        filtered_events = self.filter_events_by_timeframe(events, timeframe)
        
        # Analyze events in batches
        analyzed_events = []
        try:
            batch_size = 5
            for i in range(0, len(filtered_events), batch_size):
                batch = filtered_events[i:i + batch_size]
                analyzed_batch = self.analyze_events_importance_batch(ticker, batch)
                analyzed_events.extend(analyzed_batch)
                time.sleep(1)  # Add small delay between batches
            
            # Sort by date and importance
            analyzed_events.sort(key=lambda x: (
                x['parsed_date'],
                {'High': 3, 'Medium': 2, 'Low': 1}.get(x.get('importance', 'Low'), 0)
            ), reverse=True)
            
        except Exception as e:
            logger.error(f"Error in batch analysis: {e}")
            analyzed_events = filtered_events  # Fall back to filtered events without analysis
        
        logger.info(f"Found {len(analyzed_events)} analyzed events")
        return analyzed_events

    def analyze_events_importance_batch(self, ticker: str, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze importance for a batch of events."""
        try:
            # Create simplified event list for the prompt
            event_list = [
                {
                    'title': event['title'],
                    'description': event.get('description', 'No description'),
                    'company': event['company'],
                    'type': event.get('type', 'News')
                }
                for event in events
            ]
            
            # Prepare batch analysis prompt
            batch_prompt = f"""Analyze these {len(events)} events for {ticker}. Return ONLY a JSON array (no other text):
[{{"importance":"High/Medium/Low","analysis":"Brief explanation","key_factors":["factor1"],"confidence":0.9}}]

Events to analyze:
{json.dumps(event_list, indent=None)}"""
            
            # Get batch analysis from Claude
            response = self.claude.with_config(run_name="analyze_events_importance_batch").invoke(batch_prompt)
            
            # Clean and parse response
            cleaned_response = response.content.strip()
            cleaned_response = cleaned_response.replace('```json', '').replace('```', '')
            cleaned_response = cleaned_response.replace('"', '"').replace('"', '"')
            
            try:
                analysis_results = json.loads(cleaned_response)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed: {e}\nResponse was: {cleaned_response}")
                raise
            
            # Combine original events with analysis
            analyzed_events = []
            for event, analysis in zip(events, analysis_results):
                # Validate the analysis result
                validated = {
                    'importance': analysis.get('importance', 'Low'),
                    'analysis': analysis.get('analysis', 'No analysis available'),
                    'key_factors': analysis.get('key_factors', []),
                    'confidence': float(analysis.get('confidence', 0.5))
                }
                
                # Ensure importance is valid
                if validated['importance'] not in ['High', 'Medium', 'Low']:
                    validated['importance'] = 'Low'
                
                analyzed_events.append({
                    **event,
                    'importance': validated['importance'],
                    'analysis': validated['analysis'],
                    'key_factors': validated['key_factors'],
                    'confidence': validated['confidence']
                })
            
            return analyzed_events
            
        except Exception as e:
            logger.error(f"Error in batch analysis: {e}")
            # Return events with default values
            return [{
                **event,
                'importance': 'Low',
                'analysis': 'Error analyzing event importance',
                'key_factors': [],
                'confidence': 0.0
            } for event in events]

    def filter_events_by_timeframe(self, events: List[Dict[str, Any]], timeframe: str) -> List[Dict[str, Any]]:
        """Filter events based on the selected timeframe."""
        try:
            now = datetime.now()
            days = self.timeframe_to_days(timeframe)
            cutoff_date = now - timedelta(days=days)
            
            logger.info(f"Filtering events between {cutoff_date.date()} and {now.date()}")
            
            filtered_events = []
            for event in events:
                try:
                    # Parse the date
                    parsed_date = self.parse_date(event['date'])
                    
                    if parsed_date:
                        if cutoff_date <= parsed_date <= now:
                            # Add the parsed date to the event for later sorting
                            event['parsed_date'] = parsed_date.strftime('%Y-%m-%d')
                            filtered_events.append(event)
                            logger.debug(f"Added event: {event.get('title', 'Unknown')} on {event['parsed_date']}")
                        else:
                            logger.debug(f"Event outside timeframe: {event.get('title', 'Unknown')} on {parsed_date.date()}")
                    else:
                        logger.warning(f"Could not parse date for event: {event.get('title', 'Unknown')} - {event.get('date', 'No date')}")
                        
                except Exception as e:
                    logger.warning(f"Error processing event date: {event.get('title', 'Unknown')} - {e}")
                    continue
            
            # Sort by date (most recent first)
            filtered_events.sort(key=lambda x: x['parsed_date'], reverse=True)
            
            logger.info(f"Filtered {len(events)} events to {len(filtered_events)} within timeframe")
            return filtered_events
            
        except Exception as e:
            logger.error(f"Error filtering events: {e}")
            return events

    def parse_date(self, date_str: str) -> Optional[datetime]:
            """Parse various date formats to datetime object."""
            try:
                if not date_str:
                    return None
                    
                now = datetime.now()
                cleaned_date = date_str.strip().lower()
                
                # Handle MM/DD HH:MM AM/PM format (e.g., "9/21 12:03 PM")
                if '/' in cleaned_date and any(x in cleaned_date for x in ['am', 'pm']):
                    try:
                        date_part, time_part = cleaned_date.split(' ', 1)
                        month, day = map(int, date_part.split('/'))
                        
                        # First try with current year
                        try:
                            full_date_str = f"{now.year}-{month:02d}-{day:02d} {time_part}"
                            potential_date = datetime.strptime(full_date_str, "%Y-%m-%d %I:%M %p")
                            
                            # If date would be in future, use last year
                            if potential_date > now:
                                full_date_str = f"{now.year-1}-{month:02d}-{day:02d} {time_part}"
                                return datetime.strptime(full_date_str, "%Y-%m-%d %I:%M %p")
                            return potential_date
                        except ValueError:
                            # Try with previous year if current year fails
                            full_date_str = f"{now.year-1}-{month:02d}-{day:02d} {time_part}"
                            return datetime.strptime(full_date_str, "%Y-%m-%d %I:%M %p")
                    except Exception as e:
                        logger.debug(f"Could not parse MM/DD HH:MM AM/PM format: {e}")
                
                # Parse relative time formats
                if 'ago' in cleaned_date:
                    try:
                        # Split into number and unit
                        parts = cleaned_date.split()
                        if len(parts) >= 3:  # format: "X [hours/days/etc] ago"
                            amount = float(parts[0])
                            unit = parts[1]
                            
                            if 'hour' in unit:
                                return now - timedelta(hours=amount)
                            elif 'day' in unit:
                                return now - timedelta(days=amount)
                            elif 'week' in unit:
                                return now - timedelta(weeks=amount)
                            elif 'month' in unit:
                                return now - timedelta(days=amount * 30)
                            elif 'year' in unit:
                                return now - timedelta(days=amount * 365)
                    except Exception as e:
                        logger.debug(f"Could not parse relative date: {e}")

                # Handle short date formats (MM/DD)
                if len(cleaned_date.split('/')) == 2:
                    try:
                        month, day = map(int, cleaned_date.split('/'))
                        # Try current year first
                        potential_date = datetime(now.year, month, day)
                        # If date would be in future, use last year
                        if potential_date > now:
                            return datetime(now.year - 1, month, day)
                        return potential_date
                    except Exception as e:
                        logger.debug(f"Could not parse short date format: {e}")

                # Parse time of day format (e.g., "12:36p", "11:53a")
                if any(x in cleaned_date for x in ['a', 'p']) and ':' in cleaned_date:
                    try:
                        # Remove any non-numeric characters except : and a/p
                        time_str = ''.join(c for c in cleaned_date if c.isdigit() or c in ':ap')
                        if 'a' in time_str:
                            time_str = time_str.replace('a', ' AM')
                        if 'p' in time_str:
                            time_str = time_str.replace('p', ' PM')
                        return datetime.strptime(f"{now.strftime('%Y-%m-%d')} {time_str}", '%Y-%m-%d %I:%M %p')
                    except Exception as e:
                        logger.debug(f"Could not parse time of day: {e}")

                # Standaardiseer de input voor andere formats
                cleaned_date = (
                    date_str.strip()
                    .replace(',', '')  # Verwijder komma's
                    .replace('  ', ' ')  # Verwijder dubbele spaties
                    .replace('.', '')  # Verwijder punten
                )
                
                # Verwijder tijdzone/tijd indicaties
                time_indicators = [' ET', ' PT', ' GMT', ' UTC', ' at ', ' AM', ' PM']
                for indicator in time_indicators:
                    if indicator in cleaned_date:
                        cleaned_date = cleaned_date.split(indicator)[0].strip()

                # Handle ISO format with 'T' and 'Z'
                if 'T' in cleaned_date:
                    try:
                        # Remove timezone indicator and parse
                        iso_date = cleaned_date.split('T')[0]
                        return datetime.strptime(iso_date, '%Y-%m-%d')
                    except Exception as e:
                        logger.debug(f"Could not parse ISO date: {e}")

                # Als geen format werkt, probeer natuurlijke taal parsing
                month_mapping = {
                    'january': 1, 'jan': 1,
                    'february': 2, 'feb': 2,
                    'march': 3, 'mar': 3,
                    'april': 4, 'apr': 4,
                    'may': 5,
                    'june': 6, 'jun': 6,
                    'july': 7, 'jul': 7,
                    'august': 8, 'aug': 8,
                    'september': 9, 'sep': 9, 'sept': 9,
                    'october': 10, 'oct': 10,
                    'november': 11, 'nov': 11,
                    'december': 12, 'dec': 12
                }
                
                parts = cleaned_date.lower().split()
                if len(parts) >= 2:
                    # Check for month names first
                    for word in parts:
                        if word in month_mapping:
                            month = month_mapping[word]
                            # Find the day number
                            for part in parts:
                                if part.replace('st','').replace('nd','').replace('rd','').replace('th','').isdigit():
                                    day = int(part.replace('st','').replace('nd','').replace('rd','').replace('th',''))
                                    # Check for explicit year, otherwise use appropriate year
                                    year = now.year
                                    for part in parts:
                                        if len(part) == 4 and part.isdigit():
                                            year = int(part)
                                            break
                                    else:
                                        # No explicit year found, check if date would be in future
                                        potential_date = datetime(year, month, day)
                                        if potential_date > now:
                                            year = now.year - 1
                                    return datetime(year, month, day)

                # Try standard formats with year
                date_formats = [
                    '%Y-%m-%d',                    # 2025-01-30
                    '%Y/%m/%d',                    # 2025/01/30
                    '%d/%m/%Y',                    # 30/01/2025
                    '%m/%d/%Y',                    # 01/30/2025
                    '%B %d %Y',                    # January 30 2025
                    '%b %d %Y',                    # Jan 30 2025
                    '%d %B %Y',                    # 30 January 2025
                    '%d %b %Y',                    # 30 Jan 2025
                    '%Y%m%d',                      # 20250130
                ]
                
                # Probeer elk format
                for date_format in date_formats:
                    try:
                        return datetime.strptime(cleaned_date, date_format)
                    except ValueError:
                        continue
                
                logger.warning(f"Could not parse date format: {date_str}")
                return None
                
            except Exception as e:
                logger.error(f"Error parsing date {date_str}: {e}")
                return None

    def analyze_competitors(self, ticker: str) -> Dict[str, Any]:
        """Analyze competitor performance and market position."""
        try:
            # Get competitor data from Claude
            competitor_data = {}
            
            # Get financial metrics voor vergelijking
            metrics = {
                'marketCap': 'Market Cap',
                'peRatio': 'P/E Ratio',
                'forwardPE': 'Forward P/E',
                'profitMargins': 'Profit Margin',
                'returnOnEquity': 'ROE',
                'returnOnAssets': 'ROA',
                'revenueGrowth': 'Revenue Growth',
                'operatingMargins': 'Operating Margin',
                'debtToEquity': 'Debt/Equity'
            }
            
            comparison_data = {}
            
            # Hoofdbedrijf metrics
            try:
                main_stock = yf.Ticker(ticker)
                comparison_data[ticker] = {
                    metric_name: getattr(main_stock.info, metric_key, None)
                    for metric_key, metric_name in metrics.items()
                }
                comparison_data[ticker]['name'] = main_stock.info.get('longName', ticker)
            except Exception as e:
                logger.error(f"Error getting data for main company {ticker}: {e}")
            
            return {
                'competitor_info': competitor_data,
                'comparison_data': comparison_data
            }
            
        except Exception as e:
            logger.error(f"Error analyzing competitors: {e}")
            return {
                'competitor_info': {},
                'comparison_data': {}
            }

    @staticmethod
    def timeframe_to_days(timeframe: str) -> int:
        """Convert timeframe string to number of days."""
        mapping = {
            "1d": 1, "3d": 3, "5d": 5, "1w": 7, "2w": 14,
            "1m": 30, "3m": 90, "6m": 180, "1y": 365,
            "2y": 730, "3y": 1095, "5y": 1825
        }
        return mapping.get(timeframe, 90)  # Default to 90 days
    
def create_enhanced_gradio_interface():
    with gr.Blocks(title="Enhanced Financial Event Scanner") as iface:
        # Add state for storing analysis data
        analysis_state = gr.State(value={})
        
        gr.Markdown("## Enhanced Financial Event Scanner")
        gr.Markdown("Comprehensive financial analysis tool for fundamental research")
        
        with gr.Tab("Main Analysis"):
            with gr.Row():
                with gr.Column(scale=2):
                    ticker_input = gr.Textbox(
                        label="Ticker Symbol (e.g., AAPL)",
                        value="AAPL"
                    )
                    timeframe_input = gr.Dropdown(
                        choices=["1d", "3d", "5d", "1w", "2w", "1m", "3m", "6m", "1y", "2y", "3y", "5y"],
                        label="Analysis Timeframe",
                        value="3m"
                    )
                    
                with gr.Column(scale=3):
                    summary_output = gr.Markdown(
                        label="Market Summary",
                        value="Summary will appear here..."
                    )
                    priority_flag = gr.Textbox(
                        label="Priority Level",
                        interactive=False
                    )
            
            with gr.Row():
                events_output = gr.DataFrame(
                    label="Recent Events",
                    headers=["Date", "Event", "Company", "Importance", "Analysis"]
                )
        
        with gr.Tab("Competitor Analysis"):
            with gr.Row():
                competitor_info = gr.JSON(
                    label="Competitor Information"
                )
                comp_metrics = gr.DataFrame(
                    label="Comparative Metrics"
                )
        
        with gr.Tab("News Analysis"):
            with gr.Row():
                news_output = gr.DataFrame(
                    label="Recent News Articles",
                    headers=["Date", "Title", "Summary", "Source", "Sentiment"]
                )
                sentiment_chart = gr.Plot(
                    label="News Sentiment Trend"
                )
        
        with gr.Tab("Management Changes"):
            mgmt_output = gr.DataFrame(
                label="Management Changes",
                headers=["Date", "Title", "Description", "Source", "Importance"]
            )
        
        # Export opties
        with gr.Row():
            export_format = gr.Dropdown(
                choices=["pdf", "excel", "json"],
                label="Export Format",
                value="pdf"
            )
            export_btn = gr.Button("Export Report")
            export_status = gr.Textbox(
                label="Export Status",
                interactive=False
            )
        
        # Advanced filters
        with gr.Accordion("Advanced Filters", open=False):
            with gr.Row():
                event_types = gr.CheckboxGroup(
                    choices=[
                        "Earnings", "Management Changes",
                        "Acquisitions", "Product Launches",
                        "Regulatory News", "Market Updates"
                    ],
                    label="Event Types",
                    value=["Earnings", "Management Changes", "Acquisitions"]
                )
                min_importance = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=50,
                    label="Minimum Importance Score"
                )
        
        # Action buttons
        with gr.Row():
            clear_btn = gr.Button("Clear All")
            submit_btn = gr.Button("Run Analysis", variant="primary")

        def run_analysis(ticker, timeframe, event_filters, min_imp):
            scanner = EnhancedFinancialScanner()
            try:
                # 1. First collect all base news and event data
                logger.info(f"Starting initial data collection for {ticker}")
                news_data = scanner.scrape_financial_news(ticker, timeframe)
                
                # 2. Process events from the news data we already have
                events = []
                for article in news_data:
                    try:
                        parsed_date = scanner.parse_date(article.get('date'))
                        if parsed_date:
                            formatted_date = parsed_date.strftime('%Y-%m-%d')
                            
                            event = {
                                'date': formatted_date,
                                'title': article.get('title', ''),
                                'description': article.get('summary', 'No summary available'),
                                'company': ticker,
                                'type': 'News',
                                'url': article.get('url', ''),
                                'sentiment': article.get('sentiment', {}),
                                'parsed_date': formatted_date
                            }
                            
                            # Check for management changes in same pass
                            mgmt_keywords = ["ceo", "cfo", "president", "chairman", "chief executive", "chief financial"]
                            if any(keyword in article.get('title', '').lower() for keyword in mgmt_keywords):
                                event.update({
                                    'type': 'Management Change',
                                    'importance': 'High'
                                })
                            
                            events.append(event)
                    except Exception as e:
                        logger.error(f"Error processing article into event: {e}")
                        continue

                # 3. Filter and analyze events
                if event_filters:
                    filtered_events = [
                        event for event in events 
                        if event['type'] in event_filters and 
                        float(event.get('importance_score', 0)) >= min_imp
                    ]
                else:
                    filtered_events = events
                    
                # 4. Analyze competitors once
                comp_analysis = scanner.analyze_competitors(ticker)
                
                # 5. Extract management changes from already processed events
                mgmt_changes = [
                    event for event in filtered_events 
                    if event.get('type') == 'Management Change'
                ]
                
                # 6. Create summary using the data we already have
                summary = scanner.create_summary(filtered_events, comp_analysis, news_data)
                
                # 7. Prepare display data
                all_data = {
                    'ticker': ticker,
                    'events': filtered_events,
                    'competitor_analysis': comp_analysis,
                    'news': news_data,
                    'management_changes': mgmt_changes,
                    'summary': summary,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Convert to DataFrames for display
                events_df = pd.DataFrame([{
                    'Date': event.get('date', ''),
                    'Event': event.get('title', ''),
                    'Company': event.get('company', ''),
                    'Importance': event.get('importance', 'Low'),
                    'Analysis': event.get('analysis', '')
                } for event in filtered_events])
                
                mgmt_df = pd.DataFrame([{
                    'Date': change.get('date', ''),
                    'Title': change.get('title', ''),
                    'Description': change.get('description', ''),
                    'Source': change.get('url', ''),
                    'Importance': change.get('importance', 'High')
                } for change in mgmt_changes])
                
                news_df = pd.DataFrame([{
                    'Date': article.get('date', ''),
                    'Title': article.get('title', ''),
                    'Summary': article.get('summary', ''),
                    'Source': article.get('source_domain', ''),
                    'Sentiment': f"Pos: {article.get('sentiment', {}).get('positive', 0):.2f}, Neg: {article.get('sentiment', {}).get('negative', 0):.2f}"
                } for article in news_data])
                
                # Calculate priority flag
                flag_text = (
                    "🔴 High Priority" if any(e.get('importance') == 'High' for e in filtered_events)
                    else "🟡 Medium Priority" if any(e.get('importance') == 'Medium' for e in filtered_events)
                    else "🟢 Low Priority"
                )
                
                return (
                    all_data,
                    events_df,
                    comp_analysis['competitor_info'],
                    pd.DataFrame(comp_analysis['comparison_data']),
                    news_df,
                    create_sentiment_chart(news_data),
                    mgmt_df,
                    summary,
                    flag_text
                )
                
            except Exception as e:
                logger.error(f"Error in analysis: {e}")
                return create_empty_response()  # Return empty dataframes with correct structure

        def create_empty_response():
            """Create empty response with correct structure when analysis fails."""
            empty_events_df = pd.DataFrame(columns=['Date', 'Event', 'Company', 'Importance', 'Analysis'])
            empty_news_df = pd.DataFrame(columns=['Date', 'Title', 'Summary', 'Source', 'Sentiment'])
            empty_mgmt_df = pd.DataFrame(columns=['Date', 'Title', 'Description', 'Source', 'Importance'])
            empty_comp_df = pd.DataFrame(columns=['Metric', 'Value'])
            
            return (
                {},
                empty_events_df,
                {},
                empty_comp_df,
                empty_news_df,
                None,
                empty_mgmt_df,
                "Error during analysis. Please try again.",
                "⚠️ Analysis Error"
            )

        def create_sentiment_chart(news_data):
                    if not news_data:
                        return None
                        
                    try:
                        df = pd.DataFrame(news_data)
                        
                        # Create one scanner instance for all date parsing
                        scanner = EnhancedFinancialScanner()
                        
                        # Parse dates met onze eigen parse_date functie
                        df['parsed_date'] = df['date'].apply(lambda x: scanner.parse_date(x) if x else pd.NaT)
                        df = df.dropna(subset=['parsed_date'])  # Remove rows where date parsing failed
                        df = df.sort_values('parsed_date')
                        
                        if df.empty:
                            logger.warning("No valid dates found for sentiment chart")
                            return None

                        # Extract sentiment values safely
                        def get_sentiment_value(row, sentiment_type):
                            try:
                                return row['sentiment'].get(sentiment_type, 0.0)
                            except (AttributeError, KeyError, TypeError):
                                return 0.0
                        
                        fig = go.Figure()
                        
                        # Add sentiment traces with safe value extraction
                        fig.add_trace(go.Scatter(
                            x=df['parsed_date'],
                            y=[get_sentiment_value(row, 'positive') for _, row in df.iterrows()],
                            name='Positive',
                            line=dict(color='green')
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=df['parsed_date'],
                            y=[get_sentiment_value(row, 'negative') for _, row in df.iterrows()],
                            name='Negative',
                            line=dict(color='red')
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            title='News Sentiment Trend',
                            xaxis_title='Date',
                            yaxis_title='Sentiment Score',
                            template='plotly_white'
                        )
                        
                        return fig
                        
                    except Exception as e:
                        logger.error(f"Error creating sentiment chart: {e}")
                        return None

        def export_report(data, format_type):
            try:
                filename = f"financial_analysis_{data.get('ticker', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                if format_type == 'pdf':
                    filename = f"{filename}.pdf"
                    # Add PDF export logic here
                elif format_type == 'excel':
                    filename = f"{filename}.xlsx"
                    # Add Excel export logic here
                else:  # json
                    filename = f"{filename}.json"
                    with open(filename, 'w') as f:
                        json.dump(data, f, indent=2)
                
                return f"Report exported successfully: {filename}"
            except Exception as e:
                logger.error(f"Error exporting report: {e}")
                return f"Error exporting report: {str(e)}"

        def clear_all():
            empty_data = {
                'events_df': None,
                'competitor_info': None,
                'comp_metrics': None,
                'news_df': None,
                'sentiment_fig': None,
                'mgmt_df': None,
                'summary': "Summary will appear here...",
                'flag_text': ""
            }
            return (
                {},  # Clear state
                *[empty_data[k] for k in empty_data.keys()]
            )

        # Wire up the event handlers
        submit_btn.click(
            fn=run_analysis,
            inputs=[
                ticker_input, timeframe_input,
                event_types, min_importance
            ],
            outputs=[
                analysis_state,
                events_output,
                competitor_info,
                comp_metrics,
                news_output,
                sentiment_chart,
                mgmt_output,
                summary_output,
                priority_flag
            ]
        )
        
        export_btn.click(
            fn=export_report,
            inputs=[analysis_state, export_format],
            outputs=[export_status]
        )
        
        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[
                analysis_state,
                events_output,
                competitor_info,
                comp_metrics,
                news_output,
                sentiment_chart,
                mgmt_output,
                summary_output,
                priority_flag
            ]
        )
        
    return iface

if __name__ == "__main__":
    iface = create_enhanced_gradio_interface()
    iface.launch(share=True)
