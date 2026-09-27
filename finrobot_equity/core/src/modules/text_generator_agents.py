#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from typing import Dict, Optional
from openai import OpenAI

from modules.retail_sentiment_client import format_retail_sentiment_for_prompt


# Output-token budget handed to the chat-completions endpoint.
# Reasoning models count their hidden reasoning against this same budget, so it
# needs headroom well beyond the visible answer. 1000 used to be enough for
# non-reasoning models but is fully consumed by reasoning on e.g. DeepSeek,
# which then returns an empty message with finish_reason="length".
DEFAULT_MAX_TOKENS = 16000
# Upper bound for the automatic retry that kicks in when reasoning eats the
# whole budget before any answer text is produced.
MAX_TOKEN_CEILING = 32000


def _get_fallback_text(prompt_type: str, company_name: str) -> str:
    """Returns fallback text when agent generation fails."""
    fallbacks = {
        "tagline": f"{company_name} demonstrates strong financial fundamentals with consistent revenue growth and solid profitability metrics. The company maintains a competitive position in its market segment through operational efficiency and strategic initiatives. Strong balance sheet metrics support continued value creation for shareholders.",
        "company_overview": f"{company_name} operates as a prominent player in its industry sector, demonstrating consistent financial performance through strategic market positioning and operational excellence. The company has shown resilient growth patterns supported by strong demand dynamics and effective cost management strategies.",
        "investment_overview": f"{company_name} has delivered solid financial performance in recent periods, supported by strong operational execution and favorable market conditions. Revenue growth has been driven by robust demand and strategic initiatives, while margin improvements reflect operational efficiency gains.",
        "valuation_overview": f"{company_name} trades at reasonable valuation levels relative to its peer group, supported by strong fundamental metrics and growth prospects. The company's financial profile demonstrates consistent profitability and cash generation capabilities.",
        "risks": "Key risks include: (1) Industry competition and market share pressure, (2) Regulatory changes affecting operations, (3) Economic downturns impacting demand, (4) Technology disruption risks, (5) Supply chain and operational challenges.",
        "competitor_analysis": f"{company_name} demonstrates competitive positioning within its industry through consistent financial performance and strategic market positioning relative to key competitors in the sector.",
        "major_takeaways": f"Revenue Growth: {company_name}'s revenue growth shows consistent performance trends.\n\nGross Profit Margin: {company_name}'s gross profit margins demonstrate operational effectiveness.\n\nSG&A Expense Margin: {company_name}'s SG&A expense management shows disciplined cost control.\n\nEBITDA Margin Stability: {company_name}'s EBITDA margin stability reflects strong underlying fundamentals.",
        "news_summary": f"Recent news coverage for {company_name} reflects ongoing market interest and developments in the company's operations and strategic initiatives."
    }
    return fallbacks.get(prompt_type, f"{company_name} analysis for {prompt_type.replace('_', ' ')} section.")


# System prompts for each text section
SYSTEM_PROMPTS = {
    "tagline": "You are an equity research analyst. Create a 3-sentence professional tagline summarizing the company's financial position. Be concise and professional. Do not use markdown.",
    "company_overview": "You are a financial analyst. Write a comprehensive company overview (300-400 words) covering business model, products/services, market position, and recent performance. Use plain text, no markdown.",
    "investment_overview": "You are an investment analyst. Write an investment update (200-300 words) covering recent financial performance, growth drivers, and outlook. Use plain text, no markdown.",
    "valuation_overview": "You are a valuation analyst. Write a valuation analysis (200-300 words) covering current valuation metrics, peer comparison, and fair value assessment. Use plain text, no markdown.",
    "risks": "You are a risk analyst. List 5 key investment risks in bullet point format. Be specific and concise.",
    "competitor_analysis": "You are a competitive analyst. Write a competitor analysis (200-300 words) comparing the company to its peers. Use plain text, no markdown.",
    "major_takeaways": "You are a financial analyst. Provide 4 major takeaways covering: Revenue Growth, Gross Profit Margin, SG&A Expense Margin, and EBITDA Margin. Format each with a header followed by 1-2 sentences.",
    "news_summary": "You are a financial news analyst. Summarize the recent news (200-300 words) highlighting key developments and their investment implications. Use plain text, no markdown."
}


def _df_to_string(df: Optional[pd.DataFrame], name: str) -> str:
    """Converts a DataFrame to a markdown string for use in a prompt."""
    if df is None or df.empty:
        return f"{name}:\n[Data not available]\n"
    
    try:
        return f"{name}:\n{df.to_markdown()}\n"
    except Exception as e:
        return f"{name}:\n[Error formatting data: {e}]\n"


def _prepare_user_prompt(data: Dict, prompt_type: str, company_name: str, company_ticker: str) -> str:
    """Prepare user prompt with financial data."""
    financial_metrics = data.get('financial_metrics')
    peer_ebitda = data.get('peer_ebitda')
    peer_ev_ebitda = data.get('peer_ev_ebitda')
    company_news = data.get('company_news')
    retail_sentiment = data.get('retail_sentiment')
    
    prompt = f"Company: {company_name} ({company_ticker})\n\n"
    
    if financial_metrics is not None and not financial_metrics.empty:
        prompt += _df_to_string(financial_metrics, "Financial Metrics")
    
    if peer_ebitda is not None and not peer_ebitda.empty:
        prompt += _df_to_string(peer_ebitda, "Peer EBITDA Comparison")
        
    if peer_ev_ebitda is not None and not peer_ev_ebitda.empty:
        prompt += _df_to_string(peer_ev_ebitda, "Peer EV/EBITDA Comparison")
    
    if prompt_type == "news_summary" and company_news:
        prompt += f"\n## Recent News:\n"
        for i, article in enumerate(company_news[:10], 1):  # Limit to 10 articles
            prompt += f"{i}. {article.get('title', 'N/A')} ({article.get('publishedDate', 'N/A')[:10]})\n"
            prompt += f"   {article.get('text', 'N/A')[:200]}...\n\n"

    if prompt_type == "news_summary" and retail_sentiment:
        prompt += "\n" + format_retail_sentiment_for_prompt(retail_sentiment) + "\n"

    prompt += f"\nPlease provide the {prompt_type.replace('_', ' ')} based on the above data."
    return prompt


def generate_text_section(data: Dict, prompt_type: str, api_key: str, company_name: str, company_ticker: str,
                          base_url: str = None, model: str = None,
                          reasoning_effort: str = None, max_tokens: int = None) -> str:
    """
    Generates a specific text section for the equity report using OpenAI Chat API.
    
    Args:
        data: Financial data dictionary
        prompt_type: Type of text section to generate
        api_key: OpenAI API key
        company_name: Company name
        company_ticker: Stock ticker
        base_url: Optional API base URL (for proxy services like SiliconFlow)
        model: Optional model name (default: gpt-4o-mini or configured model)
        reasoning_effort: Optional reasoning budget hint ("none"/"low"/...).
            Only sent when set, because most providers reject unknown values.
        max_tokens: Optional output-token budget (default: DEFAULT_MAX_TOKENS)
    """
    
    print(f"🤖 Generating '{prompt_type}' text section...")
    
    # Validate API key
    if not api_key:
        print(f"⚠️ Warning: No API key provided. Using fallback text for '{prompt_type}'.")
        return _get_fallback_text(prompt_type, company_name)
    
    # Determine model to use
    default_model = "gpt-4o-mini"
    if model:
        default_model = model
    
    # Create OpenAI client
    try:
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
            print(f"📡 Using API base URL: {base_url}")
        
        client = OpenAI(**client_kwargs)
        print(f"🤖 Using model: {default_model}")
    except Exception as e:
        print(f"⚠️ Warning: Could not create OpenAI client: {e}")
        return _get_fallback_text(prompt_type, company_name)
    
    # Get system prompt
    system_prompt = SYSTEM_PROMPTS.get(prompt_type, f"You are a financial analyst. Provide {prompt_type.replace('_', ' ')} analysis.")
    
    # Prepare user prompt with data
    user_prompt = _prepare_user_prompt(data, prompt_type, company_name, company_ticker)
    
    # Call the chat-completions endpoint
    def _request(token_budget: int):
        kwargs = {
            "model": default_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": token_budget
        }
        # Only sent when configured: providers differ in which values they accept.
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        return client.chat.completions.create(**kwargs)

    try:
        budget = max_tokens or DEFAULT_MAX_TOKENS
        response = _request(budget)
        choice = response.choices[0]
        generated_text = (choice.message.content or "").strip()

        # Reasoning models can spend the entire budget on hidden reasoning and
        # return no answer text at all. Retry once with a bigger budget rather
        # than silently degrading to the static fallback text.
        if not generated_text and choice.finish_reason == "length" and budget < MAX_TOKEN_CEILING:
            retry_budget = MAX_TOKEN_CEILING
            print(f"🔁 '{prompt_type}': no answer text (finish_reason=length, {budget}-token budget "
                  f"consumed by reasoning), retrying with {retry_budget}")
            response = _request(retry_budget)
            choice = response.choices[0]
            generated_text = (choice.message.content or "").strip()

        if generated_text:
            print(f"✅ Successfully generated '{prompt_type}' ({len(generated_text)} chars)")
            return generated_text
        else:
            print(f"⚠️ Warning: Empty response for '{prompt_type}' (finish_reason={choice.finish_reason})")
            return _get_fallback_text(prompt_type, company_name)

    except Exception as e:
        print(f"❌ Error generating '{prompt_type}': {e}")
        return _get_fallback_text(prompt_type, company_name)

# Backward compatibility - keep old function signature
def _query_openai(prompt: str, api_key: str) -> str:
    """Legacy function for backward compatibility."""
    return "Text generation now handled by agents."

if __name__ == '__main__':
    print("Testing agent-based text_generator...")
