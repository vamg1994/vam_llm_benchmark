"""
Main application file for the AI Model Comparison Platform.
This Streamlit application allows users to:
1. Submit prompts to multiple AI models
2. Compare responses anonymously
3. Vote for the best response
4. View analytics on model performance
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import pytz
import logging
from typing import Dict, List, Optional
from database import engine, SessionLocal, Base, verify_database_connection # Added import
from models import Vote
from utils import get_responses, shuffle_responses, calculate_leaderboard

# Configure logging with detailed format for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize database tables
def init_db():
    """Initialize database tables with proper transaction handling"""
    from database import verify_database_connection, Base, engine

    # First verify database connection
    if not verify_database_connection():
        logger.error("Database connection verification failed")
        return False

    try:
        logger.info("Attempting to create database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False

# Initialize database before starting the application
if not init_db():
    st.error("Database connection failed. Please check your configuration.")
    st.stop()

def save_vote(input_text: str, outputs: Dict[str, str], winner: str) -> None:
    """
    Save a user's vote to the database.

    Args:
        input_text: The original prompt text
        outputs: Dictionary containing each model's response
        winner: Name of the winning model
    """
    logger.info(f"Attempting to save vote for winner: {winner}")
    db = SessionLocal()
    try:
        # Create new vote record
        vote = Vote(
            input_text=input_text,
            output_chatgpt=outputs['ChatGPT 4.0'],
            output_claude=outputs['Claude 3.5 Sonnet'],
            output_deepseek=outputs['DeepSeek v3'],
            winner=winner
        )
        logger.info("Vote object created, attempting to save to database...")
        db.add(vote)
        db.commit()
        logger.info("Vote successfully saved to database")
    except Exception as e:
        logger.error(f"Failed to save vote: {str(e)}")
        db.rollback()
        st.error("Failed to save your vote. Please try again.")
    finally:
        db.close()

@st.cache_data(ttl=300)  # Cache for 5 minutes to improve performance
def get_analytics_data() -> pd.DataFrame:
    """
    Fetch and process voting data for analytics.

    Returns:
        DataFrame containing processed voting data
    """
    logger.info("Fetching analytics data from database")
    db = SessionLocal()
    try:
        # Query all votes from database
        votes = db.query(Vote).all()
        # Process vote data for analysis
        data = [{
            'timestamp': vote.created_at,
            'winner': vote.winner,
            'input_length': len(vote.input_text),
            'prompt': vote.input_text[:100] + '...' if len(vote.input_text) > 100 else vote.input_text
        } for vote in votes]
        df = pd.DataFrame(data)
        logger.info(f"Retrieved {len(data)} vote records")
        return df
    except Exception as e:
        logger.error(f"Error fetching analytics data: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error
    finally:
        db.close()

def get_leaderboard() -> Optional[Dict[str, int]]:
    """
    Calculate and return current model rankings.

    Returns:
        Dictionary of model names and their vote counts
    """
    logger.info("Fetching leaderboard data...")
    db = None
    try:
        db = SessionLocal()
        votes = db.query(Vote).all()
        logger.info(f"Retrieved {len(votes)} votes from database")
        standings = calculate_leaderboard(votes)
        logger.info(f"Current standings: {standings}")
        return standings
    except Exception as e:
        logger.error(f"Error fetching leaderboard data: {str(e)}")
        return None
    finally:
        if db:
            db.close()

def show_chat_tab() -> None:
    """
    Display the main chat interface for model comparison.
    Handles user input, model responses, and voting functionality.
    """
    st.markdown("Compare AI models anonymously and vote for the best response!")

    # Initialize session state for managing UI state
    if 'current_responses' not in st.session_state:
        st.session_state.current_responses = None
    if 'current_mapping' not in st.session_state:
        st.session_state.current_mapping = None
    if 'has_voted' not in st.session_state:
        st.session_state.has_voted = False

    # Chat interface
    prompt = st.text_area("Enter your prompt:", key="prompt_input", 
                         help="Type your question or prompt here to get responses from all models")

    if st.button("Generate Responses", type="primary"):
        if prompt:
            with st.spinner("Generating responses from all models..."):
                try:
                    # Get and shuffle responses for anonymous comparison
                    responses = get_responses(prompt)
                    shuffled_outputs, mapping = shuffle_responses(responses)

                    # Store in session state for persistence
                    st.session_state.current_responses = responses
                    st.session_state.current_mapping = mapping
                    st.session_state.shuffled_outputs = shuffled_outputs
                    st.session_state.has_voted = False

                    st.rerun()  # Update UI with new responses
                except Exception as e:
                    st.error(f"Error generating responses: {str(e)}")
                    logger.error(f"Error in response generation: {str(e)}")

    # Display responses and voting interface
    if hasattr(st.session_state, 'shuffled_outputs') and not st.session_state.has_voted:
        st.markdown("### 🤖 Anonymous Model Responses")
        st.info("Responses are randomly ordered to ensure unbiased voting")

        for i, response in enumerate(st.session_state.shuffled_outputs):
            with st.container():
                st.markdown(f"<div class='model-response'>{response}</div>", 
                          unsafe_allow_html=True)
                if st.button(f"Vote for Response {i + 1}", 
                           key=f"vote_{i}",
                           help="Click to vote for this response"):
                    winner = st.session_state.current_mapping[i]
                    save_vote(prompt, st.session_state.current_responses, winner)
                    st.session_state.has_voted = True
                    st.rerun()

def show_analytics_tab() -> None:
    """
    Display analytics dashboard with voting statistics and trends.
    Includes visualizations for model performance and recent activity.
    """
    st.markdown("""
    ## 📊 Analytics Dashboard
    This dashboard provides detailed insights into model performance and voting patterns.
    Updates every 5 minutes with the latest data.
    """)

    try:
        # Fetch and process analytics data
        df = get_analytics_data()

        if not df.empty:
            # Calculate key metrics
            total_votes = len(df)
            now_utc = datetime.now(pytz.UTC)
            yesterday_utc = now_utc - timedelta(days=1)

            # Ensure timestamps are in UTC
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
            else:
                df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')

            recent_votes = len(df[df['timestamp'] > yesterday_utc])

            # Display metrics in columns
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                st.metric("📝 Total Votes", total_votes)
            with metrics_col2:
                st.metric("⚡ Recent Votes (24h)", recent_votes)
            with metrics_col3:
                if total_votes > 0:
                    leading_model = df['winner'].mode()[0]
                    st.metric("🏆 Current Leader", leading_model)

            # Create visualizations
            col1, col2 = st.columns(2)

            with col1:
                # Vote distribution pie chart
                st.subheader("Vote Distribution")
                fig_pie = px.pie(
                    df,
                    names='winner',
                    title='Total Votes by Model',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                # Voting trends line chart
                st.subheader("Voting Trends")
                df_time = df.set_index('timestamp').resample('D')['winner'].count().reset_index()
                fig_line = px.line(
                    df_time,
                    x='timestamp',
                    y='winner',
                    title='Daily Vote Count',
                    labels={'winner': 'Number of Votes', 'timestamp': 'Date'}
                )
                st.plotly_chart(fig_line, use_container_width=True)

            # Model performance timeline
            st.subheader("Model Performance Timeline")
            df_model_time = df.set_index('timestamp').resample('D')['winner'].value_counts().unstack().fillna(0)
            fig_area = px.area(
                df_model_time,
                title='Daily Votes by Model',
                labels={'value': 'Number of Votes', 'index': 'Date'}
            )
            st.plotly_chart(fig_area, use_container_width=True)

            # Recent activity table
            st.subheader("Recent Activity")
            recent_prompts = df.sort_values('timestamp', ascending=False).head(10)
            st.dataframe(
                recent_prompts[['timestamp', 'prompt', 'winner']],
                column_config={
                    "timestamp": st.column_config.DatetimeColumn("Time", format="D MMM YYYY, HH:mm"),
                    "prompt": "Prompt",
                    "winner": "Winning Model"
                },
                hide_index=True
            )

        else:
            st.info("No voting data available yet. Start comparing models to see analytics!")

    except Exception as e:
        logger.error(f"Error in analytics dashboard: {e}")
        st.error("An error occurred while loading the analytics dashboard. Please try again later.")

def main():
    """
    Main application entry point.
    Sets up the Streamlit interface and manages navigation.
    """
    # Configure page settings
    st.set_page_config(
        page_title="ChatBot Arena",
        page_icon="🤖",
        layout="wide"
    )

    # Custom CSS for better UI
    st.markdown("""
        <style>
        .stButton > button {
            width: 100%;
            border-radius: 5px;
            background-color: #4CAF50;
            color: white;
        }
        .model-response {
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            border: 1px solid #e0e0e0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Application header
    st.title("🤖 VAM mini-LLM Arena")
    st.markdown("### Compare and vote for the best AI responses from gpt-4o, claude-3.5-sonnet, and deepseek-v3")

    # Create tabs for navigation
    chat_tab, analytics_tab = st.tabs(["💬 Chat", "📊 Analytics"])

    with chat_tab:
        show_chat_tab()

    with analytics_tab:
        show_analytics_tab()

    # Add sidebar with leaderboard
    with st.sidebar:
        
        st.markdown("### Virgilio Madrid - Data Scientist")
        st.markdown("#### virgiliomadrid1994@gmail.com")
        st.markdown("#### https://www.linkedin.com/in/vamadrid/")
        st.markdown("#### https://portfolio-vam.vercel.app/")
        st.subheader("🏆 Current Rankings")
        standings = get_leaderboard()
        if standings:
            for model, votes in standings.items():
                st.metric(model, f"{votes} votes")
        else:
            st.info("No votes recorded yet")

if __name__ == "__main__":
    main()