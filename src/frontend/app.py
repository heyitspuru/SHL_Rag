"""
Streamlit web interface for SHL Assessment Recommender
"""

import streamlit as st
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.rag.embeddings import AssessmentEmbedder
from src.rag.vectorstore import VectorStoreManager
from src.rag.retriever import AssessmentRetriever
from src.rag.recommender import SHLRecommendationEngine

# Configure page
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .recommendation-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_recommendation_engine():
    """Load and cache the recommendation engine"""
    with st.spinner("Loading recommendation engine..."):
        # Load catalog
        catalog_df = pd.read_csv('data/raw/shl_assessments.csv')
        
        # Initialize embeddings
        embedder = AssessmentEmbedder(model_name="all-MiniLM-L6-v2")
        
        # Load vector store
        vs_manager = VectorStoreManager(embedder.embeddings)
        try:
            vectorstore = vs_manager.load_vectorstore(use_chroma=True)
        except:
            st.error("Vector store not found. Please run build_vectorstore.py first.")
            return None, None
        
        # Create retriever
        retriever = AssessmentRetriever(vectorstore)
        
        # Create recommendation engine
        engine = SHLRecommendationEngine(
            retriever=retriever,
            llm=None,
            catalog_df=catalog_df
        )
        
        return engine, catalog_df


# Load engine
engine, catalog_df = load_recommendation_engine()

# Header
st.markdown('<div class="main-header">🎯 SHL Assessment Recommendation System</div>', unsafe_allow_html=True)
st.markdown("Get intelligent assessment recommendations based on your job requirements")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    num_recommendations = st.slider(
        "Number of Recommendations",
        min_value=1,
        max_value=10,
        value=10,
        help="Select how many assessments to recommend"
    )
    
    enable_balancing = st.checkbox(
        "Enable Test Type Balancing",
        value=True,
        help="Balance recommendations across different test types (K, C, P, S)"
    )
    
    st.markdown("---")
    st.markdown("""
    ### About
    This system uses Retrieval-Augmented Generation (RAG) to recommend relevant SHL assessments.
    
    **Test Types:**
    - **K**: Knowledge/Technical Skills
    - **C**: Cognitive/Aptitude
    - **P**: Personality/Behavioral
    - **S**: Situational Judgment
    """)
    
    if catalog_df is not None:
        st.markdown(f"**Total Assessments:** {len(catalog_df)}")

# Main interface
tab1, tab2, tab3 = st.tabs(["🔍 Recommend", "📊 Batch Process", "📈 Catalog"])

with tab1:
    st.header("Enter Job Requirements")
    
    # Query input
    query = st.text_area(
        "Job Description or Requirements",
        placeholder="Example: I am hiring for Java developers who can also collaborate well in teams...",
        height=150,
        help="Enter a job description, required skills, or any natural language query"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        recommend_button = st.button("🎯 Get Recommendations", type="primary")
    
    with col2:
        clear_button = st.button("🗑️ Clear")
    
    if clear_button:
        st.rerun()
    
    if recommend_button and query:
        if engine is None:
            st.error("Recommendation engine not available. Please check setup.")
        else:
            with st.spinner("Generating recommendations..."):
                try:
                    recommendations = engine.recommend(
                        query=query,
                        top_k=num_recommendations,
                        enable_balancing=enable_balancing
                    )
                    
                    st.success(f"✅ Found {len(recommendations)} recommendations!")
                    
                    # Display recommendations
                    st.markdown("### 📋 Recommended Assessments")
                    
                    for i, rec in enumerate(recommendations, 1):
                        with st.container():
                            col_a, col_b = st.columns([4, 1])
                            
                            with col_a:
                                st.markdown(f"**{i}. {rec['assessment_name']}**")
                                st.markdown(f"*{rec['category']} | Duration: {rec['duration']}*")
                                st.markdown(f"[🔗 View Assessment]({rec['assessment_url']})")
                            
                            with col_b:
                                test_type = rec['test_type']
                                type_colors = {
                                    'K': '🟦',  # Blue for Knowledge
                                    'C': '🟩',  # Green for Cognitive
                                    'P': '🟨',  # Yellow for Personality
                                    'S': '🟧'   # Orange for Situational
                                }
                                st.markdown(f"### {type_colors.get(test_type, '⬜')} {test_type}")
                            
                            st.divider()
                    
                    # Show type distribution
                    type_dist = {}
                    for rec in recommendations:
                        t = rec['test_type']
                        type_dist[t] = type_dist.get(t, 0) + 1
                    
                    st.markdown("### 📊 Test Type Distribution")
                    dist_df = pd.DataFrame(list(type_dist.items()), columns=['Test Type', 'Count'])
                    st.bar_chart(dist_df.set_index('Test Type'))
                    
                    # Download option
                    results_df = pd.DataFrame(recommendations)
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="shl_recommendations.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error generating recommendations: {e}")
    
    elif recommend_button:
        st.warning("⚠️ Please enter a job description or query")

with tab2:
    st.header("Batch Processing")
    st.markdown("Upload a CSV file with multiple queries for batch processing")
    
    uploaded_file = st.file_uploader(
        "Upload CSV with 'Query' column",
        type=['csv'],
        help="CSV file should have a 'Query' column with job descriptions"
    )
    
    if uploaded_file is not None:
        queries_df = pd.read_csv(uploaded_file)
        
        if 'Query' not in queries_df.columns:
            st.error("❌ CSV must have a 'Query' column")
        else:
            st.success(f"✅ Loaded {len(queries_df)} queries")
            st.dataframe(queries_df.head())
            
            if st.button("🚀 Process All Queries"):
                if engine is None:
                    st.error("Recommendation engine not available.")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    all_results = []
                    
                    for i, row in queries_df.iterrows():
                        query = row['Query']
                        status_text.text(f"Processing {i+1}/{len(queries_df)}: {query[:50]}...")
                        
                        try:
                            recommendations = engine.recommend(query, top_k=num_recommendations)
                            
                            for rec in recommendations:
                                all_results.append({
                                    'Query': query,
                                    'Assessment_url': rec['assessment_url']
                                })
                        
                        except Exception as e:
                            st.warning(f"Failed for query {i+1}: {e}")
                        
                        progress_bar.progress((i + 1) / len(queries_df))
                    
                    status_text.text("✅ Processing complete!")
                    
                    results_df = pd.DataFrame(all_results)
                    st.dataframe(results_df)
                    
                    # Download
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Batch Results",
                        data=csv,
                        file_name="batch_recommendations.csv",
                        mime="text/csv"
                    )

with tab3:
    st.header("Assessment Catalog")
    
    if catalog_df is not None:
        # Filters
        col1, col2 = st.columns(2)
        
        with col1:
            selected_types = st.multiselect(
                "Filter by Test Type",
                options=catalog_df['test_type'].unique(),
                default=catalog_df['test_type'].unique()
            )
        
        with col2:
            selected_categories = st.multiselect(
                "Filter by Category",
                options=catalog_df['category'].unique(),
                default=catalog_df['category'].unique()
            )
        
        # Filter data
        filtered_df = catalog_df[
            (catalog_df['test_type'].isin(selected_types)) &
            (catalog_df['category'].isin(selected_categories))
        ]
        
        st.markdown(f"**Showing {len(filtered_df)} of {len(catalog_df)} assessments**")
        
        # Display catalog
        st.dataframe(
            filtered_df[['name', 'test_type', 'category', 'duration']],
            use_container_width=True,
            height=400
        )
        
        # Statistics
        st.markdown("### 📊 Catalog Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Test Type Distribution**")
            type_counts = filtered_df['test_type'].value_counts()
            st.bar_chart(type_counts)
        
        with col2:
            st.markdown("**Category Distribution**")
            cat_counts = filtered_df['category'].value_counts().head(10)
            st.bar_chart(cat_counts)
    else:
        st.error("Catalog not loaded")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with ❤️ using LangChain, ChromaDB, and Streamlit | "
    "<a href='https://github.com/yourusername/shl-recommender'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True
)
