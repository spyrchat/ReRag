"""
Streamlit GUI for ReRag Agent
Minimal code, reuses existing architecture
"""

import streamlit as st
from datetime import datetime
import json

# Configure page
st.set_page_config(
    page_title="ReRag Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = "standard"
if "graph" not in st.session_state:
    st.session_state.graph = None


@st.cache_resource
def load_graph(mode: str):
    """Load the agent graph (cached to avoid reloading)"""
    if mode == "self-rag":
        from agent.graph_self_rag import graph
    else:
        from agent.graph_refined import graph
    return graph


# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Mode selection
    mode = st.radio(
        "Agent Mode",
        ["standard", "self-rag"],
        help="Standard: Fast RAG | Self-RAG: Iterative refinement with verification"
    )
    
    if mode != st.session_state.mode:
        st.session_state.mode = mode
        st.session_state.graph = None  # Force reload
    
    st.divider()
    
    # Retrieval Configuration
    st.subheader("🔍 Retrieval Settings")
    
    # Quick presets
    preset = st.selectbox(
        "Quick Preset",
        ["Custom", "Fast & Light", "Balanced", "High Quality", "Maximum Recall"],
        help="Pre-configured retrieval settings"
    )
    
    # Apply preset defaults
    if preset == "Fast & Light":
        default_pipeline = "pipelines/configs/retrieval/fast_dense_bge_m3.yml"
        default_top_k = 5
        default_threshold = 0.3
        default_rerank = False
        default_rerank_k = 3
    elif preset == "Balanced":
        default_pipeline = "pipelines/configs/retrieval/dense_bge_m3.yml"
        default_top_k = 10
        default_threshold = 0.0
        default_rerank = True
        default_rerank_k = 5
    elif preset == "High Quality":
        default_pipeline = "pipelines/configs/retrieval/hybrid_optimal.yml"
        default_top_k = 20
        default_threshold = 0.0
        default_rerank = True
        default_rerank_k = 5
    elif preset == "Maximum Recall":
        default_pipeline = "pipelines/configs/retrieval/hybrid_optimal.yml"
        default_top_k = 50
        default_threshold = 0.0
        default_rerank = True
        default_rerank_k = 10
    else:  # Custom
        default_pipeline = "pipelines/configs/retrieval/hybrid_optimal.yml"
        default_top_k = 10
        default_threshold = 0.0
        default_rerank = True
        default_rerank_k = 5
    
    retrieval_config_path = st.selectbox(
        "Retrieval Pipeline",
        [
            "pipelines/configs/retrieval/hybrid_optimal.yml",
            "pipelines/configs/retrieval/dense_bge_m3.yml",
            "pipelines/configs/retrieval/fast_dense_bge_m3.yml"
        ],
        index=[
            "pipelines/configs/retrieval/hybrid_optimal.yml",
            "pipelines/configs/retrieval/dense_bge_m3.yml",
            "pipelines/configs/retrieval/fast_dense_bge_m3.yml"
        ].index(default_pipeline),
        help="Select the retrieval pipeline configuration"
    )
    
    top_k = st.slider(
        "Top-K Results",
        min_value=1,
        max_value=50,
        value=default_top_k,
        help="Number of documents to retrieve"
    )
    
    # Advanced Retrieval Options
    with st.expander("🔧 Advanced Retrieval"):
        col1, col2 = st.columns(2)
        
        with col1:
            score_threshold = st.slider(
                "Score Threshold",
                min_value=0.0,
                max_value=1.0,
                value=default_threshold,
                step=0.05,
                help="Minimum relevance score (0 = no filtering)"
            )
            
            enable_reranking = st.checkbox(
                "Enable Reranking",
                value=default_rerank,
                help="Use cross-encoder for result reranking"
            )
        
        with col2:
            reranker_top_k = st.slider(
                "Reranker Top-K",
                min_value=1,
                max_value=20,
                value=default_rerank_k,
                help="Number of results after reranking"
            )
            
            # Hybrid retrieval alpha (if using hybrid)
            if "hybrid" in retrieval_config_path:
                fusion_alpha = st.slider(
                    "Dense/Sparse Balance (α)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.8,
                    step=0.1,
                    help="0.0=Pure Sparse, 1.0=Pure Dense"
                )
    
    st.divider()
    
    # Display options
    st.subheader("👁️ Display Options")
    show_reasoning = st.checkbox("Show reasoning steps", value=True)
    show_metadata = st.checkbox("Show retrieval metadata", value=False)
    show_documents = st.checkbox("Show retrieved documents", value=False)
    
    st.divider()
    
    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Export conversation
    if st.session_state.messages:
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "mode": st.session_state.mode,
            "messages": st.session_state.messages
        }
        export_json = json.dumps(export_data, indent=2, default=str)
        
        st.download_button(
            label="💾 Export Conversation",
            data=export_json,
            file_name=f"rerag_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    st.divider()
    st.caption(f"Mode: {mode.upper()}")
    st.caption(f"Pipeline: {retrieval_config_path.split('/')[-1]}")
    st.caption(f"Time: {datetime.now().strftime('%H:%M:%S')}")


# Main chat interface
st.title("🤖 ReRag Agent")
st.caption("Retrieval-Augmented Generation with Advanced RAG Techniques")

# Show active configuration
with st.expander("📊 Active Configuration", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Agent Mode", mode.upper())
        st.metric("Top-K", top_k)
    with col2:
        st.metric("Score Threshold", f"{score_threshold:.2f}")
        st.metric("Reranking", "✓ Enabled" if enable_reranking else "✗ Disabled")
    with col3:
        pipeline_name = retrieval_config_path.split('/')[-1].replace('.yml', '')
        st.metric("Pipeline", pipeline_name)
        if "hybrid" in retrieval_config_path:
            st.metric("Dense/Sparse (α)", f"{fusion_alpha:.1f}")

st.divider()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Show settings badge for assistant messages
        if msg["role"] == "assistant" and "settings" in msg:
            settings = msg["settings"]
            badge_parts = [
                f"🔧 {settings['mode']}",
                f"📊 k={settings['top_k']}",
                f"🎯 {settings['pipeline'].replace('.yml', '')}"
            ]
            if settings.get("reranking"):
                badge_parts.append(f"🔄 rerank={settings['reranker_top_k']}")
            if settings.get("fusion_alpha") is not None:
                badge_parts.append(f"⚖️ α={settings['fusion_alpha']:.1f}")
            
            st.caption(" | ".join(badge_parts))
        
        # Show reasoning if available
        if "reasoning" in msg and show_reasoning:
            with st.expander("🔍 Reasoning Steps"):
                for key, value in msg["reasoning"].items():
                    st.markdown(f"**{key}:**")
                    if isinstance(value, dict):
                        st.json(value)
                    else:
                        st.text(value)
        
        # Show documents if available
        if "documents" in msg and show_documents:
            with st.expander(f"📄 Retrieved Documents ({len(msg['documents'])})"):
                for i, doc in enumerate(msg["documents"], 1):
                    st.markdown(f"**Document {i}** (Score: {doc.metadata.get('score', 'N/A'):.3f})")
                    st.text(doc.page_content[:300] + "...")
                    if show_metadata:
                        st.json(doc.metadata)
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process with agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Load graph
            if st.session_state.graph is None:
                st.session_state.graph = load_graph(st.session_state.mode)
            
            graph = st.session_state.graph
            
            # Build dynamic retrieval config
            retrieval_override = {
                "top_k": top_k,
                "score_threshold": score_threshold if score_threshold > 0 else None,
            }
            
            # Add reranker settings if enabled
            if enable_reranking:
                retrieval_override["reranker"] = {
                    "enabled": True,
                    "top_k": reranker_top_k
                }
            
            # Add fusion alpha if using hybrid retrieval
            if "hybrid" in retrieval_config_path:
                retrieval_override["fusion_alpha"] = fusion_alpha
            
            # Prepare state with dynamic config
            state = {
                "question": prompt,
                "chat_history": [],
                "retrieval_top_k": top_k,
                "retrieval_config_path": retrieval_config_path,
                "retrieval_override": retrieval_override
            }
            
            # Invoke agent
            try:
                final_state = graph.invoke(state)
                answer = final_state.get("answer", "[No answer returned]")
                
                # Display answer
                st.markdown(answer)
                
                # Collect reasoning steps
                reasoning = {}
                if show_reasoning:
                    if "query_analysis" in final_state:
                        reasoning["Query Analysis"] = final_state["query_analysis"]
                    if "query_type" in final_state:
                        reasoning["Query Type"] = final_state["query_type"]
                    if "needs_retrieval" in final_state:
                        reasoning["Needs Retrieval"] = final_state["needs_retrieval"]
                    if "routing_decision" in final_state:
                        reasoning["Routing Decision"] = final_state["routing_decision"]
                    if "retrieval_metadata" in final_state:
                        reasoning["Retrieval Metadata"] = final_state["retrieval_metadata"]
                    if "generation_mode" in final_state:
                        reasoning["Generation Mode"] = final_state["generation_mode"]
                    if "verification" in final_state:
                        reasoning["Verification"] = final_state["verification"]
                    if "self_rag_metadata" in final_state:
                        reasoning["Self-RAG Metadata"] = final_state["self_rag_metadata"]
                
                # Show reasoning
                if reasoning:
                    with st.expander("🔍 Reasoning Steps"):
                        for key, value in reasoning.items():
                            st.markdown(f"**{key}:**")
                            if isinstance(value, dict):
                                st.json(value)
                            else:
                                st.text(value)
                
                # Show documents
                retrieved_docs = final_state.get("retrieved_documents", [])
                if retrieved_docs and show_documents:
                    with st.expander(f"📄 Retrieved Documents ({len(retrieved_docs)})"):
                        for i, doc in enumerate(retrieved_docs, 1):
                            st.markdown(f"**Document {i}** (Score: {doc.metadata.get('score', 'N/A'):.3f})")
                            st.text(doc.page_content[:300] + "...")
                            if show_metadata:
                                st.json(doc.metadata)
                            st.divider()
                
                # Store message with settings
                msg_data = {
                    "role": "assistant",
                    "content": answer,
                    "reasoning": reasoning,
                    "documents": retrieved_docs,
                    "settings": {
                        "mode": st.session_state.mode,
                        "pipeline": retrieval_config_path.split('/')[-1],
                        "top_k": top_k,
                        "score_threshold": score_threshold,
                        "reranking": enable_reranking,
                        "reranker_top_k": reranker_top_k if enable_reranking else None,
                        "fusion_alpha": fusion_alpha if "hybrid" in retrieval_config_path else None
                    }
                }
                st.session_state.messages.append(msg_data)
                
                # Show errors if any
                if "error" in final_state:
                    st.error(f"Error: {final_state['error']}")
                
            except Exception as e:
                st.error(f"Agent failed: {str(e)}")
                import traceback
                with st.expander("Full Error"):
                    st.code(traceback.format_exc())

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"Messages: {len(st.session_state.messages)}")
with col2:
    st.caption(f"Mode: {st.session_state.mode.upper()}")
with col3:
    st.caption("Powered by ReRag")
