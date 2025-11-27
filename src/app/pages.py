from matplotlib import pyplot as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.cluster.hierarchy import linkage
from plotly.figure_factory import create_dendrogram

from src.app.io import list_configs, load_history_data, load_normalized_data, load_config, load_projection
from src.app.metrics import compute_cluster_metrics

def show_overview():
    st.header("Dashboard Overview")

    col1, col2, col3, col4 = st.columns(4)

    configs = list_configs()
    history = load_history_data()
    normalized = load_normalized_data()

    with col1:
        st.metric("Total Best Configurations", len(configs))

    with col2:
        if configs:
            best_si = max(c['SI'] for c in configs)
            st.metric("Best SI Score", f"{best_si:.4f}")
        else:
            st.metric("Best SI Score", "N/A")

    with col3:
        if history is not None:
            st.metric("Valid Search Iterations", len(history))
        else:
            st.metric("Valid Search Iterations", "N/A")

    with col4:
        if normalized is not None:
            st.metric("Provinces", len(normalized))
        else:
            st.metric("Provinces", "N/A")

    st.markdown("---")

    st.subheader("Available Model Configurations")
    if configs:
        df_configs = st.experimental_data_editor({ 'configs': configs }) if False else None
        df_configs = None
        df = { 'Config File': [c['file'] for c in configs], 'Iteration': [c['iteration'] for c in configs], 'Silhouette Index': [c['SI'] for c in configs] }
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No configurations found in model_data/configs/")

    st.markdown("---")

    st.subheader("Quick View: Best Model Projection")
    if configs:
        best_config = configs[0]
        proj = load_projection(best_config['file'])
        if proj is not None:
            fig = px.scatter(
                proj.reset_index(),
                x='Dim1',
                y='Dim2',
                color='Cluster_KM' if 'Cluster_KM' in proj.columns else None,
                text='Province' if 'Province' in proj.reset_index().columns else proj.index,
                title=f"Best Projection (SI={best_config['SI']:.4f})",
                color_continuous_scale='viridis'
            )
            fig.update_traces(textposition='top center', marker=dict(size=15))
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)


def show_search_history():
    st.header("Hyperparameter Search History")

    history = load_history_data()
    if history is None:
        st.error("History data not found at model_data/data/history_data.csv")
        return

    st.subheader("Search Progress Over Iterations")

    col1, col2 = st.columns(2)

    with col1:
        fig_si = px.line(history, x='Iteration', y='SI', title='Silhouette Index Over Iterations', markers=True)
        fig_si.add_hline(y=history['SI'].max(), line_dash="dash", line_color="green", annotation_text=f"Best: {history['SI'].max():.4f}")
        st.plotly_chart(fig_si, use_container_width=True)

    with col2:
        fig_dbi = px.line(history, x='Iteration', y='DBI', title='Davies-Bouldin Index Over Iterations', markers=True)
        fig_dbi.add_hline(y=history['DBI'].min(), line_dash="dash", line_color="green", annotation_text=f"Best: {history['DBI'].min():.4f}")
        st.plotly_chart(fig_dbi, use_container_width=True)

    st.subheader("Combined Score Progression")
    fig_score = px.line(history, x='Iteration', y='Score', title='Combined Score (SI - 0.5*DBI) Over Iterations', markers=True)
    fig_score.add_hline(y=history['Score'].max(), line_dash="dash", line_color="green")
    st.plotly_chart(fig_score, use_container_width=True)

    st.markdown("---")

    st.subheader("Performance by Configuration")
    col1, col2 = st.columns(2)
    with col1:
        fig_scaler = px.box(history, x='Scaler', y='SI', title='SI Distribution by Scaler', color='Scaler')
        st.plotly_chart(fig_scaler, use_container_width=True)
    with col2:
        fig_transformer = px.box(history, x='Transformer', y='SI', title='SI Distribution by Transformer', color='Transformer')
        st.plotly_chart(fig_transformer, use_container_width=True)

    st.subheader("Scaler vs Transformer Heatmap")
    pivot = history.pivot_table(values='SI', index='Scaler', columns='Transformer', aggfunc='mean')
    fig_heat = px.imshow(pivot, text_auto='.3f', aspect='auto', title='Average SI by Scaler and Transformer', color_continuous_scale='RdYlGn')
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    st.subheader("Top 10 Best Configurations")
    top10 = history.nlargest(10, 'SI')[['Iteration', 'Scaler', 'Transformer', 'Metric', 'SI', 'DBI', 'Score']]
    st.dataframe(top10, use_container_width=True, hide_index=True)

    st.markdown("---")

    st.subheader("Full History Data")
    st.dataframe(history, use_container_width=True, height=400)


def show_normalized_data():
    st.header("Normalized Data Analysis")

    normalized = load_normalized_data()
    if normalized is None:
        st.error("Normalized data not found at model_data/data/best_normalized_data.csv")
        return

    st.subheader("Normalized Data Table")
    st.dataframe(normalized, use_container_width=True)

    st.markdown("---")

    st.subheader("Feature Distributions")
    fig_box = go.Figure()
    for col in normalized.columns:
        fig_box.add_trace(go.Box(y=normalized[col], name=col))
    fig_box.update_layout(title='Feature Distributions (Box Plot)', showlegend=False, height=400)
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")

    st.subheader("Feature Correlation Heatmap")
    corr = normalized.corr()
    fig_corr = px.imshow(corr, text_auto='.2f', aspect='auto', title='Feature Correlation Matrix', color_continuous_scale='RdBu_r')
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")

    st.subheader("Province Feature Profiles")
    fig_profile = go.Figure()
    for idx in normalized.index:
        fig_profile.add_trace(go.Scatterpolar(
            r=normalized.loc[idx].values[:6] if len(normalized.columns) > 6 else normalized.loc[idx].values,
            theta=normalized.columns[:6] if len(normalized.columns) > 6 else normalized.columns.tolist(),
            fill='toself',
            name=idx
        ))
    fig_profile.update_layout(title='Province Feature Profiles (Radar Chart)', height=500)
    st.plotly_chart(fig_profile, use_container_width=True)

    st.markdown("---")

    st.subheader("Statistical Summary")
    st.dataframe(normalized.describe(), use_container_width=True)


def show_model_explorer():
    st.header("Model Configuration Explorer")

    configs = list_configs()
    if not configs:
        st.error("No configurations found in model_data/configs/")
        return

    config_options = [f"Iter {c['iteration']} (SI={c['SI']:.4f})" for c in configs]
    selected_idx = st.selectbox("Select Configuration", range(len(config_options)), format_func=lambda x: config_options[x])

    selected_config = configs[selected_idx]
    config_data = load_config(selected_config['file'])
    projection = load_projection(selected_config['file'])

    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Configuration Details")
        if config_data:
            st.json(config_data)
        else:
            st.warning("Could not load configuration data")

    with col2:
        st.subheader("2D Projection Visualization")
        if projection is not None:
            fig = px.scatter(projection.reset_index(), x='Dim1', y='Dim2', color='Cluster_KM' if 'Cluster_KM' in projection.columns else None, text=projection.index, title=f"Projection for Iteration {selected_config['iteration']}", color_continuous_scale='viridis')
            fig.update_traces(textposition='top center', marker=dict(size=15))
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Projection data not found")

    if projection is not None:
        st.markdown("---")
        st.subheader("Clustering Evaluation")
        X = projection[['Dim1', 'Dim2']].values
        metrics = compute_cluster_metrics(X, k=3)

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("KMeans SI", f"{metrics['KMeans']['SI']:.4f}")
        with col2:
            st.metric("KMeans DBI", f"{metrics['KMeans']['DBI']:.4f}")
        with col3:
            st.metric("Agglomerative SI", f"{metrics['Agglomerative']['SI']:.4f}")
        with col4:
            st.metric("Agglomerative DBI", f"{metrics['Agglomerative']['DBI']:.4f}")
        with col5:
            if metrics['Fuzzy C-Means']['SI'] is not None:
                st.metric("FCM SI", f"{metrics['Fuzzy C-Means']['SI']:.4f}")
            else:
                st.metric("FCM SI", "N/A")
        with col6:
            if metrics['Fuzzy C-Means']['DBI'] is not None:
                st.metric("FCM DBI", f"{metrics['Fuzzy C-Means']['DBI']:.4f}")
            else:
                st.metric("FCM DBI", "N/A")

        st.markdown("---")
        st.subheader("Cluster Assignments")
        col1, col2, col3 = st.columns(3)

        with col1:
            proj_km = projection.copy()
            proj_km['Cluster'] = metrics['KMeans']['labels']
            fig_km = px.scatter(proj_km.reset_index(), x='Dim1', y='Dim2', color='Cluster', text=proj_km.index, title='KMeans Clustering', color_continuous_scale='viridis')
            fig_km.update_traces(textposition='top center', marker=dict(size=15))
            st.plotly_chart(fig_km, use_container_width=True)

        with col2:
            proj_hc = projection.copy()
            proj_hc['Cluster'] = metrics['Agglomerative']['labels']
            fig_hc = px.scatter(proj_hc.reset_index(), x='Dim1', y='Dim2', color='Cluster', text=proj_hc.index, title='Agglomerative Clustering', color_continuous_scale='viridis')
            fig_hc.update_traces(textposition='top center', marker=dict(size=15))
            st.plotly_chart(fig_hc, use_container_width=True)

        with col3:
            if metrics['Fuzzy C-Means']['labels'] is not None:
                proj_fcm = projection.copy()
                proj_fcm['Cluster'] = metrics['Fuzzy C-Means']['labels']
                fig_fcm = px.scatter(proj_fcm.reset_index(), x='Dim1', y='Dim2', color='Cluster', text=proj_fcm.index, title='Fuzzy C-Means Clustering', color_continuous_scale='viridis')
                fig_fcm.update_traces(textposition='top center', marker=dict(size=15))
                st.plotly_chart(fig_fcm, use_container_width=True)
            else:
                st.warning("FCM clustering unavailable")
                
        if projection is not None:
            st.markdown("---")
            st.subheader("Agglomerative Clustering Dendrogram")
            X = projection[['Dim1', 'Dim2']].values
            labels = projection.index.astype(str).tolist()
            Z = linkage(X, method='ward')
            fig = create_dendrogram(X, orientation='bottom', labels=labels, linkagefun=lambda x: Z)
            
            st.plotly_chart(fig)
            
            

        if metrics['Fuzzy C-Means']['membership'] is not None:
            st.markdown("---")
            st.subheader("Fuzzy C-Means Membership Matrix")
            membership = metrics['Fuzzy C-Means']['membership']
            df_membership = __import__('pandas').DataFrame(membership, index=projection.index, columns=[f"Cluster {i}" for i in range(membership.shape[1])])
            fig_mem = px.imshow(df_membership, text_auto='.3f', aspect='auto', title='Fuzzy Membership Values', color_continuous_scale='Blues', labels={'x': 'Cluster', 'y': 'Province'})
            st.plotly_chart(fig_mem, use_container_width=True)
            if metrics['Fuzzy C-Means']['FPC'] is not None:
                st.metric("Fuzzy Partition Coefficient (FPC)", f"{metrics['Fuzzy C-Means']['FPC']:.4f}")

        st.markdown("---")
        st.subheader("Projection Data")
        st.dataframe(projection, use_container_width=True)


def show_comparison():
    st.header("Model Comparison")

    configs = list_configs()
    if len(configs) < 2:
        st.warning("Need at least 2 configurations to compare")
        return

    config_options = [f"Iter {c['iteration']} (SI={c['SI']:.4f})" for c in configs]
    col1, col2 = st.columns(2)
    with col1:
        selected_idx1 = st.selectbox("Configuration 1", range(len(config_options)), format_func=lambda x: config_options[x], key='comp1')
    with col2:
        selected_idx2 = st.selectbox("Configuration 2", range(len(config_options)), format_func=lambda x: config_options[x], index=min(1, len(configs)-1), key='comp2')

    config1 = configs[selected_idx1]
    config2 = configs[selected_idx2]
    proj1 = load_projection(config1['file'])
    proj2 = load_projection(config2['file'])
    conf1_data = load_config(config1['file'])
    conf2_data = load_config(config2['file'])

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Config 1: Iter {config1['iteration']}")
        if conf1_data:
            st.json(conf1_data.get('hyperparams', {}))
    with col2:
        st.subheader(f"Config 2: Iter {config2['iteration']}")
        if conf2_data:
            st.json(conf2_data.get('hyperparams', {}))

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if proj1 is not None:
            fig1 = px.scatter(proj1.reset_index(), x='Dim1', y='Dim2', color='Cluster_KM' if 'Cluster_KM' in proj1.columns else None, text=proj1.index, title=f"Projection 1 (SI={config1['SI']:.4f})", color_continuous_scale='viridis')
            fig1.update_traces(textposition='top center', marker=dict(size=12))
            st.plotly_chart(fig1, use_container_width=True)
    with col2:
        if proj2 is not None:
            fig2 = px.scatter(proj2.reset_index(), x='Dim1', y='Dim2', color='Cluster_KM' if 'Cluster_KM' in proj2.columns else None, text=proj2.index, title=f"Projection 2 (SI={config2['SI']:.4f})", color_continuous_scale='viridis')
            fig2.update_traces(textposition='top center', marker=dict(size=12))
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Metrics Comparison")
    if proj1 is not None and proj2 is not None:
        X1 = proj1[['Dim1', 'Dim2']].values
        X2 = proj2[['Dim1', 'Dim2']].values
        metrics1 = compute_cluster_metrics(X1, k=3)
        metrics2 = compute_cluster_metrics(X2, k=3)

        comparison_data = {
            'Metric': ['KMeans SI', 'KMeans DBI', 'Agglomerative SI', 'Agglomerative DBI', 'FCM SI', 'FCM DBI'],
            f'Config 1 (Iter {config1["iteration"]})': [
                metrics1['KMeans']['SI'],
                metrics1['KMeans']['DBI'],
                metrics1['Agglomerative']['SI'],
                metrics1['Agglomerative']['DBI'],
                metrics1['Fuzzy C-Means']['SI'] if metrics1['Fuzzy C-Means']['SI'] is not None else np.nan,
                metrics1['Fuzzy C-Means']['DBI'] if metrics1['Fuzzy C-Means']['DBI'] is not None else np.nan
            ],
            f'Config 2 (Iter {config2["iteration"]})': [
                metrics2['KMeans']['SI'],
                metrics2['KMeans']['DBI'],
                metrics2['Agglomerative']['SI'],
                metrics2['Agglomerative']['DBI'],
                metrics2['Fuzzy C-Means']['SI'] if metrics2['Fuzzy C-Means']['SI'] is not None else np.nan,
                metrics2['Fuzzy C-Means']['DBI'] if metrics2['Fuzzy C-Means']['DBI'] is not None else np.nan
            ]
        }

        df_comp = __import__('pandas').DataFrame(comparison_data)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(name=f'Config 1 (Iter {config1["iteration"]})', x=['KMeans SI', 'Agglomerative SI', 'FCM SI'], y=[metrics1['KMeans']['SI'], metrics1['Agglomerative']['SI'], metrics1['Fuzzy C-Means']['SI'] if metrics1['Fuzzy C-Means']['SI'] is not None else 0]))
        fig_comp.add_trace(go.Bar(name=f'Config 2 (Iter {config2["iteration"]})', x=['KMeans SI', 'Agglomerative SI', 'FCM SI'], y=[metrics2['KMeans']['SI'], metrics2['Agglomerative']['SI'], metrics2['Fuzzy C-Means']['SI'] if metrics2['Fuzzy C-Means']['SI'] is not None else 0]))
        fig_comp.update_layout(title='Silhouette Index Comparison', barmode='group')
        st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")
    st.subheader("All Configurations SI Ranking")
    fig_rank = px.bar(__import__('pandas').DataFrame(configs), x='iteration', y='SI', title='Silhouette Index by Configuration Iteration', labels={'iteration': 'Iteration', 'SI': 'Silhouette Index'}, color='SI', color_continuous_scale='RdYlGn')
    st.plotly_chart(fig_rank, use_container_width=True)
