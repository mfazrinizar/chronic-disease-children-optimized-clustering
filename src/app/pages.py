from matplotlib import pyplot as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.cluster.hierarchy import linkage
from plotly.figure_factory import create_dendrogram

from src.app.io import list_configs, load_history_data, load_normalized_data, load_config, load_projection, load_raw_data
from src.app.metrics import compute_cluster_metrics
from src.app.dss import (
    generate_cluster_summary,
    rank_provinces_by_priority,
    generate_action_plan,
    generate_executive_summary,
    get_cluster_profile,
    analyze_cluster_risk_levels,
    analyze_province_trends,
    analyze_population_trends,
    extract_temporal_data,
    identify_disease_hotspots,
    generate_comparative_analysis
)

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
        st.dataframe(df, width='stretch')
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
            st.plotly_chart(fig, width='stretch')


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
        st.plotly_chart(fig_si, width='stretch')

    with col2:
        fig_dbi = px.line(history, x='Iteration', y='DBI', title='Davies-Bouldin Index Over Iterations', markers=True)
        fig_dbi.add_hline(y=history['DBI'].min(), line_dash="dash", line_color="green", annotation_text=f"Best: {history['DBI'].min():.4f}")
        st.plotly_chart(fig_dbi, width='stretch')

    st.subheader("Combined Score Progression")
    fig_score = px.line(history, x='Iteration', y='Score', title='Combined Score (SI - 0.5*DBI) Over Iterations', markers=True)
    fig_score.add_hline(y=history['Score'].max(), line_dash="dash", line_color="green")
    st.plotly_chart(fig_score, width='stretch')

    st.markdown("---")

    st.subheader("Performance by Configuration")
    col1, col2 = st.columns(2)
    with col1:
        fig_scaler = px.box(history, x='Scaler', y='SI', title='SI Distribution by Scaler', color='Scaler')
        st.plotly_chart(fig_scaler, width='stretch')
    with col2:
        fig_transformer = px.box(history, x='Transformer', y='SI', title='SI Distribution by Transformer', color='Transformer')
        st.plotly_chart(fig_transformer, width='stretch')

    st.subheader("Scaler vs Transformer Heatmap")
    pivot = history.pivot_table(values='SI', index='Scaler', columns='Transformer', aggfunc='mean')
    fig_heat = px.imshow(pivot, text_auto='.3f', aspect='auto', title='Average SI by Scaler and Transformer', color_continuous_scale='RdYlGn')
    st.plotly_chart(fig_heat, width='stretch')

    st.markdown("---")

    st.subheader("Top 10 Best Configurations")
    top10 = history.nlargest(10, 'SI')[['Iteration', 'Scaler', 'Transformer', 'Metric', 'Seed', 'SI', 'DBI', 'Score']]
    st.dataframe(top10, width='stretch', hide_index=True)

    st.markdown("---")

    st.subheader("Full History Data")
    st.dataframe(history, width='stretch', height=400)


def show_normalized_data():
    st.header("Normalized Data Analysis")

    normalized = load_normalized_data()
    if normalized is None:
        st.error("Normalized data not found at model_data/data/best_normalized_data.csv")
        return

    st.subheader("Normalized Data Table")
    st.dataframe(normalized, width='stretch')

    st.markdown("---")

    st.subheader("Feature Distributions")
    fig_box = go.Figure()
    for col in normalized.columns:
        fig_box.add_trace(go.Box(y=normalized[col], name=col))
    fig_box.update_layout(title='Feature Distributions (Box Plot)', showlegend=False, height=400)
    st.plotly_chart(fig_box, width='stretch')

    st.markdown("---")

    st.subheader("Feature Correlation Heatmap")
    corr = normalized.corr()
    fig_corr = px.imshow(corr, text_auto='.2f', aspect='auto', title='Feature Correlation Matrix', color_continuous_scale='RdBu_r')
    st.plotly_chart(fig_corr, width='stretch')

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
    st.plotly_chart(fig_profile, width='stretch')

    st.markdown("---")

    st.subheader("Statistical Summary")
    st.dataframe(normalized.describe(), width='stretch')


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
            st.plotly_chart(fig, width='stretch')
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
            st.plotly_chart(fig_km, width='stretch')

        with col2:
            proj_hc = projection.copy()
            proj_hc['Cluster'] = metrics['Agglomerative']['labels']
            fig_hc = px.scatter(proj_hc.reset_index(), x='Dim1', y='Dim2', color='Cluster', text=proj_hc.index, title='Agglomerative Clustering', color_continuous_scale='viridis')
            fig_hc.update_traces(textposition='top center', marker=dict(size=15))
            st.plotly_chart(fig_hc, width='stretch')

        with col3:
            if metrics['Fuzzy C-Means']['labels'] is not None:
                proj_fcm = projection.copy()
                proj_fcm['Cluster'] = metrics['Fuzzy C-Means']['labels']
                fig_fcm = px.scatter(proj_fcm.reset_index(), x='Dim1', y='Dim2', color='Cluster', text=proj_fcm.index, title='Fuzzy C-Means Clustering', color_continuous_scale='viridis')
                fig_fcm.update_traces(textposition='top center', marker=dict(size=15))
                st.plotly_chart(fig_fcm, width='stretch')
            else:
                st.warning("FCM clustering unavailable")
                
        if projection is not None:
            st.markdown("---")
            st.subheader("Agglomerative Clustering Dendrogram")
            X = projection[['Dim1', 'Dim2']].values
            labels = projection.index.astype(str).tolist()
            Z = linkage(X, method='ward')
            fig = create_dendrogram(X, orientation='bottom', labels=labels, linkagefun=lambda x: Z)
            fig.update_layout(height=600)  # Set explicit height to avoid infinity error
            
            st.plotly_chart(fig, width='stretch')
            
            

        if metrics['Fuzzy C-Means']['membership'] is not None:
            st.markdown("---")
            st.subheader("Fuzzy C-Means Membership Matrix")
            membership = metrics['Fuzzy C-Means']['membership']
            df_membership = __import__('pandas').DataFrame(membership, index=projection.index, columns=[f"Cluster {i}" for i in range(membership.shape[1])])
            fig_mem = px.imshow(df_membership, text_auto='.3f', aspect='auto', title='Fuzzy Membership Values', color_continuous_scale='Blues', labels={'x': 'Cluster', 'y': 'Province'})
            st.plotly_chart(fig_mem, width='stretch')
            if metrics['Fuzzy C-Means']['FPC'] is not None:
                st.metric("Fuzzy Partition Coefficient (FPC)", f"{metrics['Fuzzy C-Means']['FPC']:.4f}")

        st.markdown("---")
        st.subheader("2D Projection Data")
        st.dataframe(projection, width='stretch')


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
            st.plotly_chart(fig1, width='stretch')
    with col2:
        if proj2 is not None:
            fig2 = px.scatter(proj2.reset_index(), x='Dim1', y='Dim2', color='Cluster_KM' if 'Cluster_KM' in proj2.columns else None, text=proj2.index, title=f"Projection 2 (SI={config2['SI']:.4f})", color_continuous_scale='viridis')
            fig2.update_traces(textposition='top center', marker=dict(size=12))
            st.plotly_chart(fig2, width='stretch')

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
        st.dataframe(df_comp, width='stretch', hide_index=True)

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(name=f'Config 1 (Iter {config1["iteration"]})', x=['KMeans SI', 'Agglomerative SI', 'FCM SI'], y=[metrics1['KMeans']['SI'], metrics1['Agglomerative']['SI'], metrics1['Fuzzy C-Means']['SI'] if metrics1['Fuzzy C-Means']['SI'] is not None else 0]))
        fig_comp.add_trace(go.Bar(name=f'Config 2 (Iter {config2["iteration"]})', x=['KMeans SI', 'Agglomerative SI', 'FCM SI'], y=[metrics2['KMeans']['SI'], metrics2['Agglomerative']['SI'], metrics2['Fuzzy C-Means']['SI'] if metrics2['Fuzzy C-Means']['SI'] is not None else 0]))
        fig_comp.update_layout(title='Silhouette Index Comparison', barmode='group')
        st.plotly_chart(fig_comp, width='stretch')

    st.markdown("---")
    st.subheader("All Configurations SI Ranking")
    fig_rank = px.bar(__import__('pandas').DataFrame(configs), x='iteration', y='SI', title='Silhouette Index by Configuration Iteration', labels={'iteration': 'Iteration', 'SI': 'Silhouette Index'}, color='SI', color_continuous_scale='RdYlGn')
    st.plotly_chart(fig_rank, width='stretch')


def show_decision_support():
    """Decision Support System page for actionable insights."""
    import pandas as pd
    
    st.header("Decision Support System")
    st.markdown("""
    This module provides **actionable insights** based on clustering analysis to support 
    health policy decisions for chronic disease management in children across provinces.
    
    **Enhanced Features:**
    - Data-driven risk assessment
    - Temporal trend analysis (2015-2023)
    - Early warning indicators
    - Province-specific recommendations
    """)
    
    # Load data
    configs = list_configs()
    if not configs:
        st.error("No configurations found. Please run the clustering analysis first.")
        return
    
    normalized = load_normalized_data()
    raw_data = load_raw_data()  # Load raw data for temporal analysis
    
    # Select configuration
    config_options = [f"Iter {c['iteration']} (SI={c['SI']:.4f})" for c in configs]
    selected_idx = st.selectbox(
        "Select Model Configuration for DSS Analysis",
        range(len(config_options)),
        format_func=lambda x: config_options[x],
        key='dss_config'
    )
    
    selected_config = configs[selected_idx]
    projection = load_projection(selected_config['file'])
    
    if projection is None:
        st.error("Could not load projection data for selected configuration.")
        return
    
    # Compute clustering
    X = projection[['Dim1', 'Dim2']].values
    metrics = compute_cluster_metrics(X, k=3)
    cluster_labels = metrics['KMeans']['labels']
    
    # Generate DSS outputs (with temporal data)
    cluster_summary = generate_cluster_summary(projection, cluster_labels, normalized)
    rankings = rank_provinces_by_priority(projection, cluster_labels, normalized)
    executive_summary = generate_executive_summary(cluster_summary, rankings, raw_data)
    
    st.markdown("---")
    st.subheader("Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Provinces", executive_summary["total_provinces"])
    with col2:
        st.metric("🔴 High Risk", executive_summary["risk_distribution"]["High Risk"])
    with col3:
        st.metric("🟠 Moderate Risk", executive_summary["risk_distribution"]["Moderate Risk"])
    with col4:
        st.metric("🟢 Low Risk", executive_summary["risk_distribution"]["Low Risk"])
    
    # Key Findings
    st.markdown("#### Key Findings")
    for finding in executive_summary["key_findings"]:
        st.markdown(f"- {finding}")
    
    # Immediate Actions
    st.markdown("#### Recommended Immediate Actions")
    for i, action in enumerate(executive_summary["immediate_actions"], 1):
        st.markdown(f"{i}. {action}")
    
    st.markdown("---")
    st.subheader("Risk Distribution Map")
    
    col1, col2 = st.columns([2, 1])
    
    # Get cluster profiles from data-driven analysis
    if normalized is not None:
        cluster_profiles = analyze_cluster_risk_levels(normalized, cluster_labels)
    else:
        cluster_profiles = {}
    
    with col1:
        # Scatter plot with risk colors
        plot_data = projection.copy()
        plot_data['Cluster'] = cluster_labels
        plot_data['Risk Level'] = [cluster_profiles.get(int(c), {}).get('risk_level', 'Unknown') for c in cluster_labels]
        plot_data['Color'] = [cluster_profiles.get(int(c), {}).get('color', '#95a5a6') for c in cluster_labels]
        
        fig_risk = px.scatter(
            plot_data.reset_index(),
            x='Dim1',
            y='Dim2',
            color='Risk Level',
            text=plot_data.index,
            title='Provincial Risk Classification',
            color_discrete_map={
                'Low Risk': '#2ecc71',
                'Moderate Risk': '#f39c12',
                'High Risk': '#e74c3c'
            }
        )
        fig_risk.update_traces(textposition='top center', marker=dict(size=18))
        fig_risk.update_layout(height=500)
        st.plotly_chart(fig_risk, width='stretch')
    
    with col2:
        # Risk distribution pie chart
        risk_counts = rankings['Risk Level'].value_counts()
        fig_pie = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title='Risk Distribution',
            color=risk_counts.index,
            color_discrete_map={
                'Low Risk': '#2ecc71',
                'Moderate Risk': '#f39c12',
                'High Risk': '#e74c3c'
            }
        )
        st.plotly_chart(fig_pie, width='stretch')
        
        # Resource allocation suggestion
        st.markdown("#### Suggested Resource Allocation")
        for category, pct in executive_summary["resource_allocation_suggestion"].items():
            st.progress(pct / 100, text=f"{category}: {pct}%")
    
    st.markdown("---")
    st.subheader("Province Priority Rankings")
    
    st.markdown("""
    Provinces are ranked by a **composite score** that considers:
    - Base risk level from cluster assignment
    - Anomaly score (distance from cluster centroid — higher means more unusual)
    """)
    
    # Display rankings table with styling
    display_rankings = rankings[['Rank', 'Province', 'Risk Level', 'Priority', 'Composite Score']].copy()
    st.dataframe(
        display_rankings,
        width='stretch',
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", format="%d"),
            "Composite Score": st.column_config.ProgressColumn(
                "Priority Score",
                min_value=0,
                max_value=4,
                format="%.2f"
            )
        }
    )
    
    # Bar chart of composite scores
    fig_rank_bar = px.bar(
        rankings.sort_values('Composite Score', ascending=True),
        x='Composite Score',
        y='Province',
        color='Risk Level',
        orientation='h',
        title='Province Priority Scores (Higher = More Urgent)',
        color_discrete_map={
            'Low Risk': '#2ecc71',
            'Moderate Risk': '#f39c12',
            'High Risk': '#e74c3c'
        }
    )
    fig_rank_bar.update_layout(height=600)
    st.plotly_chart(fig_rank_bar, width='stretch')
    
    if raw_data is not None:
        st.markdown("---")
        st.subheader("Temporal Trend Analysis (2015-2023)")
        
        st.markdown("""
        Analyzing how disease prevalence has changed over time helps identify:
        - 🔺 **Worsening trends** requiring urgent intervention
        - 🔻 **Improving trends** indicating successful programs
        - ⚠️ **Early warnings** of emerging concerns
        """)
        
        # Population-wide trends
        pop_trends = executive_summary.get("temporal_insights", {})
        if pop_trends:
            st.markdown("#### Population-Wide Disease Trends")
            
            # Create trend visualization
            temporal_data = extract_temporal_data(raw_data)
            
            # Line chart for each disease across years
            trend_tabs = st.tabs(list(temporal_data.keys()))
            
            for tab, (disease, df) in zip(trend_tabs, temporal_data.items()):
                with tab:
                    # Calculate population average per year
                    yearly_avg = df.mean(axis=0)
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Line chart
                        fig_trend = go.Figure()
                        
                        # Add individual province lines (faded)
                        for province in df.index:
                            fig_trend.add_trace(go.Scatter(
                                x=df.columns,
                                y=df.loc[province],
                                mode='lines',
                                name=province,
                                opacity=0.3,
                                line=dict(width=1),
                                showlegend=False
                            ))
                        
                        # Add population average (bold)
                        fig_trend.add_trace(go.Scatter(
                            x=yearly_avg.index,
                            y=yearly_avg.values,
                            mode='lines+markers',
                            name='Population Average',
                            line=dict(width=3, color='#e74c3c'),
                            marker=dict(size=8)
                        ))
                        
                        fig_trend.update_layout(
                            title=f'{disease} Prevalence Over Time',
                            xaxis_title='Year',
                            yaxis_title='Prevalence (%)',
                            height=400
                        )
                        st.plotly_chart(fig_trend, width='stretch')
                    
                    with col2:
                        trend_info = pop_trends.get("disease_trends", {}).get(disease, {})
                        
                        # Trend metrics
                        direction = trend_info.get("direction", "stable")
                        annual_change = trend_info.get("annual_change_pct", 0)
                        
                        # Direction indicator
                        if "improvement" in direction:
                            st.success(f"📉 **Trend: IMPROVING**")
                        elif "worsening" in direction:
                            st.error(f"📈 **Trend: WORSENING**")
                        else:
                            st.info(f"➡️ **Trend: STABLE**")
                        
                        st.metric(
                            "Annual Change",
                            f"{annual_change:+.2f}%/year",
                            delta=f"{annual_change:+.2f}%",
                            delta_color="inverse"  # Negative is good for disease
                        )
                        
                        st.metric(
                            "Total Change (2015-2023)",
                            f"{trend_info.get('total_change_pct', 0):+.1f}%"
                        )
                        
                        st.markdown(f"**Confidence:** {trend_info.get('confidence', 'N/A').title()}")
        
        # Early Warnings Section
        early_warnings = executive_summary.get("early_warnings", [])
        if early_warnings:
            st.markdown("#### ⚠️ Early Warning Indicators")
            for warning in early_warnings:
                severity_color = "error" if warning.get("severity") == "high" else "warning"
                if warning.get("type") == "population_trend":
                    getattr(st, severity_color)(
                        f"**{warning.get('disease', 'Unknown')}**: {warning.get('message', '')}"
                    )
                elif warning.get("type") == "province_deterioration":
                    provinces = warning.get("provinces", [])
                    getattr(st, severity_color)(
                        f"**Deteriorating Health Trends:** {', '.join(provinces)}"
                    )
    else:
        st.markdown("---")
        st.info("Temporal trend analysis requires raw data file (raw_data.csv). Place it in model_data/data/ to enable this feature.")
    
    st.markdown("---")
    st.subheader("Cluster Profiles")
    
    # Create tabs for each cluster found in data
    cluster_ids = sorted(cluster_summary["clusters"].keys())
    tab_names = [cluster_summary["clusters"][cid].get("risk_level", f"Cluster {cid}") for cid in cluster_ids]
    tabs = st.tabs(tab_names) if tab_names else []
    
    for tab, cluster_id in zip(tabs, cluster_ids):
        with tab:
            cluster_info = cluster_summary["clusters"][cluster_id]
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"### {cluster_info.get('risk_level', 'Unknown')}")
                st.markdown(f"**Cluster ID:** {cluster_id}")
                st.markdown(f"**Priority Level:** {cluster_info.get('priority', 'Medium')}")
                st.markdown(f"**Provinces in Cluster:** {cluster_info.get('count', 0)}")
                st.markdown(f"**Percentage:** {cluster_info.get('percentage', 0):.2f}%")
                
                st.markdown("**Provinces:**")
                provinces_in_cluster = cluster_info.get('provinces', [])
                for prov in provinces_in_cluster:
                    st.markdown(f"- {prov}")
            
            with col2:
                st.markdown("**Description:**")
                st.info(cluster_info.get('description', 'No description available.'))
                
                st.markdown("**Characteristics:**")
                for char in cluster_info.get('characteristics', []):
                    st.markdown(f"- {char}")
                
                # Show feature means if available
                if 'feature_means' in cluster_info:
                    with st.expander("📊 Feature Averages for this Cluster"):
                        import pandas as pd
                        feat_df = pd.DataFrame({
                            'Feature': list(cluster_info['feature_means'].keys()),
                            'Cluster Mean': list(cluster_info['feature_means'].values())
                        })
                        st.dataframe(feat_df, width='stretch', hide_index=True)
    
    st.markdown("---")
    st.subheader("Province Action Plan Generator")
    
    province_list = rankings['Province'].tolist()
    selected_province = st.selectbox(
        "Select Province for Detailed Action Plan",
        province_list,
        key='dss_province'
    )
    
    if selected_province:
        province_row = rankings[rankings['Province'] == selected_province].iloc[0]
        action_plan = generate_action_plan(
            province=selected_province,
            cluster_id=province_row['Cluster'],
            rank=province_row['Rank'],
            total_provinces=len(rankings),
            normalized_data=normalized,
            cluster_labels=cluster_labels,
            raw_data=raw_data  # Include raw data for temporal analysis
        )
        
        # Display action plan
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Risk Level", action_plan['risk_level'])
        with col2:
            st.metric("Priority Rank", f"#{action_plan['rank']} of {len(rankings)}")
        with col3:
            urgency_colors = {
                "Critical": "🔴",
                "High": "🟠",
                "Moderate": "🟡",
                "Standard": "🟢"
            }
            st.metric("Urgency", f"{urgency_colors.get(action_plan['urgency'], '')} {action_plan['urgency']}")
        
        st.info(action_plan['summary'])
        
        # Temporal trends for this province
        if action_plan.get('temporal_trends'):
            st.markdown("#### Temporal Trends for This Province (2015-2023)")
            
            with st.expander("ℹ️ About Temporal Analysis"):
                st.markdown("""
                **Why use raw data for trends?**
                - Normalized data is scaled relative to other provinces (0-1), losing actual values
                - Raw data preserves real percentages needed to calculate meaningful trends
                - Example: "Anemia dropped from 29.6% to 26.5%" is more actionable than "normalized value changed from 0.97 to 0.71"
                
                **Metrics explained:**
                - **Annual change %/yr**: Linear regression slope (accounts for all years)
                - **Total change %**: Simple start-to-end difference
                
                ---
                
                **🎯 How Confidence is Determined:**
                
                Confidence is calculated using TWO metrics:
                
                1. **R² (R-squared)**: Measures how well a straight line fits the data
                   - R² = 1.0 means perfect linear trend
                   - R² = 0.0 means no linear pattern (highly volatile)
                   - Health data is naturally noisy, so even R² > 0.4 is considered strong
                
                2. **Consistency Ratio**: Measures directional consistency
                   - Counts what % of year-over-year changes follow the overall trend direction
                   - Example: If trend is "increasing" and 6 out of 8 yearly changes are positive, ratio = 75%
                
                **Confidence Levels:**
                - 🟢 **High**: R² > 0.4 OR Consistency > 70%
                - 🟡 **Medium**: R² > 0.15 OR Consistency > 50%  
                - ⚪ **Low**: R² ≤ 0.15 AND Consistency ≤ 50% (data is volatile/unpredictable)
                """)
            
            trend_data = action_plan['temporal_trends']
            overall_trend = trend_data.get('overall_trend', 'stable')
            
            # Overall trend indicator
            if overall_trend == 'worsening':
                st.error(f"⚠️ **Overall Trend: WORSENING** — Health indicators deteriorating over time")
            elif overall_trend == 'improving':
                st.success(f"✓ **Overall Trend: IMPROVING** — Health indicators getting better over time")
            else:
                st.info(f"➡️ **Overall Trend: STABLE** — Health indicators relatively unchanged")
            
            # Disease-specific trends
            disease_trends = trend_data.get('diseases', {})
            if disease_trends:
                trend_cols = st.columns(len(disease_trends))
                for col, (disease, trend) in zip(trend_cols, disease_trends.items()):
                    with col:
                        direction = trend.get('direction', 'stable')
                        annual_change = trend.get('annual_change_pct', 0)
                        total_change = trend.get('total_change_pct', 0)
                        r_squared = trend.get('r_squared', 0)
                        consistency = trend.get('consistency_ratio', 0)
                        
                        if 'improvement' in direction:
                            icon = "📉"
                            color = "green"
                        elif 'worsening' in direction:
                            icon = "📈"
                            color = "red"
                        else:
                            icon = "➡️"
                            color = "gray"
                        
                        st.markdown(f"**{disease}** {icon}")
                        st.markdown(f"<span style='color:{color}'><b>{annual_change:+.1f}%/yr</b></span>", 
                                   unsafe_allow_html=True)
                        st.caption(f"{trend.get('start_value', 0):.1f}% → {trend.get('end_value', 0):.1f}%")
                        st.caption(f"Total: {total_change:+.1f}%")
                        confidence = trend.get('confidence', 'low')
                        conf_icon = "🟢" if confidence == "high" else ("🟡" if confidence == "medium" else "⚪")
                        st.caption(f"Confidence: {conf_icon} {confidence.title()}")
                        st.caption(f"R²={r_squared:.2f} | Cons={consistency*100:.0f}%")
            
            # Early warnings for this province
            province_warnings = action_plan.get('early_warnings', [])
            if province_warnings:
                st.markdown("##### ⚠️ Early Warnings")
                for warning in province_warnings:
                    st.warning(f"**{warning.get('disease', 'Unknown')}**: {warning.get('message', '')} — {warning.get('action', '')}")
        
        # Province-specific concerns (data-driven)
        if action_plan.get('concerns'):
            st.markdown("#### ⚠️ Key Concerns (Data-Driven)")
            for concern in action_plan['concerns']:
                severity_icon = "🔴" if concern['severity'] == 'high' else "🟠"
                st.markdown(f"{severity_icon} **{concern['feature']}**: {concern['vs_mean']} — {concern['recommendation']}")
        
        # Province strengths
        if action_plan.get('strengths'):
            st.markdown("#### Strengths")
            for strength in action_plan['strengths']:
                st.markdown(f"🟢 **{strength['feature']}**: {strength['interpretation']}")
        
        # Key actions
        st.markdown("#### ⚠️ Key Actions")
        for i, action in enumerate(action_plan['key_actions'], 1):
            st.markdown(f"**{i}.** {action}")
        
        # Monitoring info
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Monitoring Frequency:** {action_plan['monitoring_frequency']}")
        with col2:
            st.markdown(f"**Review Timeline:** {action_plan['review_timeline']}")
        
        # Detailed recommendations in expander
        with st.expander("View Detailed Recommendations"):
            for rec in action_plan['detailed_recommendations']:
                st.markdown(f"### {rec['category']}")
                st.markdown(f"**Action:** {rec['action']}")
                st.markdown(f"**Details:** {rec['details']}")
                st.markdown(f"**Timeline:** {rec['timeline']}")
                st.markdown(f"**Resources:** {rec['resources']}")
                st.markdown("---")
    
    st.markdown("---")
    st.subheader("Export DSS Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export rankings as CSV
        csv_rankings = rankings.to_csv(index=False)
        st.download_button(
            label="Download Priority Rankings (CSV)",
            data=csv_rankings,
            file_name="province_priority_rankings.csv",
            mime="text/csv"
        )
    
    with col2:
        # Export executive summary as text
        summary_text = f"""
CHRONIC DISEASE IN CHILDREN: PROVINCIAL RISK ASSESSMENT
========================================================

EXECUTIVE SUMMARY
-----------------
Total Provinces Analyzed: {executive_summary['total_provinces']}

Risk Distribution:
- High Risk: {executive_summary['risk_distribution']['High Risk']} provinces
- Moderate Risk: {executive_summary['risk_distribution']['Moderate Risk']} provinces  
- Low Risk: {executive_summary['risk_distribution']['Low Risk']} provinces

Top Priority Provinces:
{chr(10).join(f'  {i+1}. {p}' for i, p in enumerate(executive_summary['top_priority_provinces']))}

Key Findings:
{chr(10).join(f'  - {f}' for f in executive_summary['key_findings'])}

Recommended Immediate Actions:
{chr(10).join(f'  {i+1}. {a}' for i, a in enumerate(executive_summary['immediate_actions']))}

Suggested Resource Allocation:
{chr(10).join(f'  - {k}: {v}%' for k, v in executive_summary['resource_allocation_suggestion'].items())}

Generated using Clustering-based Decision Support System
Configuration: Iteration {selected_config['iteration']} (SI={selected_config['SI']:.4f})
"""
        st.download_button(
            label="Download Executive Summary (TXT)",
            data=summary_text,
            file_name="executive_summary.txt",
            mime="text/plain"
        )

