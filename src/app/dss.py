"""
Decision Support System (DSS) for Chronic Disease Clustering Analysis.

This module provides DATA-DRIVEN actionable insights based on clustering results:
- Dynamic cluster profiling based on actual feature analysis
- Risk assessment derived from feature centroids
- Province-specific recommendations based on individual feature patterns
- Intervention suggestions tailored to each province's specific issues
- TEMPORAL TREND ANALYSIS for tracking improvement/deterioration
- Disease-specific weighting for prioritization
- Early warning indicators for emerging concerns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import re


# Disease criticality weights (higher = more critical for children's health)
DISEASE_WEIGHTS = {
    "Anemia": 1.0,      # Common but manageable
    "Asthma": 1.2,      # Chronic, requires ongoing management
    "Pneumonia": 1.5    # Acute, potentially life-threatening
}

# Trend interpretation thresholds
TREND_THRESHOLDS = {
    "significant_improvement": -0.05,  # >5% annual decrease
    "mild_improvement": -0.02,         # 2-5% annual decrease
    "stable": 0.02,                    # -2% to +2%
    "mild_worsening": 0.05,            # 2-5% annual increase
    "significant_worsening": 0.10      # >5% annual increase
}

# Intervention templates by concern level
INTERVENTION_TEMPLATES = {
    "high_prevalence": [
        {
            "category": "Screening",
            "action": "Implement comprehensive screening program",
            "details": "Deploy mobile health units for early detection in affected areas.",
            "timeline": "Within 3 months",
            "resources": "Mobile clinics, diagnostic equipment, trained personnel"
        },
        {
            "category": "Treatment Access",
            "action": "Establish specialized treatment pathways",
            "details": "Create referral networks to ensure timely access to specialist care.",
            "timeline": "Within 6 months",
            "resources": "Specialist consultations, treatment protocols, referral systems"
        },
        {
            "category": "Community Outreach",
            "action": "Launch community health education campaign",
            "details": "Educate families on early warning signs and prevention measures.",
            "timeline": "Immediate",
            "resources": "Health educators, educational materials, community partnerships"
        }
    ],
    "moderate_prevalence": [
        {
            "category": "Monitoring",
            "action": "Enhanced surveillance program",
            "details": "Increase monitoring frequency to detect emerging trends early.",
            "timeline": "Within 3 months",
            "resources": "Data collection systems, monitoring personnel"
        },
        {
            "category": "Prevention",
            "action": "Strengthen preventive care programs",
            "details": "Expand vaccination, nutrition, and wellness programs.",
            "timeline": "Within 6 months",
            "resources": "Preventive care supplies, health workers"
        }
    ],
    "low_prevalence": [
        {
            "category": "Maintenance",
            "action": "Continue current health programs",
            "details": "Maintain existing successful prevention and care strategies.",
            "timeline": "Ongoing",
            "resources": "Standard allocation"
        },
        {
            "category": "Best Practices",
            "action": "Document and share successful approaches",
            "details": "Identify what works well and share with higher-risk provinces.",
            "timeline": "Within 6 months",
            "resources": "Knowledge management team"
        }
    ]
}


# =============================================================================
# TEMPORAL TREND ANALYSIS FUNCTIONS
# =============================================================================

def extract_temporal_data(raw_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Extract and organize temporal data by disease type.
    
    Args:
        raw_data: DataFrame with columns like 'Disease_Prevalence_pct_YYYY'
        
    Returns:
        Dictionary mapping disease names to DataFrames with years as columns
    """
    diseases = {}
    
    # Find all disease types from column names
    disease_pattern = re.compile(r'^(.+?)_Prevalence_pct_(\d{4})$')
    
    for col in raw_data.columns:
        match = disease_pattern.match(col)
        if match:
            disease_name = match.group(1)
            year = int(match.group(2))
            
            if disease_name not in diseases:
                diseases[disease_name] = {}
            diseases[disease_name][year] = raw_data[col]
    
    # Convert to DataFrames
    result = {}
    for disease, year_data in diseases.items():
        df = pd.DataFrame(year_data)
        df.columns = sorted(df.columns)  # Sort years
        result[disease] = df
    
    return result


def calculate_trend(series: pd.Series) -> Dict:
    """
    Calculate trend statistics for a time series.
    
    Uses linear regression slope and recent change analysis.
    
    Args:
        series: Time series data (index = years, values = prevalence)
        
    Returns:
        Dictionary with trend metrics
    """
    years = np.array(series.index).astype(float)
    values = series.values.astype(float)
    
    # Handle NaN values
    valid_mask = ~np.isnan(values)
    if valid_mask.sum() < 2:
        return {
            "slope": 0.0,
            "annual_change_pct": 0.0,
            "direction": "stable",
            "confidence": "low",
            "recent_trend": "stable",
            "start_value": float(values[0]) if len(values) > 0 else 0,
            "end_value": float(values[-1]) if len(values) > 0 else 0,
            "total_change_pct": 0.0
        }
    
    years_valid = years[valid_mask]
    values_valid = values[valid_mask]
    
    # Linear regression
    n = len(years_valid)
    mean_x = np.mean(years_valid)
    mean_y = np.mean(values_valid)
    
    numerator = np.sum((years_valid - mean_x) * (values_valid - mean_y))
    denominator = np.sum((years_valid - mean_x) ** 2)
    
    slope = numerator / denominator if denominator != 0 else 0
    
    # Calculate annual percentage change
    avg_value = np.mean(values_valid)
    annual_change_pct = (slope / avg_value * 100) if avg_value != 0 else 0
    
    # Determine trend direction
    if annual_change_pct < TREND_THRESHOLDS["significant_improvement"] * 100:
        direction = "significant_improvement"
    elif annual_change_pct < TREND_THRESHOLDS["mild_improvement"] * 100:
        direction = "mild_improvement"
    elif annual_change_pct < TREND_THRESHOLDS["stable"] * 100:
        direction = "stable"
    elif annual_change_pct < TREND_THRESHOLDS["mild_worsening"] * 100:
        direction = "mild_worsening"
    else:
        direction = "significant_worsening"
    
    # Calculate R-squared for confidence
    y_pred = slope * years_valid + (mean_y - slope * mean_x)
    ss_res = np.sum((values_valid - y_pred) ** 2)
    ss_tot = np.sum((values_valid - mean_y) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Alternative confidence: consistency of year-over-year direction
    # Aligns with the TREND_THRESHOLDS used for direction classification
    if len(values_valid) >= 2:
        year_changes = np.diff(values_valid)
        
        # Use the same threshold as direction classification (±2% = "stable")
        # Convert to absolute value threshold based on mean
        stable_threshold = abs(avg_value) * TREND_THRESHOLDS["stable"]  # 2% of average
        
        if direction == "stable":
            # For stable trends, count year-over-year changes that are small
            # (within the stable threshold range)
            consistent_years = np.sum(np.abs(year_changes) <= stable_threshold)
        elif direction in ["mild_improvement", "significant_improvement"]:
            # For improving (decreasing) trends, count negative changes
            consistent_years = np.sum(year_changes < 0)
        else:
            # For worsening (increasing) trends, count positive changes
            consistent_years = np.sum(year_changes > 0)
        
        consistency_ratio = consistent_years / len(year_changes)
    else:
        consistency_ratio = 0
    
    # Combined confidence using both R² and consistency
    # High: strong R² OR very consistent direction
    # Medium: moderate R² OR somewhat consistent
    # Low: weak R² AND inconsistent
    if r_squared > 0.4 or consistency_ratio > 0.7:
        confidence = "high"
    elif r_squared > 0.15 or consistency_ratio > 0.5:
        confidence = "medium"
    else:
        confidence = "low"
    
    # Recent trend (last 3 years)
    if len(values_valid) >= 3:
        recent_values = values_valid[-3:]
        recent_change = (recent_values[-1] - recent_values[0]) / recent_values[0] * 100 if recent_values[0] != 0 else 0
        if recent_change < -5:
            recent_trend = "improving"
        elif recent_change > 5:
            recent_trend = "worsening"
        else:
            recent_trend = "stable"
    else:
        recent_trend = "insufficient_data"
    
    # Total change
    total_change_pct = ((values_valid[-1] - values_valid[0]) / values_valid[0] * 100) if values_valid[0] != 0 else 0
    
    return {
        "slope": float(slope),
        "annual_change_pct": float(annual_change_pct),
        "direction": direction,
        "confidence": confidence,
        "r_squared": float(r_squared),
        "consistency_ratio": float(consistency_ratio),
        "recent_trend": recent_trend,
        "start_value": float(values_valid[0]),
        "end_value": float(values_valid[-1]),
        "total_change_pct": float(total_change_pct),
        "years_analyzed": int(n)
    }


def analyze_province_trends(
    province: str,
    raw_data: pd.DataFrame
) -> Dict:
    """
    Analyze temporal trends for a specific province across all diseases.
    
    Args:
        province: Name of the province
        raw_data: Raw data DataFrame with temporal columns
        
    Returns:
        Dictionary with trend analysis for each disease
    """
    if province not in raw_data.index:
        return {"error": f"Province {province} not found in data"}
    
    temporal_data = extract_temporal_data(raw_data)
    province_trends = {
        "province": province,
        "diseases": {},
        "overall_trend": "stable",
        "concerns": [],
        "improvements": [],
        "early_warnings": []
    }
    
    trend_scores = []
    
    for disease, df in temporal_data.items():
        if province in df.index:
            series = df.loc[province]
            trend = calculate_trend(series)
            trend["disease"] = disease
            trend["weight"] = DISEASE_WEIGHTS.get(disease, 1.0)
            province_trends["diseases"][disease] = trend
            
            # Weighted trend score (positive = worsening, negative = improving)
            weighted_score = trend["annual_change_pct"] * trend["weight"]
            trend_scores.append(weighted_score)
            
            # Identify concerning trends
            if trend["direction"] in ["significant_worsening", "mild_worsening"]:
                province_trends["concerns"].append({
                    "disease": disease,
                    "trend": trend["direction"],
                    "annual_change": f"+{trend['annual_change_pct']:.1f}%/year",
                    "severity": "high" if trend["direction"] == "significant_worsening" else "moderate"
                })
            
            # Identify improvements
            if trend["direction"] in ["significant_improvement", "mild_improvement"]:
                province_trends["improvements"].append({
                    "disease": disease,
                    "trend": trend["direction"],
                    "annual_change": f"{trend['annual_change_pct']:.1f}%/year"
                })
            
            # Early warning: Recent worsening despite overall stable/improving
            if trend["recent_trend"] == "worsening" and trend["direction"] not in ["significant_worsening", "mild_worsening"]:
                province_trends["early_warnings"].append({
                    "disease": disease,
                    "message": f"{disease} showing recent uptick despite historically stable trend",
                    "action": "Increased monitoring recommended"
                })
    
    # Overall trend
    if trend_scores:
        avg_score = np.mean(trend_scores)
        if avg_score < -2:
            province_trends["overall_trend"] = "improving"
        elif avg_score > 2:
            province_trends["overall_trend"] = "worsening"
        else:
            province_trends["overall_trend"] = "stable"
        province_trends["trend_score"] = float(avg_score)
    
    return province_trends


def generate_trend_recommendations(trend_analysis: Dict) -> List[Dict]:
    """
    Generate recommendations based on temporal trend analysis.
    """
    recommendations = []
    
    # Recommendations for worsening trends
    for concern in trend_analysis.get("concerns", []):
        disease = concern["disease"]
        severity = concern["severity"]
        
        if severity == "high":
            recommendations.append({
                "category": f"Urgent: {disease} Trend",
                "action": f"Emergency intervention for rising {disease} rates",
                "details": (f"{disease} prevalence is increasing at {concern['annual_change']}. "
                          f"This alarming trend requires immediate root cause analysis and intervention."),
                "timeline": "Immediate",
                "resources": "Emergency response team, epidemiological investigation",
                "priority": 1,
                "type": "trend_based"
            })
        else:
            recommendations.append({
                "category": f"Monitor: {disease} Trend",
                "action": f"Address gradual increase in {disease}",
                "details": (f"{disease} prevalence is trending upward ({concern['annual_change']}). "
                          f"Implement preventive measures to reverse this trend."),
                "timeline": "Within 2 months",
                "resources": "Prevention programs, health education",
                "priority": 2,
                "type": "trend_based"
            })
    
    # Early warning recommendations
    for warning in trend_analysis.get("early_warnings", []):
        recommendations.append({
            "category": "Early Warning",
            "action": warning["action"],
            "details": warning["message"],
            "timeline": "Within 1 month",
            "resources": "Surveillance systems, data analysis team",
            "priority": 2,
            "type": "early_warning"
        })
    
    # Recommendations to maintain improvements
    for improvement in trend_analysis.get("improvements", []):
        recommendations.append({
            "category": f"Sustain: {improvement['disease']}",
            "action": f"Maintain successful {improvement['disease']} reduction programs",
            "details": (f"{improvement['disease']} shows positive trend ({improvement['annual_change']}). "
                       f"Continue and document current interventions for replication."),
            "timeline": "Ongoing",
            "resources": "Program evaluation, documentation",
            "priority": 4,
            "type": "sustainability"
        })
    
    return recommendations


def calculate_weighted_risk_score(
    province_features: pd.Series,
    feature_columns: List[str]
) -> float:
    """
    Calculate risk score with disease-specific weighting.
    
    More critical diseases (e.g., Pneumonia) contribute more to the score.
    """
    weighted_sum = 0.0
    total_weight = 0.0
    
    for col in feature_columns:
        value = province_features.get(col, 0)
        
        # Determine disease type from column name
        weight = 1.0
        for disease, w in DISEASE_WEIGHTS.items():
            if disease in col:
                weight = w
                break
        
        weighted_sum += value * weight
        total_weight += weight
    
    return weighted_sum / total_weight if total_weight > 0 else 0.0


def identify_disease_hotspots(
    normalized_data: pd.DataFrame,
    threshold: float = 0.7
) -> Dict[str, List[str]]:
    """
    Identify provinces that are hotspots for specific diseases.
    
    Args:
        normalized_data: Normalized feature DataFrame
        threshold: Value above which a province is considered a hotspot
        
    Returns:
        Dictionary mapping disease to list of hotspot provinces
    """
    hotspots = {}
    
    for disease in DISEASE_WEIGHTS.keys():
        disease_cols = [c for c in normalized_data.columns if disease in c]
        if disease_cols:
            # Average across all years for this disease
            disease_avg = normalized_data[disease_cols].mean(axis=1)
            hotspot_provinces = disease_avg[disease_avg > threshold].index.tolist()
            if hotspot_provinces:
                hotspots[disease] = hotspot_provinces
    
    return hotspots


def generate_comparative_analysis(
    province: str,
    normalized_data: pd.DataFrame,
    cluster_labels: np.ndarray
) -> Dict:
    """
    Generate comparative analysis of a province vs its cluster and population.
    """
    if province not in normalized_data.index:
        return {"error": f"Province {province} not found"}
    
    province_idx = normalized_data.index.get_loc(province)
    cluster_id = cluster_labels[province_idx]
    
    province_data = normalized_data.loc[province]
    cluster_mask = cluster_labels == cluster_id
    cluster_data = normalized_data.iloc[cluster_mask]
    
    comparison = {
        "province": province,
        "cluster_id": int(cluster_id),
        "vs_cluster": {},
        "vs_population": {},
        "rank_in_cluster": {},
        "rank_in_population": {}
    }
    
    for disease in DISEASE_WEIGHTS.keys():
        disease_cols = [c for c in normalized_data.columns if disease in c]
        if disease_cols:
            province_avg = province_data[disease_cols].mean()
            cluster_avg = cluster_data[disease_cols].mean().mean()
            pop_avg = normalized_data[disease_cols].mean().mean()
            
            comparison["vs_cluster"][disease] = {
                "province_value": float(province_avg),
                "cluster_avg": float(cluster_avg),
                "difference_pct": float((province_avg - cluster_avg) / cluster_avg * 100) if cluster_avg != 0 else 0,
                "status": "above" if province_avg > cluster_avg * 1.1 else ("below" if province_avg < cluster_avg * 0.9 else "similar")
            }
            
            comparison["vs_population"][disease] = {
                "province_value": float(province_avg),
                "population_avg": float(pop_avg),
                "difference_pct": float((province_avg - pop_avg) / pop_avg * 100) if pop_avg != 0 else 0,
                "status": "above" if province_avg > pop_avg * 1.1 else ("below" if province_avg < pop_avg * 0.9 else "similar")
            }
            
            # Rank within cluster
            all_cluster_avgs = cluster_data[disease_cols].mean(axis=1).sort_values(ascending=False)
            rank_in_cluster = list(all_cluster_avgs.index).index(province) + 1 if province in all_cluster_avgs.index else None
            comparison["rank_in_cluster"][disease] = rank_in_cluster
            
            # Rank in population
            all_pop_avgs = normalized_data[disease_cols].mean(axis=1).sort_values(ascending=False)
            rank_in_pop = list(all_pop_avgs.index).index(province) + 1 if province in all_pop_avgs.index else None
            comparison["rank_in_population"][disease] = rank_in_pop
    
    return comparison


# =============================================================================
# ORIGINAL CLUSTER ANALYSIS FUNCTIONS (ENHANCED)
# =============================================================================

def analyze_cluster_risk_levels(
    normalized_data: pd.DataFrame,
    cluster_labels: np.ndarray
) -> Dict[int, Dict]:
    """
    Dynamically determine risk levels for each cluster based on actual feature values.
    
    Higher average feature values = higher disease burden = higher risk
    (Assumes features represent disease prevalence/rates where higher = worse)
    
    Args:
        normalized_data: DataFrame with normalized features (index=Province)
        cluster_labels: Array of cluster assignments
        
    Returns:
        Dictionary mapping cluster_id to risk profile
    """
    cluster_profiles = {}
    cluster_scores = {}
    
    # Calculate mean feature values for each cluster
    unique_clusters = np.unique(cluster_labels)
    
    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        cluster_data = normalized_data.iloc[mask]
        
        # Overall risk score = mean of all feature means
        cluster_mean = cluster_data.mean()
        overall_score = cluster_mean.mean()
        cluster_scores[int(cluster_id)] = overall_score
        
        # Store detailed feature analysis
        cluster_profiles[int(cluster_id)] = {
            "feature_means": cluster_mean.to_dict(),
            "feature_stds": cluster_data.std().to_dict(),
            "overall_score": float(overall_score),
            "count": int(mask.sum()),
            "percentage": float(mask.sum() / len(cluster_labels) * 100),
            "provinces": normalized_data.index[mask].tolist()
        }
    
    # Rank clusters by overall score to assign risk levels
    sorted_clusters = sorted(cluster_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Assign risk levels based on ranking
    risk_levels = ["High Risk", "Moderate Risk", "Low Risk"]
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    priorities = ["High", "Medium", "Low"]
    priority_scores = [3, 2, 1]
    
    for rank, (cluster_id, score) in enumerate(sorted_clusters):
        level_idx = min(rank, len(risk_levels) - 1)
        
        cluster_profiles[cluster_id].update({
            "risk_level": risk_levels[level_idx],
            "color": colors[level_idx],
            "priority": priorities[level_idx],
            "priority_score": priority_scores[level_idx],
            "risk_rank": rank + 1
        })
        
        # Generate dynamic description based on actual features
        cluster_profiles[cluster_id]["description"] = _generate_cluster_description(
            cluster_profiles[cluster_id], normalized_data.columns.tolist()
        )
        cluster_profiles[cluster_id]["characteristics"] = _generate_cluster_characteristics(
            cluster_profiles[cluster_id], normalized_data.mean()
        )
    
    return cluster_profiles


def _generate_cluster_description(cluster_profile: Dict, feature_names: List[str]) -> str:
    """Generate a data-driven description for a cluster."""
    risk_level = cluster_profile["risk_level"]
    score = cluster_profile["overall_score"]
    count = cluster_profile["count"]
    
    if risk_level == "High Risk":
        return (f"This cluster contains {count} provinces with the highest overall disease burden "
                f"(composite score: {score:.3f}). These provinces show elevated rates across multiple "
                f"health indicators and require priority intervention.")
    elif risk_level == "Moderate Risk":
        return (f"This cluster contains {count} provinces with moderate disease burden "
                f"(composite score: {score:.3f}). These provinces show mixed patterns across "
                f"health indicators and benefit from targeted prevention programs.")
    else:
        return (f"This cluster contains {count} provinces with relatively low disease burden "
                f"(composite score: {score:.3f}). These provinces maintain better health outcomes "
                f"and can serve as models for best practices.")


def _generate_cluster_characteristics(cluster_profile: Dict, population_mean: pd.Series) -> List[str]:
    """Generate data-driven characteristics based on feature comparison."""
    characteristics = []
    feature_means = cluster_profile["feature_means"]
    
    high_features = []
    low_features = []
    
    for feat, val in feature_means.items():
        pop_mean = population_mean.get(feat, 0.5)
        if val > pop_mean * 1.2:
            high_features.append(feat)
        elif val < pop_mean * 0.8:
            low_features.append(feat)
    
    if high_features:
        characteristics.append(f"Above-average rates in: {', '.join(high_features[:3])}")
    if low_features:
        characteristics.append(f"Below-average rates in: {', '.join(low_features[:3])}")
    
    characteristics.append(f"Overall composite score: {cluster_profile['overall_score']:.3f}")
    characteristics.append(f"Contains {cluster_profile['count']} provinces")
    
    return characteristics


def analyze_province_features(
    province: str,
    province_features: pd.Series,
    population_stats: Dict,
    cluster_profile: Dict
) -> Dict:
    """
    Analyze a specific province's features and generate tailored insights.
    """
    pop_mean = population_stats['mean']
    pop_std = population_stats['std']
    
    analysis = {
        "province": province,
        "concerns": [],
        "strengths": [],
        "feature_status": {}
    }
    
    for feat, val in province_features.items():
        mean = pop_mean.get(feat, 0.5)
        std = pop_std.get(feat, 0.1)
        
        z_score = (val - mean) / std if std > 0 else 0
        
        status = {
            "value": float(val),
            "population_mean": float(mean),
            "z_score": float(z_score),
            "percentile": _z_to_percentile(z_score)
        }
        
        if z_score > 1.5:
            status["level"] = "critical"
            status["interpretation"] = f"Significantly above average (top {100-status['percentile']:.0f}%)"
            analysis["concerns"].append({
                "feature": feat,
                "severity": "high",
                "value": val,
                "vs_mean": f"{((val/mean)-1)*100:.1f}% above average" if mean > 0 else "N/A",
                "recommendation": f"Priority intervention needed for {feat}"
            })
        elif z_score > 0.5:
            status["level"] = "elevated"
            status["interpretation"] = f"Above average (top {100-status['percentile']:.0f}%)"
            analysis["concerns"].append({
                "feature": feat,
                "severity": "moderate",
                "value": val,
                "vs_mean": f"{((val/mean)-1)*100:.1f}% above average" if mean > 0 else "N/A",
                "recommendation": f"Enhanced monitoring recommended for {feat}"
            })
        elif z_score < -1.0:
            status["level"] = "low"
            status["interpretation"] = f"Well below average (bottom {status['percentile']:.0f}%)"
            analysis["strengths"].append({
                "feature": feat,
                "value": val,
                "interpretation": f"Performing well in {feat}"
            })
        else:
            status["level"] = "normal"
            status["interpretation"] = "Within normal range"
        
        analysis["feature_status"][feat] = status
    
    return analysis


def _z_to_percentile(z: float) -> float:
    """Convert z-score to approximate percentile."""
    from math import erf, sqrt
    return 50 * (1 + erf(z / sqrt(2)))


def generate_province_recommendations(
    province_analysis: Dict,
    cluster_profile: Dict
) -> List[Dict]:
    """
    Generate tailored recommendations based on province-specific analysis.
    """
    recommendations = []
    
    # Add recommendations based on specific concerns
    for concern in province_analysis["concerns"]:
        severity = concern["severity"]
        feature = concern["feature"]
        
        if severity == "high":
            recommendations.append({
                "category": f"Critical: {feature}",
                "action": f"Immediate intervention for {feature}",
                "details": (f"This province shows significantly elevated {feature} "
                           f"({concern['vs_mean']}). Recommend immediate assessment "
                           f"and targeted intervention program."),
                "timeline": "Immediate - within 1 month",
                "resources": "Emergency health funds, specialist teams",
                "priority": 1
            })
        elif severity == "moderate":
            recommendations.append({
                "category": f"Monitoring: {feature}",
                "action": f"Enhanced surveillance for {feature}",
                "details": (f"This province shows elevated {feature} ({concern['vs_mean']}). "
                           f"Recommend increased monitoring and preventive measures."),
                "timeline": "Within 3 months",
                "resources": "Monitoring systems, prevention programs",
                "priority": 2
            })
    
    # Add cluster-level recommendations
    risk_level = cluster_profile.get("risk_level", "Moderate Risk")
    
    if risk_level == "High Risk":
        recommendations.extend([
            {
                "category": "Comprehensive Assessment",
                "action": "Conduct full health system assessment",
                "details": "Evaluate healthcare infrastructure, access, and capacity in the province.",
                "timeline": "Within 2 months",
                "resources": "Assessment team, evaluation tools",
                "priority": 2
            },
            {
                "category": "Resource Mobilization",
                "action": "Allocate additional health resources",
                "details": "Increase budget allocation and deploy additional healthcare workers.",
                "timeline": "Within 3 months",
                "resources": "Budget reallocation, personnel recruitment",
                "priority": 1
            }
        ])
    elif risk_level == "Moderate Risk":
        recommendations.extend([
            {
                "category": "Prevention Focus",
                "action": "Strengthen preventive care programs",
                "details": "Expand early intervention and health education initiatives.",
                "timeline": "Within 6 months",
                "resources": "Prevention program funding, health educators",
                "priority": 2
            }
        ])
    else:
        recommendations.extend([
            {
                "category": "Sustainability",
                "action": "Maintain and document best practices",
                "details": "Continue current successful programs and document approaches for replication.",
                "timeline": "Ongoing",
                "resources": "Standard allocation, documentation support",
                "priority": 3
            }
        ])
    
    # Add knowledge sharing for provinces with strengths
    if province_analysis["strengths"] and risk_level != "High Risk":
        strength_features = [s["feature"] for s in province_analysis["strengths"][:2]]
        recommendations.append({
            "category": "Knowledge Sharing",
            "action": "Share successful approaches",
            "details": f"This province excels in {', '.join(strength_features)}. "
                      f"Document and share these practices with higher-risk provinces.",
            "timeline": "Within 6 months",
            "resources": "Knowledge management team",
            "priority": 3
        })
    
    recommendations.sort(key=lambda x: x.get("priority", 99))
    return recommendations


def generate_cluster_summary(
    projection: pd.DataFrame,
    cluster_labels: np.ndarray,
    normalized_data: Optional[pd.DataFrame] = None
) -> Dict:
    """
    Generate a comprehensive summary of all clusters with data-driven profiling.
    """
    summary = {
        "total_provinces": len(projection),
        "num_clusters": len(np.unique(cluster_labels)),
        "clusters": {}
    }
    
    if normalized_data is not None:
        cluster_profiles = analyze_cluster_risk_levels(normalized_data, cluster_labels)
        summary["clusters"] = cluster_profiles
        summary["data_driven"] = True
    else:
        provinces = projection.index.tolist()
        for cluster_id in np.unique(cluster_labels):
            mask = cluster_labels == cluster_id
            cluster_provinces = [p for p, m in zip(provinces, mask) if m]
            cluster_points = projection.iloc[mask][['Dim1', 'Dim2']].values
            centroid = np.mean(cluster_points, axis=0)
            
            summary["clusters"][int(cluster_id)] = {
                "count": int(np.sum(mask)),
                "percentage": float(np.sum(mask) / len(cluster_labels) * 100),
                "provinces": cluster_provinces,
                "centroid": {"Dim1": float(centroid[0]), "Dim2": float(centroid[1])},
                "risk_level": "Unknown",
                "color": "#95a5a6",
                "priority": "Medium",
                "priority_score": 2,
                "description": f"Cluster {cluster_id} contains {np.sum(mask)} provinces.",
                "characteristics": [f"Cluster centroid at ({centroid[0]:.2f}, {centroid[1]:.2f})"]
            }
        summary["data_driven"] = False
    
    return summary


def rank_provinces_by_priority(
    projection: pd.DataFrame,
    cluster_labels: np.ndarray,
    normalized_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Rank all provinces by intervention priority using data-driven scoring.
    """
    provinces = projection.index.tolist()
    
    if normalized_data is not None:
        cluster_profiles = analyze_cluster_risk_levels(normalized_data, cluster_labels)
    else:
        cluster_profiles = {}
    
    rankings = []
    for i, (province, cluster_id) in enumerate(zip(provinces, cluster_labels)):
        cluster_id = int(cluster_id)
        
        profile = cluster_profiles.get(cluster_id, {
            "risk_level": "Unknown",
            "priority": "Medium",
            "priority_score": 2,
            "color": "#95a5a6"
        })
        
        # Anomaly score
        cluster_mask = cluster_labels == cluster_id
        cluster_points = projection.iloc[cluster_mask][['Dim1', 'Dim2']].values
        centroid = np.mean(cluster_points, axis=0)
        point = projection.iloc[i][['Dim1', 'Dim2']].values
        distance_from_centroid = np.linalg.norm(point - centroid)
        
        # Severity score from features
        if normalized_data is not None:
            province_features = normalized_data.iloc[i]
            severity_score = province_features.mean()
        else:
            severity_score = 0.5
        
        base_score = profile.get("priority_score", 2)
        anomaly_score = min(distance_from_centroid / 2, 1)
        composite_score = base_score + anomaly_score * 0.3 + severity_score * 0.5
        
        rankings.append({
            "Province": province,
            "Cluster": cluster_id,
            "Risk Level": profile.get("risk_level", "Unknown"),
            "Priority": profile.get("priority", "Medium"),
            "Base Score": base_score,
            "Severity Score": round(severity_score, 3),
            "Anomaly Score": round(anomaly_score, 3),
            "Composite Score": round(composite_score, 3),
            "Color": profile.get("color", "#95a5a6")
        })
    
    df_rankings = pd.DataFrame(rankings)
    df_rankings = df_rankings.sort_values("Composite Score", ascending=False)
    df_rankings["Rank"] = range(1, len(df_rankings) + 1)
    
    return df_rankings


def generate_action_plan(
    province: str,
    cluster_id: int,
    rank: int,
    total_provinces: int,
    normalized_data: Optional[pd.DataFrame] = None,
    cluster_labels: Optional[np.ndarray] = None,
    raw_data: Optional[pd.DataFrame] = None
) -> Dict:
    """
    Generate a detailed, DATA-DRIVEN action plan for a specific province.
    
    Enhanced with temporal trend analysis when raw_data is provided.
    """
    if normalized_data is not None and cluster_labels is not None:
        cluster_profiles = analyze_cluster_risk_levels(normalized_data, cluster_labels)
        profile = cluster_profiles.get(cluster_id, {})
        
        province_idx = normalized_data.index.get_loc(province) if province in normalized_data.index else None
        if province_idx is not None:
            province_features = normalized_data.iloc[province_idx]
            pop_stats = {
                'mean': normalized_data.mean(),
                'std': normalized_data.std()
            }
            province_analysis = analyze_province_features(
                province, province_features, pop_stats, profile
            )
            recommendations = generate_province_recommendations(province_analysis, profile)
        else:
            province_analysis = {"concerns": [], "strengths": [], "feature_status": {}}
            recommendations = []
    else:
        profile = {
            "risk_level": "Unknown",
            "priority": "Medium",
            "priority_score": 2
        }
        province_analysis = {"concerns": [], "strengths": [], "feature_status": {}}
        recommendations = []
    
    # Temporal trend analysis (if raw_data provided)
    trend_analysis = None
    trend_recommendations = []
    if raw_data is not None and province in raw_data.index:
        trend_analysis = analyze_province_trends(province, raw_data)
        trend_recommendations = generate_trend_recommendations(trend_analysis)
        # Merge trend recommendations with existing ones
        recommendations.extend(trend_recommendations)
        recommendations.sort(key=lambda x: x.get("priority", 99))
    
    # Comparative analysis
    comparative = None
    if normalized_data is not None and cluster_labels is not None:
        comparative = generate_comparative_analysis(province, normalized_data, cluster_labels)
    
    # Urgency level
    percentile = (total_provinces - rank + 1) / total_provinces * 100
    if percentile >= 80:
        urgency = "Critical"
        urgency_color = "#e74c3c"
    elif percentile >= 50:
        urgency = "High"
        urgency_color = "#f39c12"
    elif percentile >= 20:
        urgency = "Moderate"
        urgency_color = "#3498db"
    else:
        urgency = "Standard"
        urgency_color = "#2ecc71"
    
    # Generate summary
    risk_level = profile.get("risk_level", "Unknown")
    num_concerns = len(province_analysis["concerns"])
    num_strengths = len(province_analysis["strengths"])
    
    summary = (f"{province} is classified as {risk_level} (Cluster {cluster_id}) "
               f"and ranked #{rank} out of {total_provinces} provinces. ")
    
    if num_concerns > 0:
        concern_features = [c["feature"] for c in province_analysis["concerns"][:2]]
        summary += f"Key concerns: {', '.join(concern_features)}. "
    if num_strengths > 0:
        strength_features = [s["feature"] for s in province_analysis["strengths"][:2]]
        summary += f"Strengths: {', '.join(strength_features)}. "
    
    # Add trend summary
    if trend_analysis:
        overall_trend = trend_analysis.get("overall_trend", "stable")
        if overall_trend == "worsening":
            summary += "⚠️ Overall health indicators are WORSENING over time."
        elif overall_trend == "improving":
            summary += "✓ Overall health indicators are IMPROVING over time."
        else:
            summary += "Health indicators are relatively STABLE over time."
    
    result = {
        "province": province,
        "cluster": cluster_id,
        "risk_level": risk_level,
        "rank": rank,
        "percentile": round(percentile, 1),
        "urgency": urgency,
        "urgency_color": urgency_color,
        "summary": summary,
        "concerns": province_analysis["concerns"],
        "strengths": province_analysis["strengths"],
        "feature_status": province_analysis["feature_status"],
        "key_actions": [rec["action"] for rec in recommendations[:3]] if recommendations else [],
        "detailed_recommendations": recommendations,
        "monitoring_frequency": _get_monitoring_frequency(profile.get("priority", "Medium")),
        "review_timeline": _get_review_timeline(profile.get("priority", "Medium"))
    }
    
    # Add temporal analysis if available
    if trend_analysis:
        result["temporal_trends"] = trend_analysis
        result["early_warnings"] = trend_analysis.get("early_warnings", [])
    
    # Add comparative analysis if available
    if comparative:
        result["comparative_analysis"] = comparative
    
    return result


def _get_monitoring_frequency(priority: str) -> str:
    frequencies = {"Low": "Annually", "Medium": "Quarterly", "High": "Monthly"}
    return frequencies.get(priority, "Quarterly")


def _get_review_timeline(priority: str) -> str:
    timelines = {"Low": "12 months", "Medium": "6 months", "High": "3 months"}
    return timelines.get(priority, "6 months")


def generate_executive_summary(
    cluster_summary: Dict,
    rankings: pd.DataFrame,
    raw_data: Optional[pd.DataFrame] = None
) -> Dict:
    """
    Generate an executive summary for decision makers.
    
    Enhanced with temporal trend analysis when raw_data is provided.
    """
    high_risk = rankings[rankings["Risk Level"] == "High Risk"]
    moderate_risk = rankings[rankings["Risk Level"] == "Moderate Risk"]
    low_risk = rankings[rankings["Risk Level"] == "Low Risk"]
    
    summary = {
        "title": "Chronic Disease in Children: Provincial Risk Assessment",
        "total_provinces": cluster_summary["total_provinces"],
        "data_driven": cluster_summary.get("data_driven", False),
        "risk_distribution": {
            "High Risk": len(high_risk),
            "Moderate Risk": len(moderate_risk),
            "Low Risk": len(low_risk)
        },
        "top_priority_provinces": rankings.head(5)["Province"].tolist(),
        "key_findings": [],
        "immediate_actions": [],
        "resource_allocation_suggestion": _suggest_resource_allocation(rankings),
        "temporal_insights": None,
        "early_warnings": [],
        "disease_hotspots": {}
    }
    
    total = len(rankings)
    if total > 0:
        if len(high_risk) > 0:
            summary["key_findings"].append(
                f"{len(high_risk)} provinces ({len(high_risk)/total*100:.1f}%) identified as High Risk "
                f"requiring urgent intervention"
            )
        if len(moderate_risk) > 0:
            summary["key_findings"].append(
                f"{len(moderate_risk)} provinces ({len(moderate_risk)/total*100:.1f}%) identified as Moderate Risk "
                f"requiring enhanced monitoring"
            )
        if len(low_risk) > 0:
            summary["key_findings"].append(
                f"{len(low_risk)} provinces ({len(low_risk)/total*100:.1f}%) maintaining good health outcomes"
            )
    
    # Temporal analysis across all provinces
    if raw_data is not None:
        temporal_insights = analyze_population_trends(raw_data)
        summary["temporal_insights"] = temporal_insights
        
        # Add trend-based findings
        for disease, trend in temporal_insights.get("disease_trends", {}).items():
            direction = trend.get("direction", "stable")
            if direction in ["significant_worsening", "mild_worsening"]:
                summary["key_findings"].append(
                    f"⚠️ {disease} shows concerning upward trend "
                    f"({trend.get('annual_change_pct', 0):+.1f}%/year across all provinces)"
                )
                summary["early_warnings"].append({
                    "type": "population_trend",
                    "disease": disease,
                    "message": f"{disease} prevalence is increasing population-wide",
                    "severity": "high" if direction == "significant_worsening" else "moderate"
                })
            elif direction in ["significant_improvement", "mild_improvement"]:
                summary["key_findings"].append(
                    f"✓ {disease} shows positive downward trend "
                    f"({trend.get('annual_change_pct', 0):+.1f}%/year)"
                )
        
        # Identify provinces with worsening trends
        worsening_provinces = []
        for province in raw_data.index:
            province_trend = analyze_province_trends(province, raw_data)
            if province_trend.get("overall_trend") == "worsening":
                worsening_provinces.append(province)
        
        if worsening_provinces:
            summary["early_warnings"].append({
                "type": "province_deterioration",
                "provinces": worsening_provinces,
                "message": f"{len(worsening_provinces)} province(s) showing worsening health trends",
                "severity": "high"
            })
    
    if len(high_risk) > 0:
        top_high_risk = high_risk.head(3)["Province"].tolist()
        summary["immediate_actions"].append(
            f"Deploy emergency resources to: {', '.join(top_high_risk)}"
        )
    
    # Add trend-based actions
    if summary["early_warnings"]:
        summary["immediate_actions"].append(
            "Investigate root causes of worsening health trends identified by temporal analysis"
        )
    
    summary["immediate_actions"].extend([
        "Establish inter-provincial health coordination committee",
        "Initiate comprehensive data collection for continuous monitoring",
        "Develop province-specific intervention protocols based on identified concerns"
    ])
    
    return summary


def analyze_population_trends(raw_data: pd.DataFrame) -> Dict:
    """
    Analyze temporal trends across the entire population.
    
    Returns aggregated trend analysis for each disease type.
    """
    temporal_data = extract_temporal_data(raw_data)
    
    results = {
        "disease_trends": {},
        "overall_trend": "stable",
        "years_analyzed": 0,
        "provinces_analyzed": len(raw_data)
    }
    
    trend_scores = []
    
    for disease, df in temporal_data.items():
        # Calculate population-wide average for each year
        yearly_avg = df.mean(axis=0)
        trend = calculate_trend(yearly_avg)
        trend["disease"] = disease
        results["disease_trends"][disease] = trend
        results["years_analyzed"] = max(results["years_analyzed"], trend.get("years_analyzed", 0))
        
        # Weighted score
        weight = DISEASE_WEIGHTS.get(disease, 1.0)
        trend_scores.append(trend["annual_change_pct"] * weight)
    
    # Overall population trend
    if trend_scores:
        avg_score = np.mean(trend_scores)
        if avg_score < -2:
            results["overall_trend"] = "improving"
        elif avg_score > 2:
            results["overall_trend"] = "worsening"
        else:
            results["overall_trend"] = "stable"
        results["trend_score"] = float(avg_score)
    
    return results


def _suggest_resource_allocation(rankings: pd.DataFrame) -> Dict:
    high_risk = len(rankings[rankings["Risk Level"] == "High Risk"])
    moderate_risk = len(rankings[rankings["Risk Level"] == "Moderate Risk"])
    low_risk = len(rankings[rankings["Risk Level"] == "Low Risk"])
    total = len(rankings)
    
    if total > 0:
        high_weight = high_risk * 3
        moderate_weight = moderate_risk * 2
        low_weight = low_risk * 1
        total_weight = high_weight + moderate_weight + low_weight
        
        if total_weight > 0:
            return {
                "High Risk Provinces": round(high_weight / total_weight * 100, 1),
                "Moderate Risk Provinces": round(moderate_weight / total_weight * 100, 1),
                "Low Risk Provinces": round(low_weight / total_weight * 100, 1)
            }
    
    return {"High Risk Provinces": 50, "Moderate Risk Provinces": 35, "Low Risk Provinces": 15}


def get_cluster_profile(cluster_id: int, cluster_profiles: Optional[Dict] = None) -> Dict:
    """Get profile for a cluster."""
    if cluster_profiles and cluster_id in cluster_profiles:
        return cluster_profiles[cluster_id]
    return {
        "risk_level": "Unknown",
        "color": "#95a5a6",
        "priority": "Medium",
        "priority_score": 2,
        "description": f"Cluster {cluster_id}",
        "characteristics": []
    }


def get_recommendations_for_risk(risk_level: str) -> List[Dict]:
    """Get generic recommendations for a risk level."""
    if risk_level == "High Risk":
        return INTERVENTION_TEMPLATES["high_prevalence"]
    elif risk_level == "Moderate Risk":
        return INTERVENTION_TEMPLATES["moderate_prevalence"]
    else:
        return INTERVENTION_TEMPLATES["low_prevalence"]
