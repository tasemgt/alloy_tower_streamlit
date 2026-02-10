"""
Plotting and visualization utilities with best practices for data analysis.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

from app.constants import COUNTY_COL, PROPERTY_TYPE_COL, CURRENT_PRICE_COL


def plot_top_counties(df: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """
    Create a horizontal bar chart of top counties by listing count.
    
    Args:
        df: Listings DataFrame
        top_n: Number of top counties to show
        
    Returns:
        Matplotlib figure
    """
    county_counts = df[COUNTY_COL].astype(str).value_counts().head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(county_counts.index.astype(str), county_counts.values, color='#667eea')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{int(width):,}', 
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    ax.set_xlabel("Number of Listings", fontsize=11, fontweight='bold')
    ax.set_ylabel("County", fontsize=11, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    return fig


def plot_avg_price_by_property_type(df: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """
    Create a bar chart of average price by property type.
    
    Args:
        df: Listings DataFrame
        top_n: Number of top property types to show
        
    Returns:
        Matplotlib figure
    """
    avg_price_pt = (
        df.groupby(df[PROPERTY_TYPE_COL].astype(str))[CURRENT_PRICE_COL]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(avg_price_pt)), avg_price_pt.values, color='#764ba2')
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f'${height/1000:.0f}K',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xticks(range(len(avg_price_pt)))
    ax.set_xticklabels(avg_price_pt.index.astype(str), rotation=45, ha='right')
    ax.set_ylabel("Average Price ($)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Property Type", fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    return fig


def plot_price_distribution(df: pd.DataFrame, bins: int = 30) -> plt.Figure:
    """
    Create a histogram showing price distribution.
    
    Args:
        df: Listings DataFrame
        bins: Number of histogram bins
        
    Returns:
        Matplotlib figure
    """
    prices = df[CURRENT_PRICE_COL].dropna()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    n, bins_edges, patches = ax.hist(prices, bins=bins, color='#667eea', alpha=0.7, edgecolor='black')
    
    # Add median and mean lines
    median_price = prices.median()
    mean_price = prices.mean()
    
    ax.axvline(median_price, color='red', linestyle='--', linewidth=2, label=f'Median: ${median_price:,.0f}')
    ax.axvline(mean_price, color='orange', linestyle='--', linewidth=2, label=f'Mean: ${mean_price:,.0f}')
    
    ax.set_xlabel("Price ($)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Frequency", fontsize=11, fontweight='bold')
    ax.set_title("Property Price Distribution", fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    return fig


def plot_dom_distribution(df: pd.DataFrame, bins: int = 30) -> plt.Figure:
    """
    Create a histogram showing days on market distribution.
    
    Args:
        df: Listings DataFrame
        bins: Number of histogram bins
        
    Returns:
        Matplotlib figure
    """
    dom = pd.to_numeric(df['days_on_market'], errors='coerce').dropna()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    n, bins_edges, patches = ax.hist(dom, bins=bins, color='#764ba2', alpha=0.7, edgecolor='black')
    
    # Add median and mean lines
    median_dom = dom.median()
    mean_dom = dom.mean()
    
    ax.axvline(median_dom, color='red', linestyle='--', linewidth=2, label=f'Median: {median_dom:.0f} days')
    ax.axvline(mean_dom, color='orange', linestyle='--', linewidth=2, label=f'Mean: {mean_dom:.0f} days')
    
    # Add risk zones
    ax.axvspan(0, 60, alpha=0.1, color='green', label='Low Risk (≤60 days)')
    ax.axvspan(60, 120, alpha=0.1, color='yellow', label='Medium Risk (61-120 days)')
    ax.axvspan(120, dom.max(), alpha=0.1, color='red', label='High Risk (>120 days)')
    
    ax.set_xlabel("Days on Market", fontsize=11, fontweight='bold')
    ax.set_ylabel("Frequency", fontsize=11, fontweight='bold')
    ax.set_title("Days on Market Distribution", fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    return fig


def plot_price_vs_sqft_scatter(df: pd.DataFrame, sample_size: int = 1000) -> plt.Figure:
    """
    Create a scatter plot of price vs square footage.
    
    Args:
        df: Listings DataFrame
        sample_size: Number of points to plot (for performance)
        
    Returns:
        Matplotlib figure
    """
    # Sample data for performance
    plot_df = df[[CURRENT_PRICE_COL, 'square_footage']].dropna()
    if len(plot_df) > sample_size:
        plot_df = plot_df.sample(sample_size)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(
        plot_df['square_footage'], 
        plot_df[CURRENT_PRICE_COL],
        alpha=0.5, 
        c=plot_df[CURRENT_PRICE_COL],
        cmap='viridis',
        s=30
    )
    
    # Add trend line
    z = np.polyfit(plot_df['square_footage'], plot_df[CURRENT_PRICE_COL], 1)
    p = np.poly1d(z)
    ax.plot(plot_df['square_footage'].sort_values(), 
            p(plot_df['square_footage'].sort_values()), 
            "r--", alpha=0.8, linewidth=2, label='Trend')
    
    ax.set_xlabel("Square Footage", fontsize=11, fontweight='bold')
    ax.set_ylabel("Price ($)", fontsize=11, fontweight='bold')
    ax.set_title("Price vs Square Footage", fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, linestyle='--')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Price ($)', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """
    Create a correlation heatmap for numeric features.
    
    Args:
        df: Listings DataFrame
        
    Returns:
        Matplotlib figure
    """
    # Select numeric columns
    numeric_cols = ['current_price', 'square_footage', 'bedrooms', 'bathrooms', 
                    'lot_size', 'year_built', 'days_on_market', 'hoa_fee', 'price_per_sq_ft']
    
    available_cols = [col for col in numeric_cols if col in df.columns]
    corr_df = df[available_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(corr_df, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_yticks(range(len(corr_df.columns)))
    ax.set_xticklabels([col.replace('_', ' ').title() for col in corr_df.columns], rotation=45, ha='right')
    ax.set_yticklabels([col.replace('_', ' ').title() for col in corr_df.columns])
    
    # Add correlation values
    for i in range(len(corr_df.columns)):
        for j in range(len(corr_df.columns)):
            text = ax.text(j, i, f'{corr_df.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    return fig
