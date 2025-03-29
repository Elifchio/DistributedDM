import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
import os

def ensure_plots_dir():
    """Ensure the plots directory exists"""
    if not os.path.exists('./plots'):
        os.makedirs('./plots')

def plot_missing_values(df: DataFrame, figsize=(12, 6), save=True):
    """Plot top 10 features with most missing values using direct DataFrame calculation"""
    # Calculate null percentages for all columns using PySpark
    null_counts = []
    total_count = df.count()
    
    for col in df.columns:
        null_count = df.filter(F.col(col).isNull()).count()
        null_counts.append({
            'column': col,
            'null_percentage': (null_count / total_count) * 100
        })
    
    # Convert to Pandas only for final plotting
    df_nulls = pd.DataFrame(null_counts)
    df_nulls = df_nulls.sort_values('null_percentage', ascending=False).head(10)
    
    plt.figure(figsize=figsize)
    sns.barplot(data=df_nulls, x='column', y='null_percentage')
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 10 Features with Missing Values')
    plt.xlabel('Features')
    plt.ylabel('Missing Values (%)')
    plt.tight_layout()
    
    if save:
        ensure_plots_dir()
        plt.savefig('./plots/missing_values.png')
    return plt

def plot_show_name_frequency(df: DataFrame, figsize=(12, 6), top_n=10, save=True):
    """Plot frequency distribution of show names using PySpark aggregation"""
    show_counts = df.groupBy('show_name') \
                   .agg(F.count('*').alias('count')) \
                   .orderBy(F.desc('count')) \
                   .limit(top_n) \
                   .toPandas()
    
    plt.figure(figsize=figsize)
    sns.barplot(data=show_counts, x='show_name', y='count')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Top {top_n} Show Names by Frequency')
    plt.xlabel('Show Name')
    plt.ylabel('Count')
    plt.tight_layout()
    
    if save:
        ensure_plots_dir()
        plt.savefig('./plots/show_name_frequency.png')
    return plt

def plot_duration_distribution(df: DataFrame, figsize=(12, 6), bins=30, save=True):
    """Plot distribution of duration_ms using direct data"""
    duration_seconds = df.select((F.col('duration_ms') / 1000).alias('duration_seconds')) \
                        .toPandas()['duration_seconds']
    
    plt.figure(figsize=figsize)
    sns.histplot(data=duration_seconds, bins=bins, kde=True)
    plt.title('Distribution of Episode Duration')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Count')
    plt.tight_layout()
    
    if save:
        ensure_plots_dir()
        plt.savefig('./plots/duration_distribution.png')
    return plt

def plot_region_frequencies(df: DataFrame, figsize=(12, 6), save=True):
    """Plot frequencies of regions using PySpark aggregation"""
    region_counts = df.groupBy('region') \
                     .agg(F.count('*').alias('count')) \
                     .orderBy(F.desc('count')) \
                     .toPandas()
    
    plt.figure(figsize=figsize)
    sns.barplot(data=region_counts, x='region', y='count')
    plt.xticks(rotation=45, ha='right')
    plt.title('Region Frequencies')
    plt.xlabel('Region')
    plt.ylabel('Count')
    plt.tight_layout()
    
    if save:
        ensure_plots_dir()
        plt.savefig('./plots/region_frequencies.png')
    return plt

def visualize_all_stats(df: DataFrame, figsize=(20, 15), save=True):
    """Create all visualizations in a single figure using PySpark DataFrame"""
    fig = plt.figure(figsize=figsize)
    
    # Missing values plot
    plt.subplot(2, 2, 1)
    total_count = df.count()
    null_counts = [
        {
            'column': col,
            'null_percentage': (df.filter(F.col(col).isNull()).count() / total_count) * 100
        }
        for col in df.columns
    ]
    df_nulls = pd.DataFrame(null_counts).sort_values('null_percentage', ascending=False).head(10)
    sns.barplot(data=df_nulls, x='column', y='null_percentage')
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 10 Features with Missing Values')
    plt.ylabel('Missing Values (%)')
    
    # Show name frequencies
    plt.subplot(2, 2, 2)
    show_counts = df.groupBy('show_name') \
                   .agg(F.count('*').alias('count')) \
                   .orderBy(F.desc('count')) \
                   .limit(10) \
                   .toPandas()
    sns.barplot(data=show_counts, x='show_name', y='count')
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 10 Show Names by Frequency')
    
    # Duration distribution
    plt.subplot(2, 2, 3)
    duration_seconds = df.select((F.col('duration_ms') / 1000).alias('duration_seconds')) \
                        .toPandas()['duration_seconds']
    sns.histplot(data=duration_seconds, bins=30, kde=True)
    plt.title('Distribution of Episode Duration (seconds)')
    
    # Region frequencies
    plt.subplot(2, 2, 4)
    region_counts = df.groupBy('region') \
                     .agg(F.count('*').alias('count')) \
                     .orderBy(F.desc('count')) \
                     .toPandas()
    sns.barplot(data=region_counts, x='region', y='count')
    plt.xticks(rotation=45, ha='right')
    plt.title('Region Frequencies')
    
    plt.tight_layout()
    
    if save:
        ensure_plots_dir()
        plt.savefig('./plots/all_stats.png')
    return plt