from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import *
from typing import Dict, List

def schema_to_feature_mappings(df: DataFrame) -> Dict[str, List[str]]:
    """Convert DataFrame schema to feature type mappings"""
    feature_mappings = {
        'date': [],
        'numerical': [],
        'binary': [],
        'categorical': [],
        'ordinal': []
    }
    
    ordinal_columns = {'chartRankMove', 'release_date_precision'}
    
    for field in df.schema.fields:
        if isinstance(field.dataType, DateType):
            feature_mappings['date'].append(field.name)
        elif isinstance(field.dataType, (IntegerType, LongType, DoubleType, FloatType)):
            feature_mappings['numerical'].append(field.name)
        elif isinstance(field.dataType, BooleanType):
            feature_mappings['binary'].append(field.name)
        elif isinstance(field.dataType, StringType):
            if field.name in ordinal_columns:
                feature_mappings['ordinal'].append(field.name)
            else:
                feature_mappings['categorical'].append(field.name)
    
    return {k: v for k, v in feature_mappings.items() if v}

def calculate_date_statistics(df: DataFrame, columns: List[str]) -> Dict:
    """Calculate statistics for date columns"""
    stats = {}
    for col in columns:
        try:
            date_stats = df.agg(
                F.min(col).alias('min_date'),
                F.max(col).alias('max_date'),
                F.datediff(F.max(col), F.min(col)).alias('date_range_days'),
                F.count(col).alias('non_null_count'),
                (F.count(F.when(F.col(col).isNull(), True)) / F.count('*')).alias('null_percentage')
            ).first()
            
            stats[col] = {
                'min_date': date_stats['min_date'],
                'max_date': date_stats['max_date'],
                'date_range_days': date_stats['date_range_days'] if date_stats['date_range_days'] is not None else 0,
                'non_null_count': date_stats['non_null_count'],
                'null_percentage': float(date_stats['null_percentage']) if date_stats['null_percentage'] is not None else 1.0
            }
        except Exception as e:
            print(f"Warning: Error calculating date statistics for column {col}: {str(e)}")
    return stats

def calculate_numerical_statistics(df: DataFrame, columns: List[str]) -> Dict:
    """Calculate statistics for numerical columns"""
    stats = {}
    for col in columns:
        try:
            basic_stats = df.agg(
                F.min(col).alias('min'),
                F.max(col).alias('max'),
                F.mean(col).alias('mean'),
                F.stddev(col).alias('stddev'),
                F.count(col).alias('non_null_count'),
                (F.count(F.when(F.col(col).isNull(), True)) / F.count('*')).alias('null_percentage')
            ).first()
            
            percentiles = df.filter(F.col(col).isNotNull()) \
                           .stat.approxQuantile(col, [0.25, 0.5, 0.75], 0.01)
            
            if not percentiles or len(percentiles) < 3:
                percentiles = [0, 0, 0]
            
            stats[col] = {
                'min': float(basic_stats['min']) if basic_stats['min'] is not None else 0,
                'max': float(basic_stats['max']) if basic_stats['max'] is not None else 0,
                'mean': float(basic_stats['mean']) if basic_stats['mean'] is not None else 0,
                'stddev': float(basic_stats['stddev']) if basic_stats['stddev'] is not None else 0,
                'q1': float(percentiles[0]),
                'median': float(percentiles[1]),
                'q3': float(percentiles[2]),
                'iqr': float(percentiles[2] - percentiles[0]),
                'non_null_count': basic_stats['non_null_count'],
                'null_percentage': float(basic_stats['null_percentage']) if basic_stats['null_percentage'] is not None else 1.0
            }
        except Exception as e:
            print(f"Warning: Error calculating numerical statistics for column {col}: {str(e)}")
    return stats

def calculate_categorical_statistics(df: DataFrame, columns: List[str], is_binary: bool = False) -> Dict:
    """Calculate statistics for categorical or binary columns"""
    stats = {}
    for col in columns:
        try:
            value_counts = df.filter(F.col(col).isNotNull()) \
                           .groupBy(col) \
                           .agg(F.count(col).alias('count')) \
                           .orderBy(F.desc('count')) \
                           .limit(5 if not is_binary else 2) \
                           .collect()
            
            string_stats = df.select(
                F.count(col).alias('non_null_count'),
                (F.count(F.when(F.col(col).isNull(), True)) / F.count('*')).alias('null_percentage')
            ).first()
            
            unique_count = df.select(col).distinct().count()
            
            stats[col] = {
                'unique_count': unique_count,
                'non_null_count': string_stats['non_null_count'],
                'null_percentage': float(string_stats['null_percentage']) if string_stats['null_percentage'] is not None else 1.0,
                'value_counts': {
                    str(vc[0]): vc['count'] for vc in value_counts
                }
            }
            
            if not is_binary:
                length_stats = df.filter(F.col(col).isNotNull()).select(
                    F.mean(F.length(col)).alias('mean_length'),
                    F.min(F.length(col)).alias('min_length'),
                    F.max(F.length(col)).alias('max_length')
                ).first()
                
                stats[col].update({
                    'mean_length': float(length_stats['mean_length']) if length_stats['mean_length'] is not None else 0,
                    'min_length': int(length_stats['min_length']) if length_stats['min_length'] is not None else 0,
                    'max_length': int(length_stats['max_length']) if length_stats['max_length'] is not None else 0
                })
                
        except Exception as e:
            print(f"Warning: Error calculating {'binary' if is_binary else 'categorical'} statistics for column {col}: {str(e)}")
    return stats

def format_statistics(stats: Dict) -> None:
    """Display statistics in a formatted way"""
    for feature_type, type_stats in stats.items():
        print(f"\n=== {feature_type.title()} Feature Statistics ===")
        for column, col_stats in type_stats.items():
            print(f"\nColumn: {column}")
            for stat, value in col_stats.items():
                print(f"{stat}: {value}")

def analyze_podcast_dataset(df: DataFrame) -> Dict:
    """Main function to analyze podcast dataset using an existing DataFrame"""
    print("\nVerifying loaded schema:")
    df.printSchema()
    
    # Verify data types with correct list comprehension syntax
    print("\nVerifying data types and non-null counts:")
    verification_cols = []
    for col in df.columns:
        verification_cols.extend([
            F.count(col).alias(f"{col}_count"),
            F.count(F.when(F.col(col).isNotNull(), col)).alias(f"{col}_non_null_count")
        ])
    df.select(verification_cols).show()
    
    print("\nCalculating statistics...")
    feature_mappings = schema_to_feature_mappings(df)
    stats = {}
    
    if feature_mappings.get('date'):
        stats['date'] = calculate_date_statistics(df, feature_mappings['date'])
    
    if feature_mappings.get('numerical'):
        stats['numerical'] = calculate_numerical_statistics(df, feature_mappings['numerical'])
    
    if feature_mappings.get('binary'):
        stats['binary'] = calculate_categorical_statistics(df, feature_mappings['binary'], is_binary=True)
    
    if feature_mappings.get('categorical'):
        stats['categorical'] = calculate_categorical_statistics(df, feature_mappings['categorical'])
    
    if feature_mappings.get('ordinal'):
        stats['ordinal'] = calculate_categorical_statistics(df, feature_mappings['ordinal'])
    
    print("\nAnalysis Results:")
    format_statistics(stats)
    
    return stats