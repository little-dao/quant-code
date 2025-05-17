import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os

def prepare_data(df):
    """Prepare data for PCA by handling missing values and non-numeric columns"""
    # Remove date column and any other non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols]
    
    # Fill any missing values with column mean
    df_filled = df_numeric.fillna(df_numeric.mean())
    
    return df_filled

def perform_pca(df, n_components=None):
    """Perform PCA and return transformed data and explained variance"""
    # Standardize the features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)
    
    # Initialize PCA
    if n_components is None:
        n_components = min(len(df.columns), len(df))
    
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_scaled)
    
    return pca, data_pca, scaler

def plot_explained_variance(pca, output_dir, stock_name):
    """Plot cumulative explained variance ratio"""
    # Calculate cumulative variance
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    
    # Find number of components for different thresholds
    thresholds = [0.8, 0.9, 0.95, 0.99]
    components_needed = []
    for threshold in thresholds:
        n_comp = np.argmax(cumsum >= threshold) + 1
        components_needed.append(n_comp)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Cumulative variance
    ax1.plot(range(1, len(cumsum) + 1), cumsum, 'b-', label='Cumulative Explained Variance')
    ax1.axhline(y=0.95, color='r', linestyle='--', label='95% Threshold')
    
    # Add markers for different thresholds
    colors = ['g', 'y', 'r', 'purple']
    for threshold, n_comp, color in zip(thresholds, components_needed, colors):
        ax1.plot(n_comp, threshold, 'o', color=color, 
                label=f'{threshold*100}% variance at {n_comp} components')
    
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('Cumulative Explained Variance Ratio')
    ax1.set_title(f'Explained Variance vs Components - {stock_name}')
    ax1.grid(True)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Individual variance contribution
    individual_var = pca.explained_variance_ratio_[:50]  # Show first 50 components
    ax2.bar(range(1, len(individual_var) + 1), individual_var)
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Individual Explained Variance Ratio')
    ax2.set_title('Individual Component Contribution')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{stock_name}_explained_variance.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Print detailed variance information
    print("\nVariance Explained by Components:")
    print("-" * 50)
    for threshold, n_comp in zip(thresholds, components_needed):
        print(f"{threshold*100}% variance explained by {n_comp} components")
    
    # Print detailed breakdown of first 20 components
    print("\nDetailed Component Contribution:")
    print("-" * 50)
    for i in range(20):
        cumulative = cumsum[i]
        individual = pca.explained_variance_ratio_[i]
        print(f"PC{i+1}: Individual {individual:.3%}, Cumulative {cumulative:.3%}")

def find_optimal_components(pca, threshold=0.95):
    """Find number of components needed to explain threshold variance"""
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumsum >= threshold) + 1
    return n_components

def analyze_feature_importance(pca, feature_names):
    """Analyze feature importance based on PCA loadings"""
    # Get absolute values of loadings for components that explain 95% variance
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumsum >= 0.95) + 1
    print(f"\nNumber of components needed for 95% variance: {n_components}")
    
    # Show top 20 components or all if less than 20
    n_top_components = min(20, n_components)
    n_top_features = 10   # Show top 10 features per component
    
    importance_df = pd.DataFrame()
    
    for i in range(n_top_components):
        # Get absolute loadings
        loadings = pd.Series(
            np.abs(pca.components_[i]),
            index=feature_names
        )
        
        # Get top features
        top_features = loadings.nlargest(n_top_features)
        
        # Add to dataframe
        importance_df[f'PC{i+1} (Var: {pca.explained_variance_ratio_[i]:.3f})'] = pd.Series(
            [f'{name} ({val:.3f})' for name, val in top_features.items()]
        )
    
    # Also save full loadings for all components
    full_loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])],
        index=feature_names
    )
    
    return importance_df, full_loadings

def process_stock(file_path, output_dir):
    """Process a single stock's data"""
    stock_name = os.path.basename(file_path).split('_')[0]
    print(f"\nProcessing {stock_name}...")
    print("=" * 50)
    
    # Create output directory for this stock
    stock_output_dir = os.path.join(output_dir, stock_name)
    os.makedirs(stock_output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(file_path)
    print(f"Original features: {len(df.columns)-1}")  # -1 for date column
    
    # Prepare data
    data_prepared = prepare_data(df)
    feature_names = data_prepared.columns
    
    # Perform initial PCA
    pca, data_pca, scaler = perform_pca(data_prepared)
    
    # Plot and analyze explained variance
    plot_explained_variance(pca, stock_output_dir, stock_name)
    
    # Get number of components for 95% variance
    n_components_95 = find_optimal_components(pca, 0.95)
    print(f"\nUsing {n_components_95} components to explain 95% of variance")
    print(f"Reduction ratio: {n_components_95}/{len(feature_names)} = {n_components_95/len(feature_names):.2%}")
    
    # Analyze feature importance
    importance_df, full_loadings = analyze_feature_importance(pca, feature_names)
    print(f"\nTop features for first components ({stock_name}):")
    print(importance_df)
    
    # Save feature importance and full loadings
    importance_df.to_csv(os.path.join(stock_output_dir, 'feature_importance.csv'))
    full_loadings.to_csv(os.path.join(stock_output_dir, 'full_loadings.csv'))
    
    # Create reduced dataset with optimal components (95% variance)
    pca_95, data_pca_95, _ = perform_pca(data_prepared, n_components_95)
    
    # Save reduced dataset with ALL components that explain 95% variance
    reduced_df = pd.DataFrame(
        data_pca_95,
        columns=[f'PC{i+1}' for i in range(n_components_95)]
    )
    reduced_df['date'] = df['date']  # Add back date column
    reduced_df.to_csv(os.path.join(stock_output_dir, 'reduced_features.csv'), index=False)
    
    # Save summary statistics
    summary = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
        'Explained_Variance_Ratio': pca.explained_variance_ratio_,
        'Cumulative_Variance_Ratio': np.cumsum(pca.explained_variance_ratio_)
    })
    summary.to_csv(os.path.join(stock_output_dir, 'pca_summary.csv'), index=False)
    
    # Save PCA model and scaler for future use
    np.save(os.path.join(stock_output_dir, 'pca_components.npy'), pca_95.components_)
    np.save(os.path.join(stock_output_dir, 'pca_mean.npy'), pca_95.mean_)
    np.save(os.path.join(stock_output_dir, 'scaler_mean.npy'), scaler.mean_)
    np.save(os.path.join(stock_output_dir, 'scaler_scale.npy'), scaler.scale_)
    
    print(f"Completed processing {stock_name}")

def main():
    # Set up directories
    input_dir = "ibkr_features_pta_multi"
    output_dir = "pca_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all CSV files in the input directory
    for file in os.listdir(input_dir):
        if file.endswith('_pta_features.csv'):
            file_path = os.path.join(input_dir, file)
            process_stock(file_path, output_dir)

if __name__ == "__main__":
    main()




