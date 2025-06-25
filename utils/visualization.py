import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os
import textwrap

def visualize_coupled_data(save_path, source, target, generated, max_features_to_plot=10):
    """
    Visualize source, target, and generated data for coupled distribution embeddings.
    
    Args:
        save_path: Path to save the visualization
        source: Source samples [set_size, features]
        target: Target samples [set_size, features] 
        generated: Generated samples (transported from source to target) [set_size, features]
        max_features_to_plot: Maximum number of features to plot in pairplot
    """
    print(source.shape, target.shape, generated.shape)
    if len(source.shape) == 2: # [set_size, features]
        source_flat = source.numpy()
        target_flat = target.numpy()
        generated_flat = generated.numpy()
        _, features = source.shape

        if features == 1:
            # Create histogram of source, target, and generated data
            plt.hist(source_flat, bins=20, alpha=0.6, label='Source', color='blue')
            plt.hist(target_flat, bins=20, alpha=0.6, label='Target', color='green')
            plt.hist(generated_flat, bins=20, alpha=0.6, label='Generated', color='red')
            plt.legend()
            plt.title('Source → Target Transport')
            plt.savefig(save_path)
            plt.close()
        elif features == 2:
            # Scatter plot source, target, and generated data
            plt.figure(figsize=(8, 6))
            plt.scatter(source_flat[:, 0], source_flat[:, 1], label='Source', alpha=0.7, color='blue')
            plt.scatter(target_flat[:, 0], target_flat[:, 1], label='Target', alpha=0.7, color='green')
            plt.scatter(generated_flat[:, 0], generated_flat[:, 1], label='Generated', alpha=0.7, color='red')
            plt.legend()
            plt.title('Source → Target Transport')
            plt.savefig(save_path)
            plt.close()            
        elif features <= max_features_to_plot:
            # Create a pairplot of source, target, and generated data
            df = pd.DataFrame(np.concatenate([source_flat, target_flat, generated_flat], axis=0))
            df['type'] = (['source'] * len(source_flat) + 
                         ['target'] * len(target_flat) + 
                         ['generated'] * len(generated_flat))
            sns.pairplot(df, hue='type', palette=['blue', 'green', 'red'])
            plt.suptitle('Source → Target Transport', y=1.02)
            plt.savefig(save_path)
            plt.close()
        else:
            # Compare first and second moments
            source_mean = np.mean(source_flat, axis=0)
            source_std = np.std(source_flat, axis=0)
            target_mean = np.mean(target_flat, axis=0)
            target_std = np.std(target_flat, axis=0)
            generated_mean = np.mean(generated_flat, axis=0)
            generated_std = np.std(generated_flat, axis=0)

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Source vs Target means
            r2_st_mean = np.corrcoef(source_mean, target_mean)[0, 1]**2
            axes[0,0].scatter(source_mean, target_mean, color='purple')
            axes[0,0].set_title("Source vs Target Mean")
            axes[0,0].text(0.05, 0.95, f"R²: {r2_st_mean:.3f}", ha='left', va='top', transform=axes[0,0].transAxes)
            axes[0,0].set_xlabel("Source Mean")
            axes[0,0].set_ylabel("Target Mean")
            
            # Generated vs Target means (how well did we transport?)
            r2_gt_mean = np.corrcoef(generated_mean, target_mean)[0, 1]**2
            axes[0,1].scatter(generated_mean, target_mean, color='red')
            axes[0,1].set_title("Generated vs Target Mean")
            axes[0,1].text(0.05, 0.95, f"R²: {r2_gt_mean:.3f}", ha='left', va='top', transform=axes[0,1].transAxes)
            axes[0,1].set_xlabel("Generated Mean")
            axes[0,1].set_ylabel("Target Mean")
            
            # Source vs Target stds
            r2_st_std = np.corrcoef(source_std, target_std)[0, 1]**2
            axes[1,0].scatter(source_std, target_std, color='purple')
            axes[1,0].set_title("Source vs Target Std")
            axes[1,0].text(0.05, 0.95, f"R²: {r2_st_std:.3f}", ha='left', va='top', transform=axes[1,0].transAxes)
            axes[1,0].set_xlabel("Source Std")
            axes[1,0].set_ylabel("Target Std")
            
            # Generated vs Target stds
            r2_gt_std = np.corrcoef(generated_std, target_std)[0, 1]**2
            axes[1,1].scatter(generated_std, target_std, color='red')
            axes[1,1].set_title("Generated vs Target Std")
            axes[1,1].text(0.05, 0.95, f"R²: {r2_gt_std:.3f}", ha='left', va='top', transform=axes[1,1].transAxes)
            axes[1,1].set_xlabel("Generated Std")
            axes[1,1].set_ylabel("Target Std")
            
            plt.suptitle('Source → Target Transport Analysis')
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

    elif len(source.shape) == 4: # [set_size, channels, height, width]
        # Create grids of source, target, and generated images
        source_grid = make_grid(source*-1 + 1, nrow=10)
        target_grid = make_grid(target*-1 + 1, nrow=10)
        gen_grid = make_grid(generated*-1 + 1, nrow=10)
        
        # Create a figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=(30, 10))
        
        # Plot source images
        axes[0].imshow(source_grid.permute(1, 2, 0).cpu().numpy())
        axes[0].axis('off')
        axes[0].set_title("Source Samples")
        
        # Plot target images
        axes[1].imshow(target_grid.permute(1, 2, 0).cpu().numpy())
        axes[1].axis('off')
        axes[1].set_title("Target Samples")
        
        # Plot generated images
        axes[2].imshow(gen_grid.permute(1, 2, 0).cpu().numpy())
        axes[2].axis('off')
        axes[2].set_title("Generated Samples")
        
        # Add main title
        plt.suptitle('Source → Target Transport', fontsize=16)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(save_path)
        plt.close()

def visualize_text_data(output_dir, source_texts, target_texts, generated_texts):
    """
    Visualize source, target, and generated text data for coupled distribution embeddings.
    
    Args:
        output_dir: Directory to save the visualizations
        source_texts: Source texts from the dataset
        target_texts: Target texts from the dataset  
        generated_texts: Generated texts from the model (transported from source to target)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(len(source_texts)):
        df_source = pd.DataFrame(source_texts[i])
        df_source['type'] = 'source'
        df_target = pd.DataFrame(target_texts[i])
        df_target['type'] = 'target'
        df_generated = pd.DataFrame(generated_texts[i])
        df_generated['type'] = 'generated'
        df_combined = pd.concat([df_source, df_target, df_generated], axis=0)
        df_combined.to_csv(os.path.join(output_dir, f"text_samples_{i}.csv"), index=False)
    
    