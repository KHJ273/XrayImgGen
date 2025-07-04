import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional, Tuple


def visualize_nii(nii_filename: str, 
                  slice_axis: int = 2, 
                  slice_index: Optional[int] = None,
                  title: Optional[str] = None,
                  figsize: Tuple[int, int] = (10, 8),
                  save_path: Optional[str] = None,
                  show: bool = True) -> None:
    """
    Visualize a NII file by displaying slices along a specified axis.
    
    Args:
        nii_filename (str): Path to the NII file
        slice_axis (int): Axis along which to slice (0=sagittal, 1=coronal, 2=axial)
        slice_index (int, optional): Specific slice index to display. If None, displays middle slice
        title (str, optional): Title for the plot
        figsize (tuple): Figure size (width, height)
        save_path (str, optional): Path to save the visualization
        show (bool): Whether to display the plot
    """
    # Check if file exists
    if not os.path.exists(nii_filename):
        raise FileNotFoundError(f"NII file not found: {nii_filename}")
    
    # Load the NII file
    print(f"Loading NII file: {nii_filename}")
    nii_img = nib.load(nii_filename)
    data = nii_img.get_fdata()
    
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
    
    # Determine slice index if not provided
    if slice_index is None:
        slice_index = data.shape[slice_axis] // 2
    
    # Validate slice index
    if slice_index < 0 or slice_index >= data.shape[slice_axis]:
        raise ValueError(f"Slice index {slice_index} out of range for axis {slice_axis} "
                        f"(valid range: 0-{data.shape[slice_axis]-1})")
    
    # Extract the slice
    if slice_axis == 0:
        slice_data = data[slice_index, :, :]
        axis_name = "Sagittal"
    elif slice_axis == 1:
        slice_data = data[:, slice_index, :]
        axis_name = "Coronal"
    elif slice_axis == 2:
        slice_data = data[:, :, slice_index]
        axis_name = "Axial"
    else:
        raise ValueError("slice_axis must be 0, 1, or 2")
    
    # Create the plot
    plt.figure(figsize=figsize)
    plt.imshow(slice_data, cmap='gray', origin='lower')
    plt.colorbar(label='Intensity')
    
    # Set title
    if title is None:
        title = f"{os.path.basename(nii_filename)} - {axis_name} Slice {slice_index}"
    plt.title(title)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()


def visualize_nii_multiple_slices(nii_filename: str,
                                  slice_axis: int = 2,
                                  num_slices: int = 9,
                                  title: Optional[str] = None,
                                  figsize: Tuple[int, int] = (15, 10),
                                  save_path: Optional[str] = None,
                                  show: bool = True) -> None:
    """
    Visualize multiple slices of a NII file in a grid layout.
    
    Args:
        nii_filename (str): Path to the NII file
        slice_axis (int): Axis along which to slice (0=sagittal, 1=coronal, 2=axial)
        num_slices (int): Number of slices to display
        title (str, optional): Title for the plot
        figsize (tuple): Figure size (width, height)
        save_path (str, optional): Path to save the visualization
        show (bool): Whether to display the plot
    """
    # Check if file exists
    if not os.path.exists(nii_filename):
        raise FileNotFoundError(f"NII file not found: {nii_filename}")
    
    # Load the NII file
    print(f"Loading NII file: {nii_filename}")
    nii_img = nib.load(nii_filename)
    data = nii_img.get_fdata()
    
    # Calculate slice indices
    total_slices = data.shape[slice_axis]
    slice_indices = np.linspace(0, total_slices - 1, num_slices, dtype=int)
    
    # Determine grid layout
    cols = int(np.ceil(np.sqrt(num_slices)))
    rows = int(np.ceil(num_slices / cols))
    
    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Axis names
    axis_names = ["Sagittal", "Coronal", "Axial"]
    axis_name = axis_names[slice_axis]
    
    # Plot each slice
    for i, slice_idx in enumerate(slice_indices):
        # Extract the slice
        if slice_axis == 0:
            slice_data = data[slice_idx, :, :]
        elif slice_axis == 1:
            slice_data = data[:, slice_idx, :]
        elif slice_axis == 2:
            slice_data = data[:, :, slice_idx]
        
        # Plot
        im = axes[i].imshow(slice_data, cmap='gray', origin='lower')
        axes[i].set_title(f'{axis_name} Slice {slice_idx}')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_slices, len(axes)):
        axes[i].axis('off')
    
    # Set main title
    if title is None:
        title = f"{os.path.basename(nii_filename)} - Multiple {axis_name} Slices"
    fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()


def visualize_nii_3d_summary(nii_filename: str,
                             title: Optional[str] = None,
                             figsize: Tuple[int, int] = (15, 5),
                             save_path: Optional[str] = None,
                             show: bool = True) -> None:
    """
    Visualize a 3D summary of a NII file showing middle slices along all three axes.
    
    Args:
        nii_filename (str): Path to the NII file
        title (str, optional): Title for the plot
        figsize (tuple): Figure size (width, height)
        save_path (str, optional): Path to save the visualization
        show (bool): Whether to display the plot
    """
    # Check if file exists
    if not os.path.exists(nii_filename):
        raise FileNotFoundError(f"NII file not found: {nii_filename}")
    
    # Load the NII file
    print(f"Loading NII file: {nii_filename}")
    nii_img = nib.load(nii_filename)
    data = nii_img.get_fdata()
    
    # Get middle slices
    mid_x = data.shape[0] // 2
    mid_y = data.shape[1] // 2
    mid_z = data.shape[2] // 2
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Sagittal slice (YZ plane)
    axes[0].imshow(data[mid_x, :, :], cmap='gray', origin='lower')
    axes[0].set_title(f'Sagittal (X={mid_x})')
    axes[0].set_xlabel('Z')
    axes[0].set_ylabel('Y')
    
    # Coronal slice (XZ plane)
    axes[1].imshow(data[:, mid_y, :], cmap='gray', origin='lower')
    axes[1].set_title(f'Coronal (Y={mid_y})')
    axes[1].set_xlabel('Z')
    axes[1].set_ylabel('X')
    
    # Axial slice (XY plane)
    axes[2].imshow(data[:, :, mid_z], cmap='gray', origin='lower')
    axes[2].set_title(f'Axial (Z={mid_z})')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    
    # Set main title
    if title is None:
        title = f"{os.path.basename(nii_filename)} - 3D Summary"
    fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <nii_filename> [slice_axis] [slice_index]")
        print("Example: python visualize.py R_assembly/R_part11.nii 2 50")
        sys.exit(1)
    
    nii_filename = sys.argv[1]
    slice_axis = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    slice_index = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    # Visualize single slice
    visualize_nii(nii_filename, slice_axis=slice_axis, slice_index=slice_index)
    
    # Visualize 3D summary
    visualize_nii_3d_summary(nii_filename)