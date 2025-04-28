import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os

# ---------------------------------------------------------------------------
# General purpose functions 
# ---------------------------------------------------------------------------

def nematic_to_vector(Q, q):
    theta = np.arctan2(q, Q)/2 
    s = np.sqrt(q**2 + Q**2)
    n = [s*np.cos(theta), s*np.sin(theta)]
    return np.array(n)

# ---------------------------------------------------------------------------
# Visualization functions
# ---------------------------------------------------------------------------

def value2bgr(values, cmap_name, vmin, vmax):
    """
    Convert a value to a color using a given colormap.

    Parameters:
    value (float): Value to be converted
    cmap_name (string): Name of the colormap
    vmin (float): Minimum value for normalization
    vmax (float): Maximum value for normalization

    Returns:
    color (array): BGR color
    """
    norm = Normalize(vmin=vmin, vmax=vmax)
    normalized_density = norm(values)
    
    # Use matplotlib's colormap to convert to RGB
    cmap_func = plt.get_cmap(cmap_name)
    colored = cmap_func(normalized_density)
    
    # Convert to BGR for OpenCV (including alpha channel handling)
    colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    colored_bgr = cv2.cvtColor(colored_rgb, cv2.COLOR_RGB2BGR)

    return colored_bgr

def visualize_active_gel_frame(density, velocity_field, nematic_field, 
                               density_cmap='Greys', vector_scale=1.0, 
                               skip=5, arrow_width=1, tip_length=0.2, 
                               density_range=None, velocity_range=None, nematic_range=None):
    """
    Create a visualization frame for active gel simulation data.
    
    Parameters:
    -----------
    density : 2D numpy array
        Density field (will be visualized as grayscale heatmap)
    velocity_field : tuple of 2D numpy arrays (vx, vy)
        Velocity vector field components
    nematic_field : tuple of 2D numpy arrays (nx, ny)
        Nematic order vector field components
    density_cmap : str, default='Greys'
        Matplotlib colormap for density visualization
    vector_scale : float, default=1.0
        Scaling factor for vector magnitudes
    skip : int, default=5
        Sample vectors every 'skip' points for better visualization
    arrow_scale : float, default=1.0
        Scale factor for arrow size
    arrow_width : int, default=1
        Width of the arrow lines
        
    Returns:
    --------
    frame : numpy array
        RGB visualization frame
    """
    height, width = density.shape
    if density_range is None: 
        density_range = [np.min(density), np.max(density)]
    if velocity_range is None: 
        velocity_range = [np.min(velocity_field[:, :, 0]), np.max(velocity_field[:, :, 1])]
    if nematic_range is None:
        nematic_range = [np.min(nematic_field[:, :, 0]), np.max(nematic_field[:, :, 1])]
    
    # Create three separate visualizations with consistent colormap ranges
    density_vis = visualize_density(density, density_range[0], density_range[1], cmap=density_cmap)
    
    velocity_vis = visualize_vector_field(velocity_field, 
                                         height, width, velocity_range[0], velocity_range[1], 
                                         skip=skip, cmap='plasma', scale=vector_scale,
                                        arrow_width=arrow_width, tip_length=tip_length)
    
    nematic_vis = visualize_vector_field(nematic_field, 
                                        height, width, nematic_range[0], nematic_range[1], 
                                        skip=skip, cmap='viridis', scale=vector_scale,
                                        arrow_width=arrow_width, tip_length=0)
    
    # Combine the three visualizations horizontally
    composite = np.hstack([density_vis, velocity_vis, nematic_vis])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(composite, 'Density', (width//2 - 40, height + 30), 
                font, 0.7, (255, 255, 255), 2)
    cv2.putText(composite, 'Velocity', (width + width//2 - 40, height + 30), 
                font, 0.7, (255, 255, 255), 2)
    cv2.putText(composite, 'Nematic Order', (2*width + width//2 - 60, height + 30), 
                font, 0.7, (255, 255, 255), 2)
    
    return composite


def visualize_density(density, vmin, vmax, cmap='Greys', ):
    """Visualize density field as a grayscale heatmap with consistent normalization"""    
    return value2bgr(density, cmap, vmin, vmax)


def visualize_vector_field(vector_field, height, width, vmin, vmax,  
                           skip=5, cmap='viridis', scale=1.0, 
                           arrow_width=1, tip_length=0.2):
    """Visualize a vector field with unit-length vectors on a magnitude background"""
    magnitude = np.linalg.norm(vector_field, axis=2)
    
    # Create background based on magnitude with consistent normalization
    background_bgr = value2bgr(magnitude, cmap, vmin, vmax)
    
    # Draw unit-length vectors
    for i in range(0, height, skip):
        for j in range(0, width, skip):
            # Skip if magnitude is close to zero
            mag = magnitude[i, j]
            if mag < 1e-6:
                continue

            v = vector_field[i, j]/mag*scale 

            # Draw arrow
            start_point = (int(j), int(i))
            end_point = (int(j + v[0]), int(i + v[1]))
            
            cv2.arrowedLine(background_bgr, start_point, end_point, 
                            (255, 255, 255), arrow_width, tipLength=tip_length)
    
    return background_bgr


def create_active_gel_video(density_frames, velocity_frames, nematic_frames, 
                            output_file='active_gel_simulation.mp4', fps=30, skip=5,
                            vector_scale=1.0, arrow_width=1):
    """
    Create a video from active gel simulation data
    
    Parameters:
    -----------
    density_frames : list of 2D arrays
        List of density field frames
    velocity_frames : list of (vx, vy) tuples
        List of velocity field component frames
    nematic_frames : list of (nx, ny) tuples
        List of nematic order field component frames
    output_file : str, default='active_gel_simulation.mp4'
        Output video filename
    fps : int, default=30
        Frames per second
    skip : int, default=5
        Sample vectors every 'skip' points for better visualization
    vector_scale : float, default=1.0
        Scaling factor for vector magnitudes
    arrow_scale : float, default=5.0
        Scale factor for arrow size
    arrow_width : int, default=1
        Width of the arrow lines
    """
    # Compute global min/max values for consistent colormapping if requested
    density_range = [np.min(density_frames), np.max(density_frames)]
    velocity_range = [np.min(velocity_frames[:, :, :, 0]), np.max(velocity_frames[:, :, :, 1])]
    nematic_range = [np.min(nematic_frames[:, :, :, 0]), np.max(nematic_frames[:, :, :, 1])]
    
    # Create first frame to get dimensions
    first_frame = visualize_active_gel_frame(
        density_frames[0], velocity_frames[0], nematic_frames[0],
        skip=skip, vector_scale=vector_scale, 
        arrow_width=arrow_width,
        density_range=density_range,
        velocity_range=velocity_range,
        nematic_range=nematic_range
    )
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for AVI
    video = cv2.VideoWriter(output_file, fourcc, fps, 
                           (first_frame.shape[1], first_frame.shape[0]))
    
    # Add frames to video
    for i in range(len(density_frames)):
        frame = visualize_active_gel_frame(
            density_frames[i], velocity_frames[i], nematic_frames[i],
            skip=skip, vector_scale=vector_scale, 
            arrow_width=arrow_width,
            density_range=density_range,
            velocity_range=velocity_range,
            nematic_range=nematic_range
        )
        video.write(frame)
        
        # Optional: print progress
        if i % 10 == 0:
            print(f"Processing frame {i}/{len(density_frames)}")
    
    # Release resources
    video.release()
    print(f"Video saved to {output_file}")


# Example usage
if __name__ == "__main__":
    # Generate sample data (replace with your actual simulation data)
    n_frames = 60
    height, width = 400, 400
    
    # Create sample data
    density_frames = []
    velocity_frames = []
    nematic_frames = []
    
    for t in range(n_frames):
        # Sample density field: diffusing gaussian
        x, y = np.meshgrid(np.linspace(-5, 5, width), np.linspace(-5, 5, height))
        center_x = 2 * np.cos(t/10)
        center_y = 2 * np.sin(t/10)
        density = np.exp(-((x-center_x)**2 + (y-center_y)**2) / (2 + t/30))
        density_frames.append(density)
        
        # Sample velocity field: rotating vortex
        vx = -(y - center_y) * 0.1
        vy = (x - center_x) * 0.1
        v = np.stack((vx, vy), axis=-1)
        velocity_frames.append(v)
        
        # Sample nematic field: aligned in bands
        nx = np.cos(x + t/10)
        ny = np.sin(y + t/10)
        n = np.stack((nx, ny), axis=-1)
        nematic_frames.append(n)

    density_frames = np.array(density_frames)
    velocity_frames = np.array(velocity_frames) 
    nematic_frames = np.array(nematic_frames)
    
    # Create video
    create_active_gel_video(density_frames, velocity_frames, nematic_frames, 
                            output_file='../figures/active_gel_demo.mp4',
                            fps=20, skip=20, vector_scale=10, 
                            arrow_width=2)     
           