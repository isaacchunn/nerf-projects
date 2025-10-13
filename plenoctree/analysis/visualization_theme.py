#!/usr/bin/env python3
"""
Unified Visualization Theme

Professional, consistent styling for ALL visualizations.
Every analyzer imports and uses this theme for consistency.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ═══════════════════════════════════════════════════════════════════════
#                          COLOR PALETTE
# ═══════════════════════════════════════════════════════════════════════

class Colors:
    """Professional color palette - consistent across all visualizations."""
    
    # Primary Colors (for main data)
    PRIMARY_BLUE = '#2E86AB'      # Strong, professional blue
    PRIMARY_RED = '#E63946'       # Vibrant red
    PRIMARY_GREEN = '#06A77D'     # Fresh green
    PRIMARY_PURPLE = '#7209B7'    # Rich purple
    PRIMARY_ORANGE = '#F77F00'    # Warm orange
    
    # Secondary Colors (for comparisons)
    SECONDARY_BLUE = '#4EA8DE'    # Lighter blue
    SECONDARY_RED = '#FF6B6B'     # Coral red
    SECONDARY_GREEN = '#2EC4B6'   # Teal
    SECONDARY_PURPLE = '#9B59B6'  # Soft purple
    SECONDARY_ORANGE = '#FFB703'  # Golden
    
    # Accent Colors (for highlights)
    ACCENT_CYAN = '#06FFA5'       # Bright cyan
    ACCENT_PINK = '#FF006E'       # Hot pink
    ACCENT_YELLOW = '#FFD60A'     # Bright yellow
    ACCENT_LIME = '#B5E48C'       # Lime green
    
    # Neutral Colors
    DARK_GRAY = '#2B2D42'         # Almost black
    MEDIUM_GRAY = '#8D99AE'       # Medium gray
    LIGHT_GRAY = '#EDF2F4'        # Very light gray
    WHITE = '#FFFFFF'             # Pure white
    
    # Background Colors
    BG_LIGHT = '#F8F9FA'          # Light background
    BG_DARK = '#212529'           # Dark background (if needed)
    BG_ACCENT = '#E8F4FD'         # Light blue background
    
    # Semantic Colors
    SUCCESS = '#06A77D'           # Green for success/good
    WARNING = '#FFB703'           # Orange for warning
    ERROR = '#E63946'             # Red for error/bad
    INFO = '#2E86AB'              # Blue for info
    
    # Scene-Specific Colors (for cross-experiment comparisons)
    SCENES = {
        'chair': '#E63946',       # Red
        'drums': '#06A77D',       # Green
        'ficus': '#2E86AB',       # Blue
        'hotdog': '#F77F00',      # Orange
        'lego': '#7209B7',        # Purple
        'materials': '#FFB703',   # Yellow
        'mic': '#9B59B6',         # Lavender
        'ship': '#4EA8DE',        # Sky blue
    }
    
    # Metric Colors (consistent across all plots)
    PSNR = '#E63946'              # Red (quality)
    SSIM = '#2E86AB'              # Blue (structure)
    LPIPS = '#F77F00'             # Orange (perceptual)
    MEMORY = '#7209B7'            # Purple (memory)
    MEI = '#06A77D'               # Green (efficiency)
    QMT = '#FFB703'               # Yellow (tradeoff)
    MEPV = '#2EC4B6'              # Teal (spatial efficiency)
    COMBINED = '#9B59B6'          # Lavender (combined metrics)


# ═══════════════════════════════════════════════════════════════════════
#                          TYPOGRAPHY
# ═══════════════════════════════════════════════════════════════════════

class Typography:
    """Font settings for consistent, readable text."""
    
    # Font Families
    MAIN_FONT = 'DejaVu Sans'     # Clean, modern sans-serif
    MONO_FONT = 'DejaVu Sans Mono' # For code/data
    
    # Font Sizes (in points)
    TITLE_SIZE = 24               # Main figure title
    SUBTITLE_SIZE = 18            # Subplot titles
    LABEL_SIZE = 14               # Axis labels
    TICK_SIZE = 12                # Tick labels
    LEGEND_SIZE = 12              # Legend text
    ANNOTATION_SIZE = 11          # Annotations
    SMALL_SIZE = 10               # Small text
    
    # Font Weights
    TITLE_WEIGHT = 'bold'
    SUBTITLE_WEIGHT = 'bold'
    LABEL_WEIGHT = '600'          # Semi-bold
    NORMAL_WEIGHT = 'normal'


# ═══════════════════════════════════════════════════════════════════════
#                          STYLE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

def apply_theme():
    """Apply the unified theme to matplotlib and seaborn."""
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Configure matplotlib rcParams
    plt.rcParams.update({
        # Figure settings
        'figure.dpi': 200,
        'figure.facecolor': Colors.WHITE,
        'figure.edgecolor': Colors.WHITE,
        'figure.figsize': (12, 8),
        'figure.titlesize': Typography.TITLE_SIZE,
        'figure.titleweight': Typography.TITLE_WEIGHT,
        
        # Axes settings
        'axes.facecolor': Colors.BG_LIGHT,
        'axes.edgecolor': Colors.DARK_GRAY,
        'axes.labelcolor': Colors.DARK_GRAY,
        'axes.labelsize': Typography.LABEL_SIZE,
        'axes.labelweight': Typography.LABEL_WEIGHT,
        'axes.titlesize': Typography.SUBTITLE_SIZE,
        'axes.titleweight': Typography.SUBTITLE_WEIGHT,
        'axes.titlepad': 20,
        'axes.linewidth': 2,
        'axes.grid': True,
        'axes.axisbelow': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        
        # Grid settings
        'grid.alpha': 0.3,
        'grid.color': Colors.MEDIUM_GRAY,
        'grid.linewidth': 0.8,
        'grid.linestyle': '-',
        
        # Line settings
        'lines.linewidth': 3,
        'lines.markersize': 8,
        'lines.markeredgewidth': 2,
        'lines.antialiased': True,
        
        # Tick settings
        'xtick.color': Colors.DARK_GRAY,
        'ytick.color': Colors.DARK_GRAY,
        'xtick.labelsize': Typography.TICK_SIZE,
        'ytick.labelsize': Typography.TICK_SIZE,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        
        # Legend settings
        'legend.fontsize': Typography.LEGEND_SIZE,
        'legend.framealpha': 0.95,
        'legend.facecolor': Colors.WHITE,
        'legend.edgecolor': Colors.MEDIUM_GRAY,
        'legend.shadow': False,
        'legend.frameon': True,
        'legend.borderpad': 0.8,
        
        # Font settings
        'font.family': Typography.MAIN_FONT,
        'font.size': Typography.TICK_SIZE,
        'font.weight': Typography.NORMAL_WEIGHT,
        
        # Save settings
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': Colors.WHITE,
        'savefig.edgecolor': Colors.WHITE,
        'savefig.pad_inches': 0.2,
        
        # Text settings
        'text.color': Colors.DARK_GRAY,
        'text.antialiased': True,
    })


# ═══════════════════════════════════════════════════════════════════════
#                          PLOT ELEMENTS
# ═══════════════════════════════════════════════════════════════════════

class PlotElements:
    """Pre-configured plot elements for consistency."""
    
    @staticmethod
    def get_gradient_colormap(color_start, color_end, n=256):
        """Create a gradient colormap between two colors."""
        from matplotlib.colors import LinearSegmentedColormap
        colors = [color_start, color_end]
        n_bins = n
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
        return cmap
    
    @staticmethod
    def add_value_labels(ax, bars, format_str='{:.2f}', offset=0.02):
        """Add value labels on top of bars."""
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + height * offset,
                       format_str.format(height),
                       ha='center', va='bottom', 
                       fontweight='bold',
                       fontsize=Typography.ANNOTATION_SIZE,
                       color=Colors.DARK_GRAY)
    
    @staticmethod
    def create_stats_box(ax, stats_text, position=(0.1, 0.9)):
        """Create a styled statistics box."""
        ax.text(position[0], position[1], stats_text,
               transform=ax.transAxes,
               fontsize=Typography.ANNOTATION_SIZE,
               verticalalignment='top',
               horizontalalignment='left',
               fontfamily=Typography.MONO_FONT,
               bbox=dict(boxstyle="round,pad=1",
                        facecolor=Colors.BG_LIGHT,
                        edgecolor=Colors.DARK_GRAY,
                        linewidth=2,
                        alpha=0.95))
    
    @staticmethod
    def add_watermark(fig, text="Generated with PlenOctree Analysis Suite"):
        """Add subtle watermark to figure."""
        fig.text(0.99, 0.01, text,
                ha='right', va='bottom',
                fontsize=Typography.SMALL_SIZE,
                color=Colors.MEDIUM_GRAY,
                alpha=0.5)
    
    @staticmethod
    def style_axis(ax, title, xlabel, ylabel, 
                   title_color=Colors.DARK_GRAY,
                   show_grid=True):
        """Apply consistent styling to an axis."""
        ax.set_title(title, 
                    fontsize=Typography.SUBTITLE_SIZE,
                    fontweight=Typography.SUBTITLE_WEIGHT,
                    color=title_color,
                    pad=15)
        ax.set_xlabel(xlabel, 
                     fontsize=Typography.LABEL_SIZE,
                     fontweight=Typography.LABEL_WEIGHT,
                     color=Colors.DARK_GRAY)
        ax.set_ylabel(ylabel,
                     fontsize=Typography.LABEL_SIZE,
                     fontweight=Typography.LABEL_WEIGHT,
                     color=Colors.DARK_GRAY)
        
        if show_grid:
            ax.grid(True, alpha=0.3, color=Colors.MEDIUM_GRAY, linewidth=0.8)
        
        # Style spines
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color(Colors.DARK_GRAY)
            ax.spines[spine].set_linewidth(2)


# ═══════════════════════════════════════════════════════════════════════
#                          HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def get_scene_color(scene_name):
    """Get consistent color for a scene."""
    return Colors.SCENES.get(scene_name.lower(), Colors.MEDIUM_GRAY)


def get_metric_color(metric_name):
    """Get consistent color for a metric."""
    metric_lower = metric_name.lower()
    if 'psnr' in metric_lower:
        return Colors.PSNR
    elif 'ssim' in metric_lower:
        return Colors.SSIM
    elif 'lpips' in metric_lower:
        return Colors.LPIPS
    elif 'memory' in metric_lower or 'mem' in metric_lower:
        return Colors.MEMORY
    elif 'mei' in metric_lower:
        return Colors.MEI
    elif 'qmt' in metric_lower:
        return Colors.QMT
    elif 'mepv' in metric_lower:
        return Colors.MEPV
    elif 'combined' in metric_lower:
        return Colors.COMBINED
    else:
        return Colors.PRIMARY_BLUE


def create_gradient_fill(ax, x, y, color, alpha=0.3):
    """Create a gradient fill under a line."""
    ax.fill_between(x, 0, y, alpha=alpha, color=color)


def format_number(value, decimals=2):
    """Format number with consistent precision."""
    if abs(value) >= 1000:
        return f"{value/1000:.1f}K"
    elif abs(value) >= 1:
        return f"{value:.{decimals}f}"
    else:
        return f"{value:.{decimals+1}f}"


# ═══════════════════════════════════════════════════════════════════════
#                          PLOT TEMPLATES
# ═══════════════════════════════════════════════════════════════════════

class PlotTemplates:
    """Pre-configured plot templates for common visualizations."""
    
    @staticmethod
    def create_progression_plot(ax, x, y, title, ylabel, color, label=None):
        """Create a styled line plot for metric progression."""
        # Shadow effect
        ax.plot(x, y, '-', linewidth=5, color=Colors.DARK_GRAY, 
               alpha=0.2, zorder=1)
        
        # Main line
        line = ax.plot(x, y, '-o', linewidth=3, markersize=7,
                      color=color, markerfacecolor=Colors.WHITE,
                      markeredgewidth=2.5, markeredgecolor=color,
                      label=label, zorder=2)[0]
        
        # Gradient fill
        ax.fill_between(x, 0, y, alpha=0.2, color=color, zorder=0)
        
        # Style
        PlotElements.style_axis(ax, title, 'Training Step', ylabel)
        
        if label:
            ax.legend(loc='best', framealpha=0.95)
        
        return line
    
    @staticmethod
    def create_comparison_bars(ax, labels, values, title, ylabel, colors=None):
        """Create styled bar chart for comparisons."""
        if colors is None:
            colors = [Colors.PRIMARY_BLUE] * len(values)
        
        bars = ax.bar(range(len(labels)), values,
                     color=colors, alpha=0.85,
                     edgecolor=Colors.WHITE, linewidth=2.5)
        
        # Add value labels
        PlotElements.add_value_labels(ax, bars)
        
        # Style
        PlotElements.style_axis(ax, title, '', ylabel)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right',
                          fontweight='600')
        
        return bars
    
    @staticmethod
    def create_dual_axis_plot(fig, ax, x, y1, y2, 
                            ylabel1, ylabel2, 
                            color1, color2,
                            label1, label2):
        """Create styled dual-axis plot."""
        # Left axis
        line1 = ax.plot(x, y1, '-o', linewidth=3, markersize=7,
                       color=color1, markerfacecolor=Colors.WHITE,
                       markeredgewidth=2.5, markeredgecolor=color1,
                       label=label1, zorder=2)[0]
        ax.set_ylabel(ylabel1, color=color1, fontweight='bold',
                     fontsize=Typography.LABEL_SIZE)
        ax.tick_params(axis='y', labelcolor=color1)
        
        # Right axis
        ax2 = ax.twinx()
        line2 = ax2.plot(x, y2, '-s', linewidth=3, markersize=6,
                        color=color2, markerfacecolor=Colors.WHITE,
                        markeredgewidth=2.5, markeredgecolor=color2,
                        label=label2, zorder=2)[0]
        ax2.set_ylabel(ylabel2, color=color2, fontweight='bold',
                      fontsize=Typography.LABEL_SIZE)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Combined legend
        lines = [line1, line2]
        labels = [line1.get_label(), line2.get_label()]
        ax.legend(lines, labels, loc='upper left', framealpha=0.95)
        
        ax.grid(True, alpha=0.3)
        
        return ax, ax2


# ═══════════════════════════════════════════════════════════════════════
#                          INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════

# Apply theme when module is imported
apply_theme()

# Print confirmation
if __name__ != "__main__":
    print("✓ Unified visualization theme loaded")

