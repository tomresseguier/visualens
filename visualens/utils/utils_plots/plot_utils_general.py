import numpy as np
import sys
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import FancyArrow
from astropy.wcs import WCS
import corner
from ..utils_astro.get_default_cosmology import get_default_cosmology
cosmo = get_default_cosmology()


def make_palette(hue_range=None, sat_range=None, hue_fixed=None, sat_fixed=None, alpha=[1,0]) :
    if not isinstance(alpha, list) :
        alpha = [alpha, 0]
    hues = np.linspace(0, 1-1/hue_range, hue_range) if hue_range is not None else [hue_fixed]
    sats = np.linspace(1, 0, sat_range) if sat_range is not None else [sat_fixed]
    colors = []
    for sat in sats :
        for hue in hues :
            colors.append( hsv_to_rgb([hue, sat, 1.]).tolist() + alpha )
    return colors


def plot_image_mpl(image_data, wcs=None, deg_per_pix=None, wcs_projection=True, units='pixel', pos=111, \
                            make_axes_labels=True, make_grid=True, crop=None, extra_pad=None) :
    #if self.ax is None :
    #    self.fig = plt.figure()
    fig = plt.figure()
    if wcs_projection :
        ax = fig.add_subplot(pos, projection=wcs)
        if make_grid :
            ax.coords.grid(True, color='white', ls='dotted')
    else :
        ax = fig.add_subplot(pos)
        if units=='pixel' or units=='pixels' or units=='image' :
            scaling = 1
        if units=='arcsec' :
            scaling = deg_per_pix*60*60
        if units=='arcmin' :
            scaling = deg_per_pix*60
    if make_axes_labels and wcs_projection :
        ax.coords[0].set_axislabel('Right ascension')
        ax.coords[1].set_axislabel('Declination')
    elif make_axes_labels and not wcs_projection :
        ax.set_xlabel('x (' + units + ')')
        ax.set_ylabel('y (' + units + ')')
    elif not make_axes_labels and wcs_projection :
        ax.coords[0].set_axislabel(' ')
        ax.coords[1].set_axislabel(' ')
    else :
        ax.set_xlabel(' ')
        ax.set_ylabel(' ')
    if crop is not None :
        to_plot = image_data[crop[2]:crop[3], crop[0]:crop[1], :]
    elif extra_pad is not None :
        new_shape = (image_data.shape[0]+2*extra_pad, image_data.shape[1]+2*extra_pad, image_data.shape[2])
        to_plot = np.full(new_shape, 0)
        to_plot[extra_pad:-extra_pad, extra_pad:-extra_pad, :] = image_data
    #if crop :
    #    to_plot = self.image_data[int(self.image_data.shape[1]/4):int(self.image_data.shape[1]*3/4), \
    #                              int(self.image_data.shape[0]/4):int(self.image_data.shape[0]*3/4), :]
    else :
        to_plot = image_data
    if wcs_projection :
        ax.imshow(to_plot, origin="lower")
    if not wcs_projection :
        ax.imshow(to_plot, origin='lower', extent=[0, to_plot.shape[1]*scaling, 0, to_plot.shape[0]*scaling])
    #ax.figure.tight_layout()
    return fig, ax


def plot_scale_bar(ax, deg_per_pix=None, redshift=None, unit='arcmin', length=1 , color='white', linewidth=2, text_offset=0.01, position=['bottom', 'left']) :
    """
    Plots a white bar scale indicating the length of an arcminute on a given image.
    
    Parameters:
    - ax: matplotlib axis object containing the image.
    - redshift: Redshift of the object.
    - unit: Unit of scale bar ['arcsec']['arcmin']['kpc']. Default is 'arcmin'
    - length: Length of the scale bar in defined unit. Default is 1.
    - color: Color of the scale bar. Default is white.
    - linewidth: Width of the scale bar line. Default is 2.
    - text_offset: Offset for the text label above the bar. Default is 0.05.
    """
    print(cosmo)
    if unit=='arcsec' :
        length_deg = length / 3600.
        unit = '\"'
    if unit=='arcmin' :
        length_deg = length / 60.
        unit = "\'"
        
    if unit in ['pc', 'kpc'] :
        ang_diam_dist = cosmo.angular_diameter_distance(redshift).to(unit).value
        length_rad = length / ang_diam_dist
        length_deg = np.rad2deg(length_rad)
        length_arcmin = length_deg * 60.
        length_arcsec = length_deg * 3600.
    
    bar_length_pix = length_deg / deg_per_pix
    
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    
    if position[0]=='bottom' :
        start_y = y_lim[0] + 0.05 * np.abs(y_lim[1] - y_lim[0])
    elif position[0]=='top' :
        start_y = y_lim[1] - 0.05 * np.abs(y_lim[1] - y_lim[0])
    elif type(position[0])==float :
        start_y = y_lim[0] + position[0] * np.abs(y_lim[1] - y_lim[0])
    else :
        raise ValueError(f"Invalid position[0]: {position[0]}")
        
    if position[1]=='left' :
        start_x = x_lim[0] + 0.05 * np.abs(x_lim[1] - x_lim[0])
    elif position[1]=='right' :
        start_x = x_lim[1] - 0.05 * np.abs(x_lim[1] - x_lim[0]) - bar_length_pix
    elif type(position[1])==float :
        if position[1]<0.5 :
            start_x = x_lim[0] + position[1] * np.abs(x_lim[1] - x_lim[0])
        else :
            start_x = x_lim[1] + position[1] * np.abs(x_lim[1] - x_lim[0]) - bar_length_pix
    else :
        raise ValueError(f"Invalid position[1]: {position[1]}")

    end_x = start_x + bar_length_pix
    
    ax.plot([start_x, end_x], [start_y, start_y], color=color, linewidth=linewidth)
    
    text_x = (start_x + end_x) / 2
    text_y = start_y + text_offset * np.abs(y_lim[1] - y_lim[0])
    
    ax.text(text_x, text_y, f'{length}{unit}', \
            color=color, ha='center', va='bottom', fontsize=12)
    if unit=='kpc' :
        text_y_below = start_y - text_offset * np.abs(y_lim[1] - y_lim[0])
        if length_arcmin>=1. :
            ax.text(text_x, text_y_below, f"{length_arcmin:.2f}\'", \
                    color=color, ha='center', va='top', fontsize=12)
        else :
            ax.text(text_x, text_y_below, f'{length_arcsec:.2f}\"', \
                    color=color, ha='center', va='top', fontsize=12)
        
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)


def plot_NE_arrows(ax, wcs, arrow_length=0.1, color='white', fontsize=12):
    arrowprops = dict(facecolor=color, edgecolor=color, width=0.5, headwidth=4, headlength=5)
    
    #center_pixel = np.array([(ax.get_xlim()[1] - ax.get_xlim()[0]) / 2, 
    #                         (ax.get_ylim()[1] - ax.get_ylim()[0]) / 2])
    center_pixel = wcs.wcs.crpix
    
    # Convert the center pixel to world coordinates
    center_world = wcs.pixel_to_world(center_pixel[0], center_pixel[1])
    
    # Define small offsets in world coordinates for North and East directions
    north_offset = np.array([center_world.spherical.lon.degree, center_world.spherical.lat.degree + 0.01])
    east_offset = np.array([center_world.spherical.lon.degree + 0.01, center_world.spherical.lat.degree])
    
    # Convert these offsets back to pixel coordinates
    north_pixel = wcs.world_to_pixel_values(north_offset[0], north_offset[1])
    east_pixel = wcs.world_to_pixel_values(east_offset[0], east_offset[1])
    
    # Calculate the pixel direction vectors
    north_vector = np.array(north_pixel) - center_pixel
    east_vector = np.array(east_pixel) - center_pixel
    
    # Normalize the direction vectors
    north_vector /= np.linalg.norm(north_vector)
    east_vector /= np.linalg.norm(east_vector)
    
    # Calculate the end points of the arrows
    fig_scale_factor = max( ax.get_xlim()[1] - ax.get_xlim()[0], ax.get_ylim()[1] - ax.get_ylim()[0] )
    north_end = center_pixel + arrow_length * north_vector * fig_scale_factor
    east_end = center_pixel + arrow_length * east_vector * fig_scale_factor
    
    start_x = 0.95
    start_y = 0.09
    
    xy = (start_x + (north_end[0] - center_pixel[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0]),
          start_y + (north_end[1] - center_pixel[1]) / (ax.get_ylim()[1] - ax.get_ylim()[0]))
    ax.annotate('', xy=xy, xytext=(start_x, start_y), arrowprops=arrowprops, xycoords='axes fraction')
    ax.text(xy[0], xy[1]+0.01, 'N', color=color, fontsize=fontsize,
            ha='center', va='bottom', transform=ax.transAxes)
    
    xy = (start_x + (east_end[0] - center_pixel[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0]),
          start_y + (east_end[1] - center_pixel[1]) / (ax.get_ylim()[1] - ax.get_ylim()[0]))
    ax.annotate('', xy=xy, xytext=(start_x, start_y), arrowprops=arrowprops, xycoords='axes fraction')
    ax.text(xy[0]-0.01, xy[1], 'E', color=color, fontsize=fontsize,
            ha='right', va='center', transform=ax.transAxes)


def adjust_luminosity(image, factor):
    adjusted_image = image * factor
    adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)
    return adjusted_image

def adjust_contrast(image, factor, pivot=1.):
    mean = np.mean(image, axis=(0, 1))
    adjusted_image = (image - mean*pivot) * factor + mean*pivot
    adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)
    return adjusted_image

def plot_corner(df):
    # Extract the numeric columns (skip non-numeric like 'Chi2' or similar if needed)
    data = df.select_dtypes(include='number')
    
    df_range = []
    for i, col in enumerate(data.columns) :
        col_min = np.min(data[col])
        col_max = np.max(data[col])
        if col_min==col_max or col=='Chi2' or col=='Nsample' or col=='ln(Lhood)':
            #df_range.append( [col_min-1, col_min+1] )
            del data[col]
        #else :
            #df_range.append( [col_min, col_max] )
    
    figure = corner.corner( data,
                            labels=data.columns,  # Use column names from the DataFrame
                            quantiles=[0.16, 0.5, 0.84],  # Plot the 1, 2, and 3 sigma contours
                            show_titles=True,  # Show titles with mean and std
                            title_fmt=".2f",  # Format the numbers to 2 decimal places
                            title_kwargs={"fontsize": 12},  # Customize title font size
                            levels=(1 - 0.6827, 1 - 0.9545, 1 - 0.9973),  # 1, 2, 3 sigma
                            plot_contours=True)#, range=df_range)

    plt.show()




