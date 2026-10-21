import os

import numpy as np
from scipy.spatial.distance import pdist, squareform


def derived_path(path, suffix_before_ext):
    """Return ``path`` with ``suffix_before_ext`` inserted before the file extension."""
    base, ext = os.path.splitext(path)
    return f"{base}{suffix_before_ext}{ext}"


def hst_wht_fits_path(sci_fits_path):
    """Path to the HST weight map paired with a ``*_sci.fits`` science image."""
    if len(sci_fits_path) >= 8 and sci_fits_path.endswith('sci.fits'):
        return f"{sci_fits_path[:-8]}wht.fits"
    return None



def read_text_file(file_path):
    lines = []
    with open(file_path, 'r') as file:
        for line in file:
            cleaned_line = line.strip()
            if not cleaned_line.startswith("#"):
                lines.append(cleaned_line)
    return lines


def find_close_coord(cat, d):
    ids = cat['id']
    indices = np.arange(len(ids))
    x = cat['x']
    y = cat['y']
    
    points = np.vstack((x, y)).T
    distance_matrix = squareform(pdist(points))
    n = len(points)
    
    parent = list(range(n))
    
    def find(u):
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]
    
    def union(u, v):
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            parent[root_u] = root_v

    for i in range(n):
        for j in range(i + 1, n):
            if distance_matrix[i, j] < d:
                union(i, j)
    
    groups = {}
    for i in range(n):
        root = find(i)
        if root in groups:
            groups[root].append(ids[i])
            #groups[root].append(indices[i])
        else:
            groups[root] = [ids[i]]
            #groups[root] = [indices[i]]
    
    result = list(groups.values())
    
    return result


def remove_png_margins(im) :
    x_shape = im.shape[0]
    y_shape = im.shape[1]
    
    threshold = 3000.
    
    i=0
    check = False
    while not check :
        i+=1
        check = not np.sum(im[0:i,:,:])>i*y_shape*4.-threshold
        if check :
            print(np.sum(im[0:i,:,:]), i*y_shape*4.)
    left_x_margin = i-1
    
    i=0
    check = False
    while not check :
        i+=1
        check = not np.sum(im[-i-1:-1,:,:])>i*y_shape*4.-threshold
    right_x_margin = -i-1
    
    i=0
    check = False
    while not check :
        i+=1
        check = not np.sum(im[:,0:i,:])>i*x_shape*4.-threshold
    bottom_y_margin = i-1
    
    i=0
    check = False
    while not check :
        i+=1
        check = not np.sum(im[:,-i-1:-1,:])>i*x_shape*4.-threshold
    top_y_margin = -i-1
    
    im_cropped = im[left_x_margin:right_x_margin, bottom_y_margin:top_y_margin, :]
    
    return im_cropped


def has_world_coordinates(cat) :
    for col in cat.colnames :
        if col.lower() in ['ra', 'alpha_j2000', 'x_world'] :
            for col in cat.colnames :
                if col.lower() in ['dec', 'delta_j2000', 'y_world'] :
                    return True
    return False


def make_colnames_dict(catalog, use_default_names=True):
    """
    Extracts column names for positions and shape parameters from an Astropy table.
    Parameters:
    - catalog: Astropy Table
        Input catalog containing astronomical data.
    Returns:
    - column_names: list
        List of column names for positions and shape parameters present in the catalog.
    """
    
    to_test_names_dict = {}
    to_test_names_dict['ra'] = ['ra', 'ALPHA_J2000', 'X_WORLD'] #No need to add different upper/lower cases as str.lower() will be tested
    to_test_names_dict['dec'] = ['dec', 'DELTA_J2000', 'Y_WORLD'] #No need to add different upper/lower cases as str.lower() will be tested
    #to_test_names_dict['x'] = ['X_IMAGE', 'x']
    #to_test_names_dict['y'] = ['Y_IMAGE', 'y']
    to_test_names_dict['a'] = ['a', 'A_IMAGE']
    to_test_names_dict['b'] = ['b', 'B_IMAGE']
    to_test_names_dict['theta'] = ['angle', 'theta', 'THETA_IMAGE']
    
    names_list = list(to_test_names_dict.keys())
    #names_list = ['ra', 'dec', 'x', 'y', 'a', 'b']
    names_dict = {}
    names_dict_default = {}
    for name in names_list :
        names_dict[name] = []
        names_dict_default[name] = None
    for name in names_list :
        for to_test_name in to_test_names_dict[name] :
            cat_colnames_lower = [col.lower() for col in catalog.colnames]
            
            #if 'colnames' in dir(catalog) :
            #    cat_colnames_lower = [col.lower() for col in catalog.colnames]
            #else :
            #    cat_colnames_lower = [col.lower() for col in catalog.columns.names]
            
            if to_test_name.lower() in cat_colnames_lower :
                col_idx = np.where( np.array(cat_colnames_lower) == to_test_name.lower() )[0][0]
                names_dict[name].append(catalog.colnames[col_idx])
                names_dict_default[name] = catalog.colnames[col_idx]
    
    print('Columns found in catalog: \n' + str(names_dict))
    yesno = 'y' if use_default_names else input('Columns to be used: \n' + str(names_dict_default) + \
                                                '\nKeep these names? (if no, user prompted to select other columns) [y][n]')
    if yesno == 'n' :
        for name in names_list :
            if len(names_dict[name]) > 1 :
                selected_name = input("Several columns found for name " + name + ": " + str(names_dict[name]) + ". Which one should be kept (if unit, should be image pixels)?")
                names_dict[name] = selected_name
            else :
                names_dict[name] = names_dict[name][0]
    else :
        names_dict = names_dict_default
    return names_dict


def extract_line(point1, point2, image_array) :
    """
    ***Function created by Gemini AI***
    Extracts pixel values along a line defined by two points in a 2D NumPy array.
    This function implements a variation of Bresenham's line algorithm to
    identify the pixels that lie on the line segment between the two given points.
    Args:
        point1 (tuple): The (x, y) coordinates of the first point.
        point2 (tuple): The (x, y) coordinates of the second point.
        image_array (np.ndarray): The 2D NumPy array (the map) from which
                                  to extract pixel values.
    Returns:
        list: A list of pixel values along the defined line.
              Returns an empty list if points are out of bounds or invalid.
    """
    if not isinstance(image_array, np.ndarray) or image_array.ndim != 2:
        print("Error: image_array must be a 2D NumPy array.")
        return [], []

    height, width = image_array.shape
    x0, y0 = int(point1[0]), int(point1[1])
    x1, y1 = int(point2[0]), int(point2[1])

    # Check if points are within bounds
    if not (0 <= x0 < width and 0 <= y0 < height and
            0 <= x1 < width and 0 <= y1 < height):
        print("Error: One or both points are out of image bounds.")
        return [], []

    # Lists to store the pixel values and their corresponding distances
    pixel_values = []
    distances = []

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    current_x, current_y = x0, y0

    while True:
        # Add the current pixel value
        if 0 <= current_y < height and 0 <= current_x < width:
            pixel_values.append(image_array[current_y, current_x])
            # Calculate Euclidean distance from the starting point (x0, y0)
            dist = np.sqrt((current_x - x0)**2 + (current_y - y0)**2)
            distances.append(dist)
        else:
            # This case should ideally not be hit if initial bounds check is thorough
            print(f"Warning: Pixel ({current_x}, {current_y}) went out of bounds during line drawing.")
            break # Or handle as appropriate

        if current_x == x1 and current_y == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            current_x += sx
        if e2 < dx:
            err += dx
            current_y += sy

    return [distances, pixel_values]


