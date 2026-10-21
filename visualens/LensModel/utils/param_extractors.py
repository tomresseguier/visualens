import re
import pandas as pd
from io import StringIO
import numpy as np
import math
import os
import shutil
import astropy.units as u
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from ...utils.utils_astro.utils_general import world_to_relative
from ...utils.utils_astro.get_default_cosmology import get_default_cosmology
cosmo = get_default_cosmology()






def parse_lenstool_parameter_file(path) :
    """
    Main function to read a lenstool parameter file and turn it into the standard lenstool dictionary format for visualens
    Args:
        path: path to lenstool parameter file
    Returns:
        data: dictionary with the parameters from the lenstool parameter file
    """
    def convert_token(tok):
        """Try to convert to int or float; otherwise return string."""
        try :
            if '.' in tok or 'e' in tok.lower() :
                return float(tok)
            return int(tok)
        except ValueError:
            return tok

    data = {}
    current_section = None
    current_subsection = None

    with open(path, 'r') as f :
        for line in f :
            # Remove comments and whitespace
            line = line.split('#')[0].strip()
            if not line:
                continue

            tokens = line.split()

            # END of current section
            if tokens[0].lower() == 'end' :
                current_section = None
                current_subsection = None
                continue

            # New section begins
            if current_section is None :
                section = line if 'potentiel' not in line else line.replace("potentiel", "potential", 1)
                current_section = section
                data.setdefault(section, {})
                current_subsection = None
                continue

            # Inside a section
            key = tokens[0] if tokens[0]!='ellipticite' else 'ellipticity'
            values = [convert_token(v) for v in tokens[1:]]

            # Logic to handle target nesting
            target = data[current_section]
            if current_subsection is not None :
                target = target[current_subsection]

            if key in target :
                # If it's already a list of lists, just append
                if isinstance(target[key], list) and any(isinstance(i, list) for i in target[key]) :
                    target[key].append(values)
                else :
                    # Convert existing single entry into a list of lists, then add the new one
                    # We ensure 'values' is always treated as a list representing the line
                    target[key] = [target[key] if isinstance(target[key], list) else [target[key]], values]
            else :
                # First time seeing this key: store values as a single entry
                # If values has only 1 item, we store that item; otherwise the whole list
                target[key] = values if len(values) > 1 else values[0]
    
    # Remove the last item ('fini' keyword)
    if 'fini' in data :
        del data['fini']
    return data


def read_bayes_file(file_path, convert_to_kpc=True, z=None):
    """
    Reads bayes.dat and creates a dataFrame with all the parameters as columns and 
    their values in the different samples as lines
    Args:
        file_path: path to bayes.dat file
    Returns:
        df: dataFrame with all the parameters as columns and their values in the different samples as lines
    
                            Nsample  ln(Lhood)      ...    Pot0 sigma   Chi2
                       0    1        -64393.427747  ...    195.359518   128880.763
    bayes.dat -->      1    1        -64400.548013  ...    195.365226   128895.003
                       2    1        -64392.938091  ...    195.363315   128879.784
                       3    1        -64392.622949  ...    195.359561   128879.153
    """
    columns = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("#"):
                col_name = line[1:].strip()
                columns.append(col_name)
            else:
                break
    
    df = pd.read_csv(file_path, sep='\s+', comment='#', header=None)
    df.columns = columns
    
    if convert_to_kpc :
        if z is not None :
            ang_diam_dist_value = cosmo.angular_diameter_distance(z).to('kpc').value
            kpc_per_arcsec = ang_diam_dist_value * 1.*u.arcsec.to(u.rad)
            for col in df.columns :
                if 'rc (arcsec)' in col or 'rcut (arcsec)' in col :
                    df[col] = df[col]*kpc_per_arcsec
                    df = df.rename(columns={col: col[:-len('(arcsec)')] + '(kpc)'})
        else :
            raise ValueError("A redshift must be specified.")
    return df


def calculate_parameter_statistics_from_bayes(bayes_df) :
    """
    Calculates the MCMC statistics from the bayes dataFrame.
    Args:
        bayes_df: dataFrame with all the parameters as columns and their values in the different samples as lines
    Returns:
        param_dict: dictionary with calculated statistics
    """
    param_dict = {}
    for col in bayes_df.select_dtypes(include='number').columns :
        lower, median, upper = np.percentile(bayes_df[col], [16, 50, 84])
        param_dict[col] = [lower, median, upper]
    return param_dict


def make_table_dict_from_bayes(bayes_df) :
    """
    Goes from bayes dataFrame to dictionary with calculated statistics sorted per potential
    Args:
        bayes_df: dataFrame with all the parameters as columns and their values in the different samples as lines
    Returns:
        pot_dict: dictionary with calculated statistics sorted per potential
    """
    param_dict = calculate_parameter_statistics_from_bayes(bayes_df)
    
    ### Extract the potential names ###
    pot_list = []
    for col in param_dict.keys() :
        pot_name = col.split()[0]
        if pot_name.startswith('O') or pot_name.startswith('Pot') :
            pot_list.append(pot_name)
    pots = np.unique(pot_list)
    
    correspondances = { 'x_centre': 'x',
                        'y_centre': 'y',
                        'ellipticity': 'emass',
                        'angle_pos': 'theta'}
    
    check_kpc = False
    for col in bayes_df.columns :
        if 'kpc' in col :
            check_kpc = True
    if check_kpc :
        correspondances['core_radius_kpc'] = 'rc'
        correspondances['cut_radius_kpc'] = 'rcut'
    else :
        correspondances['core_radius'] = 'rc'
        correspondances['cut_radius'] = 'rcut'
    correspondances['v_disp'] = 'sigma'
    correspondances['gamma'] = 'gamma'
    
    pot_dict = {}
    for pot in pots :
        indiv_pot_dict = {}
        indiv_pot_dict['name'] = ['...' for i in range(len(correspondances.keys()))]
        indiv_pot_dict['value'] = ['...' for i in range(len(correspondances.keys()))]
        indiv_pot_dict['uncertainty1'] = ['...' for i in range(len(correspondances.keys()))]
        indiv_pot_dict['uncertainty2'] = ['...' for i in range(len(correspondances.keys()))]
        for col in param_dict.keys() :
            if col.split()[0]==pot :
                for i, name in enumerate(correspondances.keys()) :
                    indiv_pot_dict['name'][i] = name
                    if correspondances[name] in col.split() :
                        indiv_pot_dict['value'][i] = param_dict[col][1]
                        indiv_pot_dict['uncertainty1'][i] = param_dict[col][1] - param_dict[col][0]
                        indiv_pot_dict['uncertainty2'][i] = param_dict[col][2] - param_dict[col][1]
        pot_dict[pot] = pd.DataFrame(indiv_pot_dict)
        
    return pot_dict


def extract_main_pot_names(param_file_path) :
    """
    Extracts the main potentials' names from the parameter file
    Args:
        param_file_path: path to lenstool parameter file
    Returns:
        pot_names: list of main potentials' names
    """
    param_dict = parse_lenstool_parameter_file(param_file_path)
    pot_names = []
    for name in param_dict :
        if 'potential' in name : #'potentiel' in name or 
            pot_names.append(name)
    #pot_names = []
    #pattern = re.compile(r'^\s*potentiel\s+(\S+)', re.MULTILINE)
    #with open(param_file_path, 'r') as file:
    #    content = file.read()
    #pot_names = pattern.findall(content)
    return pot_names


def make_full_pot_dict(param_file_path, convert_to_kpc=False, z=None) :
    """
    Completes pot_dict with fixed values from param file
    Args:
        param_file_path
    Returns:
        full_pot_dict: dictionary with completed pot_dict
    """
    model_dir = os.path.dirname(param_file_path)
    bayes_file_path = os.path.join(model_dir, 'bayes.dat')
    bayes_df = read_bayes_file(bayes_file_path, convert_to_kpc=convert_to_kpc, z=z)
    full_pot_dict = make_table_dict_from_bayes(bayes_df)
    #param_file = pylenstool.lenstool_param_file(param_file_path)
    param_dict = parse_lenstool_parameter_file(param_file_path)
    pot_names_alt = extract_main_pot_names(param_file_path)
    
    for i, pot in enumerate(full_pot_dict.keys()) :
        if pot!='Pot0' :
            for j, value in enumerate(full_pot_dict[pot]['value']) :
                if value=='...' :
                    #if full_pot_dict[pot]['name'][j] = 'ellipticity' :
                    if full_pot_dict[pot]['name'][j] in param_dict[pot_names_alt[i]] :
                        param_value = param_dict[pot_names_alt[i]][full_pot_dict[pot]['name'][j]]
                        full_pot_dict[pot].loc[j, 'value'] = str(param_value)
                    if full_pot_dict[pot]['name'][j] + '_kpc' in param_dict[pot_names_alt[i]] :
                        full_pot_dict[pot].loc[j, 'name'] = full_pot_dict[pot]['name'][j] + '_kpc'
                        param_value_kpc = param_dict[pot_names_alt[i]][full_pot_dict[pot]['name'][j]]
                        full_pot_dict[pot].loc[j, 'value'] = str(param_value_kpc)
    return full_pot_dict


def generate_latex_table(full_pot_dict, ref_coord=None, deluxetable=False) :
    """
    Generates LaTeX string from dictionary with calculated statistics sorted per potential
    Args:
        full_pot_dict: dictionary with calculated statistics sorted per potential
        ref_coord: reference coordinates
        deluxetable: if True, output in AASTeX deluxetable* format instead of standard tabular
    Returns:
        table_str: LaTeX string
    """
    if ref_coord is not None :
        ref_ra, ref_dec = ref_coord
    
    table_rows = []

    for key in full_pot_dict.keys() :
        df = full_pot_dict[key]
        table_row = key
        
        uncertainty_from_bayes = False
        for _, row in df.iterrows() :
            if type(row['uncertainty2']) is not str :
                if abs(row['uncertainty2'])>0. :
                    uncertainty_from_bayes = True
        
        
        for name in ['x_centre', 'y_centre', 'ellipticity', 'angle_pos', 'core_radius_kpc', 'cut_radius_kpc', 'v_disp', 'gamma'] :
            i_array = np.where(df['name']==name)[0]
            if len(i_array)!=0 :
                i = i_array[0]
                
                if uncertainty_from_bayes :
                    if type(df['value'][i]) is str :
                        val_str = df['value'][i]
                        if val_str == '...' :
                            table_row = table_row + " & ..."
                        else :
                            try :
                                fixed_val = float(val_str)
                                table_row = table_row + f" & $[{fixed_val:.4g}]$"
                            except (ValueError, TypeError) :
                                table_row = table_row + f" & $[{val_str}]$"
                    else :
                        value = df['value'][i]
                        uncertainty1 = df['uncertainty1'][i]
                        uncertainty2 = df['uncertainty2'][i]
                        
                        
                        sig_fig1 = -math.floor(math.log10(abs(uncertainty1)))
                        if sig_fig1>=0 :
                            sig_fig1 = -math.floor(math.log10(abs( float(f"{uncertainty1:.{sig_fig1}f}") )))
                        if sig_fig1<0 :
                            sig_fig1 = 0
                        
                        sig_fig2 = -math.floor(math.log10(abs(uncertainty2)))
                        if sig_fig2>=0 :
                            sig_fig2 = -math.floor(math.log10(abs( float(f"{uncertainty2:.{sig_fig2}f}") )))
                        if sig_fig2<0 :
                            sig_fig2 = 0
                        
                        sig_fig = max(sig_fig1, sig_fig2)
                        
                        table_row = table_row + f" & ${value:.{sig_fig}f}" + "^{+" + f"{uncertainty2:.{sig_fig}f}" + "}_{-" + f"{uncertainty1:.{sig_fig}f}" + "}$"
                    
                else :
                    value = float(df['value'][i])
                    uncertainty = float(df['uncertainty1'][i])
                    
                    sig_fig = -math.floor(math.log10(abs(uncertainty)))
                    if sig_fig>=0 :
                        sig_fig = -math.floor(math.log10(abs( float(f"{uncertainty:.{sig_fig}f}") )))
                    if sig_fig<0 :
                        sig_fig = 0
                    
                    table_row = table_row + f" & {value:.{sig_fig}f}" + r" $\pm$ " + f"{uncertainty:.{sig_fig}f}"
                    
                    
            else :
                table_row = table_row + " & ..."
                
        table_row = table_row + " \\\\"
        table_rows.append(table_row)

    header_cols = ( "Component & $\\Delta$ R.A. (\") & $\\Delta$ Dec (\") & ellipticity & "
                    "$\\theta$ (deg) & $r_{\\rm core}$ (kpc) & $r_{\\rm cut}$ (kpc) & "
                    "$\\sigma_0$ (km s$^{-1}$) & $\\gamma$" )
    data_str = "        " + "\n        ".join(table_rows) + "\n"

    if deluxetable :
        col_heads = ( "\\colhead{Component} & \\colhead{$\\Delta$ R.A. (\")} & \\colhead{$\\Delta$ Dec (\")} & "
                      "\\colhead{ellipticity} & \\colhead{$\\theta$ (deg)} & \\colhead{$r_{\\rm core}$ (kpc)} & "
                      "\\colhead{$r_{\\rm cut}$ (kpc)} & \\colhead{$\\sigma_0$ (km s$^{-1}$)} & \\colhead{$\\gamma$}" )
        table_str = ( "\\begin{deluxetable*}{ccccccccc}\n"
                      "\\tablecaption{Parameters of the lens model. \\label{tab:SL_params}}\n"
                      "\\tablewidth{\\columnwidth}\n"
                      "\\tablehead{" + col_heads + "}\n"
                      "\\startdata\n" ) + \
                    data_str + \
                    ( "\\enddata\n"
                      "\\end{deluxetable*}" )
    else :
        table_str = ( "\\begin{table*}[htb!]\n"
                      "\\begin{center}\n"
                      "    \\begin{tabular}{c c c c c c c c c}\n"
                      "        \\hline\\hline\n"
                      "        " + header_cols + " \\\\\n"
                      "        \\hline\n" ) + \
                    data_str + \
                    ( "        \\hline\n"
                      "    \\end{tabular}\n"
                      "    \\caption{Parameters of the lens model.}\n"
                      "    \\label{tab:SL_params}\n"
                      "\\end{center}\n"
                      "\\end{table*}" )
    
    return table_str


def make_param_latex_table(param_file_path, convert_to_kpc=True, z=None, deluxetable=True) :
    """
    Returns and saves LaTeX table as text file
    Args:
        param_file_path: path to lenstool parameter file
        convert_to_kpc: whether to convert to kpc
        z: lens redshift
        deluxetable: if True, output in AASTeX deluxetable* format instead of standard tabular
    Returns:
        table_str: LaTeX string
    """
    model_dir = os.path.dirname(param_file_path)
    full_pot_dict = make_full_pot_dict(param_file_path, convert_to_kpc=convert_to_kpc, z=z)
    table_str = generate_latex_table(full_pot_dict, deluxetable=deluxetable)
    param_latex_table = os.path.join(model_dir, 'best_params.latex')
    open(param_latex_table, 'w').write(table_str)
    return table_str





###############################################################################
#Creates the best file for a single sample in order to calculate magnification uncertainties etc.
#
###############################################################################
def make_single_sample_best_file(lt, sample_index) :
    best_str = f'runmode\n\treference  3 {lt.reference[0]} {lt.reference[1]}\n'
    best_str += f'\timage  1 { os.path.basename(lt.mult_path) }\n'
    best_str += '\tend\n'
    
    """ Copy the optimized redshift info """
    best_str += 'image\n'
    for l in lt.param_best['image']['z_m_limit'] :
        best_str += f'\tz_m_limit  {" ".join(str(item) for item in l)}\n'
    best_str += '\tend\n'
    
    """ Match cosmology to param file """
    best_str += 'cosmology\n'
    for param, value in lt.param_best['cosmology'].items() :
        best_str += f'\t{param}  {value}\n'
    best_str += '\tend\n'
    
    pot_names = []
    for pot_name in lt.param_best :
        if pot_name.startswith('potential') or pot_name.startswith('potentiel') :
            pot_names.append(pot_name)
            
    best_str += f'grid\n\tnlens {len(pot_names)}\n\tend\n'
    
    match_dict = {'x_centre': 'x (arcsec)',
                  'y_centre': 'y (arcsec)',
                  'ellipticity': 'emass',
                  'angle_pos': 'theta (deg)',
                  'core_radius': 'rc (arcsec)',
                  'cut_radius': 'rcut (arcsec)',
                  'v_disp': 'sigma (km/s)',
                  'gamma': 'gamma',
                  'angle_pos': 'theta (deg)'}
    
    for pot_name in pot_names :
        best_str += f'{pot_name}\n'
        for param, value in lt.param_best[pot_name].items() :
            param_matched = match_dict.get(param)
            if param_matched is not None :
                for col in lt.samples_df.columns :
                    if pot_name.split()[1] == col.split()[0] and param_matched in col :
                        value = lt.samples_df[col][sample_index]
            best_str += f'\t{param} {value}\n'
        best_str += f'\tend\n'
    best_str += f'finish\n'
    return best_str

def write_single_sample_best_file(lt, sample_index) :
    best_str = make_single_sample_best_file(lt, sample_index)
    if not os.path.exists(os.path.join(lt.model_dir, 'samples')) :
        os.makedirs(os.path.join(lt.model_dir, 'samples'))
    sample_file_path = os.path.join(lt.model_dir, 'samples', f'best_sample_{sample_index}.par')
    with open(sample_file_path, 'w') as file :
        file.write(best_str)
    print(f'Single sample best file written to {sample_file_path}')
    return sample_file_path














###############################################################################
############# CHECK IF FOLLOWING FUNCTIONS ARE STILL USEFUL OR NOT ############
# + remove pylenstool
###############################################################################




def find_ref(param_file_path) :
    with open(param_file_path, 'r') as text_file :
        full_string = text_file.read()
    pattern = re.compile(r"reference\s+\d+\s+([-+]?\d*\.\d+)\s+([-+]?\d*\.\d+)")
    match = pattern.search(full_string)
    if match :
        ra = float(match.group(1))
        dec = float(match.group(2))
        return ra, dec
    else :
        raise ValueError("Reference RA and DEC not found in the file content.")





###############################################################################
#Extracts the potfile and reference galaxy parameters from param file
#Extracts the optimized potfile and reference galaxy parameters from bayes.dat
#Joins both dictionaries
#param_file_path --> potfile_param_dict
###############################################################################
def get_potfile_params(param_file_path) :
    param_file = pylenstool.lenstool_param_file(param_file_path)
    potfile_param_dict = {}
    potfile_name = param_file.get_parameter('potfile', 'filein')[-1]
    potfile_path = os.path.join(os.path.dirname(param_file_path), potfile_name)
    potfile_param_dict['potfile_path'] = potfile_path
    for name in ['type', 'zlens', 'mag0', 'corekpc'] :
        potfile_param_dict[name] = float(param_file.get_parameter('potfile', name)[1])
    for name in ['sigma', 'cutkpc', 'slope', 'vdslope'] :
        if int(param_file.get_parameter('potfile', name)[1])==0 :
            potfile_param_dict[name] = float(param_file.get_parameter('potfile', name)[2])
    
    full_pot_dict = make_full_pot_dict(param_file_path)
    Pot0_df = full_pot_dict['Pot0']
    for line in Pot0_df.iloc :
        if line['value']!='...' :
            potfile_param_dict[line['name']] = float(line['value'])
    return potfile_param_dict

###############################################################################
#Reads potfile and puts galaxies in a catalog
#potfile_path --> potfile_cat
###############################################################################
def read_potfile(potfile_path) :
    with open(potfile_path, 'r') as file :
        lines = file.readlines()
    potfile_cat = Table(names=['id','ra','dec','a','b','theta','mag','lum'], dtype=['int', *['float',]*7])
    for line in lines :
        columns = line.split()
        if columns[0].isdigit() :
            potfile_cat.add_row( [col for col in columns] )
    return potfile_cat



def read_arclet(arclet_path) :
    with open(arclet_path, 'r') as file :
        lines = file.readlines()
    arclet_cat = Table(names=['id','ra','dec','a','b','theta','mag','lum'], dtype=['str', *['float',]*7])
    for line in lines :
        columns = line.split()
        arclet_cat.add_row( [col for col in columns] )
    return arclet_cat
def extract_magnification_from_dist(dist_path) :
    dist_text = open(dist_path, 'r').read()
    data_lines = [line for line in dist_text.splitlines() if line.strip() and not line.strip().startswith('#')]
    table_text = "\n".join(data_lines)
    columns = ['ID', 'X', 'Y', 'R', 'EPS', 'TAU', 'AMP', 'E_AMP', 'DMAG', 'TIME[days]', 'DTIME', 'PARITY']
    table_df = pd.read_csv(StringIO(table_text), sep='\s+', names=columns)
    return table_df






