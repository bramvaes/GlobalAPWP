"""
PYTHON CODE WITH FUNCTION FOR COMPUTING AN APWP FROM SITE-LEVEL DATA
"""

# ------------------------
# Import packages
import pmagpy.pmag as pmag
import pmagpy.pmagplotlib as pmagplotlib
import pmagpy.ipmag as ipmag
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from numpy import random
from numpy import float64

def boot_ref_pole(PPs,Ks,Ns,N_poles,Nb=100,balanced=True):
  """
  Generates a reference pole based on parametrically sampled VGPs from a collection of paleopoles (with K and N)

  Input:
  PPs: nested list of paleopoles [longitude, latitude]
  kappas: list of Fisher (1953) precision parameter values for distribution of VGPs underlying each paleopole
  Ns: list of integers corresponding to number of VGPs from which each paleopole was computed
  N_poles: number of input paleopoles
  Nb: number of bootstrap runs
  balanced: choose if paleopoles are sampled using a balanced or unbalanced bootstrap (default is True)
  """
  
  # print warning if lists are of unequal size
  if len(PPs)!=N_poles or len(Ks)!=N_poles or len(Ns)!=N_poles:
    print('WARNING: INPUT LISTS DO NOT HAVE EQUAL LENGTH')
  
  # generate Nb pseudopoles from parametrically sampled VGPs
  if balanced==True:
    boot_means = [gen_pseudopoles(PPs,Ks,Ns,N_poles) for i in range(Nb)]
  else:
    boot_means = [unb_pseudopoles(PPs,Ks,Ns,N_poles) for i in range(Nb)]

  # print mean of Nb bootstrap means and associated statistical parameters
  bpars = pmag.fisher_mean(boot_means)
  # ipmag.print_pole_mean(bpars)
  boot_ref = [bpars['dec'],bpars['inc']]
  N_VGPs = sum(Ns)
  # print('N_VGPs = ', N_VGPs)

  # Compute angular distances of pseudopoles to reference pole
  D_ppoles=[]
  for j in range(Nb):
      ang_distance=pmag.angle(boot_means[j],boot_ref)
      D_ppoles.append(ang_distance[0])

  # Compute statistical parameters
  D_ppoles.sort()
  ind_95perc=int(0.95*Nb)
  V95 = D_ppoles[ind_95perc]
  # print('V95 = ', V95)

  return boot_ref, V95, N_VGPs


def boot_APWP (data, plon_label, plat_label, age_label, window_length, time_step, max_age, min_age, Nb=100, balanced=True):
  """
  Generates a bootstrapped APWP
  """

  mean_pole_ages = np.arange(min_age, max_age + time_step, time_step)

  boot_ref_poles = pd.DataFrame(columns=['age','N_VGPs','V95','plon','plat'])

  for age in mean_pole_ages:
      window_min = age - (window_length / 2.)
      if age == 0:
        window_max = 5
      else:
        window_max = age + (window_length / 2.)

      # compute bootstrapped reference pole
      poles = data.loc[(data[age_label] >= window_min) & (data[age_label] <= window_max)]
      N_poles = len(poles)
      plons = poles[plon_label].tolist()
      plats = poles[plat_label].tolist()
      #print(poles['name'],poles['K'],poles['N'])
      PPs = [ [plons[i],plats[i]] for i in range(N_poles)]
      ref_pole, V95, N_VGPs = boot_ref_pole(PPs,poles['K'].tolist(),poles['N'].tolist(),N_poles,Nb=Nb,balanced=balanced)
      
      if ref_pole: # this just ensures that dict isn't empty
          boot_ref_poles.loc[age] = [age, N_VGPs, V95, ref_pole[0], ref_pole[1]]

  boot_ref_poles.reset_index(drop=1, inplace=True)

  return boot_ref_poles

def running_mean_APWP (data, plon_label, plat_label, age_label, window_length, time_step, max_age, min_age, elong=False):
    """
    Generates a running mean apparent polar wander path from a collection of poles
    
    Required input:
    data: Pandas dataframe with pole longitude, latitude and age (in Ma)
    plon_label, plat_label, age_label: column names of pole longitude, pole latitude and age
    window_length: size of time window (in Ma)
    time_step: time step at which APWP is computed (in Ma)
    min_age, max_age: age interval for which APWP is computed (in Ma)
    
    Output:
    Pandas dataframe with APWP and associated statistical parameters
    """
    
    mean_pole_ages = np.arange(min_age, max_age + time_step, time_step)
    
    if elong == True:
        running_means = pd.DataFrame(columns=['age','N','A95','plon','plat','kappa','csd','E','mean_age'])
    else:
        running_means = pd.DataFrame(columns=['age','N','A95','plon','plat','kappa','csd','mean_age'])
    
    for age in mean_pole_ages:
        window_min = age - (window_length / 2.)
        if age == 0:
          window_max = 5
        else:
          window_max = age + (window_length / 2.)
        # store poles (or VGPs) that fall within time window
        poles = data.loc[(data[age_label] >= window_min) & (data[age_label] <= window_max)]
        plons,plats = poles[plon_label].tolist(),poles[plat_label].tolist()
        
        # compute Fisher (1953) mean
        mean = ipmag.fisher_mean(dec=plons, inc=plats)
        mean_age = poles.age.mean() # compute mean age
        
        # compute elongation
        if elong == True:
            pole_block = ipmag.make_di_block(plons,plats,unit_vector=False)
            ppars = pmag.doprinc(pole_block)
            E = ppars["tau2"] / ppars["tau3"]
        
        if mean: # this just ensures that dict isn't empty
            if elong == True:
                running_means.loc[age] = [age, mean['n'], mean['alpha95'], mean['dec'], mean['inc'], mean['k'], mean['csd'], E, mean_age]
            else:
                running_means.loc[age] = [age, mean['n'], mean['alpha95'], mean['dec'], mean['inc'], mean['k'], mean['csd'], mean_age]
        else:
            print('No pseudo-VGPs in age bin from %1.1f to %1.1f Ma' % (window_min,window_max))
    
    running_means.reset_index(drop=1, inplace=True)
    
    return running_means

# ------------------------

def get_pseudo_vgps(plons,plats,Ks,Ns,plate_IDs,age,min_age,max_age,sim_age=True):
    """
    Generates pseudopole from reference dataset of paleopoles with N and K
    """
    N_poles = len(plons)

    # generate nested list of parametrically sampled VGPs
    vgps_nested_list = [fish_VGPs(K=Ks[j],n=Ns[j],lon=plons[j],lat=plats[j]) for j in range(N_poles)]
    vgps_list = [vgps_nested_list[i][j] for i in range(N_poles) for j in range(Ns[i])]

    # create dataframe
    pseudo_VGPs = pd.DataFrame(vgps_list,columns=['plon','plat'])
    
    # generate and add plate IDs to dataframe
    plate_id_nested_list = [[plate_IDs[i]]*Ns[i] for i in range(N_poles)]
    plate_id_list = [plate_id_nested_list[i][j] for i in range(N_poles) for j in range(Ns[i])]
    pseudo_VGPs['plate_id'] = plate_id_list

    # generate and add simulated ages
    if sim_age == True:
      ages_nested_list = [random.uniform(low=min_age[i],high=max_age[i],size=Ns[i]) for i in range(N_poles)]
      ages_list = [ages_nested_list[i][j] for i in range(N_poles) for j in range(Ns[i])]
      pseudo_VGPs['age'] = ages_list

    return pseudo_VGPs

def rotate_with_ep(plon,plat,age,plate_ID,EP_data,output_type):
  """
  Rotate pole/VGP of with specified age and plate ID with Euler pole listed in .csv file

  Input:
  plon, plat: pole longitude and latitude
  age: age of pole/VGP
  plate_ID: plate number (e.g., 101 for North America)
  EP_data: multidimensional numpy array with columns plate ID, age, Euler pole latitude, Euler pole longitude, Euler pole angle
  
  Output:
  If output_type='rvgps': longitude and latitude of rotate pole/VGP
  If different output_type: latitude, longitude and angle of Euler pole
  """

  # round age to nearest integer
  rounded_age = round(age)

  # find index of Euler pole with correct plate ID and age (rounded to nearest integer value)
  indx = np.where((EP_data[:,0] == plate_ID) & (EP_data[:,1] == rounded_age))

  # rotate pole with Euler pole
  rlon,rlat = pt_rot([EP_data[indx,2][0][0], EP_data[indx,3][0][0],EP_data[indx,4][0][0]],[plat],[plon])

  # specify output
  if output_type == 'rvgps':
    return rlon,rlat
  else:
    return EP_data[indx,2][0][0], EP_data[indx,3][0][0], EP_data[indx,4][0][0]


def get_rotated_vgps(plons,plats,ages,plate_IDs,EP_data,output_type='rvgps'):
  """
  Rotate collection of poles/VGPs with Euler poles listed in file

  Input:
  plons, plats: pole/VGP longitudes and latitudes
  ages: ages of pole/VGP
  plate_IDs: plate numbers (e.g., 101 for North America)
  EP_data: multidimensional numpy array with columns plate ID, age, Euler pole latitude, Euler pole longitude, Euler pole angle
  
  Output:
  If output_type='EPs': dataframe with latitude, longitude and angle of Euler poles 
  If different output_type: nested list of rotated poles/VGPs
  """

  # get rotated VGPs or Euler poles
  output_list = [rotate_with_ep(plons[i],plats[i],ages[i],plate_IDs[i],EP_data,output_type=output_type) for i in range(len(ages))]

  # create dataframe
  if output_type == 'EPs':
    EPs = pd.DataFrame(output_list,columns=['EP_lon','EP_lat','EP_ang'])
    return EPs
  
  else:
    return output_list

def sedimentary_pole_resample(PP_lon = 90, PP_lat = 45, a95 = 2, slon = 0, slat = 0, p_std = 1, plot = False):
    """
    Generates a pseudopole with a paleomagnetic colatitude sampled from a normal distribution with given standard deviation.
    
    Parameters
    ----------
    PP_lon : paleopole longitude
    PP_lat : paleopole latitude
    slon : longitude of sampling locationn
    slat : latitude of sampling location
    p_std : standard deviation of normal distribution of paleomagnetic colatitudes

    Returns
    ---------
    plon, plat: pseudopole longitude and latitude
    """

    # determine colatitude of paleopole
    p_ref = pmag.angle([PP_lon,PP_lat],[slon,slat])

    # resample colatitude from normal distribution
    p = np.random.normal(p_ref, p_std)

    # compute paleomagnetic direction
    dec, inc = pmag.vgp_di(PP_lat,PP_lon,slat,slon)

    # convert values to radians
    rad = np.pi / 180.
    p, dec, slat, slon = p * rad, dec * rad, slat * rad, slon * rad

    # determine latitude and longitude of pseudopole
    plat = np.arcsin(np.sin(slat) * np.cos(p) + np.cos(slat) * np.sin(p) * np.cos(dec)) # compute pole latitude
    beta = np.arcsin((np.sin(p) * np.sin(dec)) / np.cos(plat)) # compute beta angle
    if np.cos(p) > np.sin(slat) * np.sin(plat):
      plong = slon + beta
    else:
      plong = slon + np.pi - beta

    # determine paleopole position
    plat = np.rad2deg(plat)
    plong = np.rad2deg(plong)

    # plot (optional)
    if plot == True:
      m = ipmag.make_orthographic_map(180,60, land_color='linen',add_ocean=True,ocean_color='azure')
      ipmag.plot_pole(m, plong, plat, a95, color='red', edgecolor=None, markersize=20, filled_pole=False, fill_alpha=0.1, outline=False, label='Pseudopole')
      ipmag.plot_pole(m, PP_lon,PP_lat, a95, color='k', edgecolor=None, markersize=20, filled_pole=False, fill_alpha=0.1, outline=False, label='Original paleopole')
      plt.legend(loc=1)
      plt.show()

    return plong, plat

def get_resampled_sed_poles(dataframe):
    """
    Generates resampled sedimentary paleopoles using the standard deviation of an assumed Gaussian distribution of paleomagnetic colatitudes obtained from the E/I correction.
    
    Parameters
    ----------
    dataframe: dataframe with input data

    Returns
    ---------
    new_dataframe: dataframe with resampled sedimentary paleopoles (plon and plat)
    """
    new_dataframe = dataframe.copy()
    
    for index, row in new_dataframe.iterrows(): # iterate over rows of dataframe
        if row.lithology == 'sedimentary':
            # resample paleopole position
            p_std = row.p_std if row.p_std>0 else 2.
            #p_std = 3.
            new_plon, new_plat = sedimentary_pole_resample(PP_lon = row.plon, PP_lat = row.plat, slon = row.slon, slat = row.slat, p_std = p_std, plot = False)
            # replace value in updated dataframe
            new_dataframe['plon'][index], new_dataframe['plat'][index] = new_plon[0],new_plat[0]
    
    return new_dataframe

# ------------------------
# 12/04/2024: ADDED FUNCTION FROM VAES_FUNC.PY TO HAVE ALL RELEVANT FUNCTIONS IN SAME FILE
# ------------------------

def fish_VGPs(K=20, n=100, lon=0, lat=90):
    """
    Generates Fisher distributed unit vectors from a specified distribution
    using the pmag.py fshdev and dodirot functions.
    Parameters
    ----------
    k : kappa precision parameter (default is 20)
    n : number of vectors to determine (default is 100)
    lon : mean longitude of distribution (default is 0)
    lat : mean latitude of distribution (default is 90)
    di_block : this function returns a nested list of [lon,lat] as the default
    if di_block = False it will return a list of lon and a list of lat
    Returns
    ---------
    di_block : a nested list of [lon,lat] (default)
    lon,lat : a list of lon and a list of lat (if di_block = False)
    """

    k = np.array(K)
    R1 = np.random.random(size=n)
    R2 = np.random.random(size=n)
    L = np.exp(-2 * k)
    a = R1 * (1 - L) + L
    fac = np.sqrt(-np.log(a)/(2 * k))
    inc = 90. - np.degrees(2 * np.arcsin(fac))
    dec = np.degrees(2 * np.pi * R2)

    DipDir, Dip = np.ones(n, dtype=float).transpose(
    )*(lon-180.), np.ones(n, dtype=float).transpose()*(90.-lat)
    data = np.array([dec, inc, DipDir, Dip]).transpose()
    drot, irot = pmag.dotilt_V(data)
    drot = (drot-180.) % 360.  #
    VGPs = np.column_stack((drot, irot))
    # rot_data = np.column_stack((drot, irot))
    # VGPs = rot_data.tolist()

    # for data in range(n):
    #     lo, la = pmag.fshdev(K)
    #     drot, irot = pmag.dodirot(lo, la, lon, lat)
    #     VGPs.append([drot, irot])
    return VGPs

def gen_pseudopoles(ref_poles,kappas,ns,N_ref_poles,N_test=0,equal_N=False):
    """
    Generates pseudopole from reference dataset of paleopoles with N and K
    """

    # generate nested list of parametrically sampled VGPs
    nested_VGPs = [fish_VGPs(K=kappas[j],n=ns[j],lon=ref_poles[j][0],lat=ref_poles[j][1]) for j in range(N_ref_poles)]
    # create single list with all simulated VGPs
    sim_VGPs = [nested_VGPs[i][j] for i in range(N_ref_poles) for j in range(ns[i])]
    #print(sim_VGPs)

    # if equal_N==True:
    #     # select random VGPs from dataset
    #     Inds = np.random.randint(len(sim_VGPs), size=N_test)
    #     #print(Inds)
    #     D = np.array(sim_VGPs)
    #     sample = D[Inds]
    # else:
    #     sample = sim_VGPs

    # compute pseudopole
    polepars = pmag.fisher_mean(sim_VGPs)
    return [polepars['dec'], polepars['inc']]

def unb_pseudopoles(ref_poles,kappas,ns,N_ref_poles,N_test=0,equal_N=False):
    """
    Generates pseudopole from reference dataset of paleopoles with N and K. Poles are randomly drawn with replacement.
    """

    # generate random integers between 0 and N_ref_poles
    inds = np.random.randint(N_ref_poles,size=N_ref_poles)
    #print(inds)

    # generate nested list of parametrically sampled VGPs
    nested_VGPs = [fish_VGPs(K=kappas[j],n=ns[j],lon=ref_poles[j][0],lat=ref_poles[j][1]) for j in inds]
    # create single list with all simulated VGPs
    # can this be done better/faster?!
    sim_VGPs = [nested_VGPs[i][j] for i in range(N_ref_poles) for j in range(len(nested_VGPs[i]))]
    #print(sim_VGPs)

    # if equal_N==True:
    #     # select random VGPs from dataset
    #     Inds = np.random.randint(len(sim_VGPs), size=N_test)
    #     #print(Inds)
    #     D = np.array(sim_VGPs)
    #     sample = D[Inds]
    # else:
    #     sample = sim_VGPs

    # compute pseudopole
    polepars = pmag.fisher_mean(sim_VGPs)
    return [polepars['dec'], polepars['inc']]

def pt_rot(EP, Lats, Lons):
    """
    Rotates points on a globe by an Euler pole rotation using method of
    Cox and Hart 1986, box 7-3.
    Parameters
    ----------
    EP : Euler pole list [lat,lon,angle] specifying the location of the pole;
    the angle is for a counterclockwise rotation about the pole
    Lats : list of latitudes of points to be rotated
    Lons : list of longitudes of points to be rotated
    Returns
    _________
    RLats : list of rotated latitudes
    RLons : list of rotated longitudes
    """
# gets user input of Rotation pole lat,long, omega for plate and converts
# to radians
    E = pmag.dir2cart([EP[1], EP[0], 1.])  # EP is pole lat,lon omega
    omega = np.radians(EP[2])  # convert to radians
    RLats, RLons = [], []
    for k in range(len(Lats)):
        if Lats[k] <= 90.:  # peel off delimiters
            # converts to rotation pole to cartesian coordinates
            A = pmag.dir2cart([Lons[k], Lats[k], 1.])
# defines cartesian coordinates of the pole A
            R = [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
            R[0][0] = E[0] * E[0] * (1 - np.cos(omega)) + np.cos(omega)
            R[0][1] = E[0] * E[1] * (1 - np.cos(omega)) - E[2] * np.sin(omega)
            R[0][2] = E[0] * E[2] * (1 - np.cos(omega)) + E[1] * np.sin(omega)
            R[1][0] = E[1] * E[0] * (1 - np.cos(omega)) + E[2] * np.sin(omega)
            R[1][1] = E[1] * E[1] * (1 - np.cos(omega)) + np.cos(omega)
            R[1][2] = E[1] * E[2] * (1 - np.cos(omega)) - E[0] * np.sin(omega)
            R[2][0] = E[2] * E[0] * (1 - np.cos(omega)) - E[1] * np.sin(omega)
            R[2][1] = E[2] * E[1] * (1 - np.cos(omega)) + E[0] * np.sin(omega)
            R[2][2] = E[2] * E[2] * (1 - np.cos(omega)) + np.cos(omega)
# sets up rotation matrix
            Ap = [0, 0, 0]
            for i in range(3):
                for j in range(3):
                    Ap[i] += R[i][j] * A[j]
# does the rotation
            Prot = pmag.cart2dir(Ap)
            RLats.append(Prot[1])
            RLons.append(Prot[0])
        else:  # preserve delimiters
            RLats.append(Lats[k])
            RLons.append(Lons[k])

    if len(Lats)==1:
        return RLons[0],RLats[0]
    else:
        return RLons, RLats