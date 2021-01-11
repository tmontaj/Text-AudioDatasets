import wget
import pandas as pd 
import numpy as np
from pathlib import Path
import hashlib
from functools import partial
import os

def load_md5sum_file(src):
  """
  load md5sum file from src file format is:

  md5sum split_name
  
  Arguments:
  src -- path to data directory

  Returns:
  ds -- dataframe of shape [splits,2] and col=["md5sum", "split"] contain
        md5sum of all splits
  """
  ds = pd.read_csv(src+"/md5sum.txt",names=['md5sum', 'split'], sep=" ")
  ds["md5sum"] = ds.index
  ds.set_index(pd.RangeIndex(start=0, stop=ds.shape[0]))

  return ds

def md5sum(file, bufsize=1<<15):
  """
  calculate single file md5sum by loading portion (bufsize) of the file into memory 
  then calculate its respective md5 and update the resulte.

  This should work well with big files that doesn't fit in memory 
  
  Arguments:
  file -- path to file
  Returns:
  d -- string of hex code of the file md5sum
  """
  d = hashlib.md5()
  for buf in iter(partial(file.read, bufsize), b''):
      d.update(buf)
  return d.hexdigest()

def calculate_md5sum(main, pattern = '*.tar.gz'):
  """
  calculate single file md5sum by loading portion (bufsize) of the file into memory 
  then calculate its respective md5 and update the resulte.

  This should work well with big files that doesn't fit in memory 
  
  Arguments:
  main -- path to directory of files (splits) .tar.gz
  pattern -- pattern in the file name to be selected (Defult = '*.tar.gz')
  Returns:
  df -- dataframe contain md5sum of downloaded splits. 
       df is of shape of shape [splits,2] and col=["md5sum", "split"] 
  """
  names = []
  md    = []
  for src in Path(main).rglob(pattern):
    f = open(src, "rb")
    name =  str(src).split("/")[-1]
    names.append(name)
    md.append(str(md5sum(f)))

  df = {
      "split" :names,
      "md5sum":md
  }
  return pd.DataFrame(df)


def check_librispeech_md5sum(src):
  """
  Check good downloaded splits, corrupted, and missing. 
  In order to achieve this this what happens under the hood:
    1- remove md5sum.txt if available 
    2- download md5sum
    3- load the md5sum file load_md5sum_file()
    4- calculate md5sum of all predownloaded splits (.tar.gz files) calculate_md5sum()
    5- calculate good downloaded splits, corrupted, and missing
  
  Arguments:
  src -- path to data directory
  Returns:
  match     -- python list of good splits names
  not_match -- python list of corrupted splits names
  missing   -- python list of missing splits names
  """
  try : os.remove(src+"/librispeech"+"/md5sum.txt")
  except : pass 
  wget.download("https://www.openslr.org/resources/12/md5sum.txt", src+"/librispeech"+"/md5sum.txt")
  
  real_sum         = load_md5sum_file(src+"/librispeech")
  calculated_sum   = calculate_md5sum(main=src+"/librispeech"+"/out", pattern = '*.tar.gz') #update this line 
  sum_intersection = real_sum.merge(calculated_sum, on=["split"], how="inner")


  match     = sum_intersection[sum_intersection.md5sum_x ==
                               sum_intersection.md5sum_y].split.tolist()

  not_match = sum_intersection[sum_intersection.md5sum_x !=
                               sum_intersection.md5sum_y].split.tolist()

  missing   = list(set(real_sum.split.tolist()).difference(set(sum_intersection.split.tolist())))

  return match, not_match, missing

def clean_split_name(splits):
  """
  format splits name from split_name.tar.gz to split_name
  Arguments:
  splits -- python list like splits names
  Returns:
  list -- python list of cleaned splits names
  """
  return [x.split(".")[0] for x in splits]

def what_to_download(src, splits):
  """
  main function of this module. Desides which splits needs to be downloaded
  or missing 

  Arguments:
  src -- path to data directory
  splits -- python list like of splits names
  Returns:
  download -- python list of splits names needs to get (re)download
  """
  match, not_match, missing = check_librispeech_md5sum(src)

  match     = clean_split_name(match)
  not_match = clean_split_name(not_match)
  missing   = clean_split_name(missing)

  for i in not_match:
    os.remove(src+"/d/"+i+".tar.gz")
    
  missing.extend(not_match)
  required = missing
  download = list(set(splits).intersection(set(required)))
  
  download_str = " ".join(download)
  if download_str == "":
    download_str = "nothing"
  print("\nShould download %s" %download_str)

  return download

 