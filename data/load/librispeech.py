import pandas as pd
import numpy as np
from pathlib import Path
import tarfile
import os, sys
import shutil
import wget
import soundfile as sf
import tensorflow_io as tfio
import tensorflow as tf

#create this bar_progress method which is invoked automatically from wget and used in deffrent code

def _bar_progress(current, total, width=80):
  progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
  # Don't use print() as it will print in new line every time.
  sys.stdout.write("\r" + progress_message)
  sys.stdout.flush()

"""##### Downloading and extracting Librispeech """

def download_librispeech(out, splits):
  """
    Downloading librispeech dataset splits

    Arguments:
    out -- path to save the dataset on
    splits -- list of splits needed to be downloaded. splits are:
                    [dev-clean
                    dev-other,
                    test-clean, 
                    test-other,
                    train-clean-100,
                    train-clean-360,
                    train-other-500]


  """
  def _splits_url(split_name):
    return "https://www.openslr.org/resources/12/"+split_name+".tar.gz"
  
  def _splits_progress(split_name, split_number, splits_count):
    progress_message = "Split: %s [%d / %d]" % (split_name, split_number, splits_count)
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message+"\n")
    sys.stdout.flush()

  print("Start downloading librispeech ...")
  split_number = 1
  splits_count = len(splits)

  for split_name in splits:
    _splits_progress(split_name, split_number, splits_count)
    wget.download(_splits_url(split_name), out=out, bar=_bar_progress)
    split_number+=1

  print("... Finish downloading librispeech")



def unzip_librispeech(out, extract_path):
  """
  extracting librispeech data

  Arguments:
  out -- path of the downloaded tar files 
  extract_path -- path to extract the files on  
  """
  dirs = os.listdir(out)

  print("Start extracting ...")

  for i in dirs:
    target_name = i.split('.')
    name = out +'/'+i
    if name.endswith('.tar.gz'):
      tar = tarfile.open(name, "r:gz")
      tar.extractall(extract_path +'/' + target_name[0])
      tar.close()

  print("... Finished extracting")



"""##### Organize directories """

def organize_dirs (extract_path, organized_path):
  """
  extracting librispeech data

  Arguments:
  extract_path -- path to extract the files on  
  organized_path -- path to organize the files in  
  """
  print("Start organize_dirs ...")

  dirs = os.listdir(extract_path)
  for dir in dirs:
    shutil.move(extract_path+ '/'+ dir+ '/' + 'LibriSpeech/'+ dir , organized_path)
  
  common_files_path = extract_path + '/' + dirs[0]+'/' + "LibriSpeech"
  dirs = os.listdir( common_files_path )

  for f in dirs:
    shutil.move(common_files_path+'/'+ f , organized_path)
  
  print("... Finished organize_dirs")



def _remove(dir_path):
  """
  thin wrapper over os.system to remove directory or file 

  Arguments:
  dir_path -- path to dirctory or file to remove  
  """
  os.system('rm -R %s' %dir_path)

def _rename(dir_path, old_name, new_name):
  """
  thin wrapper over os.system to rename directory or file 

  Arguments:
  dir_path -- path to dirctory or file to rename  
  old_name -- old name (original) for directory or file
  new_name -- new name for directory or file
  """
  os.system('mv %s %s' %(dir_path+"/"+old_name, dir_path+"/"+new_name))


def download_and_extract(out, splits, extract_path,
                         organized_path, remove_organized_path=False, download=True):
  """
  download and extract librispeech

  Arguments:
  out -- path of the downloaded tar files 
  extract_path -- path to extract the files on  
  organized_path -- path to organize the files in  
  remove_organized_path -- flag to remove organized_path (uses -R to remove all files)
  download -- flag to optionaly skip download the dataset
  splits -- list of splits needed to be downloaded. splits are:
                    [dev-clean
                    dev-other,
                    test-clean, 
                    test-other,
                    train-clean-100,
                    train-clean-360,
                    train-other-500]
  """
  if download:
    download_librispeech(out, splits)
  print("----------------------------")
  unzip_librispeech(out, extract_path)
  print("----------------------------")
  if remove_organized_path:
    _remove(organized_path)
  organize_dirs (extract_path, organized_path)
  print("----------------------------")



def load(src, splits, remove_organized_path=False, download=True):
  """
  simple download and extract librispeech

  Arguments:
  src -- path to dataset directory 
  splits -- list of splits needed to be downloaded. splits are:
                    [dev-clean
                    dev-other,
                    test-clean, 
                    test-other,
                    train-clean-100,
                    train-clean-360,
                    train-other-500]
  """
  src = src+"/librispeech"
  out = src+"/out"
  extract_path = src+"/tmp"
  organized_path = src+"/data"

  os.system("mkdir -p %s" %(src))
  if download:
    os.system("mkdir -p %s" %(out))
    _remove(out+"/*")

  os.system("mkdir -p %s" %(extract_path))
  _remove(extract_path+"/*")


  os.system("mkdir -p %s" %(organized_path))
  _remove(organized_path+"/*")

  print(out)
  download_and_extract(out=out,
                     splits=splits,
                     extract_path = extract_path, 
                     organized_path = organized_path,
                     remove_organized_path = remove_organized_path,
                     download = download
                     )
  print("CONGRATS Librispeach is ready to be used at %s" %(organized_path))

# load(src="dataset",
#      splits=["dev-clean", "dev-other"],
#      download=False)

def clean_speakers_file(src):
  """
  clean speakers file

  Arguments:
  src -- path to dataset
  """
  input=open(src+"/SPEAKERS.TXT", "r")
  dest=open(src+"/SPEAKERS_temp.TXT", "w")

  input_lines = input.readlines()

  line_num = 1
  for line in input_lines:
    if line_num == 45:
      line = line.split("|")
      line [-2] = line[-2]+" "+line[-1] 
      line.pop(-1)
      line.pop(-2)
      line = "|".join(line)

    if line_num == 12:
      line = line[1:].lower()
    
    dest.write(line)
    line_num+=1

  input.close()
  dest.close()

  _remove(src+"/SPEAKERS.TXT")
  _rename(src, "SPEAKERS_temp.TXT", "speakers.txt")


def load_metadata(data_path):
  """
  load metadata currently loads speakers.txt only 
  
  Arguments:
  data_path -- path to dataset
  """

  # use sep | and skip first 11 rows 
  speakers = pd.read_csv(data_path+"/"+'speakers.txt', sep="|", skiprows=11)
  speakers.columns = speakers.columns.map(lambda x: x.strip())
  speakers.set_index("id", inplace=True)
  return speakers



def load_trans(src, split_name):
  """
  load single file of transcription 
  
  Arguments:
  src -- path to the file
  Returns:
  df -- pandas dataframe of trans file
  """
  
  split = split_name.split("-")

  df = pd.read_csv(src,names=['data'])
  df[['id','text']] = df["data"].str.split(" ", 1, expand=True)
  df['text_len'] = df['text'].str.len().astype(np.str)
  df[['speaker', 'chapter', 'index']] = df["id"].str.split("-", expand=True)
  df[["split"]] = split[0]
  # df[["split"]] = split[0].split("/")[-1]
  df[["isClean"]] = True if split[1] == "clean" else False
  df["id"] = split_name+"/"+df["id"]
  df.pop("data")
  df["wav"] = ""
  
  return df


def load_all_trans(src):
  """
  load single file of transcription
  
  Arguments:
  src -- path to data directory
  Returns:
  all_trans -- pandas dataframe of all trans file
  """
  src = src+"/librispeech/data"
  splits = [x for x in Path(src).iterdir() if x.is_dir()]
  df = []

  for split in splits:
    split = str(split)
    for src in Path(split).rglob('*.trans.txt'):
      split = split.split("/")[-1]
      df.append(load_trans(src, split))


  return pd.concat(df)

# load_all_trans(src="dataset/librispeech/data")

def sf_load_wav(src, id):
  """
  load single wav 
  
  Arguments:
  src -- path to data directory
  id  -- id to load
  Returns:
  wav -- np array of mono sound file
  sample_rate -- sample rate for librispeech = 16000 
  """

  # split = split + ("-clean" if isClean else "-other")
  id = id.split("/")
  file_name = id[1]+".flac"
  id[1] = id[1].replace("-", "/")[:-4]
  path = os.path.join(src, id[0], id[1], file_name)
  wav, sample_rate = sf.read(path)      

  return wav, sample_rate


# def load_wav(src, id):
def load_wav(src, id):
  """
  load single wav 
  
  Arguments:
  src -- path to data directory
  id  -- id to load (i.e dev-clean/7850-111771-0000)
  Returns:
  wav -- np array of mono sound file
  sample_rate -- sample rate for librispeech = 16000 
  """
  id = tf.strings.split(id, sep="/")
  file_name = id[1] + ".flac"
  sub_folder = tf.strings.regex_replace(id[1], pattern="-", rewrite="/")
  sub_folder = tf.strings.regex_replace(sub_folder, pattern="(....)$", rewrite="") # = [:-4] remove last 4 char
  path  = src+"/librispeech/data/"+id[0]+"/"+sub_folder+"/"+file_name
  audio = tfio.audio.AudioIOTensor(path, dtype=tf.int32).to_tensor()
  audio = tf.cast(audio, dtype=tf.float32) # pylint: disable=[unexpected-keyword-arg, no-value-for-parameter]

  return audio


def get_audio_len(src, dataset):
  def load_wav_len(id):
    return load_wav(src, id).shape[0]

  id = dataset["id"]
  dataset["audio_len"] = id.apply(load_wav_len)
  return dataset

def load_split(src, split):
  """
  load single split of the dataset as pandas datafram 
  
  Arguments:
  src    -- path to data directory
  split  -- split name to be loaded (string) e.x. dev
  Returns:
  dataset -- pandas dataframe containg the dataset 
  """
  dataset = load_all_trans(src)
  # print(dataset["text_len"])
  # dataset = get_audio_len(src, dataset)

  split, clean = split.split("-")[0], split.split("-")[1] 
  clean        = True if clean == "clean" else False

  dataset = dataset[dataset.split == split]
  dataset = dataset[dataset.isClean == clean] 

  return dataset
