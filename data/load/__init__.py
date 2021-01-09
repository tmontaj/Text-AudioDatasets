def safe_load(loader, what_to_load, src, splits):
    """
    download splits and make sure tht aplits md5sum is correct,
     if not redownload them 

    Arguments:
    loader -- python function to load the dataset
    what_to_load -- function that determine what to load 
    src    -- path to data directory
    splits -- python list like of splits names
    Returns:
    download -- python list of splits names needs to get (re)download
    """
    wtl = what_to_load(src, splits)
    while len(wtl):
        loader(src, wtl, remove_organized_path=False, download=True)
        wtl = what_to_load(src, splits)