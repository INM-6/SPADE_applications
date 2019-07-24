import os


# Function to create new folders
def mkdirp(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)


# Function to split path to single folders
def split_path(path):
    folders = []
    while 1:
        path, folder = os.path.split(path)
        if folder != "":
            folders.append(folder)
        else:
            if path != "":
                folders.append(path)
            break
    folders.reverse()
    return folders
