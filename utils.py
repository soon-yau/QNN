import os
import tarfile
import requests

from imagenet_classes import imagenet_class

def print_file(dir_name, fname):
    fpath = os.path.join(dir_name, fname)
    with open(fpath, "r") as f:
        for line in f:
            print(line)

def maybe_download(dir_name, url):
    ''' create directory '''
    filename = url.split('/')[-1]

    # write to local disk 
    fpath = os.path.join(dir_name, filename)

    if os.path.isfile(fpath):
        print("Files already exists")
        return 
        
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)



    # download 
    print("Downloading %s ..."%(filename))
    req = requests.get(url)

    with open(fpath, "wb") as f:
        f.write(req.content)

    # unzip 
    print(fpath)
    tar = tarfile.open(fpath, "r:gz")
    tar.extractall(dir_name)
    tar.close()
    print("Download complete")

def imagenet_id_to_class(id):
    return imagenet_class[id-1]

if __name__ == "__main__":
    url = "http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz"
    maybe_download("models/mobilenet", url)



