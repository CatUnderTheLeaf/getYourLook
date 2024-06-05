import os, sys
import requests

def download_images(folder):
    print('download images for ' + folder)
    # read image urls
    fName = os.path.join('urls',folder + '.txt')
    with open(fName, 'r') as f:
        lines = [line.rstrip('\n') for line in f]
    
    # get set of all downloaded images
    contents = set(os.listdir(os.path.join('dataset',folder)))

    # create dir if not exist
    os.makedirs(os.path.join('dataset', folder), exist_ok=True)

    # download only new images
    for url in lines:
        f_name = url.split('/')[-1]
        if not f_name in contents:
            print('new image')
            r = requests.get(url)
            full_image_name = os.path.join('dataset', folder, f_name)
            with open(full_image_name,'wb') as f: 
                f.write(r.content)


if __name__ == '__main__':
    args = sys.argv[1:]
    download_images(args[0])