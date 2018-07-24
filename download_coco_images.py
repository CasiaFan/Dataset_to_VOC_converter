import os, time, sys, argparse, shutil, logging
from pathos.multiprocessing import Pool
import multiprocessing.pool
from functools import partial
import requests
import pandas as pd
import numpy as np


parser = argparse.ArgumentParser(description="Downloading ImageNet images from urls")
parser.add_argument("--input_dir", default=None, help="Directory containing image urls list file, like n03996004.txt")
parser.add_argument("--input_file", default=None, help="Image url list file, like n03996004.txt")
parser.add_argument("--input_file_list", default=None, help="file with each line containing synset file paths, like ./n03996004.txt")
parser.add_argument("--output_dir", default=".", help="Directory for storing output images")
parser.add_argument("--n_workers", default=1, type=int, help="The number of threads to download images")
args = parser.parse_args()
assert args.input_dir or args.input_file or args.input_file_list, "--input_dir or --input_file or --input_file_list should be given!"


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s [Line: %(lineno)s] %(levelname)s %(message)s',
                    filename="download_coco_url.log",
                    datefmt="%a, %b %d %Y %H:%M:%S")
logger = logging.getLogger()


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


class myPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def download(name_url, tarDir):
    '''
    :param tarDir (str): image saving directory name
           imgIds (tuple list): images to be downloaded (image_synset_name, url)
    :return:
    '''
    if not os.path.exists(tarDir):
        os.makedirs(tarDir)
    tic = time.time()
    fname = os.path.join(tarDir, name_url[0]+".jpg")
    if not os.path.exists(fname):
        try:
            request = requests.get(name_url[1], timeout=20, stream=True)
        except:
            logger.error("Fail to open page %s" %name_url[1])
            return
        if request.status_code == 200:
            with open(fname, "wb") as f:
                shutil.copyfileobj(request.raw, f)
            f.close()
        del request
        logger.info('downloaded {} from {} (t={:0.1f}s) @ {}'.format(name_url[0], name_url[1], time.time()-tic, time.strftime("%Y%m%d-%H%M%S")))
        time.sleep(3)


def main(n_workers=args.n_workers):
    if args.input_dir:
        files = [os.path.join(args.input_dir, fi) for fi in os.listdir(args.input_dir)]
    elif args.input_file_list:
        files = np.squeeze(np.asarray(pd.read_csv(args.input_file_list, header=None, sep="\n"))).tolist()
    elif args.input_file:
        files = [args.input_file]
    else:
        pass
    for file in files:
        save_dir = os.path.join(args.output_dir, os.path.splitext(os.path.basename(file))[0])
        df = pd.read_csv(file, header=None, sep=" ")
        name_url_zip = zip(np.squeeze(np.asarray(df.ix[:, 0])).tolist(), np.squeeze(np.asarray(df.ix[:, 1])).tolist())
        dn = partial(download, tarDir=save_dir)
        pool = Pool(processes=n_workers)
        try:
            pool.map(dn, name_url_zip)
        except Exception as e:
            logger.error("Error occurred during multiprocessing thread: %s @ time %s" %(str(e), time.strftime("%Y%m%d-%H%M%S")))


if __name__ == '__main__':
    main()