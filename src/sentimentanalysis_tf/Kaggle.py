import os
from kaggle import api
from opendatasets.utils.kaggle_direct import get_kaggle_dataset_id, is_kaggle_url
from opendatasets.utils.archive import extract_archive
import click
import json


class Kaggle:
    '''
    Set the below environment variables
    -   os.environ['KAGGLE_USERNAME']
    -   os.environ['KAGGLE_KEY']
    '''
    def download_kaggle_dataset(self,dataset_url, data_dir, force=False, dry_run=False):
        dataset_id = get_kaggle_dataset_id(dataset_url)
        id = dataset_id.split('/')[1]
        target_dir = os.path.join(data_dir, id)

        if not force and os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
            print('Skipping, found downloaded files in "{}" (use force=True to force download)'.format(
                target_dir))
            return


        if not dry_run:
            
            api.authenticate()
            if dataset_id.split('/')[0] == 'competitions' or dataset_id.split('/')[0] == 'c':
                api.competition_download_files(
                    id,
                    target_dir,
                    force=force,
                    quiet=False)
                zip_fname = target_dir + '/' + id + '.zip'
                extract_archive(zip_fname, target_dir)
                try:
                    os.remove(zip_fname)
                except OSError as e:
                    print('Could not delete zip file, got' + str(e))
            else:
                api.dataset_download_files(
                    dataset_id,
                    target_dir,
                    force=force,
                    quiet=False,
                    unzip=True)

        else:
            print("This is a dry run, skipping..")

if __name__=="__main__":
    __all__=["Kaggle"]