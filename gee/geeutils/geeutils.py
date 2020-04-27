import os
import numpy as np
import tqdm
import random
import ee

class utils(object):
    """
    utilities to handle Google Earth Engine.
    """
    
    def start_session(self):
        """
        start new gee Python API session.
        call it at the very first of your run.
        """
        ee.Initialize()
    
    def define_tasks(self, imgcollections,
                     description, dimensions, folder):
        """
        generate downloading tasks. GEE does not support
        multi-image batch, thus we need to iterate
        in serial run.
        GEE adopts a "define-and-run" programming scheme,
        so these tasks will be lazyly evaluated when starting
        the task. 

        Args:
            imgcollections (ee.ImageCollections): image collections to download.
                                                  should be clipped before.
            description (str): description of the dataset
            dimensions (str): "WIDTHxHEIGHT" of output image
        Notes:
            output format is 0000x.tif 
        """
        n = imgcollections.size().getInfo()
        collections = imgcollections.toList(n)  # this is server-object; not iterable
        tasks = []
        itr = np.arange(n).tolist()
        random.shuffle(itr)
        pbar = tqdm.tqdm(itr)
        for i in pbar:
            image = collections.get(i)
            task = self.define_task(ee.Image(image).float(),
                                   "{0:05d}".format(i),
                                   description,
                                   dimensions,
                                   folder)
            tasks.append(task)
            pbar.set_description("defining tasks {0:05d}/{1:05d}".format(i, n))      
        return tasks    
    
    def define_task(self, image, prefix,
                    description, dimensions, folder):
        task = ee.batch.Export.image.toDrive(
            image=image,
            region=image.geometry().getInfo()["coordinates"],
            description=description,
            folder=folder,
            fileNamePrefix=prefix,
            dimensions=dimensions
        )
        return task
