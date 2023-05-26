# get a list of all sax files for this patient
import glob
import logging
import os
from collections import Counter

import SimpleITK as sitk
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from src.utils.Utils_io import ensure_dir


class CMRPatient(ABC):

    def sort_dicoms(self, dicom_images, sort_for_time=True):
        """
        Sort all slices by trigger-time and origin
        if sort_for_time == False sort only by origin

        :param dicom_images:
        :param sort_for_time: Bool, if True: return (triggertime,*origin), else return origin
        :return:
        """
        trigger_time_tag = '0018|1060'

        reverse_pos = False
        if not sort_for_time: reverse_pos = True
        logging.debug('images to sort: {} , sortbytime: {}'.format(len(dicom_images), sort_for_time))

        # check for patient position, not necessary, as SITK loads them correctly if we keep that tag
        def origin_sort(dicom_image):
            """
            Helper, which returns a tuple as argument for sorted()
            if sort_for_time: return (triggertime,*origin), else return origin
            updated on march 2022: for the first sort we just need the spatial origin
            For the second sort just the trigger time
            :param dicom_image:
            :return:
            """

            if sort_for_time:
                # trigger time
                return float(dicom_image.GetMetaData(trigger_time_tag))
            else:
                # origin in world coordinate axis: x,y,z (x goes from patient perspective right to patient left)
                return (dicom_image.GetOrigin()[0])

        # we sort the trigger time from small too big
        # We sort the origin from big to small (apex to basal slices)
        sorted_dicom_images = sorted(dicom_images, key=origin_sort, reverse=reverse_pos)
        return sorted_dicom_images

    def get_timesteps(self, dicom_images):
        """
        calculate the time steps of one volume by summing up the origins
        all slices with same origin represents the time steps of one slice
        """

        origins = [img.GetOrigin() for idx, img in enumerate(dicom_images)]
        counter = Counter(origins)
        # make sure to get another time step size if this one is 1
        iter_ = iter(counter)
        steps = counter[next(iter_)]

        # in some cases we have a volume where each origin occurs n times
        # but one slice/origin is misplaced and occurs only once
        # to avoid volumes with 1 time step (except of LA volumes)
        # try to get a greater number of time steps, this is a hack
        if steps == 1:
            steps = counter[next(iter_)]
        return steps

    def describe_sitk(self, sitk_img):
        """
        log some basic informations for a sitk image
        :param sitk_img:
        :return:
        """
        if isinstance(sitk_img, np.ndarray):
            sitk_img = sitk.GetImageFromArray(sitk_img.astype(np.float32))

        logging.debug('size: {}'.format(sitk_img.GetSize()))
        logging.debug('spacing: {}'.format(sitk_img.GetSpacing()))
        logging.debug('origin: {}'.format(sitk_img.GetOrigin()))
        logging.debug('direction: {}'.format(sitk_img.GetDirection()))
        logging.debug('pixel type: {}'.format(sitk_img.GetPixelIDTypeAsString()))
        logging.debug('number of pixel components: {}'.format(sitk_img.GetNumberOfComponentsPerPixel()))

    def get_metadata_maybe(self, sitk_img, key, default='not_found'):
        # helper for unicode decode errors
        try:
            value = sitk_img.GetMetaData(key)
        except Exception as e:
            logging.debug('key not found: {}, {}'.format(key, e))
            value = default
        # need to encode/decode all values because of unicode errors in the dataset
        if not isinstance(value, int):
            value = value.encode('utf8', 'backslashreplace').decode('utf-8').replace('\\udcfc', 'ue')
        return value

    def get_sequence_length(self, dicom_files):
        """
        returns the biggest trigger time from a list of dicom images
        """
        # trigger time dicom tag
        tt_tag = '0018|1060'

        trigger_times = [float(self.get_metadata_maybe(sitk_img=img, key=tt_tag, default=0)) for img in dicom_files]
        return max(trigger_times)

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass
import re
def repl(m):
    """ Function for reformatting the date """
    return '{}{}-{}-20{}'.format(m.group(1), m.group(2), m.group(3), m.group(4))

def process_manifest(name, name2):
    """
        Read the lines in the manifest.csv file and check whether the date format contains
        a comma, which needs to be removed since it causes problems in parsing the file.
        """
    with open(name2, 'w') as f2:
        with open(name, 'r') as f:
            for line in f:
                line2 = re.sub('([A-Z])(\w{2}) (\d{1,2}), 20(\d{2})', repl, line)
                f2.write(line2)

class UKBioPatient(CMRPatient):

    def __init__(self, path_to_patient, export_root='/mnt/ssd/data/ukbio/import/'):

        assert os.path.exists(path_to_patient), 'patient path {} does not exist'.format(path_to_patient)

        self.path_to_patient = path_to_patient
        self.export_root = export_root
        self.all_files_ = sorted(glob.glob(os.path.join(self.path_to_patient, '/*.dcm')))
        self.manifest_f = glob.glob(os.path.join(self.path_to_patient, 'manifest.*'))[0]
        self.proc_manifest_f = self.manifest_f.replace('manifest', 'manifest2')
        ensure_dir(self.export_root)


        # collect relative sax file names
        process_manifest(self.manifest_f,self.proc_manifest_f)
        self.df = pd.read_csv(self.proc_manifest_f)
        self.sax_files_short = self.df[self.df['series discription'].str.upper().str.contains('SAX_B')]['filename']
        self.patient_id = self.df['patientid'].values[0].replace(' ', '') # remove whitespace
        logging.info('processing patient: {}'.format(self.patient_id))

        # create absolute filenames
        self.sax_files = [os.path.join(path_to_patient, f) for f in self.sax_files_short]

    def __call__(self):

        image_nda = []
        # load files
        dicom_images = [sitk.ReadImage(f) for f in self.sax_files]

        # make sure all CMR have the same shape
        img_shape = dicom_images[0].GetSize()
        before_filter_len = len(dicom_images)
        dicom_images = [img for img in dicom_images if img.GetSize() == img_shape]
        if len(dicom_images) != before_filter_len:
            print('before filter: {} after filter: {}'.format(before_filter_len, len(dicom_images)))

        # sort files by origin, now we have [slice_z0_t0, ...slice_z0_t, slice_z1_t0, ...slice_z1_t...]
        dicoms_sorted = self.sort_dicoms(dicom_images, sort_for_time=False)

        timesteps = int(self.get_timesteps(dicoms_sorted))
        # check if we might miss some dicom images
        if not (len(dicoms_sorted) / timesteps).is_integer():
            logging.error(
                'number of dicom slices {}/ timesteps {} is not an integer, trying to round z but maybe there are wrong dicom files, check: \n'.format
                (len(dicoms_sorted), timesteps))

        slices = int(len(dicoms_sorted) / timesteps)
        self.describe_sitk(dicoms_sorted[0])
        lower_boundary = 0
        upper_boundary = timesteps

        # build new spacing with z = slice thickness and t = 1
        spacing_3d = dicoms_sorted[0].GetSpacing()
        origin_3d = dicoms_sorted[0].GetOrigin()
        direction = dicoms_sorted[0].GetDirection() #copy direction is here a bad idea, because of RAS --> LPS orientation
        # dicoms_sorted[0] = lowest slice of first volume, use it as origin
        origin = (origin_3d[0], origin_3d[1], origin_3d[2], 0)

        z_spacing = spacing_3d[2]
        if z_spacing == 1:  # some volumes dont have z spacing, they have one series per slice, use the slice thickness (0018,0050)
            z_spacing = int(self.get_metadata_maybe(dicoms_sorted[0], '0018|0050', 6))  # default is 6
            logging.debug('no spacing given, use slice thickness: {} as z-spacing '.format(z_spacing))

        # get the temporal resolution of this volume
        sequence_length = self.get_sequence_length(dicoms_sorted)
        # divide the sequence length in ms by the number of images
        temp_spacing = sequence_length // timesteps
        spacing = (spacing_3d[0], spacing_3d[1], z_spacing, temp_spacing)
        # logging.debug('Building volumes for patient: {}'.format(args['patient']))
        logging.debug('images: {}'.format(len(dicoms_sorted)))
        logging.debug('timesteps: {}'.format(timesteps))
        logging.debug('slices: {}'.format(slices))
        logging.debug('Temporal resolution: {}'.format(temp_spacing))

        diffs = []
        # The dicoms are sorted by origin
        # By this we can slice one slice + t
        for z in range(slices):
            # take all timesteps of one slice,
            # sort timesteps of this slice by Triggertime
            image_volume_aslist = dicoms_sorted[lower_boundary:upper_boundary].copy()
            image_volume_aslist = self.sort_dicoms(image_volume_aslist, sort_for_time=True)

            img_t = []
            # for each timesteps of this slice we check if we find a contour which is mapped by the image uid
            for sitk_img in image_volume_aslist:
                img_t.append(np.squeeze(sitk.GetArrayFromImage(sitk_img), axis=0))

            image_nda.append(np.stack(img_t, axis=0))

            lower_boundary = lower_boundary + timesteps
            upper_boundary = upper_boundary + timesteps

        # currently its a list of 2D slices + t --> stack along the z axis (t,z,x,y)
        # align with the image orientation of the ACDC dataset (basal to apical)
        new_img_clean = np.stack(image_nda, axis=1).astype(np.float32)
        new_img_clean = np.flip(new_img_clean, axis=(1))

        # sitk.GetImageFromArray cant handle 4d images, join series will help
        sitk_images = [sitk.GetImageFromArray(vol) for vol in new_img_clean]
        # copy rotation/direction, we nee dto check for proper orientation
        # DICOM coordinate (LPS)
        #  x: left
        #  y: posterior
        #  z: superior
        # Nifti coordinate (RAS)
        #  x: right
        #  y: anterior
        #  z: superior
        #_ = [img.SetDirection(direction) for img in sitk_images]
        new_img_clean = sitk.JoinSeries(sitk_images)

        size = new_img_clean.GetSize()
        dimension = new_img_clean.GetDimension()
        logging.info("Image size: {}".format(size))
        logging.debug("Image dimension: {}".format(dimension))
        logging.info("Image Spacing: {}".format(spacing))
        logging.debug('Writing images ...')

        # Copy image tags to new volume
        sitk_img = dicoms_sorted[0]
        for tag in sitk_img.GetMetaDataKeys():
            value = self.get_metadata_maybe(sitk_img, tag)
            new_img_clean.SetMetaData(tag, value)

        new_img_clean.SetSpacing(spacing)
        new_img_clean.SetOrigin(origin)
        # set the temporal resolution
        new_img_clean.SetMetaData('0020|0110', str(temp_spacing))
        sitk.WriteImage(new_img_clean, os.path.join(self.export_root, '{}__4d.nrrd'.format(self.patient_id)))
        logging.info('patient {} done.'.format(self.patient_id))
