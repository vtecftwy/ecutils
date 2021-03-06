"""
Utility Functions that can be used in any context

Includes all stable utility functions that do not fit in other context.
"""

import datetime as dt
import os
import piexif
import re
import shutil

from pathlib import Path
from IPython.display import Image, display
from pprint import pprint

__all__ = ['get_date_from_file_name', 'date_is_within_year', 'exif2dt', 'add_missing_dates_to_exif']

def get_date_from_file_name(path2file, date_pattern=None):
    """Retrieve the date from the file name.

    Returns the date encoded in the file name (as per date_pattern) as a datetime.
    Default date pattern is YY-MM-DD, i.g. regex ^(\d\d)-(\d\d)-(\d\d)*

    path2file:      pathlib.Path object pointing to the file
    date_pattern:   the regex pattern for the date, if different from the default one
    returns:        date in datetime format, if found. False if not found or if file does not exist
    """
    if date_pattern is None:
        date_pattern = r"^(\d\d)-(\d\d)-(\d\d)*"

    if not path2file.is_file() or path2file is None:
        return False
    p = re.compile(date_pattern)
    results = p.match(path2file.name)

    if results is not None:
        y, m, d = results.groups()
        if y is None or m is None or d is None:
            return False
        else:
            y = int(y)
            offset = (1900 if 50 < y <= 99 else 2000)
            y = y + offset
            date = dt.datetime(year=y, month=int(m), day=int(d))
            return date
    else:
        return False


def date_is_within_year(date, year):
    """True if the passed date (datetime) is within year, False otherwise"""
    return date.year == year


def exif2dt(exif_d):
    """Transform a date in bytes format from EXIF into datetime format

    exif_d:     date in exif format, i.e. bytes. E.g. b"2018:11:21"
    returns:    the date in datetime or False if no date is detected or not well formatted
    """
    results = re.match("^(\d\d\d\d):(\d\d):(\d\d)*", exif_d.decode('utf-8'))
    if results is None:
        return False
    else:
        y, m, d = results.groups()
        if y is None or m is None or d is None:
            return False
        else:
            return dt.datetime(year=int(y), month=int(m), day=int(d))


def add_missing_dates_to_exif(path2folder, year=None, maxhours=24, do_not_update_exif=False, verbose=False):
    """Add missing EXIF original and digitized dates, based on file creation or date in file name.
    In order to better control the data changes and avoid mistaken exif updates, the process is done on
    a year by year basis, i.e. a specific year needs to be passed to the function and only dates within the passed
    year will be updated. All other dates will be disregarded.

    path2folder: ............. pathlib.Path: object pointing to the folder holding all jpg photos to handle
    year: .................... int: year used to filter all dates
    maxhours: ................ int: maximum acceptable difference (in hours) between exif dates and file dates
    do_not_update_exif: ...... bool: flag to prevent updating the exif file, used in debugging or testing
    verbose: ................. bool: flag to print original, updated and retrieved updated EXIF info

    Logic of the function:
    1. Retrieve EXIF info from image
    2. When there is no EXIF.DatetimeOriginal in the image EXIF:
        - use date from file name if exists, else
        - use date from creation or modification, whichever is earlier
    3. When there is an EXIF.DatetimeOriginal, compare with date from file, if any:
        - if difference < maxhours, do nothing
        - if difference >= maxhours, use date from filename
    4. If the date extracted from file name or file creation/modification date is not in passed year, skipped any change
    """

    if year is None:
        year = dt.datetime.now().year

    nbr_images = len([f for f in path2folder.glob('*.jpg')])
    print(f"Handling {nbr_images} images in {path2folder.name}")

    for i, jpg in enumerate(path2folder.glob('*.jpg')):
        try:
            exif_dict = piexif.load(str(jpg.absolute()))
        except piexif.InvalidImageDataError as err:
            print(f"  Cannot load image <{jpg.name}>. {err}")
            continue

        date2use = None
        date_from_file_name = get_date_from_file_name(jpg)

        if exif_dict['Exif'].get(piexif.ExifIFD.DateTimeOriginal, None) is None:
            # Image does not have an EXIF Original Date
            print(f"{i} of {nbr_images}.{'-' * 75}")
            print(f"  {jpg} has no Exif date")

            if date_from_file_name:
                date2use = date_from_file_name
                date_source = 'File Name'
            else:
                file_creation_date = dt.datetime.fromtimestamp(os.path.getctime(jpg))
                file_modified_date = dt.datetime.fromtimestamp(os.path.getmtime(jpg))
                date2use = min(file_creation_date, file_modified_date)
                date_source = 'File Creation/Modification Date'

            if not date_is_within_year(date2use, year):
                print(f"Implied date {date2use} not within {year} for {jpg.name}")
                continue

        else:
            # Image does have an EXIF Original Date
            if date_from_file_name:
                exif_date = exif2dt(exif_dict['Exif'][piexif.ExifIFD.DateTimeOriginal])
                if exif_date:
                    diff = max(exif_date, date_from_file_name) - min(exif_date, date_from_file_name)
                    if diff > dt.timedelta(hours=maxhours):
                        print(f"{i} of {nbr_images}.{'-' * 75}")
                        print(f"  {jpg.name}")
                        print(f"  Date from file name: {date_from_file_name}")
                        print(f"  Date from EXIF: .... {exif_date}")
                        print(f"  Diff: .............. {diff}")
                        if date_is_within_year(date_from_file_name, year):
                            date2use = date_from_file_name
                            date_source = 'File Name'
                        else:
                            print(f"Date from file name {date_from_file_name} not within {year} for {jpg.name}")

        if date2use is not None:
            thumbnail = exif_dict["thumbnail"]
            if thumbnail is not None:
                display(Image(thumbnail))
            else:
                print(f"  No thumbnail")

            print(f'  No existing EXIF dates. Creating one based on {date_source}')
            print(f"  Date to use: {date2use}")

            if verbose:
                print(f"  Original EXIF  {'-' * 50}")
                pprint(exif_dict['Exif'])

            # Create datetime tags based on file creation date
            exif_dict['Exif'][piexif.ExifIFD.DateTimeOriginal] = f"{date2use:%Y:%m:%d %H:%M:%S}"
            exif_dict['Exif'][piexif.ExifIFD.DateTimeDigitized] = f"{date2use:%Y:%m:%d %H:%M:%S}"

            # Ensure Scene Type has a byte format in dict, and not a int (Undefined in EXIF, byte in piexif)
            exif_dict['Exif'][piexif.ExifIFD.SceneType] = bytes([1])
            # Technical note:
            # In EXIF, the tag with code `piexif.ExifIFD.SceneType`, i.e. `41729`, needs to be treated in a special way.
            # - Exif IFD SceneType has a EXIF Type `Undefined`, and not `ASCII` or `Rational` like most others.
            # - When piexif loads the EXIF file, the value of the tag is sometimes converted into `int` for some reason,
            #   but it should be bytes (see doc as piexif.readthedocs.io/en/latest/appendices.html
            # - When the exif-dict is converted back into EXIF format, using `piexif.dump()`, it generates an error:
            #       ValueError: "dump" got wrong type of exif value.
            #       41729 in Exif IFD. Got as <class 'int'>.```
            # - This problem is solved by forcing the `int` to be a `bytes` format. In the case of SceneType, it is
            #   even easier because the value is always 1 for photos taken by a camera.
            # - This does not seem to be a problem for other tags with undefined type.

            if verbose:
                print(f"  Updated EXIF  {'-' * 50}")
                pprint(exif_dict['Exif'])

            if do_not_update_exif is False:
                exif_bytes = piexif.dump(exif_dict)
                piexif.insert(exif_bytes, str(jpg.absolute()))
                if verbose:
                    print(f"  New EXIF  {'-' * 50}")
                    new_exif_dict = piexif.load(str(jpg.absolute()))
                    pprint(new_exif_dict['Exif'])


if __name__ == '__main__':
    pass
