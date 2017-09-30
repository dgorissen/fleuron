from __future__ import print_function, division
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import matplotlib.cm as cm
import multiprocessing as mp
import argparse
from os import path
import logging
from utils import *
import _version
import json
import datetime


def write_result(file_path, contours, img):
    """
    Writes each found ornament into its own image file and metadata is saved
    in a json files.
    """
    log = get_logger()
    dir_name = path.dirname(file_path)
    file_name = path.basename(file_path)
    base_name, ext = path.splitext(file_name)

    log.debug("%s ornaments detected for %s, writing results" % (len(contours),
                                                                 file_path))

    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    metadata = {
        "fleuron_version": _version.__version__,
        "date": ts,
        "source_file": file_path,
        "num_ornaments": len(contours),
        "ornaments": []
    }

    for i, c in enumerate(contours):
        [x, y, w, h] = cv2.boundingRect(c)

        fn = path.join(dir_name, base_name + "_" + str(i) + ".png")
        ornament = img[y:y+h, x:x+w]

        # crop whitespace
        ornament = autocrop(ornament)

        cv2.imwrite(fn, ornament)

        ornament_dict = {
            "id": i,
            "file_name": fn,
            "x_bottom_left": x,
            "y_top_right": y,
            "width": w,
            "height": h
        }

        metadata["ornaments"].append(ornament_dict)

    fn = path.join(dir_name, base_name + ".json")
    with open(fn, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def remove_contours(img, cfilter, contours=None):
    """
    Remove all contours that satisfy the given filter.
    """

    if contours is None:
        contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)

    imgc = img.copy()

    for c in contours:
        if cfilter(c):
            # fill with black
            cv2.drawContours(imgc, [c], -1, (0, 0, 0), -1)

    return imgc


def prep_image(img):
    """
    Prepare the image for processing, do some small, safe cleanups.

    Note no de-skewing happens as most of the skews are non-linear and the
    added complexity was not deemed worth it.
    """
    height, width = img.shape

    # Threshold to black/white
    _, mask = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)

    # Rmove small speckles and close small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)

    min_area = 50
    for c in contours:
        area = cv2.contourArea(c)
        rect = cv2.boundingRect(c)
        [x, y, w, h] = rect

        # blank out contours on the left margin, remove border artefacts.
        # TODO: should really do this selectively but generally does not seem
        # to cause any problems.
        if (x+w) < 0.06*width:
            cv2.drawContours(mask, [c], -1, (0, 0, 0), -1)
            continue

        if area < min_area:
            n = nonzero_outside_margin(mask, rect, m=10)
            if n < 10:
                cv2.drawContours(mask, [c], -1, (0, 0, 0), -1)

    return mask


def merge_bboxes(contours, reference, original):
    """
    Merge overlapping contours and snap to edges.
    """
    mask = np.zeros(reference.shape, np.uint8)

    ref_bbox = get_content_bbox(original)
    rx, ry, text_width, text_height = ref_bbox

    my = 55
    mx = 85
    for c in contours:
        [x, y, w, h] = cv2.boundingRect(c)

        # snap to edges
        if y - ry < my:
            h = h + (y - ry)
            y = ry

        if (ry + text_height) - (y + h) < my:
            h = h + ((ry + text_height) - (y + h))

        if x - rx < mx:
            w = w + (x - rx)
            x = rx

        if (rx + text_width) - (x + w) < mx:
            w = w + ((rx + text_width) - (x + w))

        cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), 1)

    # Easiest way to merge is to draw rectangles and find contours
    merged_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)

    return merged_contours


def can_ignore(c, reference, ref_bbox):
    """
    Ignore contours that are very small, on the edges, etc.
    """

    height, width = reference.shape
    bbox = cv2.boundingRect(c)
    [x, y, w, h] = bbox
    w_center = x+w/2

    rx, ry, text_width, text_height = ref_bbox
    text_center = rx + text_width/2
    w_centered = abs(w_center - text_center) < 0.15*text_width

    # we really dont care below this
    if w*h < 100*100:
        return True

    # nor do we care about tall thin things
    if h/w > 10:
        return True

    # or larger, uncentered things
    if w*h < 170*170:
        if not w_centered:
            return True

        # small centered things should be isolated
        n = nonzero_outside_margin(reference, bbox, m=(50, 50, 200, 200))
        if n > 10:
            return True

    # ignore things right on edges
    # TODO manage margin size centrally, also, still needed if blanked?
    mw = 0.08*width
    mh = 0.04*height
    if (x + w) < mw or \
        x > (width - mw) or \
        (y+h) < mh or \
        y > (height - mh):
        return True

    return False


# TODO: remove one of 'reference' or 'original'
def filter_ornaments(contours, source, reference, original,
                     depth=0, debug=False):
    """
    Holds the core ornament detection logic. A series of morphologically based
    heuristics to guess if we are dealing with a printers ornament or not.
    """

    log = get_logger()

    def debug_log(s):
        if debug:
            # pid = mp.current_process().pid
            log.debug(s)

    # TODO: Turns out certain images are pathalogical for findcontours and
    # case a stack overflow. Hence we need to set a max recursion depth.
    max_depth = 3

    height, width = reference.shape

    ref_bbox = get_content_bbox(original)
    rx, ry, text_width, text_height = ref_bbox
    text_center = rx + text_width/2

    ornaments = []

    for i, c in enumerate(contours):
        # top left bounding rectangle
        bbox = cv2.boundingRect(c)
        [x, y, w, h] = bbox
        w_center = x+w/2
        # is the ornament roughly horizontally centered (wrt the text)
        w_centered = abs(w_center - text_center) < 0.15*text_width
        ref_patch = get_patch(bbox, reference)

        if can_ignore(c, reference, ref_bbox):
            # Too small to care about
            continue

        debug_log("(%s,%s) ornament %s, size: %d x %d = %s" % (x, y, i,
                                                               w, h, w*h))
        # an ornament does not start past half the page
        if x > text_center:
            debug_log("   fail start past half")
            continue

        # general centered ornament
        if w_centered and w < 0.7*text_width:

            # special case: check for thin, short, dividers

            # is it very isolated (i.e., surrounded by whitespace) ?
            n = nonzero_outside_margin(original, bbox, m=(50, 50, 300, 300))
            if n < 10:
                debug_log("   pass isolated general")
                ornaments.append(c)
                continue

            # small dividers may still fail the above so relax constraints
            n = nonzero_outside_margin(original, bbox, m=(50, 50, 100, 100))

            if n < 10 and h/w < 0.15:
                debug_log("   pass mini divider")
                ornaments.append(c)
                continue

            # general centered ornament

            # is it relatively isolated?
            n = nonzero_outside_margin(reference, bbox, m=(40, 40, 40, 40))

            # can we break it up horizontally or vertically?
            cr = re_blank(source, c, reference)

            if n < 10 and len(cr) < 2:
                debug_log("   pass general centered ornament")
                ornaments.append(c)
                continue
            else:
                # we can break it up but it still might be an ornament
                pass

        # wide ornament (title or divider)
        elif w >= 0.7*text_width:
            if 0.019*height < h <= 0.05*height:
                # a true divider will be surrounded by empty space
                n = nonzero_outside_margin(original, bbox, m=(35, 35, 0, 0))

                if n < 10:
                    # yep, its a divider
                    debug_log("   pass isolated divider")
                    ornaments.append(c)
                    continue
                else:
                    # Ok, its not quite isolated but it could still be an
                    # ornament. But lets first check for text.
                    # TODO: proper OCR would better but slower and also not
                    # foolproof.
                    if is_glued_text(ref_patch):
                        debug_log("   fail breakup divider")
                    else:
                        # a divider
                        debug_log("   pass non breakup divider")
                        ornaments.append(c)

                    continue

            # Some kind of title ornament but its not centered
            elif 0.05*height < h < 0.5*height:
                n = nonzero_outside_margin(reference, bbox, m=(40, 40, 0, 0))
                cr = re_blank(source, c, reference)

                if n < 10 and len(cr) < 2:
                    if is_lined(ref_patch) or is_glued_text(ref_patch):
                        debug_log("   fail isolated title is text")
                    else:
                        debug_log("   pass isolated title %s" % n)
                        ornaments.append(c)
                else:
                    # Ok we have something rather big, lets check its not just
                    # some large chunk of text glued together from the dilation.
                    if is_lined(ref_patch) or is_glued_text(ref_patch):
                        debug_log("  fail title is text")
                    else:
                        debug_log("  pass title")
                        ornaments.append(c)
                continue
            else:
                debug_log("  fail too thin or too high")
                continue
        else:
            pass

        # Perhaps its a capital letter
        # Starts close to the left edge, ends before half the page, is rougly
        # square and at least 170 high
        if (x < rx + text_width*0.2) \
           and (x+w) < 0.45*width \
           and (0.8 <= (w/h) <= 1.2) \
           and h > 170:
            # Could it be some glued text though?
            if is_glued_text(ref_patch):
                debug_log("   fail capital is text")
            else:
                debug_log("  pass captital")
                ornaments.append(c)

            continue

        if depth < max_depth:
            # not quite sure what we are left with, put back the blank lines
            # to see if that helps
            cr = re_blank(source, c, reference)
        else:
            debug_log("Max recurse depth exceeded")

        if depth < max_depth and len(cr) > 1:

            # blanking has revealed extra structure, recursively filter
            debug_log("  ornament %s blanked into %d new ones, recursing" %
                      (i, len(cr)))
            crf = filter_ornaments(cr, source, reference, original,
                                   depth=depth + 1, debug=debug)

            debug_log("  remaining after recurse: %d" % len(crf))
            ornaments.extend(crf)
        else:
            # at this point we dont really know what we are dealing with
            # try some basic checks

            # other than capital letters, ornaments should be horizontally
            # centered
            if not w_centered:
                debug_log("   fail not centered")
                continue

            # perhaps its still some glued text that made it through
            if is_glued_text(ref_patch):
                debug_log("   fail generic glued text %d" % len(cr))
                continue

            # if nothing else, be optimistic and consider it an ornament
            debug_log("   pass generic")
            ornaments.append(c)

    return ornaments


def process_image(file_path, debug=False):
    """
    The core worker function. Loads an image, cleans it up, filters out the
    ornaments, and saves the results.
    """
    log = get_logger()

    file_path = path.abspath(file_path)
    dir_name = path.dirname(file_path)
    file_name = path.basename(file_path)
    base_name, ext = path.splitext(file_name)

    img = cv2.imread(file_path, 0)
    height, width = img.shape

    # Prepare the image for processing by doing some very conservative
    # cleanups
    img_prep = prep_image(img)

    # Blanks out text lines so we have less work contouring
    # TODO: this is problematic for a number of reasons, should aim to remove.
    text_blanked = blank_text(img_prep.copy())

    # Dilate the elements in the image rather aggressively. We want to make
    # sure the constituent parts of any ornaments are glued together.
    # Note this means we are also gluing together text but those false
    # positives can be relatively easily filtered out.
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilated = cv2.dilate(text_blanked, kernel, iterations=6)
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # remove small contours
    # TODO: merge this into filter_ornaments?
    ref_bbox = get_content_bbox(img_prep)
    closing_nosmall = remove_contours(closing, lambda x: can_ignore(x,
                                                                    img_prep,
                                                                    ref_bbox))

    # Actual hard work happens here, filter the contours down to suspected
    # ornaments
    contours, hierarchy = cv2.findContours(closing_nosmall.copy(),
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)

    ornaments = filter_ornaments(contours, closing_nosmall,
                                 text_blanked, img_prep, debug=debug)

    # Merge overlapping bounding boxes, extend boxes close to the edge, etc.
    ornaments_merged = merge_bboxes(ornaments, closing_nosmall, img_prep)

    if not debug:
        write_result(file_path, ornaments_merged, img)
    else:
        # save snapshots of the processing pipeline as separate images
        mask = np.zeros(img.shape)
        cv2.drawContours(mask, ornaments, -1, (255, 255, 255), -1)
        mask = cv2.convertScaleAbs(mask)

        boxed = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB)
        for c in ornaments_merged:
            [x, y, w, h] = cv2.boundingRect(c)
            cv2.rectangle(boxed, (x, y), (x+w, y+h), (255, 0, 0), 3)

        plots = [
            ("prep", img_prep),
            ("blanked", text_blanked),
            ("closing", closing),
            ("tocontour", closing_nosmall),
            ("mask", mask),
            ("boxed", boxed)
        ]

        fig, ax = plt.subplots(2, 3)
        fig.set_size_inches(20, 20)

        fig.suptitle(file_path)
        ax = ax.ravel()

        for i, a, (n, d) in zip(range(len(plots)), ax, plots):
            a.imshow(d, cmap=cm.Greys_r)
            label = "%d_%s" % (i+1, n)
            a.set_title(label)
            save_debug_image(file_path, d, label)

        debug_fname = path.join(dir_name, base_name + "_debug.png")
        fig.savefig(debug_fname)
        plt.close(fig)


# http://stackoverflow.com/questions/1408356/keyboard-interrupts-with-pythons-multiprocessing-pool
class KeyboardInterruptError(Exception):
    pass


def _proc_img(args):
    fn, debug = args
    log = get_logger()
    log.info("Processing file %s" % fn)
    try:
        process_image(fn, debug=debug)
    except KeyboardInterrupt as e:
        raise KeyboardInterruptError()
    except Exception as e:
        log.exception("Processing of " + fn + " failed")


def process_images(dir, ncores=mp.cpu_count(), debug=False):
    """
    Process all the images in the given directory. Parallelism controlled by
    ncores.
    """
    log = get_logger()

    if path.isdir(dir):
        files = sorted(glob.glob(dir + "/*.TIF"))
    else:
        # actually only a single file was passed
        files = [dir]
        ncores = 1

    log.info("Processing %s image(s) from %s with %s core(s)" % (len(files),
                                                                 dir,
                                                                 ncores))
    try:
        pool = mp.Pool(processes=ncores)
        # unfortunately pool.map() does not allow passing (kw)args
        # so workaround by passing tuples
        args = [(x, debug) for x in files]
        pool.map(_proc_img, args)
    except KeyboardInterruptError as e:
        log.warning("Keyboard Interrupt detected, stopping")
        pool.terminate()


def get_logger():
    return logging.getLogger(path.basename(__file__))


def setup_logging(dir):
    """
    Configures the logging subsystem with a console & file handler
    """
    # create logger
    logger = get_logger()
    logger.setLevel(logging.DEBUG)

    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    # file handler
    fname = os.path.join(dir, "fleuron.log")
    fh = logging.FileHandler(fname)
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    logger.info("Logging setup with logfile %s" % fname)

    return logger


def main():
    # Read the commandline arguments
    parser = argparse.ArgumentParser(description="Fleuron: Ornament extractor",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('datadir', metavar="<directory or single file>",
                        type=str,
                        help="image directory or single image (TIF only)",
                        nargs=1)
    parser.add_argument('-np', dest="nprocs", type=int,
                        help="number of processes to use (%d detected)" %
                        mp.cpu_count(), default=mp.cpu_count())
    parser.add_argument('-d', '--debug', dest="debug", action="store_true",
                        help="debug mode (disables writing results)", default=False)
    parser.add_argument('-v', '--version', action="version",
                        version=_version.__version__)

    args = parser.parse_args()

    data_dir = path.abspath(args.datadir[0])

    if path.isfile(data_dir):
        # actually only a single file was passed
        log = setup_logging(path.dirname(data_dir))
    else:
        log = setup_logging(data_dir)

    log.info("Fleuron version %s started" % _version.__version__)
    log.debug("Called with commandline arguments %s" % args)
    log.debug("Running on system %s" % str(os.uname()))

    process_images(data_dir, ncores=args.nprocs, debug=args.debug)


if __name__ == "__main__":
    main()
