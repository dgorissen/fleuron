from __future__ import print_function, division
import cv2
import numpy as np
from skimage import measure


def autocrop(image):
    """
    Remove any surrounding whitespace.
    """
    rows = np.where(np.min(image, 0) < 255)[0]
    if rows.size:
        cols = np.where(np.min(image, 1) < 255)[0]
        image = image[cols[0]:cols[-1] + 1, rows[0]:rows[-1] + 1]
    else:
        image = image[:1, :1]

    return image


def blank_margin(img, mw=30, mh=30, absolute=True):
    """
    Blank out the pixels within the given margin.
    """
    height, width = img.shape

    if not absolute:
        mh = mh * height
        mw = mw * width

    img[:mh, :] = 0
    img[height-mh:, :] = 0

    img[:, :mw] = 0
    img[:, width-mw:] = 0

    return img


def get_line_heights(image, prop_width=1, perc_nonzero=0.1,
                     minscale=5, axis=1):
    """
    Returns the distances between any horizontal blank lines
    at least minscale apart
    """
    height, width = image.shape

    # how much of the width to look at to decide if its a blank line
    # taking < width helps a bit with warped text
    p = round(width * prop_width)
    sums = np.sum(image[:, :p], axis=axis)
    sums = sums / (width * 255)

    mask = sums < perc_nonzero
    I = np.where(mask == True)[0]

    diffs = np.diff(I)
    diffs = diffs[diffs > minscale]

    return diffs


def is_lined(image, min_lines=3, max_std=10, axis=1):
    """
    Simple heuristic for checking if an image is just lines of text.
    """
    lh = get_line_heights(image, axis=axis)
    return len(lh) >= min_lines and lh.std() <= max_std


def blank_text(image):
    """
    Simple heuristic based text line detection and deletion.
    """
    height, width = image.shape

    img = blank_margin(image, mw=0.025, mh=0.02, absolute=False)
    # how much of the width to look at to decide if its a blank line
    # taking < width helps a bit with warped text
    p = round(img.shape[1] * 0.60)
    sums = np.sum(img[:, :p], axis=1)

    mask = sums < 10
    I = np.where(mask == True)[0]

    diffs = np.diff(I)
    diffs = diffs[diffs > 5]
    med = np.median(diffs)

    max_text_height = min(med + 5, 70)

    i = 0
    for j in I:
        d = j - i
        if d > 1 and d <= max_text_height:
            nz = np.count_nonzero(image[i:j+1, :])
            pixel_density = nz/(d * image.shape[1])
            vsums = np.sum(image[i:j+1, :], axis=0)
            vsums = np.trim_zeros(vsums)
            nblank = np.count_nonzero(vsums < 5)
            if nblank > 90 and pixel_density < 0.23 \
               and not is_glued_text(image[i:j+1, :]):
                # assume a line of text, blank out
                image[i:j+1, :] = 0
        else:
            pass

        i = j

    return image


def compute_skew(image, ran=5):
    """
    Estimate the (linear) page skew by usin ght Hough transform.
    """
    horizontal = 90
    image = cv2.bitwise_not(image)
    height, width = image.shape

    lines = cv2.HoughLinesP(image, 1, np.pi/180, 100, minLineLength=0.5*width,
                            maxLineGap=40)
    angles = np.array([np.arctan2(x2 - x1, y2 - y1)*180.0/np.pi for
                       x1, y1, x2, y2 in lines[0]])
    angles = (angles + 180) % 360 - 180

    I1 = abs(angles) >= (horizontal - ran)
    I2 = abs(angles) <= (horizontal + ran)
    I = np.logical_and(I1, I2)
    am = angles[I].mean()

    return am, angles[horizontal] - am, lines[0][I]


def nonzero_outside_margin(img, rect, m=30):
    """
    How many non-zero pixels surround a bounding box.
    """
    [x, y, w, h] = rect

    height, width = img.shape

    if isinstance(m, int):
        tm, bm, lm, rm = [m]*4
    else:
        tm, bm, lm, rm = m

    xl1 = max(0, x - lm)
    xl2 = min(width, x + w + rm)

    yl1 = max(0, y - tm)
    yl2 = min(height, y + h + bm)

    view = np.zeros(img.shape)
    view[yl1:yl2, xl1:xl2] = img[yl1:yl2, xl1:xl2]
    view[y:y+h, x:x+w] = 0

    n = np.count_nonzero(view)

    return n


def re_blank(src, c, reference):
    """
    Copy all of the blank lines from the referece image into the source image,
    thus breaking any glued pieces apart.
    """
    [x, y, w, h] = cv2.boundingRect(c)

    # get the corresponding patch in the
    # source and reference images
    ref_patch = reference[y:y+h, x:x+w]
    src_patch = src[y:y+h, x:x+w]

    hsums = np.sum(ref_patch, axis=1)
    vsums = np.sum(ref_patch, axis=0)

    Ih = hsums < 1
    Iv = vsums < 1

    src_patch[Ih, :] = 0
    src_patch[:, Iv] = 0

    # find new contours
    # do this in the full size image so contour coordinates are correct and we
    # dont have to mess with offsets later on

    mask = np.zeros(src.shape, np.uint8)
    mask[y:y+h, x:x+w] = src_patch

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)

    return contours


def save_contour(source, c, suffix):
    """
    Useful debug function to save a contour as an image.
    """
    t = np.zeros(source.shape, np.uint8)
    cv2.drawContours(t, [c], -1, (255, 255, 255), 1)
    cv2.imwrite("contours-%s.png" % suffix, t)


def connected_components(image, ignore_holes=True):
    """
    Get all the connected components.
    """
    L = measure.label(image)

    if ignore_holes:
        # image is binary so multiplying removes all CCs
        # that are holes
        L = np.multiply(L, image)

    unique, counts = np.unique(L, return_counts=True)
    I = np.argsort(counts)[::-1]
    comps = np.asarray((unique[I], counts[I])).T
    return L, comps


def is_glued_text(image, max_cc_size=800, prop=0.97):
    """
    Guess if an image is just some text by looking at the height distribution
    of the connected components.
    """
    comps, counts = connected_components(image)
    props = measure.regionprops(comps, cache=False)

    heights = np.array([x.bbox[2] - x.bbox[0] for x in props])

    if heights.size > 10 \
       and heights.max() < 70 \
       and 2 <= heights.std() <= 12.5:
        return True
    else:
        return False


def get_patch(bbox, image):
    """
    Extract a rectangular section from an image.
    """
    [x, y, w, h] = bbox
    return image[y:y+h, x:x+w]


def get_content_bbox(image):
    """
    Get the content bounding box for the full image.
    """
    rows = np.where(np.max(image, 0) > 0)[0]
    cols = np.where(np.max(image, 1) > 0)[0]

    y = cols[0]
    h = cols[-1] - y + 1
    x = rows[0]
    w = rows[-1] - x + 1

    return x, y, w, h


def save_debug_image(file_path, img, suffix):
    """
    Utility function for saving debug images.
    """
    fn = file_path[:-4] + "_" + suffix + ".png"
    cv2.imwrite(fn, img)
