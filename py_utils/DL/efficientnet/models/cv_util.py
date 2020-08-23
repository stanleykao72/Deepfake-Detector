import cv2


def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized


def padding_image(img, size):
    h, w = img.shape[:2]
    if h == size and w == size:
        return img

    pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
    if w < size:
        pad_left = (size - w) // 2
        pad_right = pad_left
        if (size - w) % 2 != 0:
            pad_right = pad_left + 1
    
    if h < size:
        pad_top = (size - h) // 2
        pad_bottom = pad_top
        if (size - h) % 2 != 0:
            pad_bottom = pad_top + 1

    return cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,cv2.BORDER_CONSTANT,value=(0,0,0))
