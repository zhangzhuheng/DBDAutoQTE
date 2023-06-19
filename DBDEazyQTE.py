import math
import threading
import time

import dxcam
import keyboard
import numpy as np
import win32api
import winsound
from PIL import Image, ImageGrab

img_dir = 'img/'
delay_degree = 0
last_im_a = None
toggle = True
keyboard_switch = True
frame_rate = 60
repair_speed = 330
heal_speed = 300
wiggle_speed = 230
shot_delay = 0.006574
press_and_release_delay = 0.003206
color_sensitive = 125
delay_pixel = 0
speed_now = repair_speed
hyperfocus = False
red_threshold = 180
focus_level = 0


def find_largest_square(shape: tuple, target_points: list[tuple[int, int]]) -> tuple[int, int, int] | None:
    """
    Find the largest square that contains all the given target points using binary search.

    :param shape: Shape of the image.
    :param target_points: Coordinates of target points.
    :return: Center and side length of the largest square.
    """
    center_i, center_j = target_points[0]
    max_side_length = 0

    mark = np.zeros((shape[0], shape[1]), dtype=bool)
    for i, j in target_points:
        mark[i][j] = True

    def check(_l: int) -> bool:
        nonlocal center_i, center_j
        for _i, _j in target_points:
            if _i + _l >= shape[0] or _j + _l >= shape[1] or _j - _l < 0 or _i - _l < 0:
                continue
            if mark[_i + _l][_j + _l] and mark[_i - _l][_j - _l] and mark[_i - _l][_j + _l] and mark[_i + _l][_j - _l]:
                center_i, center_j = _i, _j
                return True
        return False

    left, right = 1, min(shape[0], shape[1])
    while left <= right:
        mid = (left + right) // 2
        if check(mid):
            left = mid + 1
            max_side_length = mid
        else:
            right = mid - 1

    if max_side_length < 1:
        return None
    return center_i, center_j, max_side_length


def find_largest_red_square(image: np.ndarray) -> tuple[int, int, int] | None:
    """
    Find the largest square that contains only red pixels in the given image.

    :param image: Input image.
    :return: Center and radius of the largest square, or None if no such square is found.
    """
    shape = image.shape

    # Create a boolean mask for red pixels
    red_mask = image[..., 0] > red_threshold
    non_red_mask = np.logical_or(image[..., 1] >= 20, image[..., 2] >= 20)
    logical_array = np.logical_and(red_mask, ~non_red_mask)

    # Create a boolean mask for pixels inside the central square
    row, col = np.ogrid[:shape[0], :shape[1]]
    center_row, center_col = shape[0] / 2, shape[1] / 2
    inner_disk_mask = ((row - center_row) ** 2 + (col - center_col) ** 2) <= (shape[0] / 2) ** 2

    # Combine the two masks
    logical_array = np.logical_and(logical_array, inner_disk_mask)

    # Find the coordinates of target points
    target_points = np.argwhere(logical_array)

    if not len(target_points):
        return None

    # Change the color of target points to pure red
    image[target_points[:, 0], target_points[:, 1]] = [255, 0, 0]

    # Find the largest square that contains only target points
    return find_largest_square(shape[:2], target_points)


def find_square(im_array):
    shape = im_array.shape
    target_points = []
    global focus_level
    for i in range(shape[0]):
        for j in range(shape[1]):
            if list(im_array[i][j]) == [255, 255, 255]:
                if shape[0] * (200 - 40) / 2 / 200 < i < shape[0] * (200 + 40) / 2 / 200 and shape[1] * (
                        200 - 140) / 2 / 200 < j < shape[1] * (200 + 140) / 2 / 200:
                    im_array[i][j] = [0, 0, 0]
                    continue
                target_points.append((i, j))
                im_array[i][j] = [0, 0, 255]

    if not target_points:
        return

    r_i, r_j, max_d = find_largest_square(shape, target_points)

    # print("white square:",r_i, r_j)
    if not r_i or not r_j:
        return
    # if max_d < 1:
    #     return
    pre_d = 0
    post_d = 0
    target = cal_degree(r_i - qte_region.height / 2, r_j - qte_region.width / 2)
    sin = math.sin(2 * math.pi * target / 360)
    cos = math.cos(2 * math.pi * target / 360)
    for i in range(max_d, 21):
        pre_i = round(r_i - sin * i)
        pre_j = round(r_j - cos * i)
        if list(im_array[pre_i][pre_j]) == [0, 0, 255]:
            pre_d = i
        else:
            break
    for i in range(max_d, 21):
        pre_i = round(r_i + sin * i)
        pre_j = round(r_j + cos * i)
        if list(im_array[pre_i][pre_j]) == [0, 0, 255]:
            post_d = i
        else:
            break
    print(pre_d, post_d)

    if pre_d + post_d < 5:
        print('merciless storm')
        # Image.fromarray(im_array).save(img_dir+'merciless.png')
        to_be_deleted = []
        for i, j in target_points:
            if abs(i - r_i) <= 20 and abs(j - r_j) <= 20:
                to_be_deleted.append((i, j))
        print('before', target_points)

        for i in to_be_deleted:
            target_points.remove(i)
        print('after', target_points)
        if not target_points:
            return
        r2_i, r2_j, max_d = find_largest_square(shape, target_points)
        if max_d < 3:
            target1 = cal_degree(r_i - qte_region.height / 2, r_j - qte_region.width / 2)
            target2 = cal_degree(r2_i - qte_region.height / 2, r2_j - qte_region.width / 2)
            print('storm points', r_i, r_j, r2_i, r2_j)
            if target1 < target2:
                pre_white = (r_i, r_j)
                post_white = (r2_i, r2_j)
            else:
                pre_white = (r2_i, r2_j)
                post_white = (r_i, r_j)
            new_white = (round((pre_white[0] + post_white[0]) / 2), round((pre_white[1] + post_white[1]) / 2))
            focus_level = 0
            return new_white, pre_white, post_white

    pre_white = (round(r_i - sin * pre_d), round(r_j - cos * pre_d))
    post_white = (round(r_i + sin * post_d), round(r_j + cos * post_d))

    new_white = (round((pre_white[0] + post_white[0]) / 2), round((pre_white[1] + post_white[1]) / 2))
    if list(im_array[new_white[0]][new_white[1]]) != [0, 0, 255]:
        print("new white error")
        return
    #

    return new_white, pre_white, post_white


def wiggle(t1, deg1, direction, im1):
    speed = wiggle_speed * direction
    target1 = 270
    target2 = 90
    delta_deg1 = (target1 - deg1) % (direction * 360)
    delta_deg2 = (target2 - deg1) % (direction * 360)
    predict_time = min(delta_deg1 / speed, delta_deg2 / speed)
    print("predict time", predict_time)
    # sleep(0.75)
    # return #debug

    click_time = t1 + predict_time - press_and_release_delay + delay_degree / abs(speed)

    delta_t = click_time - time.perf_counter()

    # print('delta_t',delta_t)
    if 0 > delta_t > -0.1:
        keyboard.press_and_release('space')
        print('quick space!!', delta_t, '\nspeed:', speed)
        time.sleep(0.13)
        return
    try:
        delta_t = click_time - time.perf_counter()
        time.sleep(delta_t)
        keyboard.press_and_release('space')
        print('space!!', delta_t, '\nspeed:', speed)
        Image.fromarray(im1).save(img_dir + 'log.png')
        time.sleep(0.13)
    except ValueError as e:

        # winsound.Beep(230,300)
        print(e, delta_t, deg1, delta_deg1, delta_deg2)


def timer(im1, t1):
    global focus_level
    if not toggle:
        return
    # print('timer',time.perf_counter())
    r1 = find_largest_red_square(im1)
    if not r1:
        return

    deg1 = cal_degree(r1[0] - qte_region.height / 2, r1[1] - qte_region.width / 2)

    # print('first seen:',deg1,t1)
    global last_im_a

    # sleep(1.5)
    # return #debug
    im2 = qte_region_camera.get_latest_frame()

    r2 = find_largest_red_square(im2)

    if not r2:
        return

    deg2 = cal_degree(r2[0] - qte_region.height / 2, r2[1] - qte_region.width / 2)
    if deg1 == deg2:
        # print("red same")
        return
    # speed = (deg2-deg1)/(t2-t1)

    if (deg2 - deg1) % 360 > 180:
        direction = -1
    else:
        direction = 1

    if speed_now == wiggle_speed:
        print("wiggle")
        return wiggle(t1, deg1, direction, im1)
    if hyperfocus:
        speed = direction * speed_now * (1 + 0.04 * focus_level)
    else:
        speed = direction * speed_now

    # im2[pre_i][pre_j][0] > 200 and im2[pre_i][pre_j][1] < 20 and im2[pre_i][pre_j][2] < 20:

    white = find_square(im1)

    if not white:
        return
    print(white)
    white, pre_white, post_white = white

    if direction < 0:
        pre_white, post_white = post_white, pre_white
    im1[r1[0]][r1[1]] = [0, 255, 0]
    im1[white[0]][white[1]] = [0, 255, 0]
    last_im_a = im1

    print('targeting_time:', time.perf_counter() - t1)
    print('speed:', speed)

    target = cal_degree(white[0] - qte_region.height / 2, white[1] - qte_region.width / 2)
    # target=180

    # if target< 45 or target > 315 or (target>135 and target<225):
    #     white_2=(white[0],white[1]-max_d)
    #     white_3=(white[0],white[1]+max_d)
    # else:
    #     white_2=(white[0]-max_d,white[1])
    #     white_3=(white[0]+max_d,white[1])

    delta_deg = (target - deg1) % (direction * 360)

    print("predict time", delta_deg / speed)
    # sleep(0.75)
    # return #debug

    click_time = t1 + delta_deg / speed - press_and_release_delay + delay_degree / abs(speed)
    # print("minus ",click_time%(1/frame_rate))
    # click_time-=click_time%(1/frame_rate)
    delta_t = click_time - time.perf_counter()

    # sin=math.sin(2*math.pi*target/360)
    # cos=math.cos(2*math.pi*target/360)
    max_d = r1[2]
    global delay_pixel
    start_point = post_white
    sin = math.sin(2 * math.pi * target / 360)
    cos = math.cos(2 * math.pi * target / 360)
    max_d += delay_pixel
    delta_i = pre_white[0] - white[0]
    delta_j = pre_white[1] - white[1]
    # if hyperfocus:
    #     delta_i*=(1+0.04*focus_level)
    #     delta_j*=(1+0.04*focus_level)
    end_point = [white[0] + round(delta_i - direction * sin * (-max_d)),
                 white[1] + round(delta_j - direction * cos * (-max_d))]
    check_points = []
    if abs(end_point[0] - start_point[0]) < abs(end_point[1] - start_point[1]):
        for j in range(start_point[1], end_point[1], 2 * np.sign(end_point[1] - start_point[1])):
            i = start_point[0] + (end_point[0] - start_point[0]) / (end_point[1] - start_point[1]) * (
                    j - start_point[1])
            i = round(i)
            check_points.append((i, j))
    elif np.sign(end_point[0] - start_point[0]) == 0:
        return
    else:
        for i in range(start_point[0], end_point[0], 2 * np.sign(end_point[0] - start_point[0])):
            j = start_point[1] + (end_point[1] - start_point[1]) / (end_point[0] - start_point[0]) * (
                    i - start_point[0])
            j = round(j)
            check_points.append((i, j))
    check_points.append(end_point)
    print('check points', check_points)
    pre_4deg_check_points = []

    if abs(end_point[0] - start_point[0]) ** 2 + abs(end_point[1] - start_point[1]) ** 2 < 20 ** 2:
        start_point = pre_white
        end_point = (end_point[0] + delta_i, end_point[1] + delta_j)
        # if the white area is too large don't use pre_4deg
        if abs(end_point[0] - start_point[0]) < abs(end_point[1] - start_point[1]):
            for j in range(start_point[1], end_point[1], 2 * np.sign(end_point[1] - start_point[1])):
                i = start_point[0] + (end_point[0] - start_point[0]) / (end_point[1] - start_point[1]) * (
                        j - start_point[1])
                i = round(i)
                pre_4deg_check_points.append((i, j))
        elif np.sign(end_point[0] - start_point[0]) == 0:
            return
        else:
            for i in range(start_point[0], end_point[0], 2 * np.sign(end_point[0] - start_point[0])):
                j = start_point[1] + (end_point[1] - start_point[1]) / (end_point[0] - start_point[0]) * (
                        i - start_point[0])
                j = round(j)
                pre_4deg_check_points.append((i, j))
        pre_4deg_check_points.append(end_point)
    else:
        print('[!]large white area detected')
        check_points.pop()

    # TODO: extend pre_4deg_check_points for more degs

    print('pre 4 deg check points', pre_4deg_check_points)

    print('delta_t', delta_t)
    if 0 > delta_t > -0.1:
        keyboard.press_and_release('space')
        print('[!]quick space!!', delta_t, '\nspeed:', speed)
        # sleep(0.5)

        if hyperfocus:
            print('focus hit:', focus_level)
            focus_level = (focus_level + 1) % 7
        return
    try:
        delta_t = click_time - time.perf_counter()
        # sleep(max(0,delta_t-0.1))

        # trying to catch
        checks_after_awake = 0
        check_times = 0
        im_array_pre_backup = None
        while True:
            out = False
            im_array_pre = qte_region_camera.get_latest_frame()
            checks_after_awake += 1

            for i, j in check_points:
                if im_array_pre[i][j][0] > red_threshold and im_array_pre[i][j][1] < 20 and im_array_pre[i][j][2] < 20:
                    out = True
                    im_array_pre[i][j] = [0, 255, 255]
                    check_times = 1
                    break
            if out:
                break

            for k in range(len(pre_4deg_check_points)):
                i, j = pre_4deg_check_points[k]
                if im_array_pre[i][j][0] > red_threshold and im_array_pre[i][j][1] < 20 and im_array_pre[i][j][2] < 20:
                    out = True
                    check_times = 2
                    im_array_pre[i][j] = [255, 255, 0]
                    t = 4 / speed_now * (1 + k) / len(pre_4deg_check_points) - press_and_release_delay
                    if t > 0:
                        time.sleep(t)
                    break
            if out:
                break
            if time.perf_counter() > click_time + 0.04:
                print('catch time out')
                break
            im_array_pre_backup = im_array_pre
        # if speed < 315:
        if im_array_pre_backup is None:
            return

        keyboard.press_and_release('space')
        print('check times', check_times)
        if checks_after_awake <= 1:
            print('[!]awake quick space!!', delta_t, '\nspeed:', speed)
            file_name = 'awake'
        else:
            print('space!!', delta_t, '\nspeed:', speed)
            file_name = ''
        print(im_array_pre[pre_white[0], pre_white[1]])
        # Image.fromarray(im_array3).show()
        # return
        r3 = find_largest_red_square(im_array_pre)
        shape = im_array_pre_backup.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                if im_array_pre_backup[i][j][0] > red_threshold and im_array_pre_backup[i][j][1] < 20 and \
                        im_array_pre_backup[i][j][2] < 20:
                    l1, l2 = i - shape[0] / 2, j - shape[1] / 2
                    if l1 * l1 + l2 * l2 > shape[0] * shape[0] / 4:
                        # print('not in circle:',i,j)
                        continue
                    im_array_pre[i][j] = [255, 0, 0]

        if not r3:
            return

        deg3 = cal_degree(r3[0] - qte_region.height / 2, r3[1] - qte_region.width / 2)
        real_delta_deg = deg3 - target

        im_array_pre[r1[0]][r1[1]] = [0, 255, 0]
        im_array_pre[white[0]][white[1]] = [0, 0, 255]

        im_array_pre[r3[0]][r3[1]] = [255, 255, 0]

        for i, j in check_points:
            im_array_pre[i][j] = [255, 255, 0]
        for i, j in pre_4deg_check_points:
            im_array_pre[i][j] = [0, 255, 0]

        im_array_pre[post_white[0]][post_white[1]] = [0, 255, 0]
        im_array_pre[pre_white[0]][pre_white[1]] = [0, 255, 0]
        if hyperfocus:
            file_name += 'log_focus' + str(focus_level) + '_' + str(real_delta_deg) + '_' + str(int(time.time()))
        else:
            file_name += 'log_' + str(real_delta_deg) + '_' + str(int(time.time()))
        file_name += 'speed_' + str(speed) + '.png'
        file_name = img_dir + file_name
        Image.fromarray(im_array_pre).save(file_name)
        # sleep(0.3)
        if hyperfocus:
            print('focus hit:', focus_level)
            focus_level = min(6, (focus_level + 1))
    except ValueError as e:
        Image.fromarray(im1).save(img_dir + 'log.png')
        # winsound.Beep(230,300)
        print(e, delta_t, deg1, deg2, target)

    # TODO: if white in im2


RESOLUTION_CROP_MAPPING = {
    (1920, 1080): (150, 150),
    (2560, 1440): (200, 200),
    (2560, 1600): (250, 250),
    (3840, 2160): (330, 330),
}


class QTERegion:
    def __init__(self):
        self.screen_w, self.screen_h = ImageGrab.grab().size
        self.width = None
        self.height = None
        self.screenshot_region = None
        self._set_size()
        self._set_qte_screenshot_region()

    def _set_size(self):
        if (self.screen_w, self.screen_h) in RESOLUTION_CROP_MAPPING.keys():
            self.width, self.height = RESOLUTION_CROP_MAPPING.get((self.screen_w, self.screen_h))
        else:
            raise Exception(f'The current resolution setting {self.screen_w, self.screen_h} is not supported.\n'
                            f'Please try setting it to the following resolution and restart: '
                            f'{list(RESOLUTION_CROP_MAPPING.keys())}')

    def _set_qte_screenshot_region(self):
        left, top = (self.screen_w - self.width) // 2, (self.screen_h - self.height) // 2
        right, bottom = left + self.width, top + self.height
        self.screenshot_region = (left, top, right, bottom)


qte_region = QTERegion()
fps = win32api.EnumDisplaySettings(win32api.EnumDisplayDevices().DeviceName, -1).DisplayFrequency
qte_region_camera = dxcam.create()
qte_region_camera.start(region=qte_region.screenshot_region, target_fps=fps, video_mode=True)


def driver():
    try:
        print('starting')
        while True:
            timer(qte_region_camera.get_latest_frame(), time.perf_counter())
    except KeyboardInterrupt:
        if last_im_a:
            Image.fromarray(last_im_a).save(img_dir + 'last_log.png')
    finally:
        qte_region_camera.stop()


def cal_degree(x, y):
    a = np.array([-1, 0])
    b = np.array([x, y])
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    a_dot_b = a.dot(b)
    cos_theta = np.arccos(a_dot_b / (a_norm * b_norm))
    degree = np.rad2deg(cos_theta)
    if b[1] < 0:
        degree = 360 - degree
    return degree


def keyboard_callback(x):
    global speed_now, delay_pixel, toggle, focus_level, hyperfocus, keyboard_switch

    if x.name == 'f1':
        if keyboard_switch:
            winsound.Beep(200, 500)
            keyboard_switch = False
            toggle = False
            print('keyboard_switch:', keyboard_switch)
        else:
            winsound.Beep(350, 500)
            keyboard_switch = True
            toggle = True
            print('keyboard_switch:', keyboard_switch)
    if not keyboard_switch:
        return
    if x.name == 'caps lock':
        if toggle:
            winsound.Beep(200, 500)
            toggle = False
            print('toggle:', toggle)
        else:
            winsound.Beep(350, 500)
            toggle = True
            print('toggle:', toggle)

    if not toggle:
        return
    if x.name in 'wasd':
        focus_level = 0
    if x.name == '3':
        toggle = True
        focus_level = 0
        print('change to repair')
        winsound.Beep(262, 500)
        speed_now = repair_speed
    if x.name == '4':
        toggle = True
        focus_level = 0
        winsound.Beep(300, 500)
        print('change to heal')
        speed_now = heal_speed
    if x.name == '5':
        toggle = True
        winsound.Beep(440, 500)
        print('change to wiggle')
        speed_now = wiggle_speed
    if x.name == '6':
        if hyperfocus:
            winsound.Beep(200, 500)
            hyperfocus = False
            print('hyperfocus disabled')
        else:
            winsound.Beep(350, 500)
            hyperfocus = True
            print('hyperfocus enabled')
    if x.name == '=':
        winsound.Beep(460, 500)
        delay_pixel += 2
        print('delay_pixel:', delay_pixel)
    if x.name == '-':
        winsound.Beep(500, 500)
        delay_pixel -= 2
        print('delay_pixel:', delay_pixel)


def main():
    # cap_test()

    import os
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    keyboard.on_press(keyboard_callback)
    threading.Thread(target=keyboard.wait)
    driver()


if __name__ == "__main__":
    main()
