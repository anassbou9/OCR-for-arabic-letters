import numpy as np
import cv2 as cv
import segmentation as my_segm
from skimage.morphology import skeletonize, thin
import preprocessing as pre
import os

CURR_PATH = os.path.dirname(os.path.abspath(__file__))
INPUT_TEST_PATH = CURR_PATH + "/../books_for_ocr/scanned_pics/test_7.PNG"
chars_path = CURR_PATH + "/../chars"



def binarize(word_img):
    _, binary_img = cv.threshold(word_img, 127, 255, cv.THRESH_BINARY)
    # _, binary_img = cv.threshold(word_img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    return binary_img // 255


#Cette fonction remplit les zones d'images entourés par des pixels blancs
def fill(binary_img, VP):
    (h, w) = binary_img.shape

    flag = 1
    while flag:
        flag = 0
        for row in range(h - 1):
            for col in range(1, w - 1):
                if (
                    binary_img[row][col] == 0
                    and binary_img[row][col - 1] == 1
                    and binary_img[row][col + 1] == 1
                    and binary_img[row + 1][col] == 1
                    and VP[col] != 0
                ):
                    binary_img[row][col] = 1
                    # flag = 1    #!!!!!!!!!!

    return binary_img


def baseline_detection(word_img):
    """Indice de ligne de base d'un mot"""

    Histo = my_segm.gray_projection(word_img, "horizontal")
    sommet = np.amax(Histo)

    # Matrice d'indices des max
    baseline_idx = np.where(Histo == sommet)[0]

    # Premier ou dernier indice
    upper_base = baseline_idx[0]
    lower_base = baseline_idx[-1]
    thickness = abs(lower_base - upper_base) + 1

    return upper_base, lower_base, thickness

#Cette fonction le nombre de transitions maximale, elle parcourt l'image à partir la ligne de base et monte
def horizontal_transitions(word_img, baseline_idx):
    max_transitions = 0
    max_transitions_idx = baseline_idx
    line_idx = baseline_idx - 1
    lines = []
 

    while line_idx >= 0:
        current_transitions = 0
        flag = 0

        horizontal_line = word_img[line_idx, :]
        for pixel in reversed(horizontal_line):
            if pixel == 1 and flag == 0:
                current_transitions += 1
                flag = 1
            elif pixel == 0 and flag == 1:
                current_transitions += 1
                flag = 0

        if current_transitions >= max_transitions:
            max_transitions = current_transitions
            lines.append(line_idx)
            max_transitions_idx = line_idx 

        line_idx -= 1

    return lines[len(lines) // 2] #!!!!!!!!!!!!!

#Calcule le nombre de transitions verticaux pour une colonne donnée
def vertical_transitions(word_img, cut):
    transitions = 0

    vertical_line = word_img[:, cut]

    flag = 0
    for pixel in vertical_line:
        if pixel == 1 and flag == 0:
            transitions += 1
            flag = 1
        elif pixel == 0 and flag == 1:
            transitions += 1
            flag = 0

    return transitions

#VP : Projection verticale
#MFV : Valeur la plus fréquente
#MTI : Indice du nombre de transitions maximale
def cut_points(word_img, VP, MFV, MTI, baseline_idx):
    # flag pour savoir le début du mot
    f = 0

    flag = 0
    (h, w) = word_img.shape
    i = w - 1
    separation_regions = []

    wrong = 0
    # boucle sur la largeur de l'image commençant par la droite puisque l'écriture arabe commence par la droite
    while i >= 0:
        pixel = word_img[MTI, i]

        if pixel == 1 and f == 0:
            f = 1
            flag = 1

        if f == 1:
            # Début et fin de la région de séparation
            if pixel == 0 and flag == 1:
                start = i + 1
                flag = 0
            elif pixel == 1 and flag == 0:
                end = i  # end maybe = i not i+1
                flag = 1

                mid = (start + end) // 2

                left_zero = -1
                left_MFV = -1
                right_zero = -1
                right_MFV = -1
                # threshold pour MFV
                T = 1

                j = mid - 1
                # boucle à gauche du milieu pour détécter k tel que VP[k]==0 ou VP[k]<= MFV +T
                while j >= end:
                    if VP[j] == 0 and left_zero == -1:
                        left_zero = j
                    if VP[j] <= MFV + T and left_MFV == -1:
                        left_MFV = j

                    # if left_zero != -1 and left_MFV != -1:
                    #     break                               

                    j -= 1

                j = mid
                # boucle à droite du milieu
                while j <= start:
                    if VP[j] == 0 and right_zero == -1:
                        right_zero = j
                    if VP[j] <= MFV + T and right_MFV == -1:
                        right_MFV = j

                    if right_zero != -1 and right_MFV != -1:
                        break

                    j += 1

                # Vérifier d'abord si VP du mileu est nulle
                if VP[mid] == 0:
                    cut_index = mid
                #Prendre le zero le plus proche
                elif left_zero != -1 and right_zero != -1:
                    if abs(left_zero - mid) <= abs(right_zero - mid):
                        cut_index = left_zero
                    else:
                        cut_index = right_zero
                elif left_zero != -1:
                    cut_index = left_zero
                elif right_zero != -1:
                    cut_index = right_zero

                # Check for VP = MFV second
                # elif VP[mid] <= MFV+T:
                #     cut_index = mid                   
                elif left_MFV != -1:
                    cut_index = left_MFV
                elif right_MFV != -1:
                    cut_index = right_MFV
                else:
                    cut_index = mid

                seg = word_img[:, end:start]
                HP = my_segm.gray_projection(seg, "horizontal")   #Histogramme de projection horizontale
                SHPA = np.sum(HP[:MTI])    #Somme de HP en dessus de la ligne de base
                SHPB = np.sum(HP[MTI + 1 :])  #Somme de HP en dessous de la ligne de base

                top = 0
                for idx, proj in enumerate(HP):
                    if proj != 0:
                        top = idx
                        break

                cnt = 0
                for k in range(end, cut_index + 1):
                    if vertical_transitions(word_img, k) > 2:
                        cnt = 1
                if SHPB == 0 and (baseline_idx - top) <= 5 and cnt == 1:
                    # breakpoint()
                    wrong = 1
                else:
                    separation_regions.append((end, cut_index, start))

        i -= 1

    return separation_regions, wrong


def check_baseline(word_img, start, end, upper_base, lower_base):
    """Vérifier s'il y a une ligne de base dans la région"""
    j = end + 1

    cnt = 0
    while j < start:
        # pixel noir (discontinuité)
        base = upper_base
        while base <= lower_base:
            pixel = word_img[base][j]
            cnt += pixel

            base += 1

        j += 1

    if cnt == 0:
        return False

    return True


def inside_hole(word_img, end_idx, start_idx):
    """Vérifie si le segment contient des trous"""

    if end_idx == 0 and start_idx == 0:
        return 0

    sk = skeletonize(word_img)
    j = end_idx + 1
    flag = 1
    while j < start_idx:
        VT = vertical_transitions(sk, j)
        if VT <= 2:
            flag = 0
            break
        j += 1

    return flag


def check_hole(segment):
    """Autre méthode pour vérifier si le segment contient un trou"""

    # no_dots = segment.copy()

    contours, hierarchy = cv.findContours(segment, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cnt = 0
    for hier in hierarchy[0]:
        if hier[3] >= 0:
            cnt += 1

    return cnt != 0


def remove_dots(word_img, threshold=11):
    #Supprimer les diacritiques en utilisant la méthodes des composantes connexes avec un threshold de 11
    no_dots = word_img.copy()

    components, labels, stats, GoCs = cv.connectedComponentsWithStats(
        no_dots, connectivity=8
    )
    char = []
    for label in range(1, components):
        _, _, _, _, size = stats[label]
        if size > threshold:
            char.append(label)
    for label in range(1, components):
        _, _, _, _, size = stats[label]
        if label not in char:
            no_dots[labels == label] = 0

    return no_dots


def check_dots(segment):
    #vérifier si le segment contient des diacritiques
    contours, hierarchy = cv.findContours(
        segment[:, 1 : segment.shape[1] - 1], cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )

    cnt = 0
    for c in contours:
        if len(c) >= 1:
            cnt += 1
    return cnt > 1


def check_stroke(no_dots_copy, segment, upper_base, lower_base, SR1, SR2):
    """Vérifier si la région de séparation est une stroke"""
    T = 1
    components, labels, stats, cen = cv.connectedComponentsWithStats(
        segment, connectivity=8
    )
    skeleton = skeletonize(segment.copy()).astype(np.uint8)
    (h, w) = segment.shape

    cnt = 0
    for label in range(1, components):
        if stats[label][4] > 3:
            cnt += 1
        else:
            segment[labels == label] = 0

    if cnt > 2 or cnt == 0:
        return False

    if (
        check_hole(segment)
        or inside_hole(no_dots_copy, SR1[0], SR1[1])
        or inside_hole(no_dots_copy, SR2[0], SR2[1])
    ):
        return False

    HP = my_segm.gray_projection(skeleton, "horizontal")
    VP = my_segm.gray_projection(segment, "vertical")

    seg_l = -1
    seg_r = -1
    for i in range(0, len(VP)):
        if VP[i] != 0:
            seg_l = i
            break
    for i in range(len(VP) - 1, -1, -1):
        if VP[i] != 0:
            seg_r = i
            break

    seg_width = seg_r - seg_l + 1
    SHPA = np.sum(HP[:upper_base])
    SHPB = np.sum(HP[lower_base + T + 1 :])
    MFV_HP = np.argmax(np.bincount(HP)[1:]) + 1
    MFV = lower_base - upper_base + 1 + T

    top_pixel = -1
    for i, proj in enumerate(HP):
        if proj != 0:
            top_pixel = i
            break
    height = upper_base - top_pixel

    VT = 0
    for i in range(w):
        if vertical_transitions(skeleton, i) > 2:
            VT += 1
    cnt = 0
    for proj in VP:
        if proj >= height:
            cnt += 2
        elif proj == height - 1:
            cnt += 1
    # abs(MFV - MFV_HP) <= 2
    if SHPB == 0 and height <= 6 and VT <= 2 and seg_width <= 6 and cnt >= 2:
        return True

    return False

#SRL : Liste des régions de séparation
#VP : Projection verticale
#MTI : Indice du nombre de transitions maximale
#upper_base : dessus de la ligne de base
#MFV : valeur la plus fréquente
#top_line : ligne du haut du mot
def filter_regions(
    word_img,
    no_dots_copy,
    SRL: list,
    VP: list,
    upper_base: int,
    lower_base: int,
    MTI: int,
    MFV: int,
    top_line: int,
):
    valid_separation_regions = []
    overlap = []

    T = 1
    components, labels = cv.connectedComponents(
        word_img[: lower_base + 5, :], connectivity=8
    )

    SR_idx = 0
    while SR_idx < len(SRL):
        SR = SRL[SR_idx]
        end_idx, cut_idx, start_idx = SR

        # Cas 1 : la projection verticale de l'indice de coupe est nulle
        if VP[cut_idx] == 0:
            valid_separation_regions.append(SR)
            SR_idx += 1
            continue

        # Cas 2 : pas de chemin connecté entre le début et la fin du segment
        # components, labels= cv.connectedComponents(word_img[:, end_idx:start_idx+1], connectivity=8)
        if labels[MTI, end_idx] != labels[MTI, start_idx]:
            valid_separation_regions.append(SR)
            overlap.append(SR)
            SR_idx += 1
            continue

        # Cas 3 : Contient des trous
        # if check_hole(no_dots_copy[:, end_idx: cut_idx]) and inside_hole(no_dots_copy, end_idx, start_idx):  !!!!!!!!!!
        cc, l = cv.connectedComponents(
            1 - (no_dots_copy[:, end_idx : start_idx + 1]), connectivity=4
        )

        if cc - 1 >= 3 and inside_hole(no_dots_copy, end_idx, start_idx):
            SR_idx += 1
            continue

        # Cas 4 : Pas de ligne de base entre le début et la fin du segment
        segment = no_dots_copy[:, end_idx + 1 : start_idx]
        segment_width = start_idx - end_idx - 1

        j = end_idx + 1
        cnt = 0
        while j < start_idx:
            # pixel noir (Discontinuité)
            base = upper_base - T
            while base <= lower_base + T:
                pixel = no_dots_copy[base][j]
                cnt += pixel

                base += 1

            j += 1

        if cnt < segment_width - 2 and segment_width > 4:
            segment_HP = my_segm.gray_projection(segment, "horizontal")

            SHPA = np.sum(segment_HP[:upper_base])
            SHPB = np.sum(segment_HP[lower_base + T + 1 :])

            if (int(SHPB) - int(SHPA)) >= 0:
                SR_idx += 1
                continue
            elif VP[cut_idx] <= MFV + T:
                valid_separation_regions.append(SR)
                SR_idx += 1
                continue
            else:
                SR_idx += 1
                continue

        # if SR_idx == 0:
        #     breakpoint()
        # Cas 5 : dernière région ou VP[nextcut] = 0
        if SR_idx == len(SRL) - 1 or VP[SRL[SR_idx + 1][1]] == 0:
            if SR_idx == len(SRL) - 1:
                segment_dots = word_img[:, : SRL[SR_idx][1] + 1]
                segment = no_dots_copy[:, : SRL[SR_idx][1] + 1]
                next_cut = 0
            else:
                next_cut = SRL[SR_idx + 1][1]
                segment_dots = word_img[:, next_cut : SRL[SR_idx][1] + 1]
                segment = no_dots_copy[:, next_cut : SRL[SR_idx][1] + 1]

            segment_HP = my_segm.gray_projection(segment, "horizontal")
            (h, w) = segment.shape

            top = -1
            for i, proj in enumerate(segment_HP):
                if proj != 0:
                    top = i
                    break
            height = upper_base - top

            # if SR_idx == len(SRL) - 1:
            # breakpoint()
            SHPA = np.sum(segment_HP[:upper_base])
            SHPB = np.sum(segment_HP[lower_base + T + 1 :])
            sk = skeletonize(segment).astype(np.uint8)
            seg_VP = my_segm.gray_projection(segment, "vertical")
            non_zero = np.nonzero(seg_VP)[0]
            cnt = 0
            # for k in range(0, (len(non_zero)//2)+(len(non_zero)%2)):
            for k in range(0, 3):
                if k >= len(non_zero):
                    break
                index = non_zero[k]
                if seg_VP[index] >= height:
                    cnt += 1

            if (SHPB <= 5 and cnt > 0 and height <= 6) or (
                len(non_zero) >= 10 and SHPB > SHPA and not check_dots(segment_dots)
            ):
                SR_idx += 1
                continue

        # Strokes
        #SEGP : segment précédant
        #SEGN : Prochaint segment
        #SEGNN: Segment après le prochain segment

        SEGP = (-1, -1)
        SEG = (-1, -1)
        SEGN = (-1, -1)
        SEGNN = (-1, -1)
        SEGP_SR1 = (0, 0)
        SEGP_SR2 = (0, 0)
        SEG_SR1 = (0, 0)
        SEG_SR2 = (0, 0)
        SEGN_SR1 = (0, 0)
        SEGN_SR2 = (0, 0)
        SEGNN_SR1 = (0, 0)
        SEGNN_SR2 = (0, 0)

        current_cut = SR[1]

        if SR_idx == 0:
            SEGP = (SRL[SR_idx][1], word_img.shape[1] - 1)
            SEGP_SR1 = (SRL[SR_idx][0], SRL[SR_idx][2])
            SEGP_SR2 = (SRL[SR_idx][1], word_img.shape[1] - 1)

        if SR_idx > 0:
            SEGP = (SRL[SR_idx][1], SRL[SR_idx - 1][1])
            SEGP_SR1 = (SRL[SR_idx][0], SRL[SR_idx][2])
            SEGP_SR2 = (SRL[SR_idx - 1][0], SRL[SR_idx - 1][2])

        if SR_idx < len(SRL) - 1:
            SEG = (SRL[SR_idx + 1][1], SRL[SR_idx][1])
            SEG_SR1 = (SRL[SR_idx][0], SRL[SR_idx][2])
            SEG_SR2 = (SRL[SR_idx + 1][0], SRL[SR_idx + 1][2])

        if SR_idx < len(SRL) - 2:
            SEGN = (SRL[SR_idx + 2][1], SRL[SR_idx + 1][1])
            SEGN_SR1 = (SRL[SR_idx + 1][0], SRL[SR_idx + 1][2])
            SEGN_SR2 = (SRL[SR_idx + 2][0], SRL[SR_idx + 2][2])
        elif SR_idx == len(SRL) - 2:
            SEGN = (0, SRL[SR_idx + 1][1])
            SEGN_SR1 = (SRL[SR_idx + 1][0], SRL[SR_idx + 1][2])
            SEGN_SR2 = (0, SRL[SR_idx + 1][2])

        if SR_idx < len(SRL) - 3:
            SEGNN = (SRL[SR_idx + 3][1], SRL[SR_idx + 2][1])
            SEGNN_SR1 = (SRL[SR_idx + 2][0], SRL[SR_idx + 2][2])
            SEGNN_SR2 = (SRL[SR_idx + 3][0], SRL[SR_idx + 3][2])

        # if SR_idx == 6:
        #     breakpoint()

        # SEG est une stroke avec diacritiques
        if SEG[0] != -1 and (
            check_stroke(
                no_dots_copy,
                no_dots_copy[:, SEG[0] : SEG[1]],
                upper_base,
                lower_base,
                SEG_SR1,
                SEG_SR2,
            )
            and check_dots(word_img[:, SEG[0] : SEG[1]])
        ):
            # breakpoint()
            # Cas où ca commence par Ø´
            if SEGP[0] != -1 and (
                (
                    check_stroke(
                        no_dots_copy,
                        no_dots_copy[:, SEGP[0] : SEGP[1]],
                        upper_base,
                        lower_base,
                        SEGP_SR1,
                        SEGP_SR2,
                    )
                    and not check_dots(word_img[:, SEGP[0] : SEGP[1]])
                )
                and (
                    SR_idx == 0
                    or VP[SRL[SR_idx - 1][1]] == 0
                    or (VP[SRL[SR_idx - 1][1]] == 0 and SRL[SR_idx - 1] in overlap)
                )
            ):
                SR_idx += 2
                continue
            else:
                valid_separation_regions.append(SR)
                SR_idx += 1
                continue

        # SEG est un stroke sans diacritiques
        elif SEG[0] != -1 and (
            check_stroke(
                no_dots_copy,
                no_dots_copy[:, SEG[0] : SEG[1]],
                upper_base,
                lower_base,
                SEG_SR1,
                SEG_SR2,
            )
            and not check_dots(word_img[:, SEG[0] : SEG[1]])
        ):
            # Cas où ca commence par Ø³
            if SEGP[0] != -1 and (
                check_stroke(
                    no_dots_copy,
                    no_dots_copy[:, SEGP[0] : SEGP[1]],
                    upper_base,
                    lower_base,
                    SEGP_SR1,
                    SEGP_SR2,
                )
                and not check_dots(word_img[:, SEGP[0] : SEGP[1]])
            ):
                SR_idx += 2
                continue

            # SEGN est un stroke sans diacritques
            if SEGN[0] != -1 and (
                check_stroke(
                    no_dots_copy,
                    no_dots_copy[:, SEGN[0] : SEGN[1]],
                    upper_base,
                    lower_base,
                    SEGN_SR1,
                    SEGN_SR2,
                )
                and not check_dots(word_img[:, SEGN[0] : SEGN[1]])
            ):
                valid_separation_regions.append(SR)
                SR_idx += 3
                continue

            # SEGN est un stroke avec diacritiques et SEGNN stroke sans diacritiques
            if (
                SEGN[0] != -1
                and (
                    check_stroke(
                        no_dots_copy,
                        no_dots_copy[:, SEGN[0] : SEGN[1]],
                        upper_base,
                        lower_base,
                        SEGN_SR1,
                        SEGN_SR2,
                    )
                    and check_dots(word_img[:, SEGN[0] : SEGN[1]])
                )
                and (
                    (
                        SEGNN[0] != -1
                        and (
                            check_stroke(
                                no_dots_copy,
                                no_dots_copy[:, SEGNN[0] : SEGNN[1]],
                                upper_base,
                                lower_base,
                                SEGNN_SR1,
                                SEGNN_SR2,
                            )
                            and not check_dots(word_img[:, SEGNN[0] : SEGNN[1]])
                        )
                    )
                    or (len(SRL) - 1 - SR_idx == 2)
                    or (len(SRL) - 1 - SR_idx == 3)
                )
            ):
                valid_separation_regions.append(SR)
                SR_idx += 3
                continue

            # SEGN pas un stroke ou stroke avec diacritiques
            if SEGN[0] != -1 and (
                (
                    not check_stroke(
                        no_dots_copy,
                        no_dots_copy[:, SEGN[0] : SEGN[1]],
                        upper_base,
                        lower_base,
                        SEGN_SR1,
                        SEGN_SR2,
                    )
                )
                or (
                    check_stroke(
                        no_dots_copy,
                        no_dots_copy[:, SEGN[0] : SEGN[1]],
                        upper_base,
                        lower_base,
                        SEGN_SR1,
                        SEGN_SR2,
                    )
                    and check_dots(word_img[:, SEGN[0] : SEGN[1]])
                )
            ):
                SR_idx += 1
                continue

            SR_idx += 1
            continue

        if (
            len(valid_separation_regions) == 0
            or len(valid_separation_regions) > 0
            and abs(cut_idx - valid_separation_regions[-1][1]) > 2
        ):
            valid_separation_regions.append(SR)
        SR_idx += 1

    return valid_separation_regions


def extract_char(img, valid_SR):
    
    img = img * 255
    h, w = img.shape

    next_cut = w
    char_imgs = []

    for SR in valid_SR:
        char_imgs.append(img[:, SR[1] : next_cut])
        next_cut = SR[1]
    char_imgs.append(img[:, 0:next_cut])

    return char_imgs


def segment(line, word_img):
    line = cv.bitwise_not(line)
    word_img = cv.bitwise_not(word_img)
    # binary_word = binarize(word_img)
    binary_word = word_img // 255
    no_dots_copy = remove_dots(binary_word)

    # l = binary_word.copy()

    VP_no_dots = my_segm.gray_projection(no_dots_copy, "vertical")
    VP = my_segm.gray_projection(binary_word, "vertical")
    binary_word = fill(binary_word, VP_no_dots)
    no_dots_copy = remove_dots(binary_word)

    # sk = skeletonize(no_dots_copy)
    upper_base, lower_base, MFV = baseline_detection(remove_dots(line))
    MTI = horizontal_transitions(no_dots_copy, upper_base)

    SRL, wrong = cut_points(binary_word, VP, MFV, MTI, upper_base)

    if wrong:
        MTI -= 1
        SRL.clear()
        SRL, wrong = cut_points(binary_word, VP, MFV, MTI, upper_base)

    HP = my_segm.gray_projection(line, "horizontal")
    top_line = -1

    valid = filter_regions(
        binary_word, no_dots_copy, SRL, VP, upper_base, lower_base, MTI, MFV, top_line
    )

    chars = extract_char(binary_word, valid)

    return chars


if __name__ == "__main__":

    original_img, preprocessed_img = pre.preprocess(INPUT_TEST_PATH)

    lines = my_segm.segment(preprocessed_img, "horizontal")
    line = lines[7]
    words = my_segm.segment(line, "vertical", 0, 6)
    word = words[1]

    cr = segment(line, word)

    for i, ch in enumerate(cr):
        cv.imwrite(
            chars_path + "/char_)" + str(i) + ".jpg",
            ch,
        )
