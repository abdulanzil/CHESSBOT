from flask import Flask, render_template
import keyboard
import chess
import webbrowser

flask_app = Flask(__name__)

# indicates whose turn it is
white_timer = True
black_timer = False

# their respective time left
white_time = 903
black_time = 900


# this function asynchronously without stopping
def timer():
    global white_timer, black_timer, white_time, black_time

    while True:
        if white_timer:
            if white_time > 0:
                white_time -= 1
                time.sleep(1)
        else:
            if black_time > 0:
                black_time -= 1
                time.sleep(1)


# returns a string which represents the chessboard after provided moves
def generate_fen(moves):
    board = chess.Board()
    moves = moves.split()
    for move in moves:
        try:
            board.push_uci(move)
        except:
            pass
    fen = board.fen().split()[0]
    return fen


fmove_for_gui = ''
gui_prompt = "WHITE'S MOVE"


@flask_app.route('/')
def home():
    global fmove_for_gui, gui_prompt, white_time, black_time, white_timer

    moves_list = fmove_for_gui.split()
    white_moves = ' '.join(moves_list[i] for i in range(len(moves_list)) if i % 2 == 0)
    black_moves = ' '.join(moves_list[i] for i in range(len(moves_list)) if i % 2 != 0)
    fen_string = generate_fen(fmove_for_gui)
    return render_template('index.html', fen_string=fen_string, white_moves=white_moves, black_moves=black_moves,
                           prompts=gui_prompt, white_time=white_time, black_time=black_time, is_white_timer=str(white_timer))


import copy

# arduino port connection for buzzer listening
import serial
ard = serial.Serial(port="COM3", baudrate=9600, timeout=0.1)


# function for listening the buzzer
def listen_for_buzzer():
    while True:
        data = ard.read()
        data = data.decode('utf-8')
        if data:
            break


import cv2
import numpy as np
from threading import Thread

from ChessBoard import ChessBoard
import time
from stockfish import Stockfish


spacepressed = False
maxchess = ChessBoard()

# Stockfish chess engine setup (change the path according to your need)
engine = Stockfish(path='E:/CHESSBOT/Project/stockfish_win.exe')
engine.set_skill_level(20)


def get():
    engine.stdin.write('isready\n')

    while True:
        text = engine.stdout.readline().strip()
        if text == 'readyok':
            break
        if text != '':
            pass
        if text[0:8] == 'bestmove':
            return text


def sget():
    # using the 'isready' command (engine has to answer 'readyok')
    # to indicate current last line of stdout
    stx = ""
    engine._put('isready\n')

    while True:
        text = engine._read_line()
        if text != '':
            pass
        if text[0:8] == 'bestmove':
            mtext = text
            return mtext


def getboard():
    """ gets a text string from the board """
    btxt = input("\n Enter a board message: ").lower()
    return btxt


def sendboard(stxt):
    """ sends a text string to the board """
    print("\n send to board: " + stxt)


def error_flag():
    global error
    error = 1


def newgame():
    maxchess.resetBoard()
    fmove = ""
    return fmove


def bmove(fmove):
    global botmove, bmessage, white_timer, black_timer, gui_prompt, fmove_for_gui
    fmove = fmove
    brdmove = bmessage[1:5].lower()
    # print('brdmove: ', brdmove)
    image11 = cv2.imread('fpg.png')
    image11 = cv2.resize(image11, (800, 800))

    if maxchess.addTextMove(brdmove) == False:
        etxt = "error" + str(maxchess.getReason()) + brdmove
        sendboard(etxt)
        error_flag()
        print("INVALID MOVE")
        maxchess.printBoard(image11)
        return fmove
    else:
        fmove_for_gui = fmove + ' ' + brdmove
        white_timer = not white_timer
        black_timer = not black_timer
        gui_prompt = "BLACK MOVING..."
        time.sleep(0.25)
        keyboard.press_and_release('ctrl+r')

        fmove = fmove + " " + brdmove
        print('fmove', fmove)
        cmove = "position startpos moves" + fmove
        put(cmove)

        # send move to engine & get engines move        
        put("go movetime " + movetime)

        print("\n Thinking...")
        text = sget()

        smove = text[9:13]
        hint = text[21:25]

        botmove = copy.deepcopy(smove)  # edited
        # print('botmove: ', botmove)

        move = ''
        if maxchess.addTextMove(smove) != True:
            stxt = "e" + str(maxchess.getReason()) + move
            print('second error')
            maxchess.printBoard(image11)
            sendboard(stxt)
        else:
            temp = fmove
            fmove = temp + " " + smove
            stx = smove + hint
            sendboard(stx)

            maxchess.printBoard(image11)
            return fmove


def put(command):
    engine._put(command + '\n')


#################################### IMAGE PROCESSING ##################################


def color_filter(img, condition):
    # returns img with color filtered acc to condition 0 for yellow and codition 1 for white
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if condition == 0:  # yellow
        lower = np.array([20, 100, 100])
        upper = np.array([30, 255, 255])
    elif condition == 1:  # white
        lower = np.array([0, 0, 200])
        upper = np.array([255, 50, 255])
    elif condition == 2:  # green
        upper = np.array([164, 255, 0])
        lower = np.array([0, 70, 0])

    mask = cv2.inRange(hsv, lower, upper)

    if condition == 1:
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 80
        max_area = 800
        filtered_contours = []
        count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # print(area)
            if area > min_area and area < max_area:
                filtered_contours.append(cnt)
                count += 1
        print(count)
        filtered_mask = np.zeros_like(mask)
        cv2.drawContours(filtered_mask, filtered_contours, -1, 255, -1)

        res = cv2.bitwise_and(img, img, mask=filtered_mask)
    else:
        res = cv2.bitwise_and(img, img, mask=mask)

    return res


def center_detect(res):
    # returns an array of centre xy for res

    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    blurred = cv2.GaussianBlur(opening, (5, 5), 0)

    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

    # find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(res, cnts[0], -1, (0, 255, 0), 3)
    # show(res)
    cnts = cnts[0]
    # if imutils.is_cv2() else cnts[1]

    size = 0
    for x in cnts:
        size += 1

    n = 0
    pt = np.zeros((size, 2), dtype=int)
    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)

        if M["m00"] == 0:
            print('Zro division error')
            return pt, size
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # stores loc of point in array
        pt[n, 0] = cX
        pt[n, 1] = cY
        n += 1
    return pt, size


def perspective_transformation(img):
    # returns image within square defined by pt
    global pt
    for x in range(0, 4):
        for y in range(0, 3):
            if pt[y][0] + pt[y][1] > pt[y + 1][0] + pt[y + 1][1]:
                temp = copy.deepcopy(pt[y])
                pt[y] = copy.deepcopy(pt[y + 1])
                pt[y + 1] = copy.deepcopy(temp)
    if pt[1][0] < pt[2][0]:
        temp = copy.deepcopy(pt[1])
        pt[1] = copy.deepcopy(pt[2])
        pt[2] = copy.deepcopy(temp)
    # print pt

    pts1 = np.float32(pt)
    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(img, M, (300, 300))

    return dst


def piece_location(loc, size):
    # returns matrix showing position of white pieces

    flag = 0
    cell = 300 / 8
    position = np.zeros((8, 8), dtype=int)

    for x in range(0, size):  # runs through the number of peices
        flag = 0
        for y in range(0, 8):  # no of rows
            for z in range(0, 8):  # no of columns
                if loc[x][0] < (y + 1) * cell and loc[x][1] < (z + 1) * cell:
                    position[z][y] = 1
                    flag = 1
                    break
            if flag == 1:
                break
    return position


def identify_move(pos1, pos2):
    d = pos2 - pos1

    castlingk = np.zeros_like(d)
    castlingq = np.zeros_like(d)

    castlingk[0] = [-1, 1, 1, -1, 0, 0, 0, 0]
    castlingq[0] = [0, 0, 0, -1, 1, 1, 0, -1]

    if np.all(d == castlingk):
        return 'e1g1'
    if np.all(d == castlingq):
        return 'e1c1'

    pick = None
    place = None

    for x in range(0, 8):
        for y in range(0, 8):
            if d[x][y] == 1:
                place = [x, y]
            elif d[x][y] == -1:
                pick = [x, y]

        if pick and place:
            break
        else:
            pass

    if not pick and not place:
        return 'invmove'

    try:
        cell1 = chr(104 - pick[1]) + str(1 + pick[0])
        cell2 = chr(104 - place[1]) + str(1 + place[0])

        move = cell1 + cell2
        return move
    except:
        print('SOMETHING FISHY, MAY BE NOT DETECTING ANY WHITE')
        return 'invmove'


def show(img):
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def setup():

    while True:
        s, frame = cap.read()
        #cv2.rectangle(frame, (78, 240), (333, 492), (0, 255, 0), 3)
        cv2.imshow('fr', frame)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    corners = color_filter(frame, 0)
    show(corners)
    pt,s = center_detect(corners)

    #pt = np.array([[333, 492], [78, 492], [333, 240], [78, 240]])
    print(pt)
    return pt, frame


def testPosition1():
    ret, frame = cap.read()

    t = [frame, frame, frame]
    pos = np.zeros((8, 8), dtype=int)
    pos1 = [pos, pos, pos]
    while True:
        trial = 0

        while trial < 3:
            trans = perspective_transformation(t[trial])
            pieces = color_filter(trans, 1)
            board1, s1 = center_detect(pieces)
            pos1[trial] = piece_location(board1, s1)
            time.sleep(1)
            trial += 1
            print("trial", trial)
        print('\n')

        if np.all(pos1[0] == pos1[1]) and np.all(pos1[1] == pos1[2]) and np.all(pos1[0] == pos1[2]):
            return pos1[0], s1


def Position2(pos1, s1):
    global gui_prompt
    while True:
        s, frame = cap.read()

        t = [frame, frame, frame]
        pos2 = [pos1, pos1, pos1]

        trial = 0

        while trial < 3:
            t[trial] = frame
            trans = perspective_transformation(t[trial])
            pieces = color_filter(trans, 1)
            board2, s2 = center_detect(pieces)
            pos2[trial] = piece_location(board2, s2)
            time.sleep(0.5)
            trial += 1
            print("trial", trial)

        if np.all(pos2[0] == pos2[1]) and np.all(pos2[1] == pos2[2]) and np.all(pos2[0] == pos2[2]):
            if np.any(pos2[0] != pos1):
                return pos2[0]
            else:
                gui_prompt = 'No moves identified. Make a move!'
                listen_for_buzzer()
                time.sleep(0.5)
                keyboard.press_and_release('ctrl+r')
                # input('NO MOVES ARE IDENTIFIED, make a move and press enter : ')


def main1():
    global error, pt, bmessage, movetime, botmove, imh, cap, frame, pos1, pos2, s1, s2, fmove, fmove_for_gui, gui_prompt, white_time, white_timer, black_timer, black_time

    movetime = '4000'

    fmove = newgame()
    cap = cv2.VideoCapture(0)

    pt, frame = setup()
    print(pt)

    timer_thread = Thread(target=timer)
    timer_thread.start()

    webbrowser.open('http://127.0.0.1:5000')
    time.sleep(1)
    keyboard.press_and_release('f11')

    while True:
        game_result = maxchess._game_result

        if white_time == 0:
            gui_prompt = '- YOU LOSE -'
            keyboard.press_and_release('ctrl+r')
            time.sleep(2)
            break
        elif black_time == 0:
            gui_prompt = '- YOU WIN -'
            keyboard.press_and_release('ctrl+r')
            time.sleep(2)
            break

        if game_result != 0:
            # print('\n-GAme ovER-')
            if game_result == 1:
                gui_prompt = '- YOU WIN -'
                keyboard.press_and_release('ctrl+r')
                break
            elif game_result == 2:
                gui_prompt = '- YOU LOSE -'
                keyboard.press_and_release('ctrl+r')
                break
            elif game_result == 3:
                gui_prompt = '- DRAW -'
                keyboard.press_and_release('ctrl+r')
                break

        pos1, s1 = testPosition1()

        print('pos1 found')
        print(pos1)

        s1 = np.count_nonzero(pos1)

        while True:
            # input('move WHITE & enter to continue : ')
            listen_for_buzzer()

            error = 0

            pos2 = Position2(pos1, s1)

            print('pos2found')
            print(pos2)

            print('Move identified!')

            mymove = identify_move(pos1, pos2)
            print(mymove)

            bmessage = 'm' + mymove

            fmove = fmove

            fmove = bmove(fmove)

            if maxchess._game_result != 0:
                break

            if error == 0:
                #fmove_for_gui = fmove + ' ' + mymove
                #white_timer = not white_timer
                #black_timer = not black_timer
                #gui_prompt = "BLACK MOVING..."
                #keyboard.press_and_release('ctrl+r')
                break

            gui_prompt = 'INVALID!! MAKE A VALID MOVE.'
            keyboard.press_and_release('ctrl+r')

        print('BLACK MOVE : ' + botmove)

        # input('move BLACK and enter : ')

        fmove_for_gui = fmove
        gui_prompt = "BLACK'S TURN"
        time.sleep(0.25)
        keyboard.press_and_release('ctrl+r')

        listen_for_buzzer()

        white_timer = not white_timer
        black_timer = not black_timer
        gui_prompt = "WHITE'S TURN"
        keyboard.press_and_release('ctrl+r')


# main1 function execution on a thread
thr1 = Thread(target=main1)
thr1.start()

# flask app for showing the GUI in browser
flask_app.run(use_reloader=False)
