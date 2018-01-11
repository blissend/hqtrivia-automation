import os
import sys
import pprint as pp # debug purposes
import time # time how long each section takes
import multiprocessing as mp # To lookup info online faster
import argparse

# Lookup word information
from vocabulary.vocabulary import Vocabulary # online dictionary
import nltk # local dictionary
import wikipediaapi # for more advance definitions
import urllib # google search
from bs4 import BeautifulSoup # google search
import requests # google search
import webbrowser # google search

# Capture source image
import cv2 # for webcam usage
from Foundation import * # For osascript crap (applescript)

# Google Tesseract OCR
from PIL import Image, ImageEnhance # photo manipulation
import pytesseract # bindings
import numpy as np # more advance photo manipulation

# Google Vision OCR
from google.cloud import vision
from google.cloud.vision import types
import io

os.environ['NO_PROXY'] = '*' # https://bugs.python.org/issue30385
VERSION = "2018.01.11.04.30"

"""
I vaguely wondered about the HQ trivia game and automating to get an edge in
the game since there is nothing stopping you from googling for answers. Why
not automate it in effort to learn? Well here's my attempt...
"""

class HQTrivia():
    """
    A simple test to see if one can automate determining the best answer
    """

    def __init__(self):

        # This determines source location on where to caputer picture
        # QuickTime - MacOS has record feature for phone (best)
        # WebCam - Use OpenCV to capture photo (untested)
        self.use_quicktime = False
        self.use_webcam = False
        self.use_input = False

        # The filename of picture (no extension means we're capturing image)
        self.picture = 'source'

        # Default location of where to work on self.picture
        self.location = os.getcwd()

        # Replace with your own auth file name
        self.google_auth_json = 'blissend.json'

        # Default the language for wikipedia searches
        self.wiki = wikipediaapi.Wikipedia('en')
        self.vb = Vocabulary()

        # The OCR text
        self.raw = ''

        # The information we'll be working with and passing around a lot
        self.question = ''
        self.question_nouns = ''
        self.answers = {}
        self.lookup_info = {}

        # For debugging
        self.times = {}
        self.verbose = False

    def debug(self, msg):
        # In multiprocessing environments, the below statement helps
        sys.stdout.flush()

        print("hqtrivia-automation.py: " + str(msg))

    def capture(self, ftype='tiff'):
        """
        Simple function to select function to capture picture
        """

        if self.verbose:
            pre = "method - capture | "
            self.debug(pre + "choosing how to capture...")

        if self.use_input:
            if self.verbose:
                self.debug(pre + "input provided, don't capture")
            return

        # Set file type
        self.picture += '.' + ftype

        if self.use_quicktime:
            if self.verbose:
                self.debug(pre + "quicktime")
            self.scan_quicktime(ftype)
        elif self.use_webcam:
            if self.verbose:
                self.debug(pre + "webcam")
            self.scan_webcam()

    def scan_quicktime(self, ftype='tiff'):
        """
        Takes screenshot of phone screen via AppleScript

        To use this open QuickTime player and do a movie recording. Select the
        drop down arrow next to record button and select your iPhone. This
        requires a wire connection to your computer using QuickTime. Remember,
        don't record anything. Having it show on screen is enough for a
        screencapture!

        1. Get window ID of QuickTime Player
        2. Tell script to run shell script command screencapture the window ID
        """

        if self.verbose:
            self.debug("method - quicktime | starting")
            start = time.time()

        full_path = os.path.join(self.location, self.picture)
        script = """tell application "QuickTime Player"
set winID to id of window 1
end tell
do shell script "screencapture -x -t tiff -l " & winID &"""
        script += ' " ' + full_path + '"'
        script = script.replace('tiff', ftype)

        s = NSAppleScript.alloc().initWithSource_(script)
        s.executeAndReturnError_(None)

        if self.verbose:
            diff = time.time() - start
            self.debug("method - quicktime | elapsed {!s}".format(diff))
            self.times['scan_quicktime'] = diff

    def scan_webcam(self):
        """
        Takes screenshot using webcam.

        This is untested but here just in case it's needed. You need to figure
        out which camera to capture which unfortnately appears to be a discovery
        process of entering in numbers from 0 to higher until found. Also note,
        not all cameras have good controls and autofocus sucks for this.
        """

        if self.verbose:
            pre = "method - webcam | "
            self.debug(pre + "starting")
            start = time.time()

        video = cv2.VideoCapture(1) # cam id (try from 0 and higher til found)
        video.set(cv2.CAP_PROP_AUTOFOCUS, 0) # turn the autofocus off
        video.set(3, 1920)
        video.set(4, 1080)
        cv2.namedWindow("HQ OCR Camera")
        #img_counter = 0
        while True:
            ret, frame = video.read()
            cv2.imshow("HQ OCR Camera", frame)
            if not ret:
                break
            k = cv2.waitKey(1)
            if k%256 == 27: # ESC pressed
                print("Escape hit, closing...")
                break
            elif k%256 == 32: # SPACE pressed
                img_name = self.picture # format with counter for multiple pics
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))
                #img_counter += 1
                break
        video.release()
        cv2.destroyAllWindows()

        if self.verbose:
            diff = time.time() - start
            self.debug(pre + "elapsed {!s}".format(diff))
            self.times["scan_webcam"] = diff

    def enhance(self):
        """
        Edit image readability for Tesseract because it's suuuuuuuuuuuuuper...

        picky!!!

        1. Eliminate background on buttons (answers)
        2. Turn to grayscale
        3. Make image BIG because Google OCR likes it big ;)
        4. Reduce the grayscale (eliminates button borders in good pictures)
        5. Make anything not white, black because google can't see color -_-
        """

        if self.verbose:
            pre = "method - enhance | "
            self.debug(pre + "starting")
            start = time.time()

        # Replace buttons (answers) background color, incease size scale/DPI
        im = Image.open(self.picture)
        im = im.convert("RGBA")
        data = np.array(im)
        red, green, blue, alpha = data.T # Unpack the bands for readability
        gray_buttons = (red > 225) & (green > 225) & (blue > 225)
        data[..., :-1][gray_buttons.T] = (255, 255, 255)
        im = Image.fromarray(data)
        width, height = im.size
        # New file since we're going to edit it
        file = self.picture.split('.')
        self.picture = "source_edited." + file[len(file)-1]
        im.crop((0, 300, width, height-400)).save(self.picture)
        #im.resize((round(width*3), round(height*3))).save(
            #self.picture, dpi=(600,600))

        # Make grayscale
        im = Image.open(self.picture)
        im = im.convert('RGBA')
        im = im.convert('L').save(self.picture)
        #exit()

        # Reduce the grayscale
        #im = Image.open(self.picture)
        #im = im.convert('RGBA')
        #data = np.array(im)
        #red, green, blue, alpha = data.T # Unpack the bands for readability
        #gray_triming = (red > 158) & (green > 158) & (blue > 158)
        #data[..., :-1][gray_triming.T] = (255, 255, 255)
        #Image.fromarray(data).save(self.picture)
        #exit()

        # Replace non white with black
        im = Image.open(self.picture)
        im = im.convert('RGBA')
        data = np.array(im)
        red, green, blue, alpha = data.T # Unpack the bands for readability
        non_white = (red < 255) & (green < 255) & (blue < 255)
        data[..., :-1][non_white.T] = (0, 0, 0)
        im = Image.fromarray(data)
        width, height = im.size
        im.resize((round(width*3), round(height*3))).save(self.picture)
        #exit()

        if self.verbose:
            diff = time.time() - start
            self.debug(pre + "elapsed {!s}".format(diff))
            self.times["enhance"] = diff

    def ocr_vision(self, queue):
        """
        Google Cloud Vision

        The better OCR tool out there but requires additional setup. It is
        free under limitations.
        """

        if self.verbose:
            pre = "method - ocr_vision | "
            start = time.time()
            self.debug(pre + "starting")

        # See if we have an auth file, if not return
        try:
            file_path = os.path.join(self.location, self.google_auth_json)
            if not os.path.isfile(file_path):
                if self.verbose:
                    self.debug(pre + "no auth file")
                queue.put("END")
                return
        except:
            if self.verbose:
                self.debug(pre + "no auth file")
            queue.put("END")
            return

        # Credentials
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = file_path

        # Instantiates a client
        client = vision.ImageAnnotatorClient() # spits out shit, don't know why

        # The image file
        if not os.path.isfile(self.picture):
            full_path = os.path.join(self.location, self.picture)
        else:
            full_path = self.picture

        # Loads the image into memory
        with io.open(full_path, 'rb') as image_file:
            content = image_file.read()

        image = types.Image(content=content)

        # Performs text detection on the image file
        response = client.text_detection(image=image)
        text = response.text_annotations

        for t in text:
            self.raw = t.description
            break

        # Clean up text
        self.raw = self.raw.split('\n')
        self.debug(pre + "raw - " + str(self.raw))
        index = 0
        while index < len(self.raw):
            value = self.raw[index].lower()
            if len(value) < 10:
                self.raw.pop(index)
                #self.debug("method - ocr | delete [" + value + "]")
            else:
                index += 1
        self.raw.pop(-1) # swipe left comment

        # Return data to parent process
        queue.put(self.raw)

        if self.verbose:
            self.debug(pre + "raw - cleaned" + str(self.raw))
            diff = time.time() - start
            self.debug(pre + "elapsed {!s}".format(diff))

        queue.put("END")

    def ocr_tesseract(self, queue):
        """
        Google Tesseract OCR

        Finally read the image text if possible
        """

        # Include the below line, if you don't have tesseract in your PATH
        # Example tesseract_cmd: '/usr/local/bin/tesseract'
        #pytesseract.pytesseract.tesseract_cmd = '<fullpath_to_tesseract>'

        if self.verbose:
            pre = "method - ocr_tesseract | "
            self.debug(pre + "starting")
            start = time.time()

        # Enhance image first since tesseract doesn't do it
        self.enhance()

        # Get text from image (OCR)
        self.raw = pytesseract.image_to_string(
            Image.open(self.picture), config="-psm 11")

        # Clean it up
        self.raw = self.raw.split('\n')

        if self.verbose:
            self.debug(pre + "raw = " + str(self.raw))

        index = 0
        while index < len(self.raw):
            value = self.raw[index].lower()
            if len(value) < 1:
                self.raw.pop(index)
                #self.debug("method - ocr | delete [" + value + "]")
            else:
                index += 1

        # Return the data to main parent process
        queue.put([self.picture, self.raw])

        if self.verbose:
            self.debug(pre + "raw - cleaned = " + str(self.raw))
            diff = time.time() - start
            self.debug(pre + "elapsed {!s}".format(diff))

        queue.put("END")

    def parse(self):
        """
        Parse the raw OCR text to find the question and answers in it.

        This is tricky because the OCR text won't always be the same. So
        adjustments may have to be tweaked here.
        """

        if self.verbose:
            pre = "method - parse | "
            self.debug(pre + "starting")
            start = time.time()

        # Parse text into question and answer variables
        check_q = True
        counter = 1 # for counting answers
        for line in self.raw:
            #print(len(line), end=' ['+line+']\n')
            if check_q:
                if len(line) > 2:
                    if '?' not in line:
                        self.question += line + ' '
                    else:
                        self.question += line
                        check_q = False
            else:
                if 'Swipe left' not in line:
                    if len(line) > 0 and line != '-':
                        ans = line
                        self.answers[ans] = {
                            "answer": ans,
                            "keywords": [],
                            "score": 0,
                            "index": str(counter)
                        }
                        self.lookup_info[ans] = []
                        counter += 1
                else:
                    break

        # Check parsed results
        if '?' not in self.question:
            self.debug(pre + "Could not find question!")
            raise
        if len(self.answers) < 1:
            self.debug(pre + "Could not find answers!")
            raise
        elif len(self.answers) > 3:
            self.debug(pre + "Found more than three answers!")
            raise

        # Get the nouns out of the question
        for q in nltk.pos_tag(nltk.word_tokenize(self.question)):
            if q[1] == 'NN' or q[1] == 'NNP':
                self.question_nouns += " " + q[0]
        self.question_nouns = self.question_nouns.strip().split(' ')

        if self.verbose:
            self.debug(pre + "question = " + str(self.question))
            self.debug(
                pre + "nouns in question - {!s}".format(self.question_nouns))
            self.debug(pre + "answer = " + str(self.answers))
            diff = time.time() - start
            self.debug(pre + "elapsed {!s}".format(diff))
            self.times["parse"] = diff

    def keywords(self, words):
        """
        Simple function to find words in a string that are also in question
        and then return those keywords found.
        """

        keywords = []
        for w in words:
            if len(w) > 2:
                if w in self.question_nouns:
                    if w not in keywords:
                        keywords.append(w)

        return keywords

    def lookup_wiki(self, queue):
        """
        Gets wikipedia information about answer

        Multiprocess so needs to return results to parent
        """

        if self.verbose:
            pre = "method - lookup_wiki | "
            self.debug(pre + "starting")
            start = time.time()

        # Loop through answers and search wikipedia
        for index, ans in self.answers.items():
            l_info = self.lookup_info[ans['answer']]

            try:
                page = self.wiki.page(ans['answer'])
                if page.exists():

                    try:
                        words = []
                        for i in page.sections:
                            words += i.text.split(' ')
                    except:
                        self.debug(
                            pre + "issue with wikipedia for {!s}"
                            .format(ans['answer']))
                    else:
                        l_info.append("[Wikipedia]: " + page.summary)
                        queue.put([ans['answer'], self.keywords(words), l_info])

                else:

                    a = ans['answer'].split(' ')
                    if len(a) < 2:

                        # Could not find page, so throw exception and move on
                        self.debug(
                            pre + "no results for {!s} in wikipedia... ".
                            format(ans['asnwer']))
                        raise

                    else:

                        # Try searching each word in answer as last resort
                        for w in a:
                            if len(w) > 3:
                                page = self.wiki.page(w)
                                if page.exists():
                                    try:
                                        words = []
                                        for i in page.sections:
                                            words += i.text.split(' ')
                                    except:
                                        self.debug(
                                            pre +
                                            "issue with wikipedia for {!s}"
                                            .format(ans['answer']))
                                    else:
                                        l_info.append(
                                            "[Wikipedia {!s}]: ".format(w) +
                                            page.summary)
                                        queue.put([
                                            ans['answer'],
                                            self.keywords(words),
                                            l_info])


            except:
                self.debug(
                    pre + "issue with wikipedia for {!s}... "
                    .format(ans['answer']))
                self.debug(sys.exc_info())

        queue.put("END")
        if self.verbose:
            self.debug(pre + "elapsed " + str(time.time() - start))

    def lookup_dict_and_syn(self, queue):
        """
        Use nltk to look up word info, if failure, use vocabulary (online)
        """

        if self.verbose:
            pre = "method - lookup_dict_and_syn | "
            self.debug(pre + "starting")
            start = time.time()

        # Get dictionary/synonyms
        for index, ans in self.answers.items():
            l_info = self.lookup_info[ans['answer']]
            a = ans['answer'].split(' ') # incase of multi word answers

            for w in a:
                # Don't look up small words (waste of time)
                if len(w) > 3:

                    # definition
                    define = nltk.corpus.wordnet.synsets(w)
                    synset_found = False
                    if len(define) < 1:
                        # Local dictionary didn't find anything so search online
                        if self.verbose:
                            self.debug(
                                pre + "nltk none for {!s}, using vocabulary"
                                .format(w))
                        try:
                            define = self.vb.meaning(w, format='list')
                            if define != False:
                                # Multiple definitions possible
                                for d in define:
                                    l_info.append(
                                        "[Meaning {!s}]: ".format(w) + d)
                                    queue.put([
                                        ans['answer'],
                                        self.keywords(d),
                                        l_info])
                        except:
                            self.debug(
                                pre + "issue with vocabulary for {!s}... "
                                .format(w))
                            self.debug(sys.exc_info())
                    else:
                        synset_found = True
                        l_info.append(
                            "[Meaning {!s}]: ".format(w) +
                            define[0].definition())
                        queue.put([
                            ans['answer'],
                            self.keywords(define[0].definition()),
                            l_info])

                    # Synonyms
                    if synset_found:
                        synonyms = [l.name() for s in define for l in s.lemmas()]

                        # Remove duplicates nltk adds
                        s = []
                        i = 0
                        while i < len(synonyms):
                            if synonyms[i] in s:
                                synonyms.pop(i)
                            else:
                                s.append(synonyms[i])
                                i += 1
                        syn = ', '.join(s)
                        l_info.append("[Synonyms {!s}]: ".format(w) + syn)
                        queue.put([ans['answer'], self.keywords(syn), l_info])
                    else:
                        # Local dictionary didn't find anything so search online
                        self.debug(
                            pre + "nltk has nothing for {!s}, using vocabulary"
                            .format(w))
                        try:
                            synonyms = self.vb.synonym(w, format='list')
                            if synonyms != False:
                                l_info.append(
                                    "[Synonyms {!s}]: ".format(w) +
                                    str(synonyms))
                                queue.put([
                                    ans['answer'],
                                    self.keywords(str(synonyms)),
                                    l_info])
                        except:
                            self.debug(
                                pre + "issue with vocabulary for {!s}... "
                                .format(w))
                            self.debug(sys.exc_info())

        queue.put("END")
        if self.verbose:
            self.debug(
                pre + "elapsed " + str(time.time() - start))

    def lookup_google_search(self, queue):
        """
        Does a google search for each answer and finds if words in results are
        found in question
        """

        if self.verbose:
            pre = "method - lookup_google_search | "
            self.debug(pre + "starting")
            start = time.time()

        # Google search
        for index, ans in self.answers.items():
            l_info = self.lookup_info[ans['answer']]
            try:
                text = urllib.parse.quote_plus(ans['answer'])
                url = 'https://google.com/search?q=' + text
                response = requests.get(url, timeout=2)
                soup = BeautifulSoup(response.text, 'lxml')
                results = ''
                for g in soup.find_all(class_='st'):
                    results += " " + g.text
                cleaned_results = results.strip().replace('\n','')
                l_info.append("[Google]: " + cleaned_results)
                queue.put([
                    ans['answer'],
                    self.keywords(cleaned_results),
                    l_info])
            except:
                self.debug(
                    pre + "issue with google search for {!s}... "
                    .format(ans['answer']))
                self.debug(sys.exc_info())

        if self.verbose:
            self.debug(
                pre + "google search elapsed " + str(time.time() - start))

    def display(self):
        # Clear the screen
        os.system('cls' if os.name == 'nt' else 'clear')

        # Text to output to screen
        output = []

        # Question
        output.append('\n\nQuestion - ' + self.question + '\n')

        # Answers & Lookup Info
        choice = {'index': [], 'score': 0, 'l_info': []}
        for a, ans in self.answers.items():
            if ans['score'] == choice['score']:
                choice['index'].append(a)
            if 'NOT' in self.question:
                if ans['score'] < choice['score']:
                    choice['index'] = [a]
                    choice['score'] = ans['score']
            else:
                if ans['score'] > choice['score']:
                    choice['index'] = [a]
                    choice['score'] = ans['score']
            output.append(
                "Choice - " + ans['answer'] +
                ' - Score ' + str(ans['score']))
            for l_info in self.lookup_info[ans['answer']]:
                for l in l_info:
                    l_index = l.split(':')[0]
                    if l_index not in choice['l_info']:
                        choice['l_info'].append(l_index)
                        if len(l) > 140:
                            output.append(l[0:140])
                        else:
                            output.append(l)
            output.append("[Keywords]: " + str(ans['keywords']))
            output.append("")

        # Highest scoring answer
        if len(choice['index']) > 0:
            choose = []
            for i in choice['index']:
                choose.append(self.answers[i]['answer'])
            msg = "Answer - " + ', '.join(choose)
            if 'NOT' in self.question:
                msg += (" - NOT keyword so lowest score is " +
                        str(choice['score']))
            else:
                msg += (" - highest score is " + str(choice['score']))
            output.append(msg)
        else:
            output.append("Answer - Unknown")
        output.append("")
        output.insert(1, msg + '\n')

        # Finally print it all
        for line in output:
            print(line)

if __name__ == '__main__':
    start = time.time()

    # Setup command line options
    parser = argparse.ArgumentParser(
        description='Automate searching for answers in HQ Trivia')
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-q', '--quicktime',
        action='store_true', default=False,
        help="Use quicktime to capture source image"
    )
    group.add_argument(
        '-w', '--webcam',
        action='store_true', default=False,
        help="Use webcam to capture source image"
    )
    group.add_argument(
        '-i', '--input',
        action='store',
        help="Use image provided instead of capturing"
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true', default=False,
        help="Spit out debug information"
    )
    parser.add_argument(
        '-V', '--version',
        action='store_true', default=False,
        help="Version of script"
    )
    options = parser.parse_args()

    # Configure class with command option
    hq = HQTrivia()
    hq.verbose = options.verbose
    if options.version:
        hq.debug("version - " + VERSION)
    if options.quicktime:
        hq.use_quicktime = options.quicktime
        hq.use_webcam = False
        hq.use_input = False
    elif options.webcam:
        hq.use_quicktime = False
        hq.use_webcam = options.webcam
        hq.use_input = False
    elif options.input:
        if len(options.input) > 0:
            hq.use_quicktime = False
            hq.use_webcam = False
            hq.use_input = True
            hq.picture = options.input
        else:
            exit()
    else:
        exit()
    if options.verbose:
        hq.verbose = options.verbose

    # Uncomment and use your own custom settings
    #hq.location = '/Full/path/to' # Defaults to script location (leave alone)
    #hq.google_auth_json = '<your_json>.json' (if you're using google vision)

    # Capture image first
    hq.capture()

    # Read the picture (use multiprocessing for multiple OCR readers)
    updated = {'tesseract': 0, 'vision': 0}
    vision_raw = ''
    tesseract_raw = ''
    edited_pic = ''
    q_vision = mp.Queue()
    q_tess = mp.Queue()
    p_vision = mp.Process(target=hq.ocr_vision, args=(q_vision,))
    p_tess = mp.Process(target=hq.ocr_tesseract, args=(q_tess,))
    p_vision.daemon = True
    p_tess.daemon = True
    start_ocr = time.time()
    p_vision.start()
    p_tess.start()
    while True:

        if not q_vision.empty():
            data = q_vision.get()
            if data != "END":
                vision_raw = data
                updated['vision'] = 1
                hq.times['ocr_vision'] = time.time() - start_ocr
            else:
                hq.times['ocr_vision'] = time.time() - start_ocr
                updated['vision'] = 2

        if not q_tess.empty():
            data = q_tess.get()
            if data != "END":
                edited_pic = data[0]
                tesseract_raw = data[1]
                updated['tesseract'] = 1
                hq.times['ocr_vision'] = time.time() - start_ocr
            else:
                hq.times['ocr_tesseract'] = time.time() - start_ocr

        # Make sure it doesn't take too long
        if ((int(time.time() - start_ocr) > 10) or
            (updated['tesseract'] > 0 and updated['vision'] > 0) or
            (updated['vision'] == 1)):
            break

    # Choose vision text over tesseract since it's better
    print(tesseract_raw)
    print(vision_raw)
    if len(vision_raw) > 0:
        hq.raw = vision_raw
        hq.debug("Using Google Vision OCR")
    elif len(tesseract_raw) > 0:
        hq.raw = tesseract_raw
        hq.picture = edited_pic
        hq.debug("Using Google Tesseract OCR")
    else:
        hq.debug("COULD NOT FIND TEXT!")
        exit()
    diff = time.time() - start_ocr
    hq.debug("OCR | elapsed {!s}".format(diff))
    hq.times["ocr"] = diff

    # Parse the picture text
    hq.parse()

    # Get information about answers (time consuming so do multiprocessing)
    q_wiki = mp.Queue()
    q_dict = mp.Queue()
    q_gsearch = mp.Queue()
    # Queue returns list of [answer_text, keyword_list, lookup_info]
    p_wiki = mp.Process(target=hq.lookup_wiki, args=(q_wiki,))
    p_dict = mp.Process(target=hq.lookup_dict_and_syn, args=(q_dict,))
    p_gsearch = mp.Process(target=hq.lookup_google_search, args=(q_gsearch,))
    p_wiki.daemon = True
    p_dict.daemon = True
    p_gsearch.deamon = True
    start_lookup = time.time()
    p_wiki.start()
    p_dict.start()
    p_gsearch.start()
    while True:

        # Thread counts finished
        count_cur = 0
        count_max = 3

        def update_display(data):
            ans = data[0]
            keys = data[1]
            l_info = data[2]
            for k in keys:
                if k not in hq.answers[ans]['keywords']:
                    hq.answers[ans]['keywords'].append(k)
            hq.answers[ans]['score'] = len(hq.answers[ans]['keywords'])
            hq.lookup_info[ans].append(l_info)
            hq.display()

        if not q_wiki.empty():
            data = q_wiki.get()
            if data != "END":
                update_display(data)
            else:
                hq.times['lookup_wiki'] = time.time() - start_lookup
                count_cur += 1

        if not q_gsearch.empty():
            data = q_gsearch.get()
            if data != "END":
                update_display(data)
            else:
                hq.times['lookup_google_search'] = time.time() - start_lookup
                count_cur += 1

        if not q_dict.empty():
            data = q_dict.get()
            if data != "END":
                update_display(data)
            else:
                hq.times['lookup_dict_and_syn'] = time.time() - start_lookup
                count_cur += 1

        # Make sure it doesn't take too long
        if int(time.time() - start_lookup) > 10 or count_cur == count_max:
            break

    # Display the final results!
    diff = time.time() - start_lookup
    hq.debug("Lookups elapsed {!s}".format(diff))
    hq.times['lookups'] = diff
    hq.display()

    # Show total times for everything
    diff = time.time() - start
    hq.times['total'] = diff
    if hq.verbose:
        hq.debug("Time")
        pp.pprint(hq.times)